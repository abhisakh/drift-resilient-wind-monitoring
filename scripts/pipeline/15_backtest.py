#!/usr/bin/env python
"""The evaluation suite: every claim this project makes, measured.

Runs the eight tests of `src/windmill/backtest.py` over the four ENGIE
turbines and writes `results/backtest.json`. Nothing here is a claim --
it is the evidence for the claims, and it is designed so that a result
which does NOT support the project is just as easy to get as one that
does.

    python scripts/pipeline/15_backtest.py            # the whole suite
    python scripts/pipeline/15_backtest.py --quick    # fewer windows

The protocol, in one place, so that every number in the output is
comparable with every other:

  * the split. Each test week is predicted by a reference fitted on the
    thirty days immediately before it, and `Window` refuses to be built
    otherwise.
  * the matched threshold. Every arm's alarm threshold is calibrated on its
    own out-of-fold training residuals at 1 % of rows, so every arm is
    judged at the same false-alarm rate rather than at one shared number
    on one scale.
  * one-sided. Only a positive residual alarms: overheating is the failure
    mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import backtest as bt  # noqa: E402
from windmill import conditioning, live, loaders, multivariate  # noqa: E402
from windmill import RATED_POWER_KW  # noqa: E402

#: The channel the case study is about.
CHANNEL = "Gb2t_avg"

#: The features the learned arm uses -- the project's own NBM feature set.
LGBM_FEATURES = ["P_norm", "Ws_avg", "era5_temp_2m"]

#: The forecast error levels the chain test is run at: ERA5 plus noise.
#: The levels are the ones a day-ahead forecast is realistically wrong by.
FORECAST_ERRORS = {
    "perfect": (0.0, 0.0),
    "good (+-0.5 m/s, +-1.0 K)": (0.5, 1.0),
    "typical (+-1.0 m/s, +-1.5 K)": (1.0, 1.5),
    "poor (+-2.0 m/s, +-2.5 K)": (2.0, 2.5),
}


def load_turbine(data_dir: Path, turbine: str) -> pd.DataFrame:
    """One turbine's SCADA, with ERA5 ambient joined on and power normalised."""
    df = loaders.load_engie(data_dir, variant="F99", turbines=(turbine,))
    era5 = loaders.load_era5(data_dir)
    e = era5.copy()
    e["ts"] = e["ts"].dt.floor("10min")
    e = (e.set_index("ts")[["era5_temp_2m"]]
         .resample("10min").ffill().reset_index())
    df = df.merge(e, on="ts", how="left")
    df["P_norm"] = (df["P_avg"] / RATED_POWER_KW).clip(0, 1)
    df = df.dropna(subset=["P_norm", CHANNEL, "era5_temp_2m"])
    return df.sort_values("ts").reset_index(drop=True)


def arm_factories() -> dict:
    """The five arms of the weekly backtest.

    They differ ONLY in what they condition on, so the difference between
    them is the thing being measured.
    """
    return {
        "constant": lambda: bt.ConstantArm(CHANNEL, "P_norm"),
        "load": lambda: bt.DeterministicArm(CHANNEL, "P_norm", "era5_temp_2m",
                                            use_ambient=False, use_wind=False,
                                            name="load"),
        "load+ambient": lambda: bt.DeterministicArm(CHANNEL, "P_norm",
                                                     "era5_temp_2m"),
        "load+ambient+wind": lambda: bt.DeterministicArm(
            CHANNEL, "P_norm", "era5_temp_2m", wind_col="Ws_avg",
            use_ambient=True, use_wind=True),
        "lightgbm": lambda: bt.LightGBMArm(CHANNEL, LGBM_FEATURES),
    }


def run_weekly_sweep(data_dir: Path, turbines: list[str], quick: bool) -> dict:
    """A.1 and A.2: the weekly backtest and the weather ablation."""
    out: dict = {"per_turbine": {}, "pooled": {}}
    pooled: list[bt.ArmResult] = []
    for turbine in turbines:
        t0 = time.time()
        df = load_turbine(data_dir, turbine)
        windows = bt.weekly_windows(df["ts"])
        if quick:
            windows = windows[::4]
        results = bt.run_weekly_backtest(df, CHANNEL, "P_norm", arm_factories(),
                                         windows=windows)
        pooled.extend(results)
        out["per_turbine"][turbine] = {
            "windows": len(windows),
            "rows": int(len(df)),
            "seconds": round(time.time() - t0, 2),
            "summary": bt.summarise(results),
            "paired": {
                "load+ambient vs constant": bt.paired_delta(
                    results, "load+ambient", "constant"),
                "load+ambient vs load": bt.paired_delta(
                    results, "load+ambient", "load"),
                "load+ambient vs lightgbm": bt.paired_delta(
                    results, "load+ambient", "lightgbm"),
                "load+ambient+wind vs load+ambient": bt.paired_delta(
                    results, "load+ambient+wind", "load+ambient"),
                "load vs constant": bt.paired_delta(results, "load", "constant"),
            },
        }
        print(f"  {turbine}: {len(windows)} windows, "
              f"{len(results)} arm-results, {time.time()-t0:.1f}s")
    out["pooled"] = {
        "summary": bt.summarise(pooled),
        "paired": {
            "load+ambient vs constant": bt.paired_delta(
                pooled, "load+ambient", "constant"),
            "load+ambient vs load": bt.paired_delta(
                pooled, "load+ambient", "load"),
            "load+ambient vs lightgbm": bt.paired_delta(
                pooled, "load+ambient", "lightgbm"),
            "load+ambient+wind vs load+ambient": bt.paired_delta(
                pooled, "load+ambient+wind", "load+ambient"),
            "load vs constant": bt.paired_delta(pooled, "load", "constant"),
        },
        "arm_results": [r.as_dict() for r in pooled],
    }
    return out


def run_forecast_chain(data_dir: Path, turbine: str, n_windows: int) -> dict:
    """A.3: what does the forecast cost, end to end?"""
    df = load_turbine(data_dir, turbine)
    windows = bt.weekly_windows(df["ts"])
    step = max(1, len(windows) // n_windows)
    sample = windows[::step][:n_windows]
    rng = np.random.default_rng(0)
    out: list[dict] = []
    for window in sample:
        train, test = bt.slice_window(df, window)
        if len(train) < bt.MIN_TRAIN_ROWS or len(test) < bt.MIN_TEST_ROWS:
            continue
        curve = conditioning.fit_power_curve(train, "Ws_avg", "P_avg")
        wind = test["Ws_avg"].to_numpy(dtype="float64")
        ambient = test["era5_temp_2m"].to_numpy(dtype="float64")
        row = {"label": window.label}
        for name, (sigma_w, sigma_t) in FORECAST_ERRORS.items():
            fw = wind + (rng.normal(0, sigma_w, len(wind)) if sigma_w else 0.0)
            fa = ambient + (rng.normal(0, sigma_t, len(ambient)) if sigma_t else 0.0)
            row[name] = bt.forecast_chain_error(
                train, test, CHANNEL, "P_norm", "Ws_avg", "era5_temp_2m",
                curve=curve, forecast_wind=fw, forecast_ambient=fa)
        out.append(row)
    summary: dict[str, dict] = {}
    for name in FORECAST_ERRORS:
        maes = [r[name]["forecast_mae_k"] for r in out if name in r]
        amps = [r[name].get("amplification") for r in out if name in r]
        amps = [a for a in amps if a is not None]
        summary[name] = {
            "median_mae_k": round(float(np.median(maes)), 4) if maes else None,
            "median_amplification": (round(float(np.median(amps)), 3)
                                      if amps else None),
        }
    return {"windows": len(out), "summary": summary, "per_window": out}


def run_reference_age(data_dir: Path, turbine: str, n_windows: int,
                      frozen_offset_days: int) -> dict:
    """B.4: the age of the reference, with everything else held fixed."""
    df = load_turbine(data_dir, turbine)
    windows = bt.weekly_windows(df["ts"])
    step = max(1, len(windows) // n_windows)
    sample = windows[::step][:n_windows]
    return bt.reference_age_test(df, CHANNEL, "P_norm", "era5_temp_2m",
                                 sample, train_end_offset_days=frozen_offset_days)


def run_reference_age_sweep(data_dir: Path, turbine: str,
                            n_windows: int) -> dict:
    """B.4: the false-alarm ratio against the age of the reference."""
    df = load_turbine(data_dir, turbine)
    windows = bt.weekly_windows(df["ts"])
    step = max(1, len(windows) // n_windows)
    sample = windows[::step][:n_windows]
    return bt.reference_age_sweep(df, CHANNEL, "P_norm", "era5_temp_2m", sample)


def run_cleaning(data_dir: Path, turbine: str) -> dict:
    """B.5: does trimming stop the reference absorbing a ramp?"""
    df = load_turbine(data_dir, turbine)
    windows = bt.weekly_windows(df["ts"])
    picks = [w for w in windows if w.label.startswith(("2015-07", "2016-07",
                                                       "2017-07"))]
    out = []
    for window in picks:
        train, test = bt.slice_window(df, window)
        if len(train) < bt.MIN_TRAIN_ROWS:
            continue
        healthy = train[(train["ts"] < window.train_end
                         - pd.Timedelta(days=10))]
        if len(healthy) < bt.MIN_TRAIN_ROWS:
            continue
        # The trim threshold has to sit BELOW the ramp, or the ramped rows
        # are never trimmed and the test cannot show anything.
        out.append(bt.trimming_test(train, CHANNEL, "P_norm", "era5_temp_2m",
                                     healthy, ramp_k=3.0, trim_k=2.0))
    shift_untrimmed = [r["untrimmed_reference_shift_k"] for r in out]
    shift_trimmed = [r["trimmed_reference_shift_k"] for r in out]
    return {
        "trials": len(out),
        "median_untrimmed_shift_k": (round(float(np.median(shift_untrimmed)), 4)
                                     if shift_untrimmed else None),
        "median_trimmed_shift_k": (round(float(np.median(shift_trimmed)), 4)
                                   if shift_trimmed else None),
        "per_trial": out,
    }


def run_fault_injection(data_dir: Path, turbine: str, n_windows: int) -> dict:
    """C.6: injected ramps, detection rate and lead time."""
    df = load_turbine(data_dir, turbine)
    # The detection question needs a horizon: the fault is injected seven
    # days into a four-week evaluation window, so there are three weeks in
    # which to catch it. The seven-day window of the prediction protocol is
    # too short to ask "how quickly was it caught".
    windows = [w for w in bt.weekly_windows(df["ts"], test_days=28)
               if w.label >= "2015-01-01"]
    step = max(1, len(windows) // n_windows)
    sample = windows[::step][:n_windows]
    factories = {k: v for k, v in arm_factories().items()
                 if k in ("load+ambient", "lightgbm")}
    rows = bt.fault_injection_test(df, CHANNEL, "P_norm", "era5_temp_2m",
                                    factories, sample)
    return {"windows": len(sample),
            "summary": bt.summarise_fault_injection(rows),
            "per_trial": rows}


def run_lead_time(data_dir: Path, turbine: str) -> dict:
    """C.7: the R80711 lead time, against the 80 C crossing and the siblings."""
    df = load_turbine(data_dir, turbine)
    peers = {}
    for peer in live.ENGIE_TURBINES:
        if peer == turbine:
            continue
        d = load_turbine(data_dir, peer)
        d = d.drop_duplicates(subset="ts", keep="first")
        peers[peer] = d.set_index("ts")

    lo = pd.Timestamp("2017-09-01", tz="UTC")
    hi = pd.Timestamp("2017-10-15", tz="UTC")
    warm = pd.Timestamp("2017-07-01", tz="UTC")
    window_df = df[(df["ts"] >= warm) & (df["ts"] <= hi)].reset_index(drop=True)
    peer_window = {k: v[(v.index >= warm) & (v.index <= hi)] for k, v in peers.items()}

    peer_median = live._peer_median_at_wind(
        window_df, peer_window, CHANNEL, "Ws_avg")

    # The alarm, day by day, with the reference re-estimated on the thirty
    # days before each day -- the protocol an operator would run.
    days = pd.date_range(lo, hi, freq="1D", tz="UTC")
    first_alarm: dict[str, str] = {}
    # The lead time depends on how many false alarms are tolerated, so it
    # is measured at two matched rates: the project's 1 %, and a looser 5 %.
    first_alarm_5pct: dict[str, str] = {}
    for arm_name, factory in arm_factories().items():
        for day in days:
            train = df[(df["ts"] >= day - pd.Timedelta(days=30)) & (df["ts"] < day)]
            if len(train) < bt.MIN_TRAIN_ROWS:
                continue
            arm = factory().fit(train)
            oof = bt.out_of_fold_residuals(factory, train, CHANNEL)
            todays = df[(df["ts"] >= day) & (df["ts"] < day + pd.Timedelta(days=1))]
            if len(todays) < 50:
                continue
            resid = todays[CHANNEL].to_numpy(dtype="float64") - arm.predict(todays)
            if not np.isfinite(resid).any():
                continue
            for rate, store in ((0.01, first_alarm), (0.05, first_alarm_5pct)):
                thr = bt.matched_threshold(oof, rate)
                if (resid > thr).any() and arm_name not in store:
                    store[arm_name] = day.strftime("%Y-%m-%d")

    # The two references the alarm has to beat.
    first_80c = None
    for day in days:
        todays = df[(df["ts"] >= day) & (df["ts"] < day + pd.Timedelta(days=1))]
        if len(todays) and todays[CHANNEL].max() > 80.0:
            first_80c = day.strftime("%Y-%m-%d")
            break
    first_divergence = None
    # `_peer_median_at_wind` returns one value per row of `window_df`, in
    # the same order, so the day's rows are selected with the same mask.
    ts_window = window_df["ts"].dt.tz_convert(None).to_numpy()
    for day in days:
        mask = ((ts_window >= np.datetime64(day.tz_convert(None)))
                & (ts_window < np.datetime64(
                    (day + pd.Timedelta(days=1)).tz_convert(None))))
        if not mask.any():
            continue
        gap = (window_df[CHANNEL].to_numpy(dtype="float64")[mask]
               - peer_median[mask])
        if np.isfinite(gap).any() and np.nanmax(gap) > 5.0:
            first_divergence = day.strftime("%Y-%m-%d")
            break

    def _lead(alarm: str | None) -> float | None:
        if not alarm or not first_80c:
            return None
        return round((pd.Timestamp(first_80c) - pd.Timestamp(alarm)).total_seconds()
                      / 86400.0, 2)

    def _lead_div(alarms: dict[str, str], reference: str | None) -> dict:
        if not reference:
            return {}
        out = {}
        for k, v in alarms.items():
            out[k] = round(
                (pd.Timestamp(reference) - pd.Timestamp(v)).total_seconds()
                / 86400.0, 2)
        return out

    return {
        "anomaly_window": "2017-09-01 to 2017-10-15",
        "first_alarm_1pct": first_alarm,
        "first_alarm_5pct": first_alarm_5pct,
        "first_80c_crossing": first_80c,
        "first_sibling_divergence_5k": first_divergence,
        "days_before_80c_1pct": {k: _lead(v) for k, v in first_alarm.items()},
        "days_before_80c_5pct": {k: _lead(v) for k, v in first_alarm_5pct.items()},
        "days_before_divergence_1pct": _lead_div(first_alarm, first_divergence),
        "note": ("The rolling 30-day reference is re-estimated every day. "
                 "The alarm is the matched one-sided rule, at 1 % and 5 %."),
    }


def run_care_thermal_subset(results_dir: Path) -> dict:
    """C.8: the CARE thermal subset.

    The CARE score is dominated by faults a thermal detector cannot see.
    The events whose windows contain rows in NORMAL operation are the ones
    where the score is even defined, and they are the thermal-observable
    subset.

    The sub-scores are recomputed on that subset with the benchmark's own
    formula, so the two numbers are comparable: coverage is the mean
    f-beta over the anomaly events, earliness the mean weighted score, and
    reliability the event-wise F-beta over every event.
    """
    path = results_dir / "multivariate.json"
    if not path.exists():
        return {"status": "results/multivariate.json is missing"}
    payload = json.loads(path.read_text())
    combiner = payload["max_combiner"]
    events = combiner["per_event"]
    scorable = {d["event_id"]: d.get("n_scorable_event_rows", 0)
                for d in payload.get("per_dataset", [])}

    def _sub_scores(rows: list[dict]) -> dict:
        anom = [r for r in rows if r["event_label"] == "anomaly"]
        norm = [r for r in rows if r["event_label"] == "normal"]
        if not anom:
            return {"n_anomaly": 0}
        detected = np.array([bool(r["anomaly_detected"]) for r in rows])
        is_anom = np.array([r["event_label"] == "anomaly" for r in rows])
        reliability = _fbeta(is_anom, detected, beta=0.5)
        return {
            "n_anomaly": len(anom),
            "n_normal": len(norm),
            "coverage": round(float(np.mean(
                [r["f_beta_score"] for r in anom
                 if r["f_beta_score"] == r["f_beta_score"]])), 4),
            "earliness": round(float(np.mean(
                [r["weighted_score"] for r in anom
                 if r["weighted_score"] == r["weighted_score"]])), 4),
            "accuracy": (round(float(np.mean([r["accuracy"] for r in norm])), 4)
                         if norm else None),
            "reliability": round(float(reliability), 4),
            "events_detected": int(sum(bool(r["anomaly_detected"]) for r in anom)),
        }

    thermal_ids = {eid for eid, n in scorable.items() if (n or 0) > 0}
    all_rows = events
    thermal_rows = [r for r in events
                    if r["event_label"] != "anomaly"
                    or r["event_id"] in thermal_ids]

    out = {
        "protocol": ("the committed CARE evaluation, restricted to the anomaly "
                     "events whose window contains rows in normal operation"),
        "published_on_all_events": {
            "care_score": combiner["care_score"],
            "sub_scores": combiner["sub_scores"],
        },
        "all_events": _sub_scores(all_rows),
        "thermal_observable_events": _sub_scores(thermal_rows),
        "n_thermal_observable_anomalies": len(
            [r for r in events if r["event_label"] == "anomaly"
             and r["event_id"] in thermal_ids]),
        "n_all_anomalies": len(
            [r for r in events if r["event_label"] == "anomaly"]),
    }
    a = out["all_events"]
    t = out["thermal_observable_events"]
    for key in ("coverage", "earliness"):
        if a.get(key) is not None and t.get(key) is not None:
            out[f"{key}_change"] = round(t[key] - a[key], 4)
    return out


def _fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
    """The event-wise F-beta, as the benchmark computes it."""
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    if tp == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "backtest.json"))
    ap.add_argument("--turbines", default="R80711,R80721,R80736,R80790")
    ap.add_argument("--quick", action="store_true",
                    help="fewer windows, for a fast look")
    ap.add_argument("--fault-windows", type=int, default=40)
    ap.add_argument("--chain-windows", type=int, default=26)
    args = ap.parse_args()

    data_dir = Path(args.data)
    turbines = [t for t in args.turbines.split(",") if t]
    t0 = time.time()

    print("A.1/A.2  weekly backtest and weather ablation")
    weekly = run_weekly_sweep(data_dir, turbines, args.quick)

    print("A.3      the forecast chain")
    chain = run_forecast_chain(data_dir, turbines[0],
                               6 if args.quick else args.chain_windows)

    print("B.4      the age of the reference")
    age = run_reference_age(data_dir, turbines[0],
                            8 if args.quick else 40,
                            frozen_offset_days=365)
    age["sweep_over_reference_age"] = run_reference_age_sweep(
        data_dir, turbines[0], 8 if args.quick else 40)

    print("B.5      the cleaning test")
    cleaning = run_cleaning(data_dir, turbines[0])

    print("C.6      fault injection")
    faults = run_fault_injection(data_dir, turbines[0],
                                 10 if args.quick else args.fault_windows)

    print("C.7      the R80711 lead time")
    lead = run_lead_time(data_dir, "R80711")

    print("C.8      the CARE thermal subset")
    care = run_care_thermal_subset(ROOT / "results")

    out = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "seconds": round(time.time() - t0, 1),
        "protocol": {
            "channel": CHANNEL,
            "train_days": 30,
            "test_days": 7,
            "target_false_alarm_rate": bt.DEFAULT_TARGET_RATE,
            "split": "the training window ends strictly before the test "
                     "window begins; enforced in code by Window",
            "threshold": "each arm's own out-of-fold training residuals, at "
                         "the 1 % quantile, one-sided (overheating only)",
            "turbines": turbines,
        },
        "A1_A2_weekly_backtest": weekly,
        "A3_forecast_chain": chain,
        "B4_reference_age": age,
        "B5_cleaning": cleaning,
        "C6_fault_injection": faults,
        "C7_lead_time": lead,
        "C8_care_thermal_subset": care,
    }

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest} ({dest.stat().st_size/1e3:.0f} KB) "
          f"in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
