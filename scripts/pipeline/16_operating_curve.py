#!/usr/bin/env python
"""The operating curve: false call-outs per year against time to catch a fault.

Runs `src/windmill/operating_curve.py` over the four ENGIE turbines and writes
`results/operating_curve.json`.

    python scripts/pipeline/16_operating_curve.py                 # the full run
    python scripts/pipeline/16_operating_curve.py --quick --out /tmp/curve.json
    python scripts/pipeline/16_operating_curve.py --skip-lightgbm --out /tmp/curve.json

The two halves of the project's goal pull against each other, so neither is
quoted alone. For every arm this sweeps the whole dial (the sensitivity of
its alarm threshold, and how many consecutive alarmed samples a dispatch
needs) and records, at each setting:

  * the false dispatches per turbine-year on healthy turbine-weeks (the cost
    side), and
  * the share of injected ramps that are caught and how many days after the
    onset the first dispatch comes (the benefit side),

with a 0 K PLACEBO ramp through the same code, so that a detection that is
merely a false alarm landing after the onset can be told from a real one.

The R80711 anomaly of autumn 2017 is not healthy, so every window that
touches it (in training or in test) is left out for every arm alike.

A partial run (a quick look, or no LightGBM) is never written over the
committed result: pass --out to put it somewhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import RATED_POWER_KW, loaders  # noqa: E402
from windmill import operating_curve as oc  # noqa: E402

#: The channel the case study is about.
CHANNEL = "Gb2t_avg"

#: The arms of the curve, in the order they are reported.
DEFAULT_ARMS = ("constant", "load", "load+ambient", "lightgbm")

#: Spans that are NOT healthy. R80711's gearbox bearing overheat runs from
#: late summer into October 2017; the span is widened to cover the month of
#: training windows that would otherwise have been fitted on it.
EXCLUDE = {"R80711": [("2017-08-15", "2017-11-30")]}

#: The false-call-out budgets the arms are compared at, per turbine-year:
#: roughly one every two months, monthly, fortnightly, weekly.
BUDGETS = (6, 12, 26, 52)

#: The ramp the budget comparison is quoted at, in kelvin.
HEADLINE_RAMP_K = 4.0

#: The one real anomaly the lead-time check is run on: R80711's gearbox
#: bearing, autumn 2017. The window is the dashboard's "Aug 15 - Oct 15 2017"
#: one, the reference is fitted on the month before it, and the onset is the
#: day R80711 first ran at least 5 K hotter than its three siblings (read from
#: the evaluation suite's C.7 result when that is present).
ANOMALY = {
    "turbine": "R80711",
    "train_start": "2017-07-16",
    "test_start": "2017-08-15",
    "test_end": "2017-10-15",
    "onset": "2017-09-01",
}

DEFAULT_OUT = ROOT / "results" / "operating_curve.json"


def load_turbine(data_dir: Path, turbine: str) -> pd.DataFrame:
    """One turbine's SCADA, with ERA5 ambient joined on and power normalised.

    The same preparation `15_backtest.py` uses, so the two evaluations are
    scored on identical rows.
    """
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


def budget_readings(points: list[dict], arms: list[str]) -> dict:
    """For each budget, each arm's best setting that stays within it.

    This is the comparison the goal needs: arms at EQUAL false call-outs per
    year, each at its own best dial, with the placebo beside it.
    """
    key = f"{HEADLINE_RAMP_K:g}"
    out: dict[str, dict] = {}
    for budget in BUDGETS:
        row: dict[str, dict | None] = {}
        for arm in arms:
            p = oc.best_under_budget(points, arm, float(budget), HEADLINE_RAMP_K)
            if p is None:
                row[arm] = None
                continue
            ramp, placebo = p["ramps"][key], p["ramps"]["0"]
            row[arm] = {
                "target_rate": p["target_rate"],
                "persistence_samples": p["persistence_samples"],
                "persistence_hours": p["persistence_hours"],
                "false_dispatches_per_turbine_year":
                    p["false_dispatches_per_turbine_year"],
                "detection_rate": ramp["detection_rate"],
                "median_days_to_dispatch": ramp["median_days_to_dispatch"],
                "placebo_detection_rate": placebo["detection_rate"],
                "placebo_median_days_to_dispatch":
                    placebo["median_days_to_dispatch"],
            }
        out[str(budget)] = row
    return out


def held_out_budgets(rows: list[dict], arms: list[str]) -> dict:
    """The budget comparison with the setting chosen on half the windows and
    scored on the other half, so the best-of-a-grid optimism is removed."""
    return {str(b): {arm: oc.held_out_reading(rows, arm, float(b),
                                              HEADLINE_RAMP_K)
                     for arm in arms}
            for b in BUDGETS}


def reference_age_readings(frames: dict, factories: dict, arms: list[str],
                           ages: tuple[int, ...], stride: int) -> dict:
    """False dispatches and detection against the AGE of the reference.

    The same windows for every age (windows with less history than the
    oldest reference are dropped for all of them), at the dashboard's dial
    (a 1 % rate) with one sample and with an hour of sustained alarm.
    """
    out: dict[str, dict] = {}
    for age in ages:
        run = oc.run_operating_curve(
            frames, CHANNEL, factories, exclude=EXCLUDE, window_stride=stride,
            reference_age_days=float(age), require_history_days=float(max(ages)),
            magnitudes=(0.0, HEADLINE_RAMP_K))
        pts = oc.summarise_points(run["rows"], (0.0, HEADLINE_RAMP_K))
        entry: dict = {"windows": int(sum(run["windows_used"].values()))}
        key = f"{HEADLINE_RAMP_K:g}"
        for arm in arms:
            entry[arm] = {}
            for m in (1, 6):
                p = next((q for q in pts if q["arm"] == arm
                          and q["target_rate"] == 0.01
                          and q["persistence_samples"] == m), None)
                if p is None:
                    continue
                entry[arm][str(m)] = {
                    "false_dispatches_per_turbine_year":
                        p["false_dispatches_per_turbine_year"],
                    "detection_rate": p["ramps"][key]["detection_rate"],
                    "median_days_to_dispatch":
                        p["ramps"][key]["median_days_to_dispatch"],
                    "placebo_detection_rate": p["ramps"]["0"]["detection_rate"],
                }
        out[str(age)] = entry
        print(f"  reference age {age} d: {entry['windows']} windows")
    # The ratio is stored, not left to the reader or the report to compute,
    # so every number the README quotes is a number the sweep wrote.
    fresh, oldest = out[str(min(ages))], out[str(max(ages))]
    ratios: dict[str, dict] = {}
    for arm in arms:
        ratios[arm] = {}
        for m in ("1", "6"):
            a, b = fresh.get(arm, {}).get(m), oldest.get(arm, {}).get(m)
            if (a and b and a["false_dispatches_per_turbine_year"]
                    and b["false_dispatches_per_turbine_year"] is not None):
                ratios[arm][m] = round(
                    b["false_dispatches_per_turbine_year"]
                    / a["false_dispatches_per_turbine_year"], 2)
    out["ratio_oldest_over_fresh"] = ratios
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--turbines", default="R80711,R80721,R80736,R80790")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--skip-lightgbm", action="store_true",
                    help="leave the learned arm out (for a machine without it)")
    ap.add_argument("--quick", action="store_true",
                    help="every third window only, for a fast look")
    ap.add_argument("--window-stride", type=int, default=1)
    ap.add_argument("--skip-extras", action="store_true",
                    help="skip the load-proportional ramp, the held-out "
                         "selection and the reference-age sweep")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a partial run (for example without LightGBM) "
                         "over the committed result; it is marked "
                         "complete_run=false")
    args = ap.parse_args()

    turbines = [t for t in args.turbines.split(",") if t]
    arms = [a for a in args.arms.split(",") if a]
    if args.skip_lightgbm:
        arms = [a for a in arms if a != "lightgbm"]
    stride = 3 if args.quick else max(1, args.window_stride)

    complete = (set(DEFAULT_ARMS) <= set(arms) and stride == 1
                and len(turbines) == 4)
    out_path = Path(args.out)
    if (not complete and out_path.resolve() == DEFAULT_OUT.resolve()
            and not args.allow_partial):
        print("this is a partial run (fewer arms, turbines or windows than "
              "the full evaluation), so it will not overwrite\n"
              f"{DEFAULT_OUT}\n"
              "pass --out <somewhere else> to write it.")
        return 2

    factories = oc.standard_arms(CHANNEL, include_lightgbm="lightgbm" in arms)
    missing = [a for a in arms if a not in factories]
    if missing:
        print(f"unknown or unavailable arms: {missing}")
        return 2
    factories = {a: factories[a] for a in arms}
    if "lightgbm" in factories:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            print("lightgbm is not installed; install it or pass "
                  "--skip-lightgbm")
            return 2

    data_dir = Path(args.data)
    t0 = time.time()
    frames: dict[str, pd.DataFrame] = {}
    for turbine in turbines:
        t1 = time.time()
        frames[turbine] = load_turbine(data_dir, turbine)
        print(f"  loaded {turbine}: {len(frames[turbine]):,} rows "
              f"in {time.time() - t1:.1f}s")

    done = {"n": 0}

    def progress(msg: str) -> None:
        done["n"] += 1
        if done["n"] % 20 == 0:
            print(f"  {done['n']} windows scored ({msg}), "
                  f"{time.time() - t0:.0f}s elapsed")

    run = oc.run_operating_curve(frames, CHANNEL, factories, exclude=EXCLUDE,
                                 window_stride=stride, progress=progress)
    points = oc.summarise_points(run["rows"])

    extras: dict = {}
    if ANOMALY["turbine"] in frames:
        print("the real anomaly")
        onset = ANOMALY["onset"]
        try:
            onset = (json.loads((ROOT / "results" / "backtest.json").read_text())
                     ["C7_lead_time"]["first_sibling_divergence_5k"] or onset)
        except (OSError, KeyError, ValueError):
            pass
        extras["r80711_lead_time"] = oc.lead_time_on_anomaly(
            frames[ANOMALY["turbine"]], CHANNEL, factories,
            ANOMALY["train_start"], ANOMALY["test_start"],
            ANOMALY["test_end"], onset)
    if not args.skip_extras:
        print("load-proportional ramp")
        run_load = oc.run_operating_curve(frames, CHANNEL, factories,
                                          exclude=EXCLUDE, window_stride=stride,
                                          ramp_shape="load")
        load_points = oc.summarise_points(run_load["rows"])
        extras["budgets_ramp_4k_load_proportional"] = budget_readings(
            load_points, arms)
        # The full sweep, not just the budget summary: the cost model prices
        # every setting, and a ranking that only knew the flat-offset fault
        # would miss that the order of the arms changes with the fault's shape.
        extras["points_ramp_load_proportional"] = load_points
        print("held-out selection")
        extras["held_out_budgets_ramp_4k"] = held_out_budgets(run["rows"], arms)
        print("reference age")
        extras["reference_age"] = reference_age_readings(
            frames, factories, arms, (0, 90, 365), stride)

    out = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "seconds": round(time.time() - t0, 1),
        "complete_run": complete,
        "protocol": {
            "channel": CHANNEL,
            "train_days": oc.TRAIN_DAYS,
            "test_days": oc.TEST_DAYS,
            "window_step_days": oc.TEST_DAYS,
            "onset_day": oc.ONSET_DAY,
            "ramp_days": oc.RAMP_DAYS,
            "sample_minutes": oc.SAMPLE_MINUTES,
            "max_gap_minutes": oc.MAX_GAP_MINUTES,
            "min_coverage": oc.MIN_COVERAGE,
            "window_stride": stride,
            "target_rates": list(oc.TARGET_RATES),
            "persistence_samples": list(oc.PERSISTENCE),
            "ramp_magnitudes_k": list(oc.MAGNITUDES),
            "arms": arms,
            "turbines": turbines,
            "excluded_as_unhealthy": {k: [list(s) for s in v]
                                      for k, v in EXCLUDE.items()},
            "dispatch": ("an episode of at least `persistence` consecutive "
                         "ten-minute alarmed samples; a gap of more than "
                         "max_gap_minutes ends an episode"),
            "false_dispatch": ("a dispatch on a healthy turbine-window, "
                               "counted over the whole window"),
            "placebo": ("a 0 K ramp through the same code: a dispatch after "
                        "the onset is a false dispatch that happened to land "
                        "there, so it is the chance level a real detection "
                        "has to be read against"),
            "threshold": ("each arm's own out-of-fold training residuals at "
                          "the stated quantile, one-sided"),
        },
        "windows_used": run["windows_used"],
        "windows_excluded_as_unhealthy": run["windows_excluded_as_unhealthy"],
        "windows_skipped_for_coverage": run["windows_skipped_for_coverage"],
        "points": points,
        "budgets_ramp_4k": budget_readings(points, arms),
        **extras,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e3:.0f} KB, "
          f"{len(points)} points) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
