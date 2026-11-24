"""The operating curve: what each false call-out buys in earlier warning.

A detector has a dial. Turn it towards sensitive and it warns sooner and
raises more false alarms; turn it the other way and it is quiet and slow. A
single figure ("it catches 90 % of faults", "it raises 1 % false alarms")
says nothing without the dial setting, and two models quoted at different
settings cannot be compared at all.

This module measures the whole dial, for every arm, on the same weeks:

  * the dial is two settings. SENSITIVITY: the share of its own training
    window an arm is calibrated to alarm on (the matched threshold of the
    evaluation suite). PERSISTENCE: how many consecutive alarmed samples it
    takes before it is worth sending someone, one sample or an hour or a day.

  * the COST side is false dispatches per turbine-year: alarm episodes of at
    least that persistence, on healthy turbine-weeks.

  * the BENEFIT side is how fast a developing fault is caught: a ramp of a
    stated size injected into a healthy week, and the time from its onset to
    the first dispatch.

  * a PLACEBO ramp of 0 K is run through the very same code. Whatever it
    "detects" is a false dispatch that happened to fall after the onset, so
    it is the chance level a real detection has to be read against. A
    detector that raises many false alarms will find any ramp quickly by
    accident, and without the placebo that looks like skill.

The point of putting both sides on one chart is that the comparison between
two models is then made at EQUAL false call-outs per year, which is the only
comparison the project's goal cares about.

This module is deliberately free of data loading and of Streamlit: it works
on frames it is handed, so it can be tested on synthetic data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from . import backtest as bt

#: The sensitivity sweep: the share of its own training window an arm is
#: calibrated to alarm on. 0.01 is the evaluation suite's standing rate. The
#: top of the range was widened (to 0.2) after the economics showed the
#: learned arm's most profitable setting sitting at the old edge of 0.05.
TARGET_RATES = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

#: The persistence sweep, in consecutive ten-minute samples: one sample,
#: an hour (the dashboard's dispatch rule), six hours, a day, two days. The
#: two days were added after the economics put the most profitable setting of
#: several arms at the old longest persistence of a day.
PERSISTENCE = (1, 6, 36, 144, 288)

#: The injected ramp sizes, in kelvin. Zero is the placebo.
MAGNITUDES = (0.0, 2.0, 4.0, 8.0)

#: The SCADA sampling interval, in minutes.
SAMPLE_MINUTES = 10.0

#: A gap longer than this between two samples ends an alarm run, so a run
#: is never stitched together across missing data.
MAX_GAP_MINUTES = 15.0

#: The window geometry. A reference fitted on the thirty days before a
#: window is held for the whole window, which is how the dashboard's monthly
#: loop works. The same geometry as the evaluation suite's fault injection.
TRAIN_DAYS = 30
TEST_DAYS = 28

#: The ramp: injected this many days into the window, growing over
#: `RAMP_DAYS` and then held.
ONSET_DAY = 7.0
RAMP_DAYS = 2.0

#: A window needs at least this share of its expected samples to count, so
#: a mostly-missing week does not pass as an exposure it was not.
MIN_COVERAGE = 0.5

#: The shapes a simulated fault can take. ADDITIVE adds the same offset at
#: every load. LOAD adds an offset proportional to the load, because heat in
#: a bearing fault tends to scale with how hard the machine is working; the
#: stated size is the offset at `REFERENCE_LOAD` of rated power.
RAMP_SHAPES = ("additive", "load")
REFERENCE_LOAD = 0.5


# --------------------------------------------------------------- the runs
def alarm_runs(alarm, times=None, max_gap_minutes: float = MAX_GAP_MINUTES):
    """The runs of consecutive alarmed samples: (start indices, lengths).

    A run ends at the first sample that is not alarmed, and ALSO at a gap in
    the timestamps, so two stretches of alarm either side of a missing day
    are two runs, not one long one.
    """
    a = np.asarray(alarm, dtype=bool)
    n = a.size
    if n == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    new_group = np.ones(n, dtype=bool)
    new_group[1:] = a[1:] != a[:-1]
    if times is not None:
        t = np.asarray(times, dtype="datetime64[ns]")
        limit = np.timedelta64(int(round(max_gap_minutes * 60)), "s")
        new_group[1:] |= np.diff(t) > limit
    starts = np.flatnonzero(new_group)
    lengths = np.diff(np.append(starts, n))
    keep = a[starts]
    return starts[keep], lengths[keep]


def count_dispatches(starts: np.ndarray, lengths: np.ndarray,
                     min_run: int) -> int:
    """How many runs are long enough to be a dispatch."""
    return int((np.asarray(lengths) >= int(min_run)).sum())


def first_dispatch_index(starts: np.ndarray, lengths: np.ndarray,
                         min_run: int):
    """The sample at which the first dispatch condition is MET, or None.

    That is the `min_run`-th sample of the first long-enough run: the
    moment at which, having waited for the persistence, a technician would
    actually be called. It is later than the start of the run, and the
    persistence is exactly how much later.
    """
    sel = np.asarray(lengths) >= int(min_run)
    if not sel.any():
        return None
    return int(np.asarray(starts)[sel][0] + int(min_run) - 1)


# ------------------------------------------------------------- one window
def inject_shaped(test: pd.DataFrame, channel: str, onset, magnitude: float,
                  ramp_days: float = RAMP_DAYS, shape: str = "additive",
                  power_col: str = "P_norm",
                  reference_load: float = REFERENCE_LOAD) -> pd.DataFrame:
    """A simulated fault of a stated size and shape.

    Both shapes grow over `ramp_days` from `onset` and are then held. They
    differ only in how the offset depends on load: the same at every load
    (ADDITIVE), or in proportion to it (LOAD), where `magnitude` is the
    offset at `reference_load` of rated power.
    """
    if shape not in RAMP_SHAPES:
        raise ValueError(f"unknown ramp shape {shape!r}; expected one of {RAMP_SHAPES}")
    if shape == "additive":
        return bt.inject_ramp(test, channel, onset, magnitude,
                              ramp_days=ramp_days)
    zero = test.copy()
    zero[channel] = 0.0
    profile = bt.inject_ramp(zero, channel, onset, 1.0,
                             ramp_days=ramp_days)[channel].to_numpy(dtype="float64")
    load = test[power_col].to_numpy(dtype="float64")
    out = test.copy()
    out[channel] = (test[channel].to_numpy(dtype="float64")
                    + magnitude * profile * (load / reference_load))
    return out


def evaluate_window(arm_name: str, factory: Callable[[], bt.Arm],
                    train: pd.DataFrame, test: pd.DataFrame, channel: str,
                    turbine: str, label: str,
                    rates=TARGET_RATES, persistence=PERSISTENCE,
                    magnitudes=MAGNITUDES, onset_day: float = ONSET_DAY,
                    ramp_days: float = RAMP_DAYS,
                    ramp_shape: str = "additive",
                    power_col: str = "P_norm") -> list[dict]:
    """One arm on one window: false dispatches, and the time to catch each ramp.

    The arm is fitted once, its residuals on the healthy window and on each
    ramped copy are computed once, and every (sensitivity, persistence)
    setting is then read off those same residuals, so the whole dial costs
    little more than one fit.

    Returns one row per (sensitivity, persistence). Each row carries the
    count of false dispatches over the whole healthy window, and for every
    ramp size the days from onset to the first dispatch (None if none).
    """
    arm = factory().fit(train)
    oof = bt.out_of_fold_residuals(factory, train, channel)
    thresholds = {r: bt.matched_threshold(oof, r) for r in rates}
    if not all(np.isfinite(v) for v in thresholds.values()):
        return []

    times = bt._utc(test["ts"])
    onset = pd.to_datetime(test["ts"], utc=True).min() + pd.Timedelta(days=onset_day)
    onset64 = np.datetime64(onset.tz_convert(None))
    after = times >= onset64
    t_after = times[after]

    residuals: dict[float, np.ndarray] = {}
    for mag in magnitudes:
        rows = test if mag == 0 else inject_shaped(
            test, channel, onset, mag, ramp_days=ramp_days, shape=ramp_shape,
            power_col=power_col)
        residuals[mag] = (rows[channel].to_numpy(dtype="float64")
                          - arm.predict(rows))
    if not np.isfinite(residuals[magnitudes[0]]).any():
        return []

    healthy = residuals[0.0] if 0.0 in residuals else residuals[magnitudes[0]]
    exposure_days = len(test) * SAMPLE_MINUTES / 1440.0

    out: list[dict] = []
    for rate in rates:
        thr = thresholds[rate]
        with np.errstate(invalid="ignore"):
            h_starts, h_lengths = alarm_runs(healthy > thr, times)
            ramp_runs = {
                mag: alarm_runs((res[after] > thr), t_after)
                for mag, res in residuals.items()
            }
        for m in persistence:
            days: dict[float, float | None] = {}
            for mag, (s, ln) in ramp_runs.items():
                idx = first_dispatch_index(s, ln, m)
                days[mag] = (None if idx is None else
                             float((t_after[idx] - onset64)
                                   / np.timedelta64(1, "D")))
            out.append({
                "arm": arm_name,
                "turbine": turbine,
                "label": label,
                "rate": rate,
                "persistence": m,
                "exposure_days": exposure_days,
                "false_dispatches": count_dispatches(h_starts, h_lengths, m),
                "days": days,
            })
    return out


# ------------------------------------------------------------ the sweep
def _spans(exclude: dict | None, turbine: str) -> list[tuple]:
    spans = []
    for lo, hi in (exclude or {}).get(turbine, []):
        spans.append((pd.Timestamp(lo, tz="UTC") if pd.Timestamp(lo).tzinfo is None
                      else pd.Timestamp(lo).tz_convert("UTC"),
                      pd.Timestamp(hi, tz="UTC") if pd.Timestamp(hi).tzinfo is None
                      else pd.Timestamp(hi).tz_convert("UTC")))
    return spans


def _overlaps(window: bt.Window, spans: list[tuple],
              shift: pd.Timedelta | None = None) -> bool:
    """Does any excluded span touch the window's training OR test rows?

    Training counts too: a reference fitted on a faulted month has absorbed
    the fault, so the weeks that follow it are not healthy either. When the
    reference is deliberately OLDER than the window (`shift`), it is the
    shifted training window that is checked.
    """
    shift = shift or pd.Timedelta(0)
    ref_start = window.train_start - shift
    ref_end = window.train_end - shift
    for lo, hi in spans:
        if window.test_start < hi and window.test_end > lo:
            return True
        if ref_start < hi and ref_end > lo:
            return True
    return False


def run_operating_curve(frames: dict[str, pd.DataFrame], channel: str,
                        factories: dict[str, Callable[[], bt.Arm]],
                        rates=TARGET_RATES, persistence=PERSISTENCE,
                        magnitudes=MAGNITUDES, train_days: int = TRAIN_DAYS,
                        test_days: int = TEST_DAYS,
                        onset_day: float = ONSET_DAY,
                        ramp_days: float = RAMP_DAYS,
                        exclude: dict | None = None,
                        min_coverage: float = MIN_COVERAGE,
                        window_stride: int = 1,
                        ramp_shape: str = "additive",
                        reference_age_days: float = 0.0,
                        require_history_days: float = 0.0,
                        progress: Callable[[str], None] | None = None) -> dict:
    """The operating curve for every arm over every turbine's windows.

    `frames` maps a turbine name to its SCADA frame (with `ts`, the channel
    and whatever the arms condition on). `exclude` maps a turbine name to
    (start, end) spans that are NOT healthy, for example a documented
    anomaly; any window whose training or test rows touch a span is left
    out, for every arm alike.

    Windows do not overlap (the step is the test length), so the exposure in
    turbine-years is a plain sum.
    """
    expected = test_days * 24 * 60 / SAMPLE_MINUTES
    rows: list[dict] = []
    used: dict[str, int] = defaultdict(int)
    skipped_excluded = 0
    skipped_coverage = 0
    skipped_history = 0
    shift = pd.Timedelta(days=reference_age_days)
    for turbine, df in frames.items():
        spans = _spans(exclude, turbine)
        t_min = pd.to_datetime(df["ts"], utc=True).min()
        ts_all = pd.to_datetime(df["ts"], utc=True)
        windows = bt.weekly_windows(df["ts"], train_days=train_days,
                                    test_days=test_days, step_days=test_days)
        windows = windows[::max(1, int(window_stride))]
        for window in windows:
            if (require_history_days and window.train_start
                    - pd.Timedelta(days=require_history_days) < t_min):
                skipped_history += 1
                continue
            if _overlaps(window, spans, shift):
                skipped_excluded += 1
                continue
            train, test = bt.slice_window(df, window)
            if reference_age_days:
                # An OLDER reference: the same length of history, taken
                # `reference_age_days` further back. The rows must come
                # from the whole series, since the shifted range lies
                # entirely before the sliced training window.
                train = df[(ts_all >= window.train_start - shift)
                           & (ts_all < window.train_end - shift)]
            if len(train) < bt.MIN_TRAIN_ROWS or len(test) < min_coverage * expected:
                skipped_coverage += 1
                continue
            used[turbine] += 1
            for name, factory in factories.items():
                rows.extend(evaluate_window(
                    name, factory, train.reset_index(drop=True),
                    test.reset_index(drop=True), channel, turbine,
                    window.label, rates, persistence, magnitudes,
                    onset_day, ramp_days, ramp_shape))
            if progress:
                progress(f"{turbine} {window.label}")
    return {
        "rows": rows,
        "windows_used": dict(used),
        "windows_excluded_as_unhealthy": skipped_excluded,
        "windows_skipped_for_coverage": skipped_coverage,
        "windows_skipped_for_history": skipped_history,
        "reference_age_days": reference_age_days,
    }


def summarise_points(rows: list[dict], magnitudes=MAGNITUDES) -> list[dict]:
    """One point per (arm, sensitivity, persistence), pooled over windows.

    The false-dispatch rate is the total count over the total exposure, in
    turbine-years. The detection rate and the time to the first dispatch
    are reported for every ramp size, including the 0 K placebo.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["rate"], r["persistence"])].append(r)

    points: list[dict] = []
    for (arm, rate, m), g in sorted(groups.items()):
        exposure = float(sum(x["exposure_days"] for x in g))
        fd = int(sum(x["false_dispatches"] for x in g))
        years = exposure / 365.25
        by_turbine: dict[str, dict] = {}
        for x in g:
            d = by_turbine.setdefault(
                x["turbine"], {"false_dispatches": 0, "exposure_days": 0.0})
            d["false_dispatches"] += x["false_dispatches"]
            d["exposure_days"] += x["exposure_days"]
        for d in by_turbine.values():
            d["exposure_days"] = round(d["exposure_days"], 2)
        ramps: dict[str, dict] = {}
        for mag in magnitudes:
            days = [x["days"].get(mag) for x in g]
            hit = [d for d in days if d is not None]
            ramps[f"{mag:g}"] = {
                "trials": len(days),
                "detected": len(hit),
                "detection_rate": (round(len(hit) / len(days), 4)
                                   if days else None),
                "median_days_to_dispatch": (round(float(np.median(hit)), 3)
                                            if hit else None),
                "p90_days_to_dispatch": (round(float(np.quantile(hit, 0.9)), 3)
                                         if hit else None),
            }
        points.append({
            "arm": arm,
            "target_rate": rate,
            "persistence_samples": m,
            "persistence_hours": round(m * SAMPLE_MINUTES / 60.0, 3),
            "windows": len(g),
            "exposure_turbine_years": round(years, 4),
            "false_dispatches": fd,
            "false_dispatches_per_turbine_year": (round(fd / years, 3)
                                                  if years > 0 else None),
            "by_turbine": by_turbine,
            "ramps": ramps,
        })
    return points


# ------------------------------------------------------------ the readings
def best_under_budget(points: list[dict], arm: str, budget: float,
                      magnitude: float = 4.0):
    """The best setting of one arm that stays within a false-call-out budget.

    "Best" is the highest detection rate at that ramp size, then the
    earliest median dispatch, then the fewest false dispatches. None when no
    setting of the arm fits the budget. This is the comparison the goal
    needs: arms at EQUAL false call-outs per year, each at its own best dial.
    """
    key = f"{magnitude:g}"
    best = None
    best_rank = None
    for p in points:
        if p["arm"] != arm:
            continue
        fd = p.get("false_dispatches_per_turbine_year")
        ramp = (p.get("ramps") or {}).get(key)
        if fd is None or fd > budget or not ramp or ramp["detection_rate"] is None:
            continue
        days = ramp["median_days_to_dispatch"]
        rank = (-ramp["detection_rate"],
                float("inf") if days is None else days,
                fd)
        if best_rank is None or rank < best_rank:
            best, best_rank = p, rank
    return best


def frontier(points: list[dict], arm: str, magnitude: float = 4.0) -> list[dict]:
    """The settings of one arm not beaten on both axes.

    Ordered by false dispatches; a setting is kept only if it detects more
    of the ramp than every setting with fewer false dispatches. Everything
    else is a worse way of spending the same false call-outs.
    """
    key = f"{magnitude:g}"
    mine = [p for p in points if p["arm"] == arm
            and p.get("false_dispatches_per_turbine_year") is not None
            and (p.get("ramps") or {}).get(key, {}).get("detection_rate") is not None]
    mine.sort(key=lambda p: (p["false_dispatches_per_turbine_year"],
                             -p["ramps"][key]["detection_rate"]))
    out: list[dict] = []
    best = -1.0
    for p in mine:
        det = p["ramps"][key]["detection_rate"]
        if det > best:
            out.append(p)
            best = det
    return out


# -------------------------------------------------------- held-out selection
def split_rows(rows: list[dict], parity: int) -> list[dict]:
    """Half the windows, alternating in time order, for held-out selection.

    Windows are ranked by (turbine, start date) and every other one is kept,
    so both halves span every turbine and every season.
    """
    keys = sorted({(r["turbine"], r["label"]) for r in rows})
    chosen = {k for i, k in enumerate(keys) if i % 2 == parity}
    return [r for r in rows if (r["turbine"], r["label"]) in chosen]


def held_out_reading(rows: list[dict], arm: str, budget: float,
                     magnitude: float = 4.0,
                     magnitudes=MAGNITUDES):
    """An arm's best setting under a budget, chosen on one half of the windows
    and scored on the other half, in both directions.

    The dial is chosen from a grid, so the best setting on the weeks it was
    chosen on is optimistic. Scoring it on weeks it never saw removes that.
    Both directions are returned, averaged, so neither half is privileged.
    """
    key = f"{magnitude:g}"
    folds = []
    for select_parity in (0, 1):
        sel = summarise_points(split_rows(rows, select_parity), magnitudes)
        best = best_under_budget(sel, arm, budget, magnitude)
        if best is None:
            continue
        ev = summarise_points(split_rows(rows, 1 - select_parity), magnitudes)
        match = next((p for p in ev if p["arm"] == arm
                      and p["target_rate"] == best["target_rate"]
                      and p["persistence_samples"] == best["persistence_samples"]),
                     None)
        if match is None:
            continue
        folds.append({"selected": best, "evaluated": match})
    if not folds:
        return None

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "folds": len(folds),
        "false_dispatches_per_turbine_year": _mean(
            [f["evaluated"]["false_dispatches_per_turbine_year"] for f in folds]),
        "detection_rate": _mean(
            [f["evaluated"]["ramps"][key]["detection_rate"] for f in folds]),
        "median_days_to_dispatch": _mean(
            [f["evaluated"]["ramps"][key]["median_days_to_dispatch"] for f in folds]),
        "placebo_detection_rate": _mean(
            [f["evaluated"]["ramps"]["0"]["detection_rate"] for f in folds]),
        "selected_settings": [
            {"target_rate": f["selected"]["target_rate"],
             "persistence_samples": f["selected"]["persistence_samples"]}
            for f in folds],
    }


# -------------------------------------------------------- the standard arms
def standard_arms(channel: str, lgbm_features: list[str] | None = None,
                  power_col: str = "P_norm", ambient_col: str = "era5_temp_2m",
                  include_lightgbm: bool = True) -> dict:
    """The arms of the curve: the floor, the operating point, the deployed
    model, and the learned comparison.

    They differ only in what they condition on, exactly as in the
    evaluation suite, so any difference between them is the model.
    """
    arms: dict[str, Callable[[], bt.Arm]] = {
        "constant": lambda: bt.ConstantArm(channel, power_col),
        "load": lambda: bt.DeterministicArm(
            channel, power_col, ambient_col, use_ambient=False,
            use_wind=False, name="load"),
        "load+ambient": lambda: bt.DeterministicArm(
            channel, power_col, ambient_col),
    }
    if include_lightgbm:
        feats = lgbm_features or [power_col, "Ws_avg", ambient_col]
        arms["lightgbm"] = lambda: bt.LightGBMArm(channel, feats)
    return arms


# ------------------------------------------------- the real anomaly, same dial
def lead_time_on_anomaly(df: pd.DataFrame, channel: str,
                         factories: dict[str, Callable[[], bt.Arm]],
                         train_start, test_start, test_end, onset,
                         hot_c: float = 80.0, min_hot_samples: int = 30,
                         rates=TARGET_RATES,
                         persistence=PERSISTENCE,
                         power_col: str = "P_norm",
                         diagnose_arm: str = "load+ambient",
                         high_load: float = 0.6) -> dict:
    """The operating curve's dial, run on ONE real anomaly instead of a ramp.

    Each arm is fitted on the healthy stretch before `test_start`, then scored
    on `test_start` to `test_end`, a window that holds a quiet stretch (before
    `onset`) and the anomaly (after it). Nothing is injected.

    For every setting of the dial this records:

      * the first dispatch AT OR AFTER the onset, and how many days before the
        first sustained hot spell it fell (negative: after);
      * the dispatches in the quiet stretch, which are false by construction.

    A "sustained hot spell" is the first day at or after the onset with at
    least `min_hot_samples` samples above `hot_c` (30 is five hours). A single
    sample over 80 C is not an event: R80711 had hundreds of them in 2017
    BEFORE its anomaly, so the line itself is crossed in ordinary operation.

    The second is what stops "it dispatched early" from being a prize that a
    jumpy detector can win: an arm that dispatches a week before the onset has
    raised a false alarm, not given a warning.

    Unlike the injected ramps this is a single case, so there is no detection
    rate and no chance level: it is one machine, once. It is the check that
    the dial settings the simulation prefers also behave on something real.
    """
    ts = pd.to_datetime(df["ts"], utc=True)

    def _utc(x):
        x = pd.Timestamp(x)
        return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")

    train_start, test_start, test_end, onset = (
        _utc(train_start), _utc(test_start), _utc(test_end), _utc(onset))
    train = df[(ts >= train_start) & (ts < test_start)].reset_index(drop=True)
    test = df[(ts >= test_start) & (ts < test_end)].reset_index(drop=True)
    if len(train) < bt.MIN_TRAIN_ROWS or len(test) < bt.MIN_TEST_ROWS:
        return {"rows": [], "reason": "not enough rows to fit and score"}

    times = bt._utc(test["ts"])
    onset64 = np.datetime64(onset.tz_convert(None))
    values = test[channel].to_numpy(dtype="float64")
    days_of = times.astype("datetime64[D]")
    hot_mask = (values > hot_c) & (times >= onset64)
    first_hot = None
    if hot_mask.any():
        hot_days, counts = np.unique(days_of[hot_mask], return_counts=True)
        sustained = hot_days[counts >= int(min_hot_samples)]
        if sustained.size:
            first_hot = times[hot_mask & (days_of == sustained[0])][0]
    one_day = np.timedelta64(1, "D")

    # Which samples belong to which phase, for the load profile below: before
    # the onset, on the hot-spell days themselves, and on the other days after
    # the onset. The phases are read off the data, not chosen by hand.
    spell_days = (np.unique(days_of[hot_mask])[
        np.unique(days_of[hot_mask], return_counts=True)[1] >= int(min_hot_samples)]
        if hot_mask.any() else np.array([], dtype="datetime64[D]"))
    after_onset = times >= onset64
    phases = {
        "before_onset": ~after_onset,
        "hot_spell_days": after_onset & np.isin(days_of, spell_days),
        "other_days_after_onset": after_onset & ~np.isin(days_of, spell_days),
    }
    load_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    residual_by_load = None
    high_load_coverage = None

    rows: list[dict] = []
    for name, factory in factories.items():
        arm = factory().fit(train)
        oof = bt.out_of_fold_residuals(factory, train, channel)
        resid = values - arm.predict(test)
        if name == diagnose_arm and power_col in test.columns:
            load = test[power_col].to_numpy(dtype="float64")
            residual_by_load = {"arm": name, "load_bins": load_edges[:-1],
                                "phases": {}}
            for phase, mask in phases.items():
                cells = []
                for lo, hi in zip(load_edges[:-1], load_edges[1:]):
                    sel = mask & (load >= lo) & (load < hi) & np.isfinite(resid)
                    cells.append({"n": int(sel.sum()),
                                  "mean_residual_k": (round(float(resid[sel].mean()), 3)
                                                      if sel.any() else None)})
                residual_by_load["phases"][phase] = cells

            # Observability: the excess shows mainly at high load, so what
            # matters is WHEN the turbine first ran hard, and how much of that
            # regime the reference was ever shown.
            hi = (load >= high_load) & np.isfinite(resid)
            before_spell = (times < first_hot if first_hot is not None
                            else np.ones(len(times), dtype=bool))
            between = after_onset & before_spell & hi
            train_load = train[power_col].to_numpy(dtype="float64")
            first_day = (onset.tz_convert(None).to_datetime64()
                         .astype("datetime64[D]") - np.timedelta64(7, "D"))
            daily = []
            for d in np.arange(first_day, first_day + np.timedelta64(28, "D")):
                sel = (days_of == d) & hi
                daily.append({"day": str(d), "n_high_load": int(sel.sum()),
                              "mean_residual_k": (round(float(resid[sel].mean()), 3)
                                                  if sel.any() else None)})
            n_train_hi = int((train_load >= high_load).sum())
            high_load_coverage = {
                "arm": name,
                "high_load": high_load,
                "training_samples": int(len(train)),
                "training_high_load_samples": n_train_hi,
                "training_high_load_share": round(n_train_hi / max(len(train), 1), 4),
                "samples_between_onset_and_hot_spell": int(between.sum()),
                "daily": daily,
            }
        for rate in rates:
            thr = bt.matched_threshold(oof, rate)
            if not np.isfinite(thr):
                continue
            with np.errstate(invalid="ignore"):
                starts, lengths = alarm_runs(resid > thr, times)
            for m in persistence:
                sel = lengths >= m
                dispatch_at = (times[np.asarray(starts)[sel] + (m - 1)]
                               if sel.any() else np.array([], dtype="datetime64[ns]"))
                quiet = int((dispatch_at < onset64).sum())
                after = dispatch_at[dispatch_at >= onset64]
                first = after[0] if after.size else None
                rows.append({
                    "arm": name,
                    "target_rate": rate,
                    "persistence_samples": m,
                    "dispatches_before_onset": quiet,
                    "dispatches_after_onset": int(after.size),
                    "first_dispatch_after_onset": (None if first is None
                                                   else str(first)[:16]),
                    "days_after_onset": (None if first is None
                                         else round(float((first - onset64) / one_day), 3)),
                    "days_before_hot_spell": (
                        None if first is None or first_hot is None
                        else round(float((first_hot - first) / one_day), 3)),
                })
    return {
        "rows": rows,
        "train": [str(train_start)[:10], str(test_start)[:10]],
        "test": [str(test_start)[:10], str(test_end)[:10]],
        "onset": str(onset)[:10],
        "hot_c": hot_c,
        "min_hot_samples": int(min_hot_samples),
        "first_hot_spell": None if first_hot is None else str(first_hot)[:16],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "residual_by_load": residual_by_load,
        "high_load_coverage": high_load_coverage,
    }
