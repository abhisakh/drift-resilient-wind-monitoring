"""The historical confirmation: train on last month, predict this month, check.

THE IDEA
--------
Every month a fresh reference is fitted on the previous month's healthy data
only. It learns how each sensor behaves when nothing is wrong, including how
load and the weather move it. It is then asked what temperature a healthy
machine would show in the NEXT month, given that month's load and weather. An
alarm is raised when the sensor runs hotter than that healthy expectation,
NOT when it merely crosses a fixed number: a fixed limit is also crossed by
perfectly healthy machines on hot or windy days, and the point of the model is
not to call those out.

Because the data are historical, the next month is already known, so the
prediction can be CONFIRMED against what actually happened. That is all this
module does. For every month it records:

  * how well the healthy model predicted the month (the error, in kelvin);
  * how many dispatches the model raised, against how many a plain fixed
    temperature limit would have raised;
  * how many of the fixed limit's alarms were EXPLAINED by load and weather,
    meaning the model expected that temperature and stayed quiet.

It works on what `live.monthly_cross_check` returns (the same engine the
dashboard runs for one month), so the history and the dashboard cannot
disagree. A month in which the machine is known to have had a real anomaly is
kept apart from the healthy ones: in a healthy month every alarm is a false
alarm, and in an anomaly month an alarm is what is wanted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .live import DISPATCH_MIN_RUN, dispatch_episodes

#: The fixed temperature limit an operator might set, in degrees Celsius.
HOT_C = 80.0

#: Rows below this share of rated power are standstill: no load, no information.
OPERATING_LOAD = 0.02

#: The alarm settings the history is re-scored at, without refitting anything:
#: every month already stores each row's z-score, so any threshold can be read
#: off it. SUSTAIN is how many consecutive ten-minute samples make a dispatch
#: (6 is an hour, 36 is six hours).
Z_GRID = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
SUSTAIN_GRID = (6, 36)

#: Spans in which a machine is known NOT to be healthy. R80711's gearbox
#: bearing overheat runs from late summer 2017; the span is widened to cover
#: the training month that would otherwise be fitted on it. It runs to the
#: end of the record: against its siblings R80711 ran hot on 11 of 31 days in
#: December 2017 and 6 of 11 in January 2018, so the fault had not ended.
ANOMALY_SPANS = {"R80711": [("2017-08-15", "2018-02-01")]}

#: The sibling comparison. Two turbines are compared at the same ten-minute
#: moment, only when their power output is similar (within this share of
#: rated power), so they feel the same weather and do the same work.
SIBLING_LOAD_MATCH = 0.10
#: A day counts only with at least this many comparable readings (6 hours).
SIBLING_MIN_READINGS = 36
#: The daily change is smoothed by a median over this many past days.
SIBLING_SMOOTH_DAYS = 5
#: The change from the turbine's normal gap, in kelvin, that raises a flag.
SIBLING_THRESHOLDS = (2.0, 2.5, 3.0)
#: For a CONFIRMED sibling call-out the model must also find unexplained heat
#: on that day: a z-score above this, somewhere in the day.
SIBLING_CONFIRM_Z = 2.0
#: The veto the full strategy uses: the setting that kept the limit's timing
#: on the real fault.
STRATEGY_VETO = "z2_s6"
#: A sibling call-out is known only once the day is complete, so it is
#: dispatched at the end of the day.
END_OF_DAY = " 23:50"


def daily_sibling_gap(temps: pd.DataFrame, loads: pd.DataFrame,
                      load_match: float = SIBLING_LOAD_MATCH,
                      min_readings: int = SIBLING_MIN_READINGS) -> dict:
    """Each turbine's daily gap to its siblings, at the same moment.

    `temps` and `loads` hold one column per turbine on one ten-minute UTC
    index. At each moment a turbine is compared with its siblings' median, but
    only where it is running (above standstill) and its power is within
    `load_match` of rated of the siblings' median: same weather, similar work.
    The day's value is the MEDIAN of those gaps; a day with fewer than
    `min_readings` comparable moments is left out.

    Returns {turbine: DataFrame indexed by UTC day, columns gap (K) and n}.
    """
    out = {}
    for t in temps.columns:
        peers = [p for p in temps.columns if p != t]
        if not peers:
            continue
        gap = temps[t] - temps[peers].median(axis=1)
        similar = (((loads[t] - loads[peers].median(axis=1)).abs() < load_match)
                   & (loads[t] > OPERATING_LOAD))
        g = gap[similar].dropna()
        day = g.groupby(g.index.floor("D")).agg(["median", "size"])
        day.columns = ["gap", "n"]
        out[t] = day[day["n"] >= min_readings]
    return out


def sibling_change(daily: pd.DataFrame, train_start, train_end, month_start,
                   month_end, smooth_days: int = SIBLING_SMOOTH_DAYS):
    """The smoothed change in a turbine's gap to its siblings, for one month.

    The turbine's NORMAL gap is the median daily gap over the same training
    window the model was fitted on, so both parts learn from the same healthy
    data. Each day's change from it is smoothed by a median over the past
    `smooth_days` days, which may reach back into the training window: past
    data only, so nothing is seen early. None if the training window holds
    too few comparable days to know the normal gap.
    """
    def utc(x):
        x = pd.Timestamp(x)
        return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")

    train_start, train_end = utc(train_start), utc(train_end)
    month_start, month_end = utc(month_start), utc(month_end)
    base = daily[(daily.index >= train_start) & (daily.index <= train_end)]["gap"]
    if len(base) < 10:
        return None
    normal = float(base.median())
    change = (daily["gap"] - normal)[daily.index < month_end]
    smooth = change.rolling(f"{smooth_days}D", min_periods=2).median()
    return smooth[(smooth.index >= month_start)]


def _runs(flagged_days: list) -> list[list]:
    """Consecutive calendar days, grouped into runs: one call-out per run."""
    runs: list[list] = []
    for d in sorted(flagged_days):
        if runs and (d - runs[-1][-1]) == pd.Timedelta(days=1):
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs


def month_is_healthy(turbine: str, month: str, spans: dict | None = None) -> bool:
    """Is every row of this turbine-month assumed healthy?

    The public data carry no fault labels, so health is an assumption: all of
    every turbine's record, except a documented anomaly.
    """
    spans = ANOMALY_SPANS if spans is None else spans
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    return not any(start < pd.Timestamp(hi) and end > pd.Timestamp(lo)
                   for lo, hi in spans.get(turbine, []))


def _first(frame: pd.DataFrame, episodes: list[dict]):
    return None if not episodes else str(frame["ts"].iloc[episodes[0]["start"]])[:16]


def summarise_month(result: dict, channel: str = "Gb2t_avg",
                    hot_c: float = HOT_C,
                    min_run: int = DISPATCH_MIN_RUN,
                    z_grid=Z_GRID, sustain_grid=SUSTAIN_GRID,
                    sibling=None,
                    sibling_thresholds=SIBLING_THRESHOLDS) -> dict:
    """One month, as `monthly_cross_check` returned it, reduced to its facts.

    `sibling` is the month's smoothed change in the turbine's gap to its
    siblings (`sibling_change`), or None when it cannot be computed.
    """
    f = result["frame"].reset_index(drop=True)
    prov = result["provenance"]
    actual = f[channel].to_numpy(dtype="float64")
    expected = f["predicted_weather"].to_numpy(dtype="float64")
    resid = actual - expected
    operating = (f["P_norm"].to_numpy(dtype="float64") > OPERATING_LOAD) \
        & np.isfinite(resid)
    r = resid[operating]

    model_flags = f["alarm_weather"].to_numpy(dtype=bool)
    hot = actual > hot_c
    model_eps = dispatch_episodes(model_flags, min_run)
    fixed_eps = dispatch_episodes(hot, min_run)
    # A fixed-limit alarm is EXPLAINED when the healthy model, given the load
    # and weather of that moment, expected that temperature and stayed quiet.
    explained = [e for e in fixed_eps
                 if not model_flags[e["start"]:e["end"] + 1].any()]
    with_hot = [e for e in model_eps if hot[e["start"]:e["end"] + 1].any()]

    # The same month re-scored at other alarm settings. Nothing is refitted: the
    # engine stored every row's z-score, so a stricter threshold is a filter.
    settings: dict = {}
    if "z_weather" in f.columns:
        z = f["z_weather"].to_numpy(dtype="float64")
        finite = np.isfinite(z)
        for zv in z_grid:
            flags = finite & (z > zv)
            for sustain in sustain_grid:
                eps = dispatch_episodes(flags, sustain)
                settings[f"z{zv:g}_s{sustain}"] = {
                    "dispatches": len(eps),
                    "with_hot_rows": sum(1 for e in eps
                                         if hot[e["start"]:e["end"] + 1].any()),
                    "first": _first(f, eps),
                }
    # And the plain fixed limit, held for the same lengths of time, so the two
    # are compared like for like.
    fixed_settings = {}
    for sustain in sustain_grid:
        eps = dispatch_episodes(hot, sustain)
        fixed_settings[f"s{sustain}"] = {"dispatches": len(eps),
                                         "first": _first(f, eps)}

    # The two-stage rule: the fixed limit triggers, the model vetoes. A limit
    # dispatch stands only if the healthy model ALSO finds the temperature
    # unexpected somewhere inside it; one that load and weather explain is
    # dropped. It is dispatched at the first moment both conditions hold: the
    # limit has been held long enough AND the model has flagged.
    #
    # A model can only veto what it can predict. Where it has no prediction
    # for most of the episode (a load it has hardly seen, so no reference),
    # the temperature is NOT explained, it is unobservable, and the limit
    # alarm stands.
    two_stage: dict = {}
    veto_times: list = []
    if "z_weather" in f.columns:
        for zv in z_grid:
            flags = finite & (z > zv)
            for sustain in sustain_grid:
                kept = []
                unobservable = 0
                for e in dispatch_episodes(hot, sustain):
                    seg = slice(e["start"], e["end"] + 1)
                    hit = np.flatnonzero(flags[seg])
                    if hit.size:
                        kept.append(max(e["start"] + sustain - 1,
                                        e["start"] + int(hit[0])))
                    elif finite[seg].mean() < 0.5:
                        kept.append(e["start"] + sustain - 1)
                        unobservable += 1
                kept.sort()
                key = f"z{zv:g}_s{sustain}"
                if key == STRATEGY_VETO:
                    veto_times = [str(f["ts"].iloc[i])[:16] for i in kept]
                two_stage[key] = {
                    "dispatches": len(kept),
                    "kept_unobservable": unobservable,
                    "first": (None if not kept
                              else str(f["ts"].iloc[kept[0]])[:16]),
                }

    # The sibling warning, and the full strategy. A run of consecutive flagged
    # days is ONE call-out, dispatched at the end of its first day (a daily
    # median is known only once the day is over). A CONFIRMED call-out also
    # needs the model to find unexplained heat that day, and is dispatched on
    # the first day of the run where it does. The full strategy is the veto
    # PLUS the sibling call-outs, counted once per day.
    sib: dict = {}
    sib_conf: dict = {}
    strategy: dict = {}
    strategy_conf: dict = {}
    if sibling is not None and "z_weather" in f.columns:
        day_of = pd.to_datetime(f["ts"], utc=True).dt.floor("D")
        zmax = pd.Series(np.where(np.isfinite(z), z, -np.inf)).groupby(
            day_of.to_numpy()).max()
        zmax.index = pd.to_datetime(zmax.index, utc=True)
        veto_days = {t[:10] for t in veto_times}

        def combine(callouts):
            extra = [c for c in callouts if c[:10] not in veto_days]
            events = sorted(veto_times + extra)
            return {"dispatches": len(events),
                    "first": events[0] if events else None}

        for thr in sibling_thresholds:
            flagged = [d for d, v in sibling.items() if np.isfinite(v) and v > thr]
            runs = _runs(flagged)
            plain = [str(run[0])[:10] + END_OF_DAY for run in runs]
            confirmed = []
            for run in runs:
                ok = [d for d in run if float(zmax.get(d, -np.inf)) > SIBLING_CONFIRM_Z]
                if ok:
                    confirmed.append(str(ok[0])[:10] + END_OF_DAY)
            key = f"t{thr:g}"
            sib[key] = {"dispatches": len(plain),
                        "first": plain[0] if plain else None}
            sib_conf[key] = {"dispatches": len(confirmed),
                             "first": confirmed[0] if confirmed else None}
            strategy[key] = combine(plain)
            strategy_conf[key] = combine(confirmed)

    return {
        "rows": int(len(f)),
        "operating_rows": int(operating.sum()),
        "train_rows": int(prov.get("train_rows", 0)),
        "clean_rows": int(prov.get("clean_rows", 0)),
        "dropped_rows": int(prov.get("dropped_rows", 0)),
        "mae_k": round(float(np.abs(r).mean()), 3) if r.size else None,
        "rmse_k": round(float(np.sqrt((r ** 2).mean())), 3) if r.size else None,
        "bias_k": round(float(r.mean()), 3) if r.size else None,
        "p95_abs_k": round(float(np.quantile(np.abs(r), 0.95)), 3) if r.size else None,
        "hot_rows": int(hot.sum()),
        "model_dispatches": len(model_eps),
        "model_dispatches_with_hot_rows": len(with_hot),
        "fixed_dispatches": len(fixed_eps),
        "fixed_dispatches_explained": len(explained),
        "first_model_dispatch": _first(f, model_eps),
        "first_fixed_dispatch": _first(f, fixed_eps),
        "settings": settings,
        "fixed_settings": fixed_settings,
        "two_stage": two_stage,
        "sibling": sib,
        "sibling_confirmed": sib_conf,
        "strategy": strategy,
        "strategy_confirmed": strategy_conf,
    }


def _years(months: int) -> float:
    return months / 12.0


def aggregate(rows: list[dict]) -> dict:
    """Per turbine and for the fleet.

    The false-alarm figures use HEALTHY months only, where every alarm is a
    false alarm by assumption; the anomaly months are reported separately.
    """
    def block(sel: list[dict]) -> dict:
        healthy = [r for r in sel if r["healthy_assumed"]]
        anomaly = [r for r in sel if not r["healthy_assumed"]]
        maes = [r["mae_k"] for r in healthy if r["mae_k"] is not None]
        biases = [r["bias_k"] for r in healthy if r["bias_k"] is not None]
        yrs = _years(len(healthy))
        model = sum(r["model_dispatches"] for r in healthy)
        fixed = sum(r["fixed_dispatches"] for r in healthy)
        explained = sum(r["fixed_dispatches_explained"] for r in healthy)

        def per_year(n):
            return round(n / yrs, 3) if yrs > 0 else None

        def sweep(field: str) -> dict:
            """Per alarm setting: the false dispatches in healthy months, and how
            many anomaly months were alarmed at all."""
            keys = sorted({k for r in sel for k in (r.get(field) or {})})
            out = {}
            for k in keys:
                hn = sum((r[field].get(k) or {}).get("dispatches", 0)
                         for r in healthy if r.get(field))
                an = [(r[field].get(k) or {}).get("dispatches", 0)
                      for r in anomaly if r.get(field)]
                out[k] = {
                    "healthy_dispatches": hn,
                    "healthy_dispatches_per_turbine_year": per_year(hn),
                    "anomaly_months": len(an),
                    "anomaly_months_alarmed": sum(1 for x in an if x > 0),
                    "anomaly_dispatches": sum(an),
                }
            return out

        return {
            "months": len(sel),
            "healthy_months": len(healthy),
            "anomaly_months": len(anomaly),
            "median_mae_k": round(float(np.median(maes)), 3) if maes else None,
            "p90_mae_k": round(float(np.quantile(maes, 0.9)), 3) if maes else None,
            "median_bias_k": round(float(np.median(biases)), 3) if biases else None,
            "share_months_within_1k_bias": (
                round(float(np.mean([abs(b) < 1.0 for b in biases])), 4)
                if biases else None),
            "healthy_model_dispatches": model,
            "healthy_fixed_limit_dispatches": fixed,
            "healthy_fixed_limit_dispatches_explained": explained,
            "healthy_model_dispatches_per_turbine_year": per_year(model),
            "healthy_fixed_limit_dispatches_per_turbine_year": per_year(fixed),
            "anomaly_model_dispatches": sum(r["model_dispatches"] for r in anomaly),
            "anomaly_fixed_limit_dispatches": sum(r["fixed_dispatches"] for r in anomaly),
            "settings": sweep("settings"),
            "fixed_settings": sweep("fixed_settings"),
            "two_stage": sweep("two_stage"),
            "sibling": sweep("sibling"),
            "sibling_confirmed": sweep("sibling_confirmed"),
            "strategy": sweep("strategy"),
            "strategy_confirmed": sweep("strategy_confirmed"),
        }

    turbines = sorted({r["turbine"] for r in rows})
    return {
        "fleet": block(rows),
        "by_turbine": {t: block([r for r in rows if r["turbine"] == t])
                       for t in turbines},
    }


def sibling_episodes(daily: pd.DataFrame, threshold: float = 2.0,
                     smooth_days: int = SIBLING_SMOOTH_DAYS,
                     baseline_days: int = 30) -> list[dict]:
    """Every stretch in which a turbine ran hotter than its siblings than usual.

    For the whole record: each day's gap against the median gap of the
    previous `baseline_days` days (its normal at the time), smoothed by a
    median over the past `smooth_days` days. A run of consecutive days above
    `threshold` K is one episode. This is for inspection, not alarming: it
    lists the dates so a reader can judge what they were.
    """
    normal = daily["gap"].rolling(f"{baseline_days}D", closed="left",
                                  min_periods=10).median()
    change = (daily["gap"] - normal).dropna()
    smooth = change.rolling(f"{smooth_days}D", min_periods=2).median()
    episodes = []
    for run in _runs([d for d, v in smooth.items() if v > threshold]):
        seg = smooth[(smooth.index >= run[0]) & (smooth.index <= run[-1])]
        episodes.append({"start": str(run[0])[:10], "end": str(run[-1])[:10],
                         "days": len(run), "peak_k": round(float(seg.max()), 2)})
    return episodes
