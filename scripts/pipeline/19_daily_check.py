#!/usr/bin/env python
"""The daily check, over every day of the four ENGIE turbines.

Runs the policy of `src/windmill/daily_check.py` and writes
`results/daily_check.json` (the figures) and `results/daily_series.json` (every
day of every turbine at the dashboard's tolerance, compact, for the dashboard).

    python scripts/pipeline/19_daily_check.py                       # everything
    python scripts/pipeline/19_daily_check.py --turbines R80711 --out /tmp/a.json
    python scripts/pipeline/19_daily_check.py --merge /tmp/a.json /tmp/b.json ...

The policy in one paragraph. Each month the model is trained on at least a
month of healthy readings from before the month (never standstill, never the
warm-up after a restart, never a flagged day, never data from before the last
maintenance) and then frozen for the month. Every day it predicts what a
healthy sensor should read from that day's power and weather, and the day's
mean difference from what the sensor shows is compared with a tolerance. A
run of hot days is one call-out. A confirmed fault keeps the model frozen
until maintenance; after maintenance the model is blank until a new month of
healthy data exists, and only the fixed limit protects the turbine.

The public data have no alarm, inspection or maintenance records: a stop of
three days or more stands in for maintenance, and the documented anomaly span
stands in for what an inspection would find. A partial run is never written
over the committed result; `--merge` assembles per-turbine runs.
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

from windmill import daily_check as dc  # noqa: E402
from windmill import live  # noqa: E402
from windmill.confirmation import ANOMALY_SPANS  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "daily_check.json"
SERIES_OUT = ROOT / "results" / "daily_series.json"
#: The tolerance the dashboard shows, run on every sensor with the forecast.
PRIMARY = 2.0
#: The forecast leads kept for every sensor in the series (all leads for the
#: primary sensor).
SERIES_LEADS = (1, 3, 7)
TURBINES = ("R80711", "R80721", "R80736", "R80790")
#: The case study's days kept in full, so the report can show them.
CASE = "R80711"
CASE_FROM, CASE_TO = "2017-07-01", "2018-02-01"


def _frame(data_dir: Path, turbine: str) -> pd.DataFrame:
    return dc.with_era5(live._monthly_frame(data_dir, "engie", turbine), data_dir)


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def series(sim: dict) -> dict:
    """One turbine's days, column by column, for the dashboard."""
    days = sim["days"]
    out = {"day": [d["day"] for d in days],
           "status": [d.get("status") for d in days],
           "reason": [d.get("reason") for d in days],
           "state": [d.get("state") for d in days],
           "flag": [d.get("flag") for d in days],
           "maintenance": [d.get("maintenance") for d in days],
           "collected": [d.get("collected") for d in days],
           "fault_period": [d["fault_period"] for d in days],
           "gearbox_hot": [dc.gearbox_hot(d, sim["tolerance"]) if d.get("ch") else None
                           for d in days],
           "sensors": {}}
    for c in dc.SENSORS:
        col = {"actual": [_r((d.get("ch") or {}).get(c, [None])[0]) for d in days],
               "pred": [_r(((d.get("ch") or {}).get(c) or [None, None])[1]) for d in days]}
        leads = dc.LEADS if c == dc.CHANNEL else SERIES_LEADS
        for L in leads:
            col[f"f{L}"] = [_r(((d.get("fc") or {}).get(str(L)) or {}).get(c))
                            for d in days]
        if any(v is not None for v in col["actual"]):
            out["sensors"][c] = col
    return out


def run(data_dir: Path, turbines: list[str], tolerances: list[float],
        warmup_hours: float) -> dict:
    t0 = time.time()
    rows, case_days, months, ser, all_months = [], {}, {}, {}, {}
    for turbine in turbines:
        frame = _frame(data_dir, turbine)
        limit = dc.limit_call_outs(frame, turbine)
        for tol in tolerances:
            full = abs(tol - PRIMARY) < 1e-9
            sim = dc.simulate(frame, turbine, tol, warmup_hours=warmup_hours,
                              channels=list(dc.SENSORS) if full else None,
                              leads=dc.LEADS if full else ())
            rows.append(dc.summarise(sim, limit))
            if full:
                ser[turbine] = series(sim)
                all_months[turbine] = sim["months"]
            if turbine == CASE:
                case_days[f"{tol:g}"] = [
                    {k: v for k, v in d.items() if k not in ("ch", "fc")}
                    for d in sim["days"] if CASE_FROM <= d["day"] < CASE_TO]
                months[f"{tol:g}"] = sim["months"]
            print(f"  {turbine} tolerance {tol:g} K: "
                  f"{rows[-1]['false_call_outs_per_year']} false call-outs a "
                  f"year, first on the fault {rows[-1]['first_fault_call_out']}"
                  f"  ({time.time() - t0:.0f}s)")
    return {"rows": rows, "case_days": case_days, "case_months": months,
            "series": ser, "months_primary": all_months,
            "seconds": round(time.time() - t0, 1)}


def siblings(data_dir: Path, months_by_turbine: dict) -> tuple[dict, dict]:
    """Each turbine against its siblings, with and without the wind-direction
    normal: the call-outs per threshold, and the smoothed daily change."""
    frames = {t: _frame(data_dir, t).set_index("ts") for t in TURBINES}
    temps = pd.DataFrame({t: f[dc.CHANNEL] for t, f in frames.items()})
    loads = pd.DataFrame({t: f["P_norm"] for t, f in frames.items()})
    direction = pd.concat([f["era5_wdir_100m"] for f in frames.values()],
                          axis=1).median(axis=1)
    summary, changes = {}, {}
    for t in TURBINES:
        changes[t], summary[t] = {}, {}
        for mode, aware in (("direction", True), ("overall", False)):
            ch = dc.sibling_change(temps, loads, direction, t,
                                   months_by_turbine[t], by_direction=aware)
            changes[t][mode] = {str(k)[:10]: round(float(v), 2)
                                for k, v in ch.items() if pd.notna(v)}
            healthy = [k for k in ch.index if pd.notna(ch[k])
                       and not dc.in_fault(t, k)]
            years = len(healthy) / 365.25
            per = {}
            for thr in dc.SIBLING_THRESHOLDS:
                calls = dc.sibling_call_outs(ch, thr, t)
                false = [c for c in calls if not c["fault_period"]]
                fault = [c for c in calls if c["fault_period"]]
                per[f"{thr:g}"] = {
                    "false_call_outs_per_year": round(len(false) / years, 3) if years else None,
                    "first_fault_call_out": fault[0]["first"] if fault else None,
                }
            summary[t][mode] = {"days": len(ch.dropna()), "by_threshold": per}
    return summary, changes


def write_series(path: Path, ser: dict, changes: dict) -> None:
    for t, s in ser.items():
        c = changes.get(t, {})
        s["sibling"] = [c.get("direction", {}).get(d) for d in s["day"]]
        s["sibling_overall"] = [c.get("overall", {}).get(d) for d in s["day"]]
    doc = {"generated": pd.Timestamp.now(tz="UTC").isoformat(),
           "tolerance_k": PRIMARY, "primary": dc.CHANNEL,
           "sensor_names": dc.SENSOR_NAMES, "leads": list(dc.LEADS),
           "series_leads": list(SERIES_LEADS),
           "month_of_readings": dc.MONTH_OF_READINGS,
           "forecast_sigma": {str(k): v for k, v in dc.FORECAST_SIGMA.items()},
           "anomaly_spans": {k: [list(x) for x in v] for k, v in ANOMALY_SPANS.items()},
           "turbines": ser}
    path.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def fleet(rows: list[dict]) -> dict:
    """Per tolerance, the healthy siblings pooled, beside the case turbine."""
    out: dict = {}
    for tol in sorted({r["tolerance"] for r in rows}):
        sel = [r for r in rows if r["tolerance"] == tol]
        sib = [r for r in sel if r["turbine"] != CASE]
        case = next((r for r in sel if r["turbine"] == CASE), None)
        years = sum(r["healthy_running_days"] for r in sib) / 365.25
        out[f"{tol:g}"] = {
            "siblings_false_call_outs_per_year": (
                round(sum(r["false_call_outs"] for r in sib) / years, 3)
                if years else None),
            "siblings_cold_days_per_year": (
                round(sum(r["cold_days_per_year"] * r["healthy_running_days"]
                          / 365.25 for r in sib) / years, 3) if years else None),
            "case_false_call_outs_per_year": (case or {}).get("false_call_outs_per_year"),
            "case_first_fault_call_out": (case or {}).get("first_fault_call_out"),
            "case_fault_days_hot": (case or {}).get("fault_days_hot"),
            "case_fault_days_checked": (case or {}).get("fault_days_checked"),
        }
    return out


def build(part: dict, turbines: list[str], tolerances: list[float],
          warmup_hours: float) -> dict:
    return {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "seconds": part["seconds"],
        "complete_run": set(turbines) == set(TURBINES),
        "protocol": {
            "channel": dc.CHANNEL,
            "training": ("at least a month of healthy readings before the "
                         "month: the whole previous month, reached back a week "
                         f"at a time to at most {dc.MAX_LOOKBACK_DAYS} days and "
                         "never before the last maintenance; standstill, the "
                         f"{warmup_hours:g} h warm-up after a restart and every "
                         "flagged day left out; rows trimmed as in the engine"),
            "frozen": "the month's model never learns from the month itself",
            "check": ("the day's mean of actual minus predicted, over running "
                      f"readings outside the warm-up, at least {dc.MIN_DAY_ROWS} "
                      "of them; hot above the tolerance, cold below minus it"),
            "call_out": ("a run of consecutive hot days, dispatched at the end "
                         "of its first day"),
            "confirmed_fault": ("a hot day inside a documented anomaly span "
                                "(the stand-in for an inspection); the model "
                                "then stays frozen until maintenance"),
            "false_alarm": ("a hot day outside any anomaly span: the stand-in "
                            "inspection clears it, and the day is never "
                            "trained on"),
            "maintenance": (f"a restart after {dc.MAINTENANCE_DAYS:g} days or "
                            "more without running; the model is blank until a "
                            "new month of healthy data exists"),
            "restart": (f"the first running reading after "
                        f"{dc.OFF_HOURS_FOR_RESTART:g} h without running"),
            "warmup_hours": warmup_hours,
            "tolerances_k": tolerances,
            "fixed_limit": f"{dc.HOT_C:g} C held for an hour, for comparison",
            "anomaly_spans": {k: [list(s) for s in v]
                              for k, v in ANOMALY_SPANS.items()},
            "turbines": turbines,
            "sensors": list(dc.SENSORS),
            "every_sensor_at_k": PRIMARY,
            "forecast": ("ERA5 plus a stated error, a bias per day and noise "
                         "per hour, whose sigma is "
                         f"{dc.FORECAST_SIGMA[dc.LEADS[0]]:g} a day ahead rising "
                         f"to {dc.FORECAST_SIGMA[dc.LEADS[-1]]:g} a week ahead "
                         "(m/s for wind, K for temperature); power from the "
                         "turbine's own power curve against ERA5 wind; the "
                         "model in force when the forecast is issued"),
            "siblings": ("the turbine minus its siblings' median at the same "
                         f"moment, power within {dc.SIBLING_LOAD_MATCH:g} of "
                         "rated; the normal gap learned from the model's "
                         "training window, per 45-degree ERA5 wind sector "
                         "('direction') or overall; the daily median smoothed "
                         f"over {dc.SIBLING_SMOOTH_DAYS} days"),
        },
        "by_turbine": part["rows"],
        "fleet": fleet(part["rows"]),
        "case_days": part["case_days"],
        "case_months": part["case_months"],
        "siblings": part.get("siblings"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--turbines", default=",".join(TURBINES))
    ap.add_argument("--tolerances", default=",".join(f"{t:g}" for t in dc.TOLERANCES))
    ap.add_argument("--warmup-hours", type=float, default=dc.WARMUP_HOURS)
    ap.add_argument("--merge", nargs="+", metavar="FILE")
    args = ap.parse_args()
    out_path = Path(args.out)

    if args.merge:
        parts = [json.loads(Path(p).read_text()) for p in args.merge]
        warm = {p["protocol"]["warmup_hours"] for p in parts}
        tols = {tuple(p["protocol"]["tolerances_k"]) for p in parts}
        if len(warm) != 1 or len(tols) != 1:
            raise SystemExit("cannot merge parts with different settings")
        turbines = [t for t in TURBINES
                    if any(t in p["protocol"]["turbines"] for p in parts)]
        part = {"rows": [r for p in parts for r in p["by_turbine"]],
                "case_days": next((p["case_days"] for p in parts if p["case_days"]), {}),
                "case_months": next((p["case_months"] for p in parts if p["case_months"]), {}),
                "series": {t: s for p in parts for t, s in (p.get("series") or {}).items()},
                "months_primary": {t: m for p in parts
                                   for t, m in (p.get("months_primary") or {}).items()},
                "seconds": sum(p["seconds"] for p in parts)}
        tolerances, warmup = list(tols.pop()), warm.pop()
    else:
        turbines = [t for t in args.turbines.split(",") if t]
        tolerances = [float(t) for t in args.tolerances.split(",") if t]
        warmup = args.warmup_hours
        part = run(Path(args.data), turbines, tolerances, warmup)

    complete = set(turbines) == set(TURBINES)
    if complete and set(part["months_primary"]) == set(TURBINES):
        part["siblings"], changes = siblings(Path(args.data), part["months_primary"])
        for t, s in part["siblings"].items():
            print(f"  siblings {t}: " + ", ".join(
                f"{m} {s[m]['by_threshold'][f'{PRIMARY:g}']['false_call_outs_per_year']}/yr"
                for m in s))
    out = build(part, turbines, tolerances, warmup)
    if not out["complete_run"] and out_path.resolve() == DEFAULT_OUT.resolve():
        print(f"a partial run will not overwrite {DEFAULT_OUT}; pass --out")
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out["complete_run"] and part.get("siblings"):
        write_series(out_path.parent / SERIES_OUT.name, part["series"], changes)
    else:
        # A part keeps what the merge needs.
        out["series"] = part["series"]
        out["months_primary"] = part["months_primary"]
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
