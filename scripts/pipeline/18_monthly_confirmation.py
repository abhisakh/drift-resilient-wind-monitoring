#!/usr/bin/env python
"""The historical confirmation: train on last month, predict this month, check.

Runs the dashboard's monthly loop over EVERY month of the four ENGIE turbines
and writes `results/monthly_confirmation.json`.

    python scripts/pipeline/18_monthly_confirmation.py                  # all months
    python scripts/pipeline/18_monthly_confirmation.py --stride 4 --out /tmp/c.json
    python scripts/pipeline/18_monthly_confirmation.py --turbines R80711 --out /tmp/a.json
    python scripts/pipeline/18_monthly_confirmation.py --merge /tmp/a.json /tmp/b.json ...

The project in one paragraph. Every month a fresh reference is fitted on the
previous month's healthy data only. It learns how each sensor behaves when
nothing is wrong, including how load and weather move it. It is then asked what
a healthy machine would show next month, given that month's load and weather,
and an alarm is raised when the sensor runs hotter than that expectation, not
merely above a fixed number, because a fixed limit is also crossed by healthy
machines on hot or windy days.

The data are historical, so the next month is already known and the prediction
can be CONFIRMED. For every turbine-month this records how well the healthy
model predicted the month, how many dispatches it raised, how many a plain
fixed limit would have raised, and how many of those fixed-limit alarms were
explained by load and weather (the model expected that temperature and stayed
quiet). Months in which a machine is known to have had a real anomaly are kept
apart from the healthy ones.

It runs the same engine the dashboard uses, so the history and the dashboard
cannot disagree. A partial run (a stride, or fewer turbines) is never written
over the committed result; pass --out to put it somewhere else. `--merge`
assembles per-turbine runs into the complete result, which is useful on a
machine where one run is too slow to do in a sitting.
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

from windmill import RATED_POWER_KW, confirmation, live, loaders  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "monthly_confirmation.json"
TURBINES = ("R80711", "R80721", "R80736", "R80790")

#: The alarm threshold, in robust standard deviations: the dashboard's default.
Z_THRESHOLD = 3.0


def build(rows: list[dict], skipped: list[dict], turbines: list[str],
          stride: int, z: float, seconds: float, episodes=None) -> dict:
    """The artefact: the protocol, every month, and the aggregate."""
    return {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "seconds": round(seconds, 1),
        "complete_run": stride == 1 and set(turbines) == set(TURBINES),
        "protocol": {
            "channel": "Gb2t_avg",
            "train": ("at least one month (4,320) of healthy ten-minute "
                      "readings: the whole previous calendar month, reached "
                      "further back a week at a time (at most 92 days) until "
                      "the rows the reference itself calls anomalous are "
                      "trimmed out and a month of readings remains"),
            "predict": ("the whole month, from the turbine's measured load and "
                        "the ERA5 ambient temperature"),
            "z_threshold": z,
            "alarm": f"a robust z-score above {z:g} on the healthy model's residual",
            "dispatch": (f"an alarm sustained for at least "
                         f"{live.DISPATCH_MIN_RUN} consecutive ten-minute "
                         "samples"),
            "fixed_limit_c": confirmation.HOT_C,
            "healthy_assumed": ("every month of every turbine, except the "
                                "documented anomaly spans"),
            "anomaly_spans": {k: [list(s) for s in v]
                              for k, v in confirmation.ANOMALY_SPANS.items()},
            "turbines": turbines,
            "stride": stride,
            "sibling": {
                "compare": ("the turbine's bearing temperature against its "
                            "siblings' median at the same ten-minute moment, "
                            "only where its power is within "
                            f"{confirmation.SIBLING_LOAD_MATCH:g} of rated of "
                            "theirs"),
                "day": (f"the median gap over the day, with at least "
                        f"{confirmation.SIBLING_MIN_READINGS} comparable "
                        "readings"),
                "normal": "the median daily gap over the model's training window",
                "smoothing_days": confirmation.SIBLING_SMOOTH_DAYS,
                "thresholds_k": list(confirmation.SIBLING_THRESHOLDS),
                "call_out": ("a run of consecutive flagged days is one "
                             "call-out, dispatched at the end of its first day"),
                "confirmed": (f"the model must also find a z-score above "
                              f"{confirmation.SIBLING_CONFIRM_Z:g} that day"),
                "strategy": (f"the veto ({confirmation.STRATEGY_VETO}) plus the "
                             "sibling call-outs, counted once per day"),
            },
        },
        "months": rows,
        "skipped": skipped,
        "summary": confirmation.aggregate(rows),
        "r80711_sibling_episodes": episodes,
    }


def sibling_gaps(data_dir: Path) -> dict:
    """Every turbine's daily gap to its siblings, at the same moment.

    The comparison needs all four turbines, so they are loaded together even
    when only one turbine is being scored.
    """
    d = loaders.load_engie(data_dir, variant="F99", turbines=TURBINES)
    d["ts"] = pd.to_datetime(d["ts"], utc=True).dt.floor("10min")
    d["P_norm"] = (d["P_avg"] / RATED_POWER_KW).clip(0, 1)
    temps = d.pivot_table(index="ts", columns="turbine", values="Gb2t_avg")
    loads = d.pivot_table(index="ts", columns="turbine", values="P_norm")
    return confirmation.daily_sibling_gap(temps, loads)


def run_turbines(data_dir: Path, turbines: list[str], stride: int, z: float):
    t0 = time.time()
    rows: list[dict] = []
    skipped: list[dict] = []
    gaps = sibling_gaps(data_dir)
    for turbine in turbines:
        months = live.available_months(data_dir, "engie", turbine)
        # The first month has no month before it to train on.
        for month in months[1::stride]:
            try:
                res = live.monthly_cross_check(
                    month, data_dir, source="engie", turbine=turbine,
                    z_threshold=z)
            except live.LiveEngineError as exc:
                skipped.append({"turbine": turbine, "month": month,
                                "reason": str(exc)[:160]})
                continue
            prov = res["provenance"]
            m_start = pd.Timestamp(month + "-01", tz="UTC")
            sibling = (confirmation.sibling_change(
                gaps[turbine], prov["train_start"], prov["train_end"], m_start,
                m_start + pd.offsets.MonthBegin(1))
                if turbine in gaps and "train_start" in prov else None)
            rows.append({
                "turbine": turbine,
                "month": month,
                "healthy_assumed": confirmation.month_is_healthy(turbine, month),
                **confirmation.summarise_month(res, sibling=sibling),
            })
            if len(rows) % 20 == 0:
                print(f"  {len(rows)} turbine-months scored "
                      f"({turbine} {month}), {time.time() - t0:.0f}s elapsed")
    episodes = (confirmation.sibling_episodes(gaps["R80711"])
                if "R80711" in turbines and "R80711" in gaps else None)
    return rows, skipped, time.time() - t0, episodes


def merge(paths: list[str]):
    """Assemble per-turbine runs into one. Every part must be a full-stride run
    with the same alarm threshold, so the whole is exactly what one run gives."""
    parts = [json.loads(Path(p).read_text()) for p in paths]
    strides = {p["protocol"]["stride"] for p in parts}
    zs = {p["protocol"]["z_threshold"] for p in parts}
    if strides != {1} or len(zs) != 1:
        raise SystemExit(f"cannot merge: strides {sorted(strides)}, "
                         f"thresholds {sorted(zs)}; every part must be stride 1 "
                         "with one threshold")
    seen = [t for p in parts for t in p["protocol"]["turbines"]]
    if len(seen) != len(set(seen)):
        raise SystemExit(f"cannot merge: a turbine appears twice in {seen}")
    rows = [r for p in parts for r in p["months"]]
    skipped = [s for p in parts for s in p["skipped"]]
    turbines = [t for t in TURBINES if t in seen]
    episodes = next((p.get("r80711_sibling_episodes") for p in parts
                     if p.get("r80711_sibling_episodes") is not None), None)
    return (rows, skipped, turbines, 1, zs.pop(),
            sum(p["seconds"] for p in parts), episodes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--turbines", default=",".join(TURBINES))
    ap.add_argument("--z", type=float, default=Z_THRESHOLD)
    ap.add_argument("--stride", type=int, default=1,
                    help="score every n-th month only, for a fast look")
    ap.add_argument("--merge", nargs="+", metavar="FILE",
                    help="assemble these per-turbine runs instead of running")
    args = ap.parse_args()
    out_path = Path(args.out)

    if args.merge:
        rows, skipped, turbines, stride, z, seconds, episodes = merge(args.merge)
    else:
        turbines = [t for t in args.turbines.split(",") if t]
        stride, z = max(1, args.stride), args.z
        if (stride != 1 or set(turbines) != set(TURBINES)) \
                and out_path.resolve() == DEFAULT_OUT.resolve():
            print("this is a partial run (a stride, or fewer turbines), so it "
                  f"will not overwrite\n{DEFAULT_OUT}\npass --out <somewhere "
                  "else> to write it, then --merge the parts.")
            return 2
        rows, skipped, seconds, episodes = run_turbines(Path(args.data),
                                                        turbines, stride, z)

    out = build(rows, skipped, turbines, stride, z, seconds, episodes)
    if not out["complete_run"] and out_path.resolve() == DEFAULT_OUT.resolve():
        print("the merged parts do not cover every turbine, so they will not "
              f"overwrite\n{DEFAULT_OUT}")
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    f = out["summary"]["fleet"]
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e3:.0f} KB): "
          f"{len(rows)} turbine-months, {len(skipped)} skipped")
    print(f"  healthy months: median error {f['median_mae_k']} K; "
          f"dispatches per turbine-year: model "
          f"{f['healthy_model_dispatches_per_turbine_year']}, fixed "
          f"{confirmation.HOT_C:g} C limit "
          f"{f['healthy_fixed_limit_dispatches_per_turbine_year']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
