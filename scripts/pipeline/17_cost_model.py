#!/usr/bin/env python
"""Price the operating curve: what each arm is worth, under stated assumptions.

Reads `results/operating_curve.json` and writes `results/cost_model.json`.
It needs no raw data and runs in a moment, so it is cheap to re-run whenever
the curve changes (`make curve`, then `make cost`, then `make report`).

    python scripts/pipeline/17_cost_model.py

What is measured, and what is assumed, is kept strictly apart:

  MEASURED   the false dispatches per turbine-year and the share of simulated
             faults caught, at every setting of every arm. They come from the
             operating curve, not from here.
  ASSUMED    the cost of a false dispatch, the value of catching a fault
             early, and how often a fault turns up. The public data has no
             maintenance log, so these are placeholders for the reader to
             replace; `windmill.cost` holds them and the dashboard lets the
             reader move them.

For each arm this records its most profitable setting under the default
assumptions, what it would be worth at the standing setting, how its best
value changes with the fault rate, and in what share of the assumption grid
it is the most profitable arm. The arithmetic is all in `windmill.cost`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import cost  # noqa: E402

CURVE = ROOT / "results" / "operating_curve.json"
OUT = ROOT / "results" / "cost_model.json"

#: The ramp the detection figures are taken at, in kelvin.
RAMP_K = 4.0


def price_arm(points: list[dict], arm: str, a: "cost.CostAssumptions") -> dict:
    """One arm's economics for one fault shape."""
    return {
        "best_setting": cost.best_setting(points, arm, a, RAMP_K),
        "standing_setting": cost.setting_at(
            points, arm, cost.STANDING_RATE, cost.STANDING_PERSISTENCE,
            a, RAMP_K),
        "net_value_by_fault_rate": cost.net_value_curve(
            points, arm, cost.FAULT_RATES, a, RAMP_K),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--curve", default=str(CURVE))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    curve_path, out_path = Path(args.curve), Path(args.out)
    if not curve_path.exists():
        print(f"{curve_path} not found; run `make curve` first")
        return 2
    curve = json.loads(curve_path.read_text())
    points = curve["points"]
    arms = curve["protocol"]["arms"]
    a = cost.CostAssumptions()
    # The fault shapes the curve was swept for. The flat offset is always
    # there; the load-proportional sweep is there once `make curve` has run
    # with the extras.
    shapes = {"flat": points}
    if curve.get("points_ramp_load_proportional"):
        shapes["load_proportional"] = curve["points_ramp_load_proportional"]

    out = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": "results/operating_curve.json",
        "fault_shapes": list(shapes),
        "curve_complete_run": curve.get("complete_run"),
        "ramp_k": RAMP_K,
        "assumptions": {
            **a.as_dict(),
            "status": ("ASSUMED, not measured: the public data has no "
                       "maintenance log. They are placeholders to be replaced."),
        },
        "standing_setting": {
            "target_rate": cost.STANDING_RATE,
            "persistence_samples": cost.STANDING_PERSISTENCE,
        },
        "fault_rate_axis": list(cost.FAULT_RATES),
        "ranking_robustness": {
            "assumption_grid_cells": (len(cost.DISPATCH_COSTS)
                                      * len(cost.FAULT_RATES)
                                      * len(cost.EARLY_VALUES)),
            "share_where_arm_is_most_profitable":
                cost.share_of_assumptions_where_best(points, arms,
                                                     ramp_k=RAMP_K),
        },
        "arms": {},
    }
    if "load_proportional" in shapes:
        out["ranking_robustness"]["share_where_arm_is_most_profitable_load_proportional"] = (
            cost.share_of_assumptions_where_best(
                shapes["load_proportional"], arms, ramp_k=RAMP_K))
        out["ranking_robustness"]["share_where_arm_is_best_in_the_worse_case"] = (
            cost.share_where_best_in_the_worse_case(shapes, arms,
                                                    ramp_k=RAMP_K))
    for arm in arms:
        out["arms"][arm] = price_arm(points, arm, a)
        if "load_proportional" in shapes:
            out["arms"][arm]["load_proportional"] = price_arm(
                shapes["load_proportional"], arm, a)

    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e3:.0f} KB)")
    for arm in arms:
        b, s = out["arms"][arm]["best_setting"], out["arms"][arm]["standing_setting"]
        edge = f"  [edge: {', '.join(b['at_grid_edge'])}]" if b["at_grid_edge"] else ""
        print(f"  {arm:13s} best {b['net_value']:>9,.0f}   standing {s['net_value']:>9,.0f}{edge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
