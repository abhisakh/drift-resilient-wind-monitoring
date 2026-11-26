#!/usr/bin/env python
"""A frozen model in the wrong season, and last year's same month as the fix.

Writes `results/frozen_fallback.json`.

    python scripts/pipeline/25_frozen_fallback.py   (after `make daily`; needs the raw ENGIE release)

After a fault is confirmed the turbine's model is frozen until maintenance,
so it cannot learn the fault. R80711's was trained in May-July 2017 and then
used every day to January 2018: a summer model on winter days. On some winter
days the fixed 80 C limit fired and that model did not: a summer model has
almost no high-load readings to learn from, so on a windy winter day it
cannot judge the hottest hours at all, and it is a season out of date on the
rest.

The fix (windmill.daily_check.simulate, `last_year_when_frozen`): in a frozen
month, judge the turbine with its own healthy model for the same month a year
earlier. Same season, one year old, trained before the fault. It is used only
if the component still behaves as it did then (`daily_check.last_year_ok`).

The daily check is run here both ways on R80711, over the months after the
freeze. The control: each healthy sibling judged with its own last-year
models over the same months, where every hot day is a false one.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import daily_check as dc  # noqa: E402
from windmill import live  # noqa: E402

OUT = ROOT / "results" / "frozen_fallback.json"
TURBINES = ("R80711", "R80721", "R80736", "R80790")
FAULTY = "R80711"
FROM, TO = "2017-09-01", "2018-02-01"
JUDGED = ("2017-09", "2017-10", "2017-11", "2017-12", "2018-01")
TOL = 2.0


def frame(turbine: str) -> pd.DataFrame:
    data_dir = ROOT / "data"
    return dc.with_era5(live._monthly_frame(data_dir, "engie", turbine), data_dir)


def limit_days(f: pd.DataFrame) -> set:
    """Days the fixed limit fired: HOT_C held for LIMIT_HOLD readings, as in
    scripts/pipeline/24_limit_vs_model.py."""
    f = f.sort_values("ts")
    hot = (f[dc.CHANNEL] > dc.HOT_C).astype(int)
    run = hot.groupby((hot != hot.shift()).cumsum()).cumsum() * hot
    return set(f.loc[run >= dc.LIMIT_HOLD, "ts"].dt.tz_convert(None).dt.strftime("%Y-%m-%d"))


def hourly(f: pd.DataFrame, sims: dict) -> dict:
    """R80711 hour by hour over the judged months: the sensor and each run's
    prediction, from the model that run judged the month with. An hour the
    model has no healthy reference for (a load band it never saw) has no
    prediction: that is the frozen summer model's blind spot in winter."""
    g, _ = dc.mark_restarts(f)
    g = g[(g["ts"] >= pd.Timestamp(FROM, tz="UTC")) & (g["ts"] < pd.Timestamp(TO, tz="UTC"))
          & g["running"] & ~g["warmup"]].copy()
    month = g["ts"].dt.strftime("%Y-%m")
    cols = {}
    for name, sim in sims.items():
        pred = pd.Series(np.nan, index=g.index)
        for m, model in sim["models"].items():
            sel = (month == m).to_numpy()
            if model is None or not sel.any() or dc.CHANNEL not in model["refs"]:
                continue
            pred[sel] = dc.predict(model["refs"][dc.CHANNEL], g.loc[sel, "P_norm"],
                                   g.loc[sel, "era5_temp_2m"])
        cols[name] = pred
    hour = g["ts"].dt.tz_convert(None).dt.floor("h")
    frame_ = pd.DataFrame({"actual": g[dc.CHANNEL].astype("float64"), "load": g["P_norm"],
                           **{k: v for k, v in cols.items()}})
    agg = frame_.groupby(hour.values).mean()
    # An hour is predicted only if the model covered most of its readings.
    cover = frame_[list(cols)].notna().groupby(hour.values).mean()
    for k in cols:
        agg.loc[cover[k] < 0.5, k] = np.nan

    def r(v):
        return [None if pd.isna(x) else round(float(x), 2) for x in v]
    return {"hour": [t.strftime("%Y-%m-%d %H:%M") for t in agg.index],
            "actual_c": r(agg["actual"]), "load": r(agg["load"]),
            **{f"{k}_pred_c": r(agg[k]) for k in cols}}


def checked(sim: dict) -> dict:
    return {d["day"]: d for d in sim["days"]
            if FROM <= d["day"] < TO and d.get("status") == "checked"}


def main() -> int:
    t0 = time.time()
    out = {"tolerance_k": TOL, "gate_k": dc.LAST_YEAR_GATE_K,
           "limit_rule": f"{dc.HOT_C:g} C held for {dc.LIMIT_HOLD} ten-minute readings",
           "judged": [FROM, "2018-01-31"], "turbines": {}}
    for t in TURBINES:
        f = frame(t)
        # The limit rule is off in both runs: this compares the models alone.
        off = dc.simulate(f, t, TOL, last_year_when_frozen=False, keep_models=t == FAULTY)
        on = dc.simulate(f, t, TOL, last_year_when_frozen=True,
                         last_year_months=() if t == FAULTY else JUDGED,
                         keep_models=t == FAULTY)
        a, b = checked(off), checked(on)
        days = sorted(set(a) & set(b))
        frz = {m["month"]: m for m in off["months"]}
        used = {m["month"]: m["model_from"] for m in on["months"] if m["month"] in JUDGED}
        out["turbines"][t] = {
            "role": "faulty, frozen after its confirmed fault" if t == FAULTY
                    else "healthy, judged with its own last-year models (control: every hot day is false)",
            "days_judged": len(days),
            "hot_days": {"frozen": sum(a[d]["flag"] == "hot" for d in days),
                         "last_year": sum(b[d]["flag"] == "hot" for d in days)},
            "model_by_month": {m: {"state": frz[m]["model"],
                                   "trained": [frz[m]["model_train_start"], frz[m]["model_train_end"]],
                                   "last_year_model": used.get(m)} for m in JUDGED if m in frz},
        }
        if t == FAULTY:
            lim = limit_days(f)
            out["hourly"] = hourly(f, {"frozen": off, "last_year": on})
            out["days"] = [{"day": d, "actual_c": a[d]["actual_k"],
                            "frozen_pred_c": a[d]["pred_k"], "last_year_pred_c": b[d]["pred_k"],
                            "frozen_excess_k": a[d]["mean_k"], "last_year_excess_k": b[d]["mean_k"],
                            "frozen_readings": a[d]["n"], "last_year_readings": b[d]["n"],
                            "limit_day": d in lim} for d in days]
            missed = [x for x in out["days"] if x["limit_day"] and x["frozen_excess_k"] <= TOL]
            out["limit_days_missed_by_frozen"] = {
                "days": [x["day"] for x in missed],
                "caught_by_last_year": [x["day"] for x in missed if x["last_year_excess_k"] > TOL]}
            out["limit_days"] = sum(x["limit_day"] for x in out["days"])
            # A model trained in summer has no healthy reference for the high
            # loads of winter, so those readings (the hottest) go unjudged.
            winter = [x for x in out["days"] if x["day"] >= "2017-11-01"]
            out["winter_readings_judged"] = {
                "from": "2017-11-01",
                "frozen": sum(x["frozen_readings"] for x in winter),
                "last_year": sum(x["last_year_readings"] for x in winter)}
        print(f"  {t}: {out['turbines'][t]['hot_days']}  ({time.time() - t0:.0f}s)")
    healthy = [t for t in TURBINES if t != FAULTY]
    out["healthy"] = {"days_judged": sum(out["turbines"][t]["days_judged"] for t in healthy),
                      "false_hot_days": {k: sum(out["turbines"][t]["hot_days"][k] for t in healthy)
                                         for k in ("frozen", "last_year")}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("days", "turbines")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
