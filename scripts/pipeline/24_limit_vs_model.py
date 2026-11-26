#!/usr/bin/env python
"""The fixed 80 C limit against the daily check, day by day, on R80711 (ENGIE).

Writes `results/limit_vs_model.json`.

    python scripts/pipeline/24_limit_vs_model.py   (after `make daily`; needs the raw ENGIE release)

For each day: did the fixed limit fire (the gearbox bearing above 80 C for an
hour, read from the 10-minute data), and did the daily check flag the day hot
(its daily mean more than 2 K above the healthy prediction,
results/daily_series.json)? Counted inside R80711's anomaly span and in the
ten weeks before it. The span is this project's (from the sibling
comparison); the ENGIE release has no fault label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import daily_check as dc  # noqa: E402
from windmill import live  # noqa: E402

OUT = ROOT / "results" / "limit_vs_model.json"
TURBINE = "R80711"
BEFORE = ("2017-06-01", "2017-08-15")


def main() -> int:
    # The same ENGIE record the daily check reads (scripts/pipeline/19_daily_check.py).
    data_dir = ROOT / "data"
    f = dc.with_era5(live._monthly_frame(data_dir, "engie", TURBINE), data_dir)
    f = f.sort_values("ts").reset_index(drop=True)
    f["day"] = f["ts"].dt.tz_convert(None).dt.floor("D").dt.strftime("%Y-%m-%d")
    hot = (f[dc.CHANNEL] > dc.HOT_C).astype(int)
    run = hot.groupby((hot != hot.shift()).cumsum()).cumsum() * hot
    limit_days = set(f.loc[run >= dc.LIMIT_HOLD, "day"])

    series = json.loads((ROOT / "results" / "daily_series.json").read_text())
    d = series["turbines"][TURBINE]
    rows = pd.DataFrame({k: d[k] for k in ("day", "status", "flag", "fault_period", "mean_k", "limit")})
    x = d["sensors"][dc.CHANNEL]
    rows["actual"], rows["pred"] = x["actual"], x["pred"]
    checked = rows[rows["status"] == "checked"]
    model_days = set(checked.loc[checked["flag"] == "hot", "day"])

    span = checked[checked["fault_period"]]
    in_span = set(span["day"])
    m, lim = model_days & in_span, limit_days & in_span
    only = span[span["day"].isin(m - lim)]
    # The limit is part of the daily rule (daily_check.simulate, limit_rule):
    # every limit day is a hot day. How many would the 2 K average alone have
    # caught, and how many fell on days too thin to average at all?
    fault_days = rows[rows["fault_period"]]
    hot_any = set(fault_days.loc[fault_days["flag"] == "hot", "day"])
    lim_any = limit_days & set(fault_days["day"])
    by_average = set(span.loc[span["mean_k"] > series["tolerance_k"], "day"])
    # Each rule on its own, and the model's rule (both together).
    before_rows = rows[(rows["day"] >= BEFORE[0]) & (rows["day"] < BEFORE[1])]
    before_avg = before_rows[(before_rows["status"] == "checked")
                             & (before_rows["mean_k"] > series["tolerance_k"])]
    alone = {"limit": len(lim_any), "average": len(by_average), "rule": len(hot_any),
             "rule_increase_over_limit_pct": round(100 * (len(hot_any) - len(lim_any)) / len(lim_any))
             if lim_any else None,
             "before_limit": int(before_rows["limit"].sum()), "before_average": len(before_avg)}
    before = checked[(checked["day"] >= BEFORE[0]) & (checked["day"] < BEFORE[1])]
    mb = model_days & set(before["day"])
    # How much the model checked, so the counts can be weighed: the record,
    # the abnormal period's dates, and the checked days on every turbine.
    record_from = min(series["turbines"][t]["day"][0] for t in series["turbines"])
    record_to = max(series["turbines"][t]["day"][-1] for t in series["turbines"])
    span_from, span_to = series["anomaly_spans"][TURBINE][0]
    checked_days = {t: sum(1 for x in v["status"] if x == "checked")
                    for t, v in series["turbines"].items()}
    siblings = {t: n for t, n in checked_days.items() if t != TURBINE}
    out = {
        "record": {"from": record_from, "to": record_to,
                   "checked_days": checked_days,
                   "siblings_checked_days": sum(siblings.values()),
                   "siblings_checked_years": round(sum(siblings.values()) / 365.25, 1)},
        "turbine": TURBINE, "tolerance_k": series["tolerance_k"],
        "limit_rule": f"{dc.HOT_C:g} C held for {dc.LIMIT_HOLD} ten-minute readings",
        "anomaly_span": {
            "from": span_from, "to": min(span_to, record_to),
            "days_checked": int(len(span)), "model_hot_days": len(m), "limit_days": len(lim),
            "both": len(m & lim), "model_only": len(m - lim), "limit_only": len(lim - m),
            "limit_days_all": len(lim_any), "limit_days_flagged": len(lim_any & hot_any),
            "limit_days_over_the_average_too": len(lim & by_average),
            "limit_days_not_averaged": len(lim_any - in_span),
            "model_only_daily_mean_c_median": round(float(only["actual"].median()), 1),
            "model_only_excess_k_median": round(float((only["actual"] - only["pred"]).median()), 1),
        },
        "days_flagged": alone,
        "before_span": {"from": BEFORE[0], "to": BEFORE[1], "model_hot_days": len(mb),
                        "of_them_limit_days": len(mb & limit_days)},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
