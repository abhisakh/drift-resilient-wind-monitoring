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
    rows = pd.DataFrame({k: d[k] for k in ("day", "status", "flag", "fault_period")})
    x = d["sensors"][dc.CHANNEL]
    rows["actual"], rows["pred"] = x["actual"], x["pred"]
    checked = rows[rows["status"] == "checked"]
    model_days = set(checked.loc[checked["flag"] == "hot", "day"])

    span = checked[checked["fault_period"]]
    in_span = set(span["day"])
    m, lim = model_days & in_span, limit_days & in_span
    only = span[span["day"].isin(m - lim)]
    before = checked[(checked["day"] >= BEFORE[0]) & (checked["day"] < BEFORE[1])]
    mb = model_days & set(before["day"])
    out = {
        "turbine": TURBINE, "tolerance_k": series["tolerance_k"],
        "limit_rule": f"{dc.HOT_C:g} C held for {dc.LIMIT_HOLD} ten-minute readings",
        "anomaly_span": {
            "days_checked": int(len(span)), "model_hot_days": len(m), "limit_days": len(lim),
            "both": len(m & lim), "model_only": len(m - lim), "limit_only": len(lim - m),
            "model_only_daily_mean_c_median": round(float(only["actual"].median()), 1),
            "model_only_excess_k_median": round(float((only["actual"] - only["pred"]).median()), 1),
        },
        "before_span": {"from": BEFORE[0], "to": BEFORE[1], "model_hot_days": len(mb),
                        "of_them_limit_days": len(mb & limit_days)},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
