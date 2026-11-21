"""Per-turbine reference curve: is the curve effect a calibration difference or degradation?

Two candidate explanations for a turbine's curve sitting away from the site's pooled
curve:
  (a) a fixed per-turbine calibration difference (anemometer offset, curve fit) --
      harmless, must be modelled away per turbine;
  (b) progressive degradation -- the thing we actually want to detect.

We fit each turbine's OWN curve per year and look at how the level moves over time.
A constant offset = (a). A monotone trend = (b).

    python scripts/pipeline/02c_per_turbine_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import loaders  # noqa: E402
from windmill import conditioning as cond  # noqa: E402

# Bins where the power curve is steep and well populated: the informative region.
CORE_BINS = range(6, 14)  # 3.0 - 7.0 m/s


def turbine_curve_level(g: pd.DataFrame) -> dict:
    """Level of one turbine's own power curve in the core wind range."""
    n = g.groupby("bin")["P_avg"].size()
    c = g.groupby("bin")["P_avg"].median()
    keep = n[n >= 30].index
    c = c.loc[keep]
    core = c.reindex(list(CORE_BINS)).dropna()
    if len(core) < 4:
        return {}
    return {
        "level_kw": float(core.mean()),
        "n_bins": int(len(core)),
        "curve": {int(k): round(float(v), 2) for k, v in c.items()},
    }


def main() -> int:
    scada = loaders.load_engie(ROOT / "data")
    core = scada[scada["year"].between(2013, 2017)].copy()
    core["bin"] = cond.wind_bin(core["Ws_avg"])

    print("=== each turbine's OWN curve level per year (mean over 3-7 m/s bins, kW) ===")
    levels = {}
    for t, gt in core.groupby("turbine"):
        row = {}
        for y, gy in gt.groupby("year"):
            r = turbine_curve_level(gy)
            if r:
                row[int(y)] = round(r["level_kw"], 2)
        levels[t] = row
        vals = "  ".join(f"{y}:{v:+8.2f}" for y, v in sorted(row.items()))
        print(f"  {t}: {vals}")

    print("\n=== year-on-year change in each turbine's own curve level (kW) ===")
    print("   a fixed offset gives a flat row; degradation gives a trend")
    deltas = {}
    for t, row in levels.items():
        ys = sorted(row)
        d = {ys[i]: round(row[ys[i]] - row[ys[i - 1]], 2) for i in range(1, len(ys))}
        deltas[t] = d
        vals = "  ".join(f"{y}:{v:+7.2f}" for y, v in sorted(d.items()))
        print(f"  {t}: {vals}")

    print("\n=== within-year spread ACROSS turbines (kW) -- is any one turbine an outlier? ===")
    print("   a large spread means a pooled/site curve is the wrong reference")
    df_lv = pd.DataFrame(levels).T
    for y in sorted(df_lv.columns):
        col = df_lv[y].dropna()
        print(f"  {y}: min={col.min():+7.2f}  max={col.max():+7.2f}  spread={col.max()-col.min():6.2f}"
              f"  |  {', '.join(f'{t}={v:+.1f}' for t, v in col.items())}")

    print("\n=== per-turbine vs pooled-curve mean difference (the earlier 'curve effect') ===")
    ref_year = 2014
    ref = core[core["year"] == ref_year]
    ref_curve = ref.groupby("bin")["P_avg"].median()
    nr = ref.groupby("bin")["P_avg"].size()
    ref_curve = ref_curve.loc[nr[nr >= 30].index]
    out = {}
    for t, gt in core.groupby("turbine"):
        row = {}
        for y, gy in gt.groupby("year"):
            n = gy.groupby("bin")["P_avg"].size()
            c = gy.groupby("bin")["P_avg"].median().loc[n[n >= 30].index]
            cb = c.index.intersection(ref_curve.index)
            if len(cb) < 5:
                continue
            row[int(y)] = round(float((c.loc[cb] - ref_curve.loc[cb]).mean()), 2)
        out[t] = row
        vals = "  ".join(f"{y}:{v:+7.2f}" for y, v in sorted(row.items()))
        print(f"  {t}: {vals}")

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "per_turbine_curve.json").write_text(json.dumps(
        {"own_curve_level_kw": levels, "year_on_year_delta_kw": deltas,
         "vs_pooled_curve_kw": out}, indent=2) + "\n")
    print(f"\nwrote {res/'per_turbine_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
