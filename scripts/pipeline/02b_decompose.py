"""Why does 2016 show a false signal with almost no climatology shift?

Decomposes the static-baseline error into:
  (a) DISTRIBUTION effect  -- reweighting the same curve by a different wind
                               distribution. This is the weather part.
  (b) CURVE effect         -- the power curve itself changing shape at the same
                               wind speed. This is NOT weather: it is the machine
                               (or the site) changing.

If (b) is large, the "false degradation" signal is not false at all and the
conditioning story needs to be told differently.

    python scripts/02b_decompose.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import RATED_POWER_KW, loaders  # noqa: E402
from windmill import conditioning as cond  # noqa: E402


def main() -> int:
    scada = loaders.load_engie(ROOT / "data")
    core = scada[scada["year"].between(2013, 2017)].copy()
    core["bin"] = cond.wind_bin(core["Ws_avg"])

    ref_year = 2014
    ref = core[core["year"] == ref_year]
    ref_curve = ref.groupby("bin")["P_avg"].median()
    ref_counts = ref.groupby("bin")["P_avg"].size()
    ref_keep = ref_counts[ref_counts >= 30].index
    ref_curve = ref_curve.loc[ref_keep]
    ref_dist = (ref.groupby("bin")["P_avg"].size() / len(ref)).loc[ref_keep]
    ref_dist = ref_dist / ref_dist.sum()

    print("=== per-year power CURVE shape vs the 2014 reference ===")
    print("   (curve effect: median power at the SAME wind speed, per bin)")
    curves = {}
    for y, g in core.groupby("year"):
        c = g.groupby("bin")["P_avg"].median()
        n = g.groupby("bin")["P_avg"].size()
        curves[y] = c.loc[n[n >= 30].index]

    common = ref_curve.index
    rows = []
    for y, c in curves.items():
        cc = c.reindex(common).dropna()
        if len(cc) < 5:
            continue
        diff_kw = (cc - ref_curve.loc[cc.index])
        w = ref_dist.reindex(cc.index)
        w = w / w.sum()
        rows.append({
            "year": int(y),
            "curve_mean_diff_kw": round(float(diff_kw.mean()), 2),
            "curve_weighted_diff_kw": round(float((diff_kw * w).sum()), 2),
            "curve_shape_corr": round(float(np.corrcoef(cc, ref_curve.loc[cc.index])[0, 1]), 4),
            "n_bins": int(len(cc)),
        })
    for r in rows:
        print(f"   {r['year']}: mean bin diff {r['curve_mean_diff_kw']:+7.2f} kW | "
              f"wind-weighted {r['curve_weighted_diff_kw']:+7.2f} kW | "
              f"shape corr {r['curve_shape_corr']:.4f} | {r['n_bins']} bins")

    print("\n=== decomposition of the static-baseline error ===")
    print("   total_err = dist_effect + curve_effect")
    print(f"   {'year':<6}{'total_kw':>12}{'dist_kw':>12}{'curve_kw':>12}{'share_curve%':>14}")
    decomp = []
    for y, g in core.groupby("year"):
        n = g.groupby("bin")["P_avg"].size()
        y_keep = n[n >= 30].index
        y_curve = g.groupby("bin")["P_avg"].median().loc[y_keep]
        y_dist = (g.groupby("bin")["P_avg"].size() / len(g)).loc[y_keep]
        y_dist = y_dist / y_dist.sum()

        common_bins = ref_curve.index.intersection(y_curve.index)
        if len(common_bins) < 5:
            continue
        c_ref = ref_curve.loc[common_bins]
        c_y = y_curve.loc[common_bins]
        d_ref = ref_dist.reindex(common_bins); d_ref = d_ref / d_ref.sum()
        d_y = y_dist.reindex(common_bins); d_y = d_y / d_y.sum()

        # expected power under the reference curve and each distribution
        e_ref = float((c_ref * d_ref).sum())
        e_y = float((c_ref * d_y).sum())          # distribution effect only
        e_yy = float((c_y * d_y).sum())          # both
        total = e_yy - e_ref
        dist_eff = e_y - e_ref
        curve_eff = e_yy - e_y
        share = (abs(curve_eff) / abs(total) * 100) if abs(total) > 1e-9 else float("nan")
        decomp.append({
            "year": int(y), "total_kw": round(total, 2),
            "dist_effect_kw": round(dist_eff, 2),
            "curve_effect_kw": round(curve_eff, 2),
            "share_curve_pct": round(share, 1),
        })
        print(f"   {y:<6}{total:>12.2f}{dist_eff:>12.2f}{curve_eff:>12.2f}{share:>14.1f}")

    print("\n=== is the curve effect a site-wide shift or one turbine? ===")
    print("   per-turbine mean bin-level difference vs the site's 2014 curve")
    for y in [2013, 2015, 2016, 2017]:
        line = f"   {y}: "
        for t in sorted(core["turbine"].unique()):
            g = core[(core["year"] == y) & (core["turbine"] == t)]
            n = g.groupby("bin")["P_avg"].size()
            c = g.groupby("bin")["P_avg"].median().loc[n[n >= 30].index]
            cb = c.index.intersection(ref_curve.index)
            if len(cb) < 5:
                continue
            d = (c.loc[cb] - ref_curve.loc[cb]).mean()
            line += f"{t}={d:+6.2f}  "
        print(line)

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "decomposition.json").write_text(json.dumps(
        {"curve_comparison": rows, "decomposition": decomp}, indent=2) + "\n")
    print(f"\nwrote {res/'decomposition.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
