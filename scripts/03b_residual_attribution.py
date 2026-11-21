"""Why does 2016 still show the highest conditioned alarm rate when its weather
shift is ~zero?

Hypothesis: the residual tracks AIR TEMPERATURE, not wind. The power curve is
fitted on 2014; if the site-wide curve level moved after 2014 (all four turbines
together, per results/per_turbine_curve.json), a fixed 2014 reference leaves a
systematic residual in every later year.

Test: regress the scheme-D residual on ERA5 wind deviation and on ERA5 temperature
deviation, per year.

    python scripts/03b_residual_attribution.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location("d3", ROOT / "scripts" / "03_detect.py")
d3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d3)

from windmill import RATED_POWER_KW, loaders  # noqa: E402
from windmill import conditioning as cond  # noqa: E402


def main() -> int:
    scada = loaders.load_engie(ROOT / "data")
    era5 = loaders.load_era5(ROOT / "data")
    df = d3.join_era5(scada, era5)
    df = d3.normalise_to_standard_density(df)
    df = df[df["year"].between(2013, 2017)].copy()

    print("=== ERA5 climatology per year (the drivers) ===")
    e = era5.copy()
    e["year"] = e["ts"].dt.year
    ey = e.groupby("year").agg(
        ws=("era5_ws_100m", "mean"), temp=("era5_temp_2m", "mean"),
        press=("era5_pressure", "mean"))
    ey["rho"] = (ey["press"] * 100.0) / (287.058 * (ey["temp"] + 273.15))
    print(ey.loc[2013:2017].round(4).to_string())

    print("\n=== density factor actually applied (RHO_REF / rho) ===")
    df["dens"] = df["density_factor"]
    print(df.groupby("year")["dens"].agg(["mean", "min", "max"]).round(5).to_string())

    print("\n=== scheme-D residual vs the drivers, per year ===")
    ref = d3.build_reference(df, 2014, per_turbine=True, season_conditioned=False)
    out = d3.apply_reference(df, ref, k=3.0, conditioned_band=True)
    out = out.dropna(subset=["resid"])
    out["era_ws_dev"] = out.groupby("year")["era5_ws"].transform(
        lambda s: s - s.mean())
    out["era_temp_dev"] = out.groupby("year")["era5_temp_2m"].transform(
        lambda s: s - s.mean())

    print(f"  {'year':<6}{'mean_resid_kw':>16}{'corr_era_ws':>14}{'corr_era_temp':>16}{'n':>10}")
    rows = []
    for y, g in out.groupby("year"):
        cw = float(np.corrcoef(g["era_ws_dev"], g["resid"])[0, 1])
        ct = float(np.corrcoef(g["era_temp_dev"], g["resid"])[0, 1])
        rows.append({"year": int(y), "mean_resid_kw": round(float(g["resid"].mean()), 3),
                     "corr_era_ws": round(cw, 4), "corr_era_temp": round(ct, 4),
                     "n": int(len(g))})
        print(f"  {y:<6}{g['resid'].mean():>16.3f}{cw:>14.4f}{ct:>16.4f}{len(g):>10}")

    print("\n=== is the residual the SAME for all four turbines (site-wide) or one? ===")
    print(f"  {'year':<6}" + "".join(f"{t:>12}" for t in sorted(out['turbine'].unique())))
    for y, g in out.groupby("year"):
        per = g.groupby("turbine")["resid"].mean()
        print(f"  {y:<6}" + "".join(f"{per.get(t, np.nan):>12.3f}"
                                   for t in sorted(out['turbine'].unique())))

    print("\n=== KEY TEST: does a per-turbine RE-FIT on each year remove the bias? ===")
    print("   (if yes, the 2014 reference is stale, not the turbine degrading)")
    year_matched = {}
    for y, g in out.groupby("year"):
        ref_y = df[df["year"] == y]
        r2 = d3.build_reference(ref_y, y, per_turbine=True, season_conditioned=False)
        o2 = d3.apply_reference(g, r2, k=3.0, conditioned_band=True)
        o2 = o2.dropna(subset=["resid"])
        val = float(o2["resid"].mean())
        year_matched[int(y)] = round(val, 4)
        print(f"   {y}: mean resid with year-matched reference = {val:+.3f} kW"
              f"   (was {g['resid'].mean():+.3f} kW)")

    res = ROOT / "results"
    (res / "residual_attribution.json").write_text(json.dumps(
        {"era5_per_year": {int(y): {k: round(float(v), 4) for k, v in r.items()}
                           for y, r in ey.loc[2013:2017].iterrows()},
         "per_year": rows,
         "year_matched_reference_kw": year_matched}, indent=2) + "\n")
    print(f"\nwrote {res/'residual_attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
