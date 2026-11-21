"""Test the falsifiable claim.

    python scripts/02_condition.py

Does a baseline calibrated on one period invent a degradation signal whose size
tracks the weather shift, and does climatology conditioning remove it?

Outputs results/conditioning.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import RATED_POWER_KW, SITE_LAT, SITE_LON, loaders  # noqa: E402
from windmill import conditioning as cond  # noqa: E402


def annual_wind_climatology(scada: pd.DataFrame, era5: pd.DataFrame) -> dict:
    """How much does the site's wind actually move year to year, and does ERA5 see it?"""
    sc = scada.groupby("year").agg(ws=("Ws_avg", "mean"), p=("P_avg", "mean"))
    sc["cf"] = sc["p"] / RATED_POWER_KW * 100
    era = era5.copy()
    era["year"] = era["ts"].dt.year
    ea = era.groupby("year").agg(era_ws=("era5_ws_100m", "mean"),
                                 era_temp=("era5_temp_2m", "mean"))
    j = sc.join(ea, how="inner")
    full = j[(j.index >= 2013) & (j.index <= 2017)]
    if full.empty:
        full = j
    # Deviations are taken against each series' OWN mean over the full period:
    # ERA5 deviations against the ERA5 mean, SCADA deviations against the SCADA
    # mean. Using one base for both would manufacture an offset and corrupt the
    # amplification ratio.
    base_ws, base_cf = full["ws"].mean(), full["cf"].mean()
    base_era = full["era_ws"].mean()
    j["ws_dev_pct"] = (j["ws"] / base_ws - 1) * 100
    j["era_ws_dev_pct"] = (j["era_ws"] / base_era - 1) * 100
    j["cf_dev_pct"] = (j["cf"] / base_cf - 1) * 100
    j["amplification"] = j["cf_dev_pct"] / j["era_ws_dev_pct"].replace(0, np.nan)
    # Partial years (first/last, <300 days of data) are excluded from the stats.
    j["n_days"] = scada.groupby("year")["ts"].apply(lambda s: s.dt.normalize().nunique())
    full_years = j[(j["n_days"] >= 300)]
    return {
        "per_year": {
            int(y): {k: (None if pd.isna(v) else round(float(v), 4))
                     for k, v in row.items()}
            for y, row in j.iterrows()
        },
        "full_years_only": sorted(int(y) for y in full_years.index),
        "corr_era5_ws_dev_vs_cf_dev": round(float(np.corrcoef(
            full_years["era_ws_dev_pct"], full_years["cf_dev_pct"])[0, 1]), 4),
        "corr_scada_ws_dev_vs_cf_dev": round(float(np.corrcoef(
            full_years["ws_dev_pct"], full_years["cf_dev_pct"])[0, 1]), 4),
        "mean_amplification_cf_per_wind": round(
            float(full_years["amplification"].mean()), 3),
        "annual_cf_range_pp": round(float(full_years["cf"].max() - full_years["cf"].min()), 3),
        "annual_cf_cv_pct": round(
            float(full_years["cf"].std() / full_years["cf"].mean() * 100), 3),
    }


def static_vs_conditioned(scada: pd.DataFrame, reference_year: int = 2014) -> dict:
    """The core comparison: a single-period baseline vs climatology-conditioned.

    Both are evaluated against the same observed wind. The difference is what the
    conditioning buys.
    """
    core = scada[scada["year"].between(2013, 2017)].copy()

    # --- A. static baseline: curve fitted on the reference year only ----------
    ref = core[core["year"] == reference_year]
    curve = cond.fit_power_curve(ref)
    core["p_static"] = cond.apply_power_curve(core, curve)
    core["resid_static"] = core["P_avg"] - core["p_static"]

    # --- B. climatology-conditioned ------------------------------------------
    # The reference power curve is the site's own, pooled over the climatology
    # period; the conditioning happens on the *expected* side (what the weather
    # implies the period's output should be), while the residual is still computed
    # against the observed wind.
    clim = cond.climatology_weights(ref)
    expected = cond.expected_power_from_climatology(curve, clim, RATED_POWER_KW)

    # Per-year: what the year's own wind distribution implies, vs what the
    # reference-period distribution implies.
    per_year = {}
    for y, g in core.groupby("year"):
        y_clim = cond.climatology_weights(g)
        e_year = cond.expected_power_from_climatology(curve, y_clim, RATED_POWER_KW)
        e_ref = cond.expected_power_from_climatology(curve, clim, RATED_POWER_KW)
        per_year[int(y)] = {
            "expected_from_own_climatology_kw": round(e_year, 2),
            "expected_from_reference_climatology_kw": round(e_ref, 2),
            "climatology_shift_kw": round(e_year - e_ref, 2),
            "cf_pct": round(float(g["P_avg"].mean() / RATED_POWER_KW * 100), 3),
            "measured_mean_kw": round(float(g["P_avg"].mean()), 2),
        }

    stat = core.dropna(subset=["p_static"]).groupby("year").agg(
        n=("resid_static", "size"),
        static_bias_kw=("resid_static", "mean"),
        static_mae_kw=("resid_static", lambda s: s.abs().mean()),
    )
    stat["static_false_cf_pp"] = stat["static_bias_kw"] / RATED_POWER_KW * 100

    rows = []
    for y in sorted(per_year):
        s = stat.loc[y] if y in stat.index else None
        rows.append({
            "year": y,
            "cf_pct": per_year[y]["cf_pct"],
            "climatology_shift_kw": per_year[y]["climatology_shift_kw"],
            "static_false_cf_pp": round(float(s["static_false_cf_pp"]), 4) if s is not None else None,
            "static_mae_kw": round(float(s["static_mae_kw"]), 2) if s is not None else None,
        })

    # Correlation: does the false signal track the weather shift?
    arr = [(r["climatology_shift_kw"], r["static_false_cf_pp"]) for r in rows
           if r["static_false_cf_pp"] is not None]
    corr = None
    if len(arr) >= 3:
        x = np.array([a[0] for a in arr])
        yv = np.array([a[1] for a in arr])
        if x.std() > 0 and yv.std() > 0:
            corr = round(float(np.corrcoef(x, yv)[0, 1]), 4)

    return {
        "reference_year": reference_year,
        "reference_expected_kw": round(expected, 2),
        "per_year": rows,
        "corr_climatology_shift_vs_static_false_signal": corr,
        "max_static_false_cf_pp": round(float(stat["static_false_cf_pp"].abs().max()), 4),
    }


def era5_scada_agreement(scada: pd.DataFrame, era5: pd.DataFrame) -> dict:
    """How well does ERA5 at the site agree with the turbine's own anemometer?

    Hourly resolution only -- comparing hourly ERA5 to 10-min SCADA directly would
    manufacture false precision.
    """
    s = scada[scada["turbine"] == "R80711"].copy()
    s["hr"] = s["ts"].dt.floor("h")
    h = s.groupby("hr").agg(ws=("Ws_avg", "mean"), p=("P_avg", "mean")).reset_index()
    h = h.rename(columns={"hr": "ts"})
    e = era5[["ts", "era5_ws_100m", "era5_temp_2m"]].copy()
    e["hr"] = e["ts"].dt.floor("h")
    e = e.groupby("hr").agg(era_ws=("era5_ws_100m", "mean"), era_t=("era5_temp_2m", "mean")).reset_index()
    e = e.rename(columns={"hr": "ts"})
    j = h.merge(e, on="ts").dropna()
    if j.empty:
        return {"n_hours": 0}
    r = float(np.corrcoef(j["ws"], j["era_ws"])[0, 1])
    slope, intercept = np.polyfit(j["era_ws"], j["ws"], 1)
    return {
        "n_hours": int(len(j)),
        "pearson_r_scada_vs_era5": round(r, 4),
        "r2": round(r * r, 4),
        "bias_ms_scada_minus_era5": round(float((j["ws"] - j["era_ws"]).mean()), 3),
        "slope_scada_on_era5": round(float(slope), 3),
        "intercept": round(float(intercept), 3),
        "note": (
            "SCADA wind is a nacelle anemometer at 80 m hub height; ERA5 is a 0.25 deg "
            "grid cell at 100 m. The intercept reflects the nacelle anemometer's "
            "low-wind floor, so ERA5 cannot substitute for the turbine's own sensor "
            "below ~4 m/s -- it can only condition the long-term reference."
        ),
    }


def main() -> int:
    scada = loaders.load_engie(ROOT / "data")
    era5 = loaders.load_era5(ROOT / "data")

    out = {
        "site": {"lat": SITE_LAT, "lon": SITE_LON, "rated_kw": RATED_POWER_KW},
        "scada_rows": int(len(scada)),
        "scada_span": [str(scada["ts"].min()), str(scada["ts"].max())],
        "era5_hours": int(len(era5)),
        "annual_climatology": annual_wind_climatology(scada, era5),
        "static_vs_conditioned": static_vs_conditioned(scada),
        "era5_scada_agreement": era5_scada_agreement(scada, era5),
    }

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "conditioning.json").write_text(json.dumps(out, indent=2) + "\n")

    a = out["annual_climatology"]
    print("\n=== ANNUAL CLIMATOLOGY (SCADA + ERA5) ===")
    for y, row in a["per_year"].items():
        print(f"  {y}: CF={row['cf']:.2f}%  scada_ws={row['ws']:.3f}  "
              f"era5_ws={row['era_ws']:.3f}  cf_dev={row['cf_dev_pct']:+.2f}%  "
              f"era_dev={row['era_ws_dev_pct']:+.2f}%  amp={row['amplification']}")
    print(f"\n  full years used for stats: {a['full_years_only']}")
    print(f"  corr(ERA5 wind dev, CF dev) = {a['corr_era5_ws_dev_vs_cf_dev']}")
    print(f"  corr(SCADA wind dev, CF dev) = {a['corr_scada_ws_dev_vs_cf_dev']}")
    print(f"  mean amplification = {a['mean_amplification_cf_per_wind']}x")
    print(f"  annual CF range = {a['annual_cf_range_pp']} pp, CV = {a['annual_cf_cv_pct']}%")

    print("\n=== STATIC BASELINE vs CLIMATOLOGY CONDITIONING ===")
    print(f"  {'year':<6}{'CF%':>8}{'clim_shift_kw':>16}{'static_false_cf_pp':>20}")
    for r in out["static_vs_conditioned"]["per_year"]:
        print(f"  {r['year']:<6}{r['cf_pct']:>8.2f}{r['climatology_shift_kw']:>16.2f}"
              f"{r['static_false_cf_pp']:>20.3f}")
    sc = out["static_vs_conditioned"]
    print(f"\n  corr(climatology shift, false signal) = {sc['corr_climatology_shift_vs_static_false_signal']}")
    print(f"  max |static false signal| = {sc['max_static_false_cf_pp']} pp of CF")

    ag = out["era5_scada_agreement"]
    print(f"\n=== ERA5 vs SCADA anemometer ===\n  r={ag.get('pearson_r_scada_vs_era5')} "
          f"R2={ag.get('r2')} bias={ag.get('bias_ms_scada_minus_era5')} m/s "
          f"slope={ag.get('slope_scada_on_era5')}")

    print(f"\nwrote {res/'conditioning.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
