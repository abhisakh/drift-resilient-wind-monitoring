"""Detection: residuals, operating-point-conditioned bands, deterministic gates.

The measured mechanism (see results/conditioning.json, results/decomposition.json):

  * A single-year, SITE-POOLED reference curve invents up to 0.44 pp of capacity
    factor of false "degradation" signal. Part of that is weather (the wind
    distribution changed) and part is a per-turbine calibration offset -- the four
    turbines' own curves sit 17-22 kW apart in the core wind range, so a pooled
    curve is the wrong reference.

  * The fix is threefold:
      1. PER-TURBINE reference curve (removes the calibration offset).
      2. AIR-DENSITY normalisation to standard density (removes the
         temperature/pressure-driven curve shift; needs ERA5 surface pressure
         and temperature, because the ENGIE release ships no ambient sensors).
      3. OPERATING-POINT-CONDITIONED band (removes the wind-distribution effect on
         the alarm rate).

This script measures the false-alarm rate on healthy data per year, under each
scheme, and shows that the year-to-year spread collapses once all three are in.

Outputs results/detection.json
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
from windmill.conditioning import (add_density, apply_reference,  # noqa: E402
                                build_reference, join_era5,
                                normalise_to_standard_density)
from windmill import detector  # noqa: E402

RHO_REF = 1.225   # kg/m3, standard air density (IEC 61400-12-1)
R_DRY = 287.058   # J/(kg K)
KELVIN = 273.15


def false_alarm_rate_per_year(df: pd.DataFrame, flag_col: str = "flag") -> pd.DataFrame:
    """Alarm rate per year, on the healthy subset (all four turbines are healthy
    over the study period -- no turbine shows a monotone curve degradation)."""
    g = df.groupby("year").agg(
        n=("flag", "size"),
        alarms=("flag", "sum"),
    )
    g["alarm_rate_pct"] = g["alarms"] / g["n"] * 100
    g["alarms_per_10min"] = g["alarms"] / (g["n"] * 10 / 10)
    return g.round(5)


def spread(values: pd.Series) -> float:
    """Peak-to-peak spread of the per-year rates -- the quantity to minimise."""
    v = values.dropna()
    if len(v) < 2:
        return float("nan")
    return float(v.max() - v.min())


def main() -> int:
    scada = loaders.load_engie(ROOT / "data")
    era5 = loaders.load_era5(ROOT / "data")
    df = join_era5(scada, era5)
    df = normalise_to_standard_density(df)
    df = df[df["year"].between(2013, 2017)].copy()

    REFERENCE_YEAR = 2014
    schemes = {
        "A_pooled_static": dict(per_turbine=False, conditioned_band=False, density=False, season=False),
        "B_perturbine_static": dict(per_turbine=True, conditioned_band=False, density=False, season=False),
        "C_perturbine_density": dict(per_turbine=True, conditioned_band=False, density=True, season=False),
        "D_perturbine_density_band": dict(per_turbine=True, conditioned_band=True, density=True, season=False),
        "E_perturbine_density_band_season": dict(per_turbine=True, conditioned_band=True, density=True, season=True),
    }

    # The weather-shift metric each scheme's alarm rate should NOT depend on.
    # It is computed per turbine from that turbine's OWN curve (a pooled site
    # curve is the wrong reference -- the four turbines sit 17-22 kW apart), and
    # is the difference in expected power between the year's own wind
    # distribution and the reference year's.
    clim_shift = {}
    for t in sorted(df["turbine"].unique()):
        ref_t = df[(df["year"] == REFERENCE_YEAR) & (df["turbine"] == t)]
        curve_t = cond.fit_power_curve(ref_t, power_col="P_std")
        ref_clim_t = cond.climatology_weights(ref_t)
        for y in sorted(df["year"].unique()):
            g = df[(df["year"] == y) & (df["turbine"] == t)]
            y_clim = cond.climatology_weights(g)
            e_y = cond.expected_power_from_climatology(curve_t, y_clim, RATED_POWER_KW)
            e_ref = cond.expected_power_from_climatology(curve_t, ref_clim_t, RATED_POWER_KW)
            clim_shift.setdefault(int(y), []).append(e_y - e_ref)
    clim_shift = {y: float(np.mean(v)) for y, v in clim_shift.items()}

    results = {}
    print("=== FALSE-ALARM RATE PER YEAR, BY SCHEME (healthy turbines, 2013-2017) ===")
    print(f"  {'scheme':<34}{'min%':>8}{'max%':>8}{'spread_pp':>12}{'per_10min':>12}{'corr_wx':>10}")

    for name, cfg in schemes.items():
        work = df.copy()
        if not cfg["density"]:
            work["P_std"] = work["P_avg"]
        ref = build_reference(work, REFERENCE_YEAR, per_turbine=cfg["per_turbine"],
                              season_conditioned=cfg["season"])
        out = apply_reference(work, ref, k=3.0, conditioned_band=cfg["conditioned_band"])
        # Only rows in a generating state are eligible to alarm.
        gen = (out["P_avg"] >= RATED_POWER_KW * 0.05) & (out["Ws_avg"] >= 2.0)
        out = out[gen]
        rate = false_alarm_rate_per_year(out)
        # The falsification test: does the alarm rate still track the weather shift?
        ys = [y for y in rate.index if y in clim_shift]
        corr = None
        if len(ys) >= 3:
            x = np.array([clim_shift[y] for y in ys])
            yv = rate.loc[ys, "alarm_rate_pct"].to_numpy()
            if np.std(x) > 0 and np.std(yv) > 0:
                corr = round(float(np.corrcoef(x, yv)[0, 1]), 4)
        results[name] = {
            "per_year": {int(y): {k: (None if pd.isna(v) else float(v))
                                  for k, v in row.items()}
                         for y, row in rate.iterrows()},
            "spread_pp": round(spread(rate["alarm_rate_pct"]), 5),
            "min_rate_pct": round(float(rate["alarm_rate_pct"].min()), 5),
            "max_rate_pct": round(float(rate["alarm_rate_pct"].max()), 5),
            "mean_alarms_per_10min": round(float(rate["alarms_per_10min"].mean()), 5),
            "corr_alarm_rate_vs_climatology_shift": corr,
        }
        r = results[name]
        print(f"  {name:<34}{r['min_rate_pct']:>8.3f}{r['max_rate_pct']:>8.3f}"
              f"{r['spread_pp']:>12.3f}{r['mean_alarms_per_10min']:>12.4f}"
              f"{(corr if corr is not None else float('nan')):>10.4f}")

    print("\n=== per-year alarm rate for the two extreme schemes ===")
    print(f"  {'year':<6}{'A_pooled%':>12}{'D_conditioned%':>18}{'E_season%':>14}{'clim_shift_kw':>15}")
    for y in sorted(results["A_pooled_static"]["per_year"]):
        a = results["A_pooled_static"]["per_year"][y]["alarm_rate_pct"]
        d = results["D_perturbine_density_band"]["per_year"][y]["alarm_rate_pct"]
        e = results["E_perturbine_density_band_season"]["per_year"][y]["alarm_rate_pct"]
        print(f"  {y:<6}{a:>12.4f}{d:>18.4f}{e:>14.4f}{clim_shift[y]:>15.2f}")

    # --- deterministic gates on the best scheme --------------------------------
    print("\n=== DETERMINISTIC GATES (scheme D, per turbine-year) ===")
    ref = build_reference(df, REFERENCE_YEAR, per_turbine=True)
    out = apply_reference(df, ref, k=3.0, conditioned_band=True)
    gen = (out["P_avg"] >= RATED_POWER_KW * 0.05) & (out["Ws_avg"] >= 2.0)
    out = out[gen]

    clim_ws = df["Ws_avg"]
    clim_mean, clim_sd = float(clim_ws.mean()), float(clim_ws.std())
    gate_rows = []
    for (t, y), g in out.groupby(["turbine", "year"]):
        g = g.sort_values("ts").copy()
        g["run"] = g["flag"].astype(int).groupby((~g["flag"]).cumsum()).cumsum()
        g["persist3"] = g["run"] >= 3
        g["flag_persist"] = g["persist3"]
        g1 = detector.operating_state_gate(g)
        g2 = detector.persistence_gate(g, flag_col="flag_persist", consecutive=3)
        g3 = detector.climatology_consistency_gate(g, clim_mean=clim_mean, clim_sd=clim_sd)
        g4 = detector.alarm_budget_gate(g, flag_col="flag_persist")
        allowed = all(x.passed for x in (g1, g2, g3, g4))
        n_raw = int(g["flag"].sum())
        n_persist = int(g["flag_persist"].sum())
        n_final = n_persist if allowed else 0
        gate_rows.append({
            "turbine": t, "year": int(y),
            "n_samples": int(len(g)),
            "alarms_raw": n_raw,
            "alarms_after_persistence": n_persist,
            "alarms_final": n_final,
            "raw_rate_pct": round(n_raw / len(g) * 100, 4),
            "final_rate_pct": round(n_final / len(g) * 100, 4),
            "gates": {x.name: x.passed for x in (g1, g2, g3, g4)},
            "gate_detail": {x.name: x.detail for x in (g1, g2, g3, g4)},
        })
    gr = pd.DataFrame(gate_rows)
    print(f"  {'turbine':<9}{'year':<6}{'raw%':>9}{'persist%':>11}{'final%':>9}  gates")
    for _, r in gr.iterrows():
        print(f"  {r['turbine']:<9}{r['year']:<6}{r['raw_rate_pct']:>9.4f}"
              f"{r['alarms_after_persistence']/r['n_samples']*100:>11.4f}"
              f"{r['final_rate_pct']:>9.4f}  "
              f"{[k for k, v in r['gates'].items() if not v] or 'all pass'}")

    out_payload = {
        "reference_year": REFERENCE_YEAR,
        "schemes": results,
        "gates": gate_rows,
        "climatology": {"ws_mean": round(clim_mean, 4), "ws_sd": round(clim_sd, 4)},
    }
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "detection.json").write_text(json.dumps(out_payload, indent=2) + "\n")
    print(f"\nwrote {res/'detection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
