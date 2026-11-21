"""ENGIE arm: false-alarm rate on healthy turbines, for each reference strategy.

    python scripts/08c_engie_multivariate.py

The same multivariate thermal method applied to the ENGIE La Haute Borne site, so
the two datasets are measured with one method rather than two.

Three reference strategies, because the choice of reference is the whole game
here:

  1. `frozen_2014`    -- a reference fitted on 2014 and applied to every year.
                         This is what a naive deployment does, and it is the
                         failure the project is about: each turbine's thermal
                         channels move by 5-9 K between years, so a frozen
                         reference leaves a systematic offset that reads as
                         overheating on every channel at once.
  2. `rolling_30d`    -- the reference for each row is that row's own recent
                         history at the same load bin. The reference is
                         re-estimated continuously, so it cannot go stale.
  3. `power_only`     -- the power-curve residual, for reference (from
                         scripts/03_detect.py).

Every turbine is healthy over the study period (no monotone curve degradation --
see results/per_turbine_curve.json), so ANY alarm is a false alarm and the
false-alarm rate per year is the quantity of interest.

Outputs results/engie_multivariate.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import RATED_POWER_KW, loaders  # noqa: E402
from windmill import multivariate  # noqa: E402

#: The ENGIE release's temperature channels (the 16 columns it actually ships).
ENGIE_THERMAL = [
    "Git_avg",   # gearbox inlet temperature
    "Gost_avg",  # gearbox oil sump temperature
    "Gb2t_avg",  # gearbox bearing 2 temperature
    "Gb1t_avg",  # gearbox bearing 1 temperature
    "Db2t_avg",  # generator bearing 2 temperature
    "Db1t_avg",  # generator bearing 1 temperature
    "Dst_avg",   # generator stator temperature
    "Yt_avg",    # nacelle temperature
    "Rbt_avg",   # rotor bearing temperature
    "Rt_avg",    # hub temperature
]

REFERENCE_YEAR = 2014
Z_THRESHOLD = 4.0
WINDOW_DAYS = 30


def rate_by_year(test: pd.DataFrame, flags: np.ndarray) -> list[dict]:
    years = test["year"].to_numpy()
    rows = []
    for y in sorted(set(years)):
        m = years == y
        rows.append({"year": int(y), "n_samples": int(m.sum()),
                     "n_flags": int(flags[m].sum()),
                     "rate_pct": round(float(flags[m].mean() * 100), 4)})
    return rows


def main() -> int:
    scada = loaders.load_engie(ROOT / "data")
    era5 = loaders.load_era5(ROOT / "data")

    # The load bins are defined on NORMALISED power (CARE ships power scaled by
    # the rated power; ENGIE ships kilowatts).
    scada = scada.copy()
    scada["P_norm"] = scada["P_avg"] / RATED_POWER_KW

    e = era5.copy()
    e["ts"] = e["ts"].dt.floor("10min")
    e = e.set_index("ts")[["era5_temp_2m"]].resample("10min").ffill().reset_index()
    df = scada.merge(e, on="ts", how="left")
    df = df[df["year"].between(2013, 2017)].copy()

    strategies = {}
    for t, g in df.groupby("turbine"):
        g = g.sort_values("ts").reset_index(drop=True)
        gen = ((g["P_avg"] >= RATED_POWER_KW * 0.05) & (g["Ws_avg"] >= 2.0)).to_numpy()

        # --- 1. frozen 2014 reference --------------------------------------
        train = g[g["year"] == REFERENCE_YEAR]
        test = g[g["year"] != REFERENCE_YEAR]
        frozen = multivariate.detect(train, test, ENGIE_THERMAL, "P_norm",
                                     ambient_col="era5_temp_2m",
                                     quantile=0.90, z_threshold=None)
        if frozen.channels_used:
            gen_test = ((test["P_avg"] >= RATED_POWER_KW * 0.05)
                        & (test["Ws_avg"] >= 2.0)).to_numpy()
            f = frozen.flags_max.to_numpy() & gen_test
            for r in rate_by_year(test, f):
                strategies.setdefault("frozen_2014", {}).setdefault(r["year"], r)

        # --- 2. rolling local reference -------------------------------------
        roll = multivariate.detect_rolling(g, ENGIE_THERMAL, "P_norm",
                                           ambient_col="era5_temp_2m",
                                           window_days=WINDOW_DAYS,
                                           z_threshold=Z_THRESHOLD)
        if roll.channels_used:
            r_flags = roll.flags_max.to_numpy() & gen
            for r in rate_by_year(g, r_flags):
                strategies.setdefault("rolling_30d", {}).setdefault(r["year"], r)

    out = {"arm": "Deterministic Multivariate Bins on ENGIE (healthy turbines)",
           "reference_year": REFERENCE_YEAR,
           "z_threshold_rolling": Z_THRESHOLD,
           "window_days": WINDOW_DAYS,
           "n_channels_available": len(ENGIE_THERMAL),
           "strategies": {}}

    for name, per_year in strategies.items():
        rates = [v["rate_pct"] for v in per_year.values()]
        out["strategies"][name] = {
            "per_year": {int(y): v for y, v in per_year.items()},
            "min_rate_pct": round(min(rates), 4),
            "max_rate_pct": round(max(rates), 4),
            "spread_pp": round(max(rates) - min(rates), 4),
            "mean_rate_pct": round(float(np.mean(rates)), 4),
        }

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "engie_multivariate.json").write_text(json.dumps(out, indent=2) + "\n")

    print("=== ENGIE MULTIVARIATE: false-alarm rate on healthy turbines ===")
    for name, s in out["strategies"].items():
        print(f"\n  strategy: {name}")
        print(f"  {'year':<8}{'rate %':>10}")
        for y, v in sorted(s["per_year"].items()):
            print(f"  {y:<8}{v['rate_pct']:>10.4f}")
        print(f"  spread {s['spread_pp']} pp | mean {s['mean_rate_pct']} %")
    print(f"\nwrote {res_dir/'engie_multivariate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
