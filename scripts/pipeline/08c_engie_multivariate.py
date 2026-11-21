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

TWO DETECTORS ARE MEASURED, so the deterministic/learned comparison is made on
the same footing on both datasets rather than on CARE alone:

  * `deterministic` -- load- and ambient-conditioned references with a robust
    spread, combined by max-z (src/windmill/multivariate.py).
  * `lightgbm`      -- a learned normal-behaviour model: one LightGBM regressor
    per thermal channel, predicting that channel from the operating point, with
    the residual turned into a z-score the same way (scripts/07_lightgbm_nbm.py
    uses the identical formulation on CARE).

Outputs results/engie_multivariate.json
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
from windmill import multivariate  # noqa: E402
from windmill import nbm  # noqa: E402

#: The ENGIE release's temperature channels. Defined once, in windmill.loaders,
#: so the detector, the demo and the reporting agree.
ENGIE_THERMAL = loaders.ENGIE_THERMAL

REFERENCE_YEAR = 2014
Z_THRESHOLD = 4.0
WINDOW_DAYS = 30
TRAIN_QUANTILE = 0.90
MAX_TRAIN_ROWS = 60_000


def rate_by_year(test: pd.DataFrame, flags: np.ndarray) -> list[dict]:
    years = test["year"].to_numpy()
    rows = []
    for y in sorted(set(years)):
        m = years == y
        rows.append({"year": int(y), "n_samples": int(m.sum()),
                     "n_flags": int(flags[m].sum()),
                     "rate_pct": round(float(flags[m].mean() * 100), 4)})
    return rows


# --- the learned arm --------------------------------------------------------
def lightgbm_scores(train: pd.DataFrame, test: pd.DataFrame,
                    channels: list[str], feats: list[str],
                    quantile: float = TRAIN_QUANTILE):
    """Fit the NBM on `train`, score `test`. See src/windmill/nbm.py.

    The score is max-z and the training score is cross-fitted, exactly as the
    deterministic arm does, so the two detectors are compared with the same
    statistic and the same threshold rule.
    """
    return nbm.cross_fitted_scores(train, test, channels, feats)


def rolling_lightgbm_scores(g: pd.DataFrame, channels: list[str], feats: list[str],
                            window_days: int = WINDOW_DAYS,
                            refit_days: int = 7,
                            quantile: float = TRAIN_QUANTILE):
    """The learned arm with a ROLLING reference.

    The deterministic arm's rolling reference re-estimates each row's expected
    value from its own trailing 30 days, so the reference cannot go stale. The
    learned arm needs the same treatment, otherwise the comparison is not like
    for like: a model fitted once in 2014 and applied to 2017 carries the same
    staleness the deterministic arm is criticised for.

    The model is refitted every `refit_days` on the trailing `window_days` of
    that turbine's own history, and applied to the following week. A weekly
    cadence is a compromise: it keeps the reference current at a tractable
    cost, and it is stated here rather than hidden.

    Returns a Series of scores over `g`'s rows (NaN where no model was
    available yet).
    """
    g = g.sort_values("ts").reset_index(drop=True)
    out = pd.Series(np.nan, index=g.index, dtype="float64")
    ts = g["ts"]

    # Week boundaries: one model per turbine per week.
    weeks = ts.dt.floor("D") - pd.to_timedelta(ts.dt.dayofweek, unit="D")
    boundaries = sorted(weeks.dropna().unique())

    for w in boundaries:
        block = g[weeks == w]
        if block.empty:
            continue
        w0 = pd.Timestamp(w)
        hist = g[(ts >= w0 - pd.Timedelta(days=window_days)) & (ts < w0)]
        if len(hist) < 2000:
            continue
        hist = hist.dropna(subset=feats)

        z_parts = []
        for c in channels:
            out_c = nbm.fit_channel_model(hist, block, c, feats)
            if out_c is None:
                continue
            _, block_z = out_c
            z_parts.append(block_z)

        if not z_parts:
            continue
        # max-z, matching the deterministic arm's statistic exactly.
        block_score = nbm.max_z(pd.concat(z_parts, axis=1))
        out.loc[block_score.index] = block_score.to_numpy()

    return out


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

    strategies: dict[str, dict[int, dict]] = {}
    engines: dict[str, dict[str, dict[int, dict]]] = {}

    for t, g in df.groupby("turbine"):
        g = g.sort_values("ts").reset_index(drop=True)
        gen = ((g["P_avg"] >= RATED_POWER_KW * 0.05) & (g["Ws_avg"] >= 2.0)).to_numpy()

        # --- 1. frozen 2014 reference --------------------------------------
        train = g[g["year"] == REFERENCE_YEAR]
        test = g[g["year"] != REFERENCE_YEAR]
        frozen = multivariate.detect(train, test, ENGIE_THERMAL, "P_norm",
                                     ambient_col="era5_temp_2m",
                                     quantile=TRAIN_QUANTILE, z_threshold=None)
        if frozen.channels_used:
            gen_test = ((test["P_avg"] >= RATED_POWER_KW * 0.05)
                        & (test["Ws_avg"] >= 2.0)).to_numpy()
            f = frozen.flags_max.to_numpy() & gen_test
            for r in rate_by_year(test, f):
                strategies.setdefault("frozen_2014", {}).setdefault(r["year"], r)

            # --- the learned arm, same frozen reference --------------------
            feats = [c_ for c_ in ("P_norm", "Ws_avg", "era5_temp_2m")
                     if c_ in g.columns]
            score, train_score = lightgbm_scores(
                train, test, ENGIE_THERMAL, feats, quantile=TRAIN_QUANTILE)
            if score is not None:
                thr = float(train_score.quantile(TRAIN_QUANTILE))
                lg_flags = (score.to_numpy() > thr) & gen_test
                for r in rate_by_year(test, lg_flags):
                    strategies.setdefault("frozen_2014_lightgbm", {}).setdefault(
                        r["year"], r)

        # --- 2. rolling local reference -------------------------------------
        roll = multivariate.detect_rolling(g, ENGIE_THERMAL, "P_norm",
                                           ambient_col="era5_temp_2m",
                                           window_days=WINDOW_DAYS,
                                           z_threshold=Z_THRESHOLD)
        if roll.channels_used:
            r_flags = roll.flags_max.to_numpy() & gen
            for r in rate_by_year(g, r_flags):
                strategies.setdefault("rolling_30d", {}).setdefault(r["year"], r)

        # --- 3. the learned arm with a rolling reference --------------------
        feats = [c_ for c_ in ("P_norm", "Ws_avg", "era5_temp_2m")
                 if c_ in g.columns]
        roll_score = rolling_lightgbm_scores(g, ENGIE_THERMAL, feats)
        if roll_score.notna().any():
            # The SAME threshold rule as the deterministic rolling arm: a fixed
            # z, meaning "the most deviant channel exceeds N robust deviations".
            # The two arms are therefore compared with the same statistic
            # (max-z), the same threshold (z > 4) and the same reference
            # strategy (trailing 30 days); only the model differs.
            valid = roll_score.notna().to_numpy()
            lg_flags = (roll_score.to_numpy() > Z_THRESHOLD) & gen & valid
            for r in rate_by_year(g[valid], lg_flags[valid]):
                strategies.setdefault("rolling_30d_lightgbm", {}).setdefault(
                    r["year"], r)

    out = {"arm": "Multivariate thermal detection on ENGIE (healthy turbines)",
           "reference_year": REFERENCE_YEAR,
           "z_threshold_rolling": Z_THRESHOLD,
           "window_days": WINDOW_DAYS,
           "train_quantile": TRAIN_QUANTILE,
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
