"""LightGBM Thermal NBM: a learned normal-behaviour model on the thermal channels.

    python scripts/07_lightgbm_nbm.py

The deterministic arm (Option 1) models each temperature channel as a load-
conditioned median with a robust band. This arm learns the same normal behaviour
with gradient boosting instead, and the comparison is the point: if a learned
model does not beat the deterministic one on this benchmark, the deterministic
method is the better engineering choice and should be the headline.

Method
------
Per dataset, per thermal channel c:
  * a LightGBM regressor predicts c from the operating state (active power, wind
    speed) and the other thermal channels' *load-conditioned* values;
  * trained ONLY on the dataset's own training portion, normal-operation rows
    only;
  * the prediction residual is the anomaly score for that channel;
  * residuals are normalised by the training residual spread (MAD-based) and
    combined over channels by the max of the one-sided z (overheating is the
    failure mode), or the L2 norm of positive deviations;
  * the combined score is thresholded to give per-timestamp anomaly flags.

The threshold is chosen on the training portion at a fixed quantile, so the
training data never influences the score through the threshold.

Outputs results/lightgbm_nbm.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care  # noqa: E402
from windmill import scoring  # noqa: E402

CARE_ROOT = care.care_root(ROOT / "data")
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
SEP = ";"
NORMAL_STATUS_IDS = (0, 2)
LOAD_BIN = 0.05

LGBM_PARAMS = {
    "n_estimators": 60,
    "learning_rate": 0.12,
    "num_leaves": 15,
    "min_child_samples": 80,
    "subsample": 0.6,
    "colsample_bytree": 0.7,
    "reg_lambda": 2.0,
    "verbose": -1,
    "n_jobs": 4,
    # Explicit seeds. `subsample` and `colsample_bytree` draw randomness, and
    # without a fixed seed the arm's CARE score would not be reproducible --
    # fatal for a portfolio project whose whole claim is that its numbers can
    # be re-derived. Both the sklearn-level `random_state` and LightGBM's own
    # `seed` are pinned. Verified by scripts/07a_check_determinism.py and
    # tests/test_lightgbm_determinism.py.
    "random_state": 0,
    "seed": 0,
}

#: How many thermal channels to model per dataset. The full set is 24-76
#: depending on the farm; a subset keeps the sweep tractable and the result
#: honest about what was actually modelled.
MAX_CHANNELS = 12

#: Cap the rows used for fitting, so 95 datasets stay tractable.
MAX_TRAIN_ROWS = 12000

ALL_COLUMNS: dict[str, list[str]] = {}


def feature_rows(farm_dir: Path) -> tuple[str | None, list[str], list[str], str | None]:
    """(power channel, thermal channels, wind-speed channels, ambient channel)."""
    from windmill import care as _care
    columns = _care.dataset_columns(
        sorted((farm_dir / "datasets").glob("*.csv"))[0])
    return _care.feature_rows(farm_dir, columns)


def l2norm_positive(z: pd.DataFrame) -> pd.Series:
    pos = z.clip(lower=0)
    return np.sqrt((pos ** 2).sum(axis=1, skipna=True)).fillna(0.0)


def run_dataset(path: Path, farm_dir: Path, info: pd.DataFrame, pcol: str,
                thermal: list[str], wind: list[str],
                quantile: float = 0.90) -> dict | None:
    """Fit the NBM on train, predict on test, return per-event scores."""
    import lightgbm as lgb

    eid = int(path.stem)
    row = info[info["event_id"].astype(int) == eid]
    if row.empty:
        return None
    row = row.iloc[0]
    label = str(row["event_label"]).strip().lower()

    df = pd.read_csv(path, sep=SEP, low_memory=False).set_index("id")
    train = df[df["train_test"] == "train"]
    test = df[df["train_test"] == "prediction"]
    train = train[train["status_type_id"].isin(NORMAL_STATUS_IDS)]
    if train.empty or test.empty or pcol is None:
        return None

    channels = [c for c in thermal if c in df.columns]
    if len(channels) < 2:
        return None
    # Cap the channel count so a farm with 118 thermal sensors stays tractable.
    channels = channels[:MAX_CHANNELS]

    feats = [pcol] + [w for w in wind if w in df.columns]
    tr = train.dropna(subset=feats)
    te = test.dropna(subset=feats)
    if len(tr) > MAX_TRAIN_ROWS:
        tr = tr.sample(n=MAX_TRAIN_ROWS, random_state=0).sort_index()
    if len(tr) < 500 or te.empty:
        return None

    z_cols = {}
    train_l2_parts = []
    for c in channels:
        t = train.dropna(subset=[c])
        if len(t) < 500:
            continue
        if len(t) > MAX_TRAIN_ROWS:
            t = t.sample(n=MAX_TRAIN_ROWS, random_state=0).sort_index()
        X_tr = t[feats]
        X_te = te[feats]
        y_tr = t[c].to_numpy()
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_tr, y_tr)
        r_tr = y_tr - model.predict(X_tr)
        r_te = te[c].to_numpy() - model.predict(X_te)
        med = float(np.median(r_tr))
        mad = float(np.median(np.abs(r_tr - med)))
        spread = 1.4826 * mad if mad > 0 else (float(np.std(r_tr)) or np.nan)
        if not np.isfinite(spread) or spread <= 0:
            continue
        z_cols[c] = pd.Series((r_te - med) / spread, index=te.index)
        train_l2_parts.append(pd.Series((r_tr - med) / spread, index=t.index))

    if not z_cols:
        return None
    Z = pd.DataFrame(z_cols)
    combined = l2norm_positive(Z)

    # Threshold from the TRAINING score distribution at a fixed quantile -- the
    # same rule the deterministic arm uses, and it never sees an event it is
    # later scored on, so no event label influences it.
    train_l2 = l2norm_positive(pd.concat(train_l2_parts, axis=1))
    thr = float(train_l2.quantile(quantile))

    flags = combined > thr
    out = pd.Series(False, index=test.index)
    out.loc[flags.index] = flags.to_numpy()

    # The event window is expressed in the TEST frame's id space, but `combined`
    # is indexed by `te` (the rows with all the features present). The two have
    # different lengths, so the masks must be built against `te`.
    te_ids = te.index.to_numpy()
    s, e = int(row["event_start_id"]), int(row["event_end_id"])
    in_event = (te_ids >= s) & (te_ids <= e)
    status = te["status_type_id"].to_numpy()
    scorable = in_event & np.isin(status, NORMAL_STATUS_IDS)

    return {
        "event_id": eid, "farm": farm_dir.name, "label": label,
        "description": str(row.get("event_description", ""))[:100],
        "n_channels": len(z_cols),
        "threshold": round(thr, 4),
        "n_flags_test": int(flags.sum()),
        "n_scorable_event_rows": int(scorable.sum()),
        "combined_median_event": round(float(np.nanmedian(combined.to_numpy()[scorable])), 4)
        if scorable.sum() else None,
        "combined_median_outside": round(float(np.nanmedian(combined.to_numpy()[~in_event])), 4),
        "_flags": out,
        "_combined": combined,
        "_train": train_l2,
        "_normal_index": pd.Series(test["status_type_id"].isin(NORMAL_STATUS_IDS).to_numpy(),
                                   index=test.index),
    }


def main() -> int:
    results = []
    for farm in FARMS:
        farm_dir = CARE_ROOT / farm
        if not farm_dir.exists():
            continue
        info = pd.read_csv(farm_dir / "event_info.csv", sep=SEP)
        sample = sorted((farm_dir / "datasets").glob("*.csv"))
        if not sample:
            continue
        ALL_COLUMNS[farm] = pd.read_csv(sample[0], sep=SEP, nrows=0).columns.tolist()
        pcol, thermal, wind, ambient = feature_rows(farm_dir)
        print(f"[{farm}] power={pcol} thermal={len(thermal)} wind={len(wind)} "
              f"ambient={ambient}")

        for path in sample:
            try:
                r = run_dataset(path, farm_dir, info, pcol, thermal, wind)
            except Exception as e:
                print(f"   [{farm}/{path.stem}] skipped: {type(e).__name__}: {e}")
                continue
            if r:
                results.append(r)

    if not results:
        print("no datasets processed")
        return 1

    # --- score with the CARE score -------------------------------------------
    # The event window ids are needed for scoring; read them from event_info.
    windows = {}
    for farm in FARMS:
        farm_dir = CARE_ROOT / farm
        if not farm_dir.exists():
            continue
        info = pd.read_csv(farm_dir / "event_info.csv", sep=SEP)
        for _, row in info.iterrows():
            windows[int(row["event_id"])] = (int(row["event_start_id"]), int(row["event_end_id"]))

    score = scoring.CAREScore()
    for r in results:
        s, e = windows[r["event_id"]]
        score.evaluate_event(
            event_start=s, event_end=e, event_label=r["label"],
            predicted_anomalies=r["_flags"], normal_index=r["_normal_index"],
            event_id=r["event_id"],
        )

    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    df = pd.DataFrame(rows)

    # Reliability needs the event-level detection flag: max criticality >= 72.
    det = []
    for r in results:
        crit = scoring.calculate_criticality(r["_flags"], r["_normal_index"])
        det.append(int(crit.max()) >= scoring.CRITICALITY_THRESHOLD)
    df["anomaly_detected"] = det

    from sklearn.metrics import fbeta_score
    reliability = float(fbeta_score(
        y_true=(df["label"] == "anomaly").to_numpy(),
        y_pred=df["anomaly_detected"].to_numpy(), beta=0.5, zero_division=0))

    subs = score.sub_scores()
    care = score.get_final_score()

    out = {
        "arm": "LightGBM thermal NBM",
        "n_datasets": int(len(df)),
        "n_anomaly": int((df["label"] == "anomaly").sum()),
        "n_normal": int((df["label"] == "normal").sum()),
        "care_score": round(float(care), 4),
        "sub_scores": subs,
        "published_baselines": {"AE (published)": 0.66, "random": 0.50},
        "beats_published_AE": bool(care > 0.66),
        "per_dataset": df.to_dict("records"),
    }
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "lightgbm_nbm.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

    print(f"\n=== LightGBM THERMAL NBM ({len(df)} datasets) ===")
    for k, v in subs.items():
        print(f"  {k:<12} {v:.4f}")
    print(f"  CARE SCORE   {care:.4f}")
    print(f"  beats published AE (0.66)? {out['beats_published_AE']}")
    print(f"\nwrote {res/'lightgbm_nbm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
