"""CARE benchmark arm: score our detector with the authors' own implementation.

    python scripts/pipeline/04_score.py

The CARE to Compare dataset (Zenodo 10.5281/zenodo.15846963, CC BY-SA 4.0) gives 95
event datasets: 44 with a labelled anomaly event and 51 with normal behaviour.

Protocol, matching the reference implementation in
`energy-fault-detector` (Fraunhofer IEE, the dataset's authors) exactly:

  * the reference is built ONLY from the dataset's own training portion
    (`train_test == 'train'`), on rows whose status-ID marks normal operation;
  * the detector predicts on the prediction portion (`train_test == 'prediction'`);
  * each event is then scored with the CARE score: Coverage (pointwise F_1/2),
    Accuracy (on normal datasets), Reliability (event-wise F_1/2 via the criticality
    counter), and Earliness (weighted score);
  * the final score is the weighted average with weights 1/1/1/2
    (coverage/earliness/reliability/accuracy).

Scoring is done by the authors' `CAREScore` class so the numbers are directly
comparable with the published mini-benchmark (AE 0.66, random 0.5, iForest < 0.5,
trivial 0). Our own re-implementation in `src/windmill/scoring.py` is checked
against it event by event in the same run, so the claim is verifiable.

Outputs results/care_score.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care  # noqa: F401
from windmill import scoring  # noqa: E402

CARE_ROOT = care.care_root(ROOT / "data")
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
SEP = ";"

PUBLISHED_BASELINES = {
    "AE (published)": 0.66,
    "random": 0.50,
    "iForest": "< 0.5",
    "all anomaly": 0.0,
    "all normal": 0.0,
}

#: Status-IDs the dataset treats as normal operation (paper table 3.3).
NORMAL_STATUS_IDS = (0, 2)

ALL_COLUMNS: dict[str, list[str]] = {}


def pick_columns(farm_dir: Path) -> tuple[str, str]:
    """Choose the active-power and wind-speed columns for one wind farm.

    feature_description.csv lists the BASE sensor name (`power_29`); the CSV
    columns carry the statistic suffix (`power_29_avg`), so the base name has to
    be mapped onto the actual column.
    """
    fd = pd.read_csv(farm_dir / "feature_description.csv", sep=SEP)
    fd["sensor_name"] = fd["sensor_name"].astype(str)
    fd["description"] = fd["description"].astype(str)

    def _has(df, stat="average"):
        return df["statistics_type"].astype(str).str.contains(stat, case=False, regex=False)

    def _resolve(base: str, columns: list[str]) -> str | None:
        for suffix in ("_avg", ""):
            cand = f"{base}{suffix}"
            if cand in columns:
                return cand
        return None

    pow_rows = fd[fd["sensor_name"].str.startswith("power_")]
    active = pow_rows[pow_rows["description"].str.contains("active power", case=False, na=False)]
    pow_candidates = active if len(active) else pow_rows
    pow_candidates = pow_candidates[_has(pow_candidates)]
    power = None
    for base in pow_candidates["sensor_name"].tolist():
        resolved = _resolve(base, ALL_COLUMNS.get(farm_dir.name, []))
        if resolved:
            power = resolved
            break

    ws_rows = fd[fd["sensor_name"].str.contains("wind_speed")]
    primary = ws_rows[~ws_rows["description"].str.contains(r"estimated|1\+2", case=False, na=False)]
    if not len(primary):
        primary = ws_rows
    primary = primary[_has(primary)]
    wind = None
    for base in primary["sensor_name"].tolist():
        resolved = _resolve(base, ALL_COLUMNS.get(farm_dir.name, []))
        if resolved:
            wind = resolved
            break
    return power, wind


def build_reference(
    train: pd.DataFrame, power_col: str, wind_col: str, min_bin_count: int = 20
) -> tuple[pd.Series, dict]:
    """Power curve (method of bins) and the per-wind-bin residual band."""
    w = train[wind_col].to_numpy(dtype="float64")
    p = train[power_col].to_numpy(dtype="float64")
    ok = np.isfinite(w) & np.isfinite(p) & (w >= 0)
    w, p = w[ok], p[ok]
    bins = np.floor(w / 0.5).astype("int64")

    df = pd.DataFrame({"b": bins, "p": p})
    counts = df.groupby("b")["p"].size()
    curve = df.groupby("b")["p"].median()
    curve = curve[counts[counts >= min_bin_count].index]
    if len(curve) < 3:
        return curve, {}

    df["p_ref"] = df["b"].map(curve).to_numpy()
    df["resid"] = df["p"] - df["p_ref"]
    band = {}
    for b, g in df.groupby("b"):
        r = g["resid"].to_numpy()
        if len(r) < min_bin_count:
            continue
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        sd = float(np.std(r))
        sp = 1.4826 * mad if mad > 0 else (sd if sd > 0 else np.nan)
        band[int(b)] = (med, sp)
    return curve, band


def predict_anomalies(
    train: pd.DataFrame, test: pd.DataFrame, power_col: str, wind_col: str, k: float = 3.0
) -> pd.Series:
    """Flag anomalies on the prediction portion, using a reference built on train only."""
    curve, band = build_reference(train, power_col, wind_col)
    if len(curve) < 3 or not band:
        return pd.Series(False, index=test.index)

    t = test.dropna(subset=[power_col, wind_col])
    if t.empty:
        return pd.Series(False, index=test.index)

    wb = np.floor(t[wind_col].to_numpy(dtype="float64") / 0.5).astype("int64")
    p_ref = curve.reindex(wb).to_numpy()
    resid = t[power_col].to_numpy(dtype="float64") - p_ref
    meds = np.array([band.get(int(b), (np.nan, np.nan))[0] for b in wb])
    sprs = np.array([band.get(int(b), (np.nan, np.nan))[1] for b in wb])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (resid - meds) / sprs
    flags = pd.Series(np.isfinite(z) & (np.abs(z) > k), index=t.index)

    out = pd.Series(False, index=test.index)
    out.loc[t.index] = flags
    return out


def main() -> int:
    if not CARE_ROOT.exists():
        print(f"CARE not extracted at {CARE_ROOT}")
        return 1

    try:
        from energy_fault_detector.evaluation.care_score import CAREScore as RefCAREScore
    except ImportError:
        print("energy-fault-detector is not installed; cannot score against the "
              "reference implementation. Install it with:\n"
              "  pip install 'energy-fault-detector==0.8.1'  (requires Python < 3.12)")
        return 1

    results = []
    mismatches = []
    for farm in FARMS:
        farm_dir = CARE_ROOT / farm
        if not farm_dir.exists():
            continue
        info = pd.read_csv(farm_dir / "event_info.csv", sep=SEP)
        info["asset_id"] = info["asset_id"] if "asset_id" in info.columns else info["asset"]
        sample = sorted((farm_dir / "datasets").glob("*.csv"))
        if not sample:
            continue
        ALL_COLUMNS[farm] = pd.read_csv(sample[0], sep=SEP, nrows=0).columns.tolist()
        power_col, wind_col = pick_columns(farm_dir)
        print(f"[{farm}] power={power_col} wind={wind_col} events={len(info)}")
        if not power_col or not wind_col:
            continue

        for path in sample:
            event_id = int(path.stem)
            row = info[info["event_id"].astype(int) == event_id]
            if row.empty:
                continue
            row = row.iloc[0]
            label = str(row["event_label"]).strip().lower()

            df = pd.read_csv(path, sep=SEP, low_memory=False).sort_values("id")
            df = df.set_index("id")
            train = df[df["train_test"] == "train"]
            test = df[df["train_test"] == "prediction"]
            if train.empty or test.empty:
                continue

            normal_train = train["status_type_id"].isin(NORMAL_STATUS_IDS)
            normal_test = test["status_type_id"].isin(NORMAL_STATUS_IDS)

            # The reference is built on normal-operation training rows only.
            train_ref = train[normal_train]
            anomalies = predict_anomalies(train_ref, test, power_col, wind_col)

            # Score with the authors' implementation.
            ref = RefCAREScore()
            ref.evaluate_event(
                event_start=int(row["event_start_id"]),
                event_end=int(row["event_end_id"]),
                event_label=label,
                predicted_anomalies=anomalies,
                normal_index=normal_test,
                event_id=event_id,
            )
            ref_eval = ref.evaluated_events.iloc[-1]

            # ... and with our own re-implementation, to prove they agree.
            own = scoring.CAREScore()
            own_eval = own.evaluate_event(
                event_start=int(row["event_start_id"]),
                event_end=int(row["event_end_id"]),
                event_label=label,
                predicted_anomalies=anomalies,
                normal_index=normal_test,
                event_id=event_id,
            )
            for key in ("f_beta_score", "accuracy", "max_criticality", "weighted_score"):
                a, b = ref_eval[key], own_eval[key]
                if pd.isna(a) and pd.isna(b):
                    continue
                if abs(float(a) - float(b)) > 1e-6:
                    mismatches.append({"event": event_id, "metric": key,
                                        "reference": float(a), "ours": float(b)})

            results.append({
                "event_id": event_id,
                "farm": farm,
                "label": label,
                "description": str(row.get("event_description", ""))[:120],
                "n_test_rows": int(len(test)),
                "n_anomalies": int(anomalies.sum()),
                "f_beta_score": None if pd.isna(ref_eval["f_beta_score"]) else round(float(ref_eval["f_beta_score"]), 4),
                "accuracy": round(float(ref_eval["accuracy"]), 4),
                "weighted_score": None if pd.isna(ref_eval["weighted_score"]) else round(float(ref_eval["weighted_score"]), 4),
                "max_criticality": int(ref_eval["max_criticality"]),
                "anomaly_detected": bool(ref_eval["max_criticality"] >= 72),
            })

    if not results:
        print("no CARE datasets scored")
        return 1

    # Final score, computed from the per-event metrics the reference implementation
    # produced (the weights are the CARE paper's: accuracy counts double).
    df = pd.DataFrame(results)
    anomaly = df[df["label"] == "anomaly"]
    normal = df[df["label"] == "normal"]

    # Reliability: event-wise F_1/2 over all 95 events.
    from sklearn.metrics import fbeta_score
    reliability = float(fbeta_score(
        y_true=(df["label"] == "anomaly").to_numpy(),
        y_pred=df["anomaly_detected"].to_numpy(),
        beta=0.5, zero_division=0))

    avg_accuracy = float(normal["accuracy"].mean()) if len(normal) else float("nan")
    avg_f = float(anomaly["f_beta_score"].mean()) if len(anomaly) else float("nan")
    avg_ws = float(anomaly["weighted_score"].mean()) if len(anomaly) else float("nan")

    if avg_accuracy <= 0.5:
        care_score = avg_accuracy
    elif df["anomaly_detected"].sum() == 0:
        care_score = 0.0
    else:
        care_score = ((avg_accuracy * 2.0 + avg_ws * 1.0 + avg_f * 1.0 + reliability * 1.0) / 5.0)

    out = {
        "n_datasets_scored": int(len(df)),
        "n_anomaly": int(len(anomaly)),
        "n_normal": int(len(normal)),
        "care_score": round(float(care_score), 4),
        "sub_scores": {
            "coverage": round(float(avg_f), 4),
            "accuracy": round(float(avg_accuracy), 4),
            "reliability": round(reliability, 4),
            "earliness": round(float(avg_ws), 4),
        },
        "published_baselines": PUBLISHED_BASELINES,
        "beats_published_AE": bool(care_score > 0.66),
        "reference_impl_mismatches": mismatches,
        "n_mismatches": len(mismatches),
        "per_dataset": df.to_dict("records"),
    }

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "care_score.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

    print(f"\n=== CARE SCORE ({len(df)} events: {len(anomaly)} anomaly, {len(normal)} normal) ===")
    print(f"  coverage    {out['sub_scores']['coverage']:.4f}")
    print(f"  accuracy    {out['sub_scores']['accuracy']:.4f}")
    print(f"  reliability {out['sub_scores']['reliability']:.4f}")
    print(f"  earliness   {out['sub_scores']['earliness']:.4f}")
    print(f"  CARE SCORE  {out['care_score']:.4f}")
    print(f"\n  published baselines: {PUBLISHED_BASELINES}")
    print(f"  beats published AE (0.66)? {out['beats_published_AE']}")
    print(f"  our re-implementation vs authors' reference: "
          f"{len(mismatches)} mismatches over {len(df)} events")
    if mismatches:
        for m in mismatches[:5]:
            print(f"    {m}")
    print(f"\nwrote {res/'care_score.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
