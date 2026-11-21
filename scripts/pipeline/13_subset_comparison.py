"""Score this project's detector on the SAME subset as the AE audit.

    python scripts/pipeline/13_subset_comparison.py [--datasets N]

WHY
---
The AE audit (scripts/pipeline/12_ae_audit.py) runs the benchmark authors'
own autoencoder on a label-stratified subset of the CARE datasets and scores
it with their own CARE scorer. The published AE figure (0.66) and this
project's figure (0.5028) are over all 95 datasets -- different populations,
so they cannot be subtracted from one another.

This script scores THIS PROJECT's detector on the identical subset, so the
two numbers describe the same events. Only then does the comparison mean
anything.

The protocol is the same one the full benchmark uses: the threshold is the
quantile of the CROSS-FITTED TRAINING score distribution, so no scored event
influences it.

Outputs results/subset_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care, multivariate, scoring  # noqa: E402

#: The training quantile the scored arm uses (scripts/pipeline/08_multivariate.py).
TRAIN_QUANTILE = 0.90
#: Which of the audit's configurations to compare against.
AE_CONFIG = "published_s0"
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")


def score_subset(data_dir: Path, picks: list) -> list[dict]:
    """Run the deterministic multivariate arm over the chosen datasets."""
    out = []
    for path, farm_dir, row in picks:
        eid = int(row["event_id"])
        label = str(row.get("event_label", "")).strip().lower()

        df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
        train = df[df["train_test"] == "train"]
        test = df[df["train_test"] == "prediction"]
        train = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]
        if train.empty or test.empty:
            continue

        columns = care.dataset_columns(path)
        pcol, thermal, wind, ambient = care.feature_rows(farm_dir, columns)
        if not pcol or len(thermal) < 2:
            continue

        res = multivariate.detect(
            train, test, thermal, pcol, ambient_col=ambient,
            quantile=TRAIN_QUANTILE, max_channels=24,
        )
        if not res.channels_used:
            continue

        flags = res.flags_max.reindex(test.index).fillna(False).astype(bool)
        normal_test = test["status_type_id"].isin(care.NORMAL_STATUS_IDS)
        s, e = int(row["event_start_id"]), int(row["event_end_id"])
        ids = test.index.to_numpy()
        in_event = (ids >= s) & (ids <= e)
        scorable = in_event & normal_test.to_numpy()

        scorer = scoring.CAREScore()
        ev = scorer.evaluate_event(
            event_start=s, event_end=e, event_label=label,
            predicted_anomalies=flags, normal_index=normal_test, event_id=eid,
        )
        out.append({
            "label": label,
            "event_id": eid,
            "farm": farm_dir.name,
            "description": str(row.get("event_description", ""))[:80],
            "n_scorable_rows": int(scorable.sum()),
            "n_flags": int(flags.sum()),
            "coverage": float(ev["f_beta_score"]) if ev["f_beta_score"] == ev["f_beta_score"] else None,
            "accuracy": float(ev["accuracy"]) if ev["accuracy"] is not None else None,
            "earliness": float(ev["weighted_score"]) if ev["weighted_score"] == ev["weighted_score"] else None,
            "max_criticality": int(ev["max_criticality"]),
            "anomaly_detected": bool(ev["max_criticality"] >= scorer.criticality_threshold),
        })
    return out


def care_score(rows: list[dict]) -> dict:
    """The CARE score, with the paper's two special cases."""
    from sklearn.metrics import fbeta_score

    df = pd.DataFrame(rows)
    anomaly = df[df["label"] == "anomaly"]
    normal = df[df["label"] == "normal"]
    reliability = float(fbeta_score(
        y_true=(df["label"] == "anomaly").to_numpy(),
        y_pred=df["anomaly_detected"].to_numpy(),
        beta=0.5, zero_division=0))
    avg_acc = float(normal["accuracy"].mean()) if len(normal) else float("nan")
    avg_f = float(anomaly["coverage"].mean()) if len(anomaly) else float("nan")
    avg_ws = float(anomaly["earliness"].mean()) if len(anomaly) else float("nan")
    if avg_acc != avg_acc:
        score = float("nan")
    elif avg_acc <= 0.5:
        score = avg_acc
    elif df["anomaly_detected"].sum() == 0:
        score = 0.0
    else:
        score = (avg_acc * 2.0 + avg_ws + avg_f + reliability) / 5.0
    return {
        "n_datasets": int(len(df)),
        "n_anomaly": int(len(anomaly)),
        "n_normal": int(len(normal)),
        "care_score": round(float(score), 4) if score == score else None,
        "coverage": round(avg_f, 4) if avg_f == avg_f else None,
        "accuracy": round(avg_acc, 4) if avg_acc == avg_acc else None,
        "reliability": round(reliability, 4),
        "earliness": round(avg_ws, 4) if avg_ws == avg_ws else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=4,
                    help="datasets per farm (half anomaly, half normal)")
    args = ap.parse_args()

    data_dir = ROOT / "data"
    picks = care.audit_subset(data_dir, args.datasets or None)
    if not picks:
        print("CARE not extracted -- run scripts/pipeline/01_build_dataset.py")
        return 1

    ids = [int(r[2]["event_id"]) for r in picks]
    print(f"scoring this project's arm on {len(picks)} datasets: {ids}")

    rows = score_subset(data_dir, picks)
    ours = care_score(rows)

    # The AE's numbers on the SAME datasets, from the audit artefact.
    ae = json.loads((ROOT / "results" / "ae_audit.json").read_text())
    # Match ONE configuration exactly. The audit now runs several (different
    # feature sets, different seeds), and a prefix match would pull in every
    # one of them -- double-counting each event.
    ae_rows = [r for r in ae.get("per_dataset", [])
               if r.get("event_id") in set(ids)
               and r.get("config") == AE_CONFIG]
    theirs = care_score([{**r, "label": r.get("label", "")} for r in ae_rows
                          if r.get("label") in ("anomaly", "normal")])

    out = {
        "purpose": "score every arm on the SAME subset of CARE datasets, so the "
                   "numbers are comparable",
        "dataset_ids": ids,
        "n_datasets": len(picks),
        "this_project": {
            "arm": "deterministic multivariate bins (max z)",
            "protocol": f"threshold = quantile {TRAIN_QUANTILE} of the "
                        f"CROSS-FITTED TRAINING score distribution",
            **ours,
        },
        "published_ae_baseline_measured_here": {
            "arm": "autoencoder (the benchmark authors' own implementation)",
            "protocol": "the authors' base_config.yaml, quantile 0.95, "
                        "fit_on_val=true",
            **theirs,
        },
        "published_ae_baseline_on_all_95": 0.66,
        "this_project_on_all_95": 0.5028,
        "per_dataset": rows,
    }
    path = ROOT / "results" / "subset_comparison.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    print()
    print(f"  {'arm':<46}{'n':>4}{'CARE':>9}{'cov':>8}{'acc':>8}{'rel':>8}{'earl':>8}")
    print("  " + "-" * 90)
    for label, v in (("this project (deterministic, max z)", ours),
                     ("published AE (authors' implementation)", theirs)):
        print(f"  {label:<46}{v['n_datasets']:>4}"
              f"{(v['care_score'] if v['care_score'] is not None else float('nan')):>9.4f}"
              f"{(v['coverage'] if v['coverage'] is not None else float('nan')):>8.4f}"
              f"{(v['accuracy'] if v['accuracy'] is not None else float('nan')):>8.4f}"
              f"{v['reliability']:>8.4f}"
              f"{(v['earliness'] if v['earliness'] is not None else float('nan')):>8.4f}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
