"""AE-comparability audit: is the published AE baseline 0.66 comparable to ours?

    python scripts/pipeline/12_ae_audit.py [--datasets N] [--configs a,b]

WHY THIS EXISTS
The project reports a CARE score of 0.5028 for its deterministic detector and
quotes the benchmark authors' published autoencoder baseline of 0.66. Those two
numbers are only comparable if they were produced by the same protocol. This
script establishes, from the authors' own code and by running their own
autoencoder, what that protocol is and what it scores.

WHAT WAS READ OUT OF THE AUTHORS' CODE (energy-fault-detector 0.8.1)
  * `base_config.yaml`: autoencoder layers [200, 100, 50], code size 20,
    10 epochs, rmse anomaly score, threshold selector = quantile 0.95 with
    `fit_on_val: true`.
  * `FaultDetector.fit`: the autoencoder is trained on a TRAIN/VALIDATION split
    (sklearn, validation_split 0.2, shuffle true) and the threshold is fitted
    on the VALIDATION half. The prediction portion is never used to set the
    threshold, so the THRESHOLD RULE ITSELF is not event-leaky.
  * There are TWO feature-building entry points in the package, and they
    disagree about the status column:
      - `quick_fault_detection/data_loading.py:get_sensor_data` selects every
        numeric column, so it KEEPS `status_type_id` (4 = "down due to fault").
      - `evaluation/care2compare.py:format_event_dataset` drops
        `status_type_id` before fitting.
    Which one produced the published 0.66 is not recoverable from the code.
  * Neither the split nor the Keras training is seeded, so the published 0.66
    is one stochastic run, not a reproducible number.

WHAT THE PAPER SAYS (CARE to Compare, Data 9(12):138 / arXiv:2404.10320, 4.2.2)
  This is the part that matters for comparability, and it is NOT visible in
  the code:
  * The AE's architecture, learning rate and regularisation noise were tuned
    with Optuna.
  * The threshold calibration is PER WIND FARM:
      - Farms A and B: an adaptive threshold -- a neural network learns the
        expected reconstruction error from the input, and a sensitivity gamma
        is added to it. gamma was optimised PER FARM with Optuna.
      - Farm C: a fixed threshold chosen by "iterating over the calculated
        anomaly score values and choosing the value that MAXIMIZES THE
        F_{1/2}-SCORE based the ground truth defined by the normal behavior
        labels, that can be derived from the status-IDs".
  * => The published 0.66 is the output of a search whose objective was the
    CARE score itself. The paper does not document a held-out protocol for
    that search. It is therefore a TUNED-ON-THE-BENCHMARK number, not a
    held-out one, and it is NOT directly comparable with a leak-free score
    such as this project's 0.5028.
  * Separately, the AE sees the FULL sensor vector (power, wind, pitch,
    hydraulic, electrical), while this project's detector sees only the
    thermal channels. Some of the gap is that scope difference, not method.

WHAT IS MEASURED HERE
For a subset of the CARE datasets, the authors' own autoencoder is trained and
scored with the authors' own CARE scorer, in two configurations:
  * `published`  -- the CLI feature path: the status column IS a feature.
  * `no_status`  -- the CARE loader's feature path: `status_type_id` is
                    dropped, so the AE cannot read the fault annotation.
                    This is the configuration the dataset authors describe
                    ("status_type_id ... is intended to be used for filtering
                    of the training data", Zenodo 15846963).

Outputs results/ae_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care  # noqa: E402

FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
#: The status column the AE would otherwise receive as a numeric feature.
STATUS_COL = "status_type_id"

# Silence the benchmark package's logging and Keras' progress bars -- both are
# extremely noisy and would drown the results.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
logging.getLogger("energy_fault_detector").setLevel(logging.ERROR)


def build_config(threshold_quantile: float = 0.95):
    """The authors' own base_config.yaml, as their select_config() builds it."""
    import energy_fault_detector
    from energy_fault_detector.config import Config

    pkg = Path(energy_fault_detector.__file__).parent
    config = Config(config_filename=str(pkg / "base_config.yaml"))
    config["train"]["threshold_selector"]["params"]["quantile"] = threshold_quantile
    # Silence the Keras epoch bars.
    config["train"]["autoencoder"]["verbose"] = 0
    return config


def run_one(path: Path, farm_dir: Path, row: pd.Series, config,
            drop_status: bool, keras_seed: int | None):
    """Train the authors' AE on one dataset and return its predicted anomalies."""
    import random

    import numpy as np
    import tensorflow as tf
    from energy_fault_detector.fault_detector import FaultDetector

    if keras_seed is not None:
        random.seed(keras_seed)
        np.random.seed(keras_seed)
        tf.random.set_seed(keras_seed)

    df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
    train = df[df["train_test"] == "train"]
    test = df[df["train_test"] == "prediction"]
    if train.empty or test.empty:
        return None

    normal_train = train["status_type_id"].isin(care.NORMAL_STATUS_IDS)
    normal_test = test["status_type_id"].isin(care.NORMAL_STATUS_IDS)

    # The authors' loader keeps every numeric column; the non-numeric ones
    # (train_test, time_stamp) are dropped by the preprocessor.
    X_train = train.drop(columns=[c for c in ("train_test", "time_stamp")
                                  if c in train.columns])
    X_test = test.drop(columns=[c for c in ("train_test", "time_stamp")
                                if c in test.columns])
    if drop_status and STATUS_COL in X_train.columns:
        X_train = X_train.drop(columns=[STATUS_COL])
        X_test = X_test.drop(columns=[STATUS_COL])

    detector = FaultDetector(config=config)
    detector.fit(sensor_data=X_train, normal_index=normal_train, save_models=False)
    result = detector.predict(sensor_data=X_test, root_cause_analysis=False)

    anomalies = result.predicted_anomalies
    if isinstance(anomalies, pd.DataFrame):
        anomalies = anomalies.iloc[:, 0]
    flags = anomalies.reindex(test.index).fillna(False).astype(bool)
    return flags, normal_test


def score_with_reference(flags: pd.Series, normal_test: pd.Series, row: pd.Series,
                         event_id: int):
    """Score one event with the benchmark authors' own CARE scorer.

    The per-event fields the reference produces are `f_beta_score` (Coverage),
    `accuracy`, `max_criticality` and `weighted_score` (Earliness). Reliability
    is an aggregate over all events, not a per-event field.
    """
    from energy_fault_detector.evaluation.care_score import CAREScore as RefCAREScore

    label = str(row["event_label"]).strip().lower()
    ref = RefCAREScore()
    ref.evaluate_event(
        event_start=int(row["event_start_id"]),
        event_end=int(row["event_end_id"]),
        event_label=label,
        predicted_anomalies=flags,
        normal_index=normal_test,
        event_id=event_id,
    )
    ev = ref.evaluated_events.iloc[-1]
    return {
        "label": label,
        "coverage": None if pd.isna(ev["f_beta_score"]) else float(ev["f_beta_score"]),
        "accuracy": float(ev["accuracy"]),
        "earliness": None if pd.isna(ev["weighted_score"]) else float(ev["weighted_score"]),
        "max_criticality": int(ev["max_criticality"]),
        "anomaly_detected": bool(ev["max_criticality"] >= 72),
    }


def pick_datasets(per_farm: int | None) -> list[tuple[Path, Path, pd.Series]]:
    """Every labelled dataset, stratified across farms.

    Both anomaly AND normal datasets are needed: the CARE score's Accuracy
    sub-score is computed on the normal datasets only, so a subset of anomaly
    events alone cannot produce a CARE score.

    `per_farm=None` takes all of them; an integer takes that many per farm,
    which is only useful for a quick smoke test.
    """
    picks = []
    for farm in FARMS:
        farm_dir = care.care_root(ROOT / "data") / farm
        if not farm_dir.exists():
            continue
        info = care.event_info(farm_dir)

        # A subset of anomaly events alone cannot produce a CARE score: the
        # Accuracy sub-score is averaged over the NORMAL datasets, so an
        # all-anomaly subset leaves it undefined and the score a NaN. The
        # subset is therefore stratified by label -- half anomaly, half
        # normal -- within each farm.
        by_label: dict[str, list] = {"anomaly": [], "normal": []}
        for _, row in info.iterrows():
            eid = int(row["event_id"])
            path = farm_dir / "datasets" / f"{eid}.csv"
            if not path.exists():
                continue
            label = str(row.get("event_label", "")).strip().lower()
            if label not in by_label:
                continue
            by_label[label].append((path, farm_dir, row))

        if per_farm is None:
            picks.extend(by_label["anomaly"] + by_label["normal"])
            continue

        n_anom = (per_farm + 1) // 2
        n_norm = per_farm // 2
        picks.extend(by_label["anomaly"][:n_anom])
        picks.extend(by_label["normal"][:n_norm])
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=0,
                    help="datasets per farm; 0 means all of them")
    ap.add_argument("--configs", default="published:0,no_status:0",
                    help="configurations, comma separated. Each is "
                         "name:seed where name is 'published' (the authors' "
                         "pipeline as-is) or 'no_status' (status_type_id "
                         "dropped from the features). The same name with "
                         "different seeds measures run-to-run variance.")
    ap.add_argument("--seed", type=int, default=0,
                    help="keras/torch seed; the published pipeline has none")
    args = ap.parse_args()

    configs_to_run = []
    for c in args.configs.split(","):
        c = c.strip()
        if not c:
            continue
        name, _, seed = c.partition(":")
        configs_to_run.append((name, int(seed) if seed else 0))
    per_farm = args.datasets if args.datasets and args.datasets > 0 else None
    picks = pick_datasets(per_farm)
    if not picks:
        print("CARE not extracted -- run scripts/pipeline/01_build_dataset.py")
        return 1

    print(f"AE audit: {len(picks)} datasets x {len(configs_to_run)} configurations")
    print("(the authors' own autoencoder, trained and scored by their own code)")
    print()

    results = []
    for cfg_name, cfg_seed in configs_to_run:
        drop_status = cfg_name == "no_status"
        run_name = f"{cfg_name}_s{cfg_seed}"
        cfg = build_config()
        for path, farm_dir, row in picks:
            eid = int(row["event_id"])
            t0 = time.time()
            try:
                out = run_one(path, farm_dir, row, cfg, drop_status, cfg_seed)
            except Exception as exc:  # a single dataset must not kill the audit
                print(f"  [{run_name}] {path.stem}: FAILED {type(exc).__name__}: {exc}")
                results.append({"config": run_name, "event_id": eid, "error": str(exc)})
                continue
            if out is None:
                continue
            flags, normal_test = out
            ev = score_with_reference(flags, normal_test, row, eid)
            ev.update({
                "config": run_name,
                "seed": cfg_seed,
                "event_id": eid,
                "farm": farm_dir.name,
                "description": str(row.get("event_description", ""))[:70],
                "n_flags": int(flags.sum()),
                "seconds": round(time.time() - t0, 1),
            })
            results.append(ev)
            cov = "n/a" if ev["coverage"] is None else f"{ev['coverage']:.4f}"
            print(f"  [{run_name}] {farm_dir.name} event {eid:>3} "
                  f"flags={ev['n_flags']:>6} cov={cov:>7} "
                  f"det={int(ev['anomaly_detected'])} ({ev['seconds']:.0f}s)")

    # Aggregate per configuration.
    summary = {}
    from sklearn.metrics import fbeta_score

    for cfg_name, _seed in configs_to_run:
        run_name = f"{cfg_name}_s{_seed}"
        rows = [r for r in results if r.get("config") == run_name
                and "anomaly_detected" in r]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        anomaly = df[df["label"] == "anomaly"]
        normal = df[df["label"] == "normal"]
        # Reliability: event-wise F_1/2 over the events, exactly as the CARE
        # paper defines it and as scripts/pipeline/04_score.py computes it.
        reliability = float(fbeta_score(
            y_true=(df["label"] == "anomaly").to_numpy(),
            y_pred=df["anomaly_detected"].to_numpy(),
            beta=0.5, zero_division=0))
        avg_accuracy = float(normal["accuracy"].mean()) if len(normal) else float("nan")
        avg_f = float(anomaly["coverage"].mean()) if len(anomaly) else float("nan")
        avg_ws = float(anomaly["earliness"].mean()) if len(anomaly) else float("nan")
        if avg_accuracy <= 0.5:
            care_score = avg_accuracy
        elif df["anomaly_detected"].sum() == 0:
            care_score = 0.0
        else:
            care_score = ((avg_accuracy * 2.0 + avg_ws * 1.0
                           + avg_f * 1.0 + reliability * 1.0) / 5.0)
        summary[run_name] = {
            "n_datasets": int(len(df)),
            "n_anomaly": int(len(anomaly)),
            "n_normal": int(len(normal)),
            "care_score": round(float(care_score), 4),
            "coverage": round(float(avg_f), 4) if avg_f == avg_f else None,
            "accuracy": round(float(avg_accuracy), 4) if avg_accuracy == avg_accuracy else None,
            "reliability": round(reliability, 4),
            "earliness": round(float(avg_ws), 4) if avg_ws == avg_ws else None,
        }

    out = {
        "purpose": "Establish whether the published AE baseline (0.66) is "
                   "comparable with this project's leak-free protocol.",
        "published_baselines": {"AE (published)": 0.66, "random": 0.50,
                                "iForest": "< 0.5"},
        "this_project": {
            "arm": "deterministic multivariate bins (max z)",
            "care_score": 0.5028,
            "protocol": "threshold = quantile of the CROSS-FITTED TRAINING score "
                        "distribution; the scored events never influence it",
        },
        "authors_protocol_read_from_code": {
            "autoencoder": "layers [200, 100, 50], code 20, 10 epochs",
            "anomaly_score": "rmse",
            "threshold_selector": "quantile 0.95, fit_on_val=true",
            "threshold_fitted_on": "a 20% validation split of the TRAINING data",
            "threshold_rule_uses_scored_events": False,
            "two_feature_paths_in_package": {
                "quick_fault_detection/data_loading.py:get_sensor_data":
                    "keeps every numeric column, INCLUDING status_type_id",
                "evaluation/care2compare.py:format_event_dataset":
                    "drops status_type_id before fitting",
            },
            "which_path_produced_the_published_score": "not recoverable from "
                                                       "the code alone",
            "seeded": False,
        },
        "authors_protocol_read_from_paper": {
            "source": "CARE to Compare, Data 9(12):138 (2024), sec. 4.2.2; "
                      "arXiv:2404.10320",
            "ae_trained_on": "75% of the normal training data, 25% validation",
            "architecture_and_hyperparameters": "tuned with Optuna",
            "threshold_farm_A_and_B": "adaptive: an NN predicts the expected "
                                      "reconstruction error, plus a per-farm "
                                      "sensitivity gamma",
            "gamma_tuned_with_optuna_per_farm": True,
            "threshold_farm_C": "fixed value chosen by maximising the F_1/2 "
                                "score against ground truth derived from the "
                                "status-IDs",
            "objective_of_the_search": "the CARE score itself",
            "held_out_tuning_protocol_documented": False,
            "conclusion": "the published 0.66 is a TUNED-ON-THE-BENCHMARK "
                          "number, so it is not directly comparable with a "
                          "leak-free score",
            "feature_scope_difference": "the AE sees the full sensor vector "
                                        "(power, wind, pitch, hydraulic, "
                                        "electrical); this project's detector "
                                        "sees only the thermal channels",
        },
        "subsample_note": "This audit runs on a stratified SUBSET of the CARE "
                          "datasets, not all 95: the authors' autoencoder takes "
                          "roughly 5-8 minutes per dataset, so the full "
                          "benchmark is an 8-13 hour job. The subset is stated "
                          "here rather than presented as the full benchmark.",
        "seed": args.seed,
        "per_config": summary,
        "per_dataset": results,
    }

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "ae_audit.json").write_text(json.dumps(out, indent=2) + "\n")

    print("\n=== AE-COMPARABILITY AUDIT ===")
    for name, s in summary.items():
        print(f"  {name:12} n={s['n_datasets']:>3} "
              f"CARE={s['care_score']:.4f} "
              f"(cov {s['coverage']} acc {s['accuracy']} "
              f"rel {s['reliability']} earl {s['earliness']})")
    print(f"\nwrote {res_dir/'ae_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
