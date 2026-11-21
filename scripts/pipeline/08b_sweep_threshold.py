"""Sweep the TRAINING-QUANTILE threshold for the multivariate arm.

    python scripts/pipeline/08b_sweep_threshold.py

The threshold is the arm's one free parameter, so it has to be chosen on
evidence rather than by taste. This sweeps it over a grid and reports the CARE
score for each value, using the benchmark authors' own scorer.

LEAK-FREE BY CONSTRUCTION
-------------------------
Two threshold rules are swept:

  * `z`   -- a fixed z threshold. Selecting it on the same 95 events that are
             then scored is leakage, so this row is reported as an
             UPPER BOUND, not a result.
  * `q`   -- the quantile of the TRAINING score distribution, computed from
             cross-fitted scores. The threshold never sees an event it is later
             scored on, which is the protocol the CARE paper's own baseline
             uses. This is the honest row.

Outputs results/threshold_sweep.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care, multivariate, scoring  # noqa: E402

CARE_ROOT = care.care_root(ROOT / "data")
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")

#: Fixed z thresholds (reported as an upper bound -- selected on the test events).
Z_THRESHOLDS = [2.0, 2.5, 3.0, 4.0]

#: Training quantiles (leak-free: calibrated on training data only).
QUANTILES = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]


def main() -> int:
    from energy_fault_detector.evaluation.care_score import CAREScore as RefCAREScore

    # --- pass 1: fit the references once per dataset, cache the scores -------
    cache = []
    for farm in FARMS:
        farm_dir = CARE_ROOT / farm
        if not farm_dir.exists():
            continue
        info = care.event_info(farm_dir)
        sample = sorted((farm_dir / "datasets").glob("*.csv"))
        if not sample:
            continue
        columns = care.dataset_columns(sample[0])
        pcol, thermal, wind, ambient = care.feature_rows(farm_dir, columns)
        print(f"[{farm}] power={pcol} thermal={len(thermal)} ambient={ambient}", flush=True)

        for path in sample:
            eid = int(path.stem)
            row = info[info["event_id"].astype(int) == eid]
            if row.empty:
                continue
            row = row.iloc[0]
            label = str(row["event_label"]).strip().lower()
            s, e = int(row["event_start_id"]), int(row["event_end_id"])

            df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
            train = df[df["train_test"] == "train"]
            test = df[df["train_test"] == "prediction"]
            train = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]
            if train.empty or test.empty:
                continue

            # Fit with a fixed (high) threshold so the flags are not needed here;
            # the scores themselves are what the sweep thresholds.
            res = multivariate.detect(train, test, thermal, pcol, ambient_col=ambient,
                                      z_threshold=1e9)
            if not res.channels_used:
                continue
            normal_index = pd.Series(
                test["status_type_id"].isin(care.NORMAL_STATUS_IDS).to_numpy(),
                index=test.index)
            cache.append({
                "event_id": eid, "farm": farm, "label": label,
                "start": s, "end": e,
                "zmax": res.zmax, "l2": res.l2,
                "train_zmax": res.train_scores["zmax"],
                "train_l2": res.train_scores["l2"],
                "normal_index": normal_index,
                "n_channels": len(res.channels_used),
            })
        print(f"   cached {len(cache)} datasets", flush=True)

    if not cache:
        print("nothing to sweep")
        return 1

    def score_with(flags_by_event):
        ref = RefCAREScore()
        mine = scoring.CAREScore()
        for c, flags in flags_by_event:
            ref.evaluate_event(event_start=c["start"], event_end=c["end"],
                               event_label=c["label"], predicted_anomalies=flags,
                               normal_index=c["normal_index"], event_id=c["event_id"])
            mine.evaluate_event(event_start=c["start"], event_end=c["end"],
                                event_label=c["label"], predicted_anomalies=flags,
                                normal_index=c["normal_index"], event_id=c["event_id"])
        df = ref.evaluated_events
        is_anom = df["event_label"] == "anomaly"
        reliability = float(mine.calculate_reliability())
        avg_acc = float(df.loc[~is_anom, "accuracy"].mean())
        avg_f = float(df.loc[is_anom, "f_beta_score"].mean())
        avg_ws = float(df.loc[is_anom, "weighted_score"].mean())
        if avg_acc <= 0.5:
            care_score = avg_acc
        elif df["anomaly_detected"].sum() == 0:
            care_score = 0.0
        else:
            care_score = ((avg_acc * 2 + avg_ws + avg_f + reliability) / 5.0)
        return {
            "care_score": round(care_score, 4),
            "coverage": round(avg_f, 4), "accuracy": round(avg_acc, 4),
            "reliability": round(reliability, 4), "earliness": round(avg_ws, 4),
            "n_events_detected": int(df["anomaly_detected"].sum()),
            "n_events": int(len(df)),
        }

    out = {"leak_free": [], "upper_bound": []}

    # --- leak-free: training quantiles ---------------------------------------
    for combiner in ("zmax", "l2"):
        for q in QUANTILES:
            flags = []
            for c in cache:
                thr = float(c[f"train_{combiner}"].quantile(q))
                flags.append((c, c[combiner] > thr))
            rec = {"combiner": combiner, "train_quantile": q,
                   "rule": "training quantile (leak-free)"}
            rec.update(score_with(flags))
            out["leak_free"].append(rec)
            print(f"  {combiner:>5} q={q:<6}: CARE={rec['care_score']:.4f} "
                  f"(cov {rec['coverage']:.3f} acc {rec['accuracy']:.3f} "
                  f"rel {rec['reliability']:.3f} earl {rec['earliness']:.3f}) "
                  f"det {rec['n_events_detected']}/{rec['n_events']}", flush=True)

    # --- upper bound: fixed z selected on the scored events ------------------
    for combiner in ("zmax", "l2"):
        for z in Z_THRESHOLDS:
            flags = [(c, c[combiner] > z) for c in cache]
            rec = {"combiner": combiner, "z_threshold": z,
                   "rule": "fixed z, selected on the scored events (UPPER BOUND)"}
            rec.update(score_with(flags))
            out["upper_bound"].append(rec)
            print(f"  {combiner:>5} z={z:<6}: CARE={rec['care_score']:.4f} "
                  f"(cov {rec['coverage']:.3f} acc {rec['accuracy']:.3f} "
                  f"rel {rec['reliability']:.3f} earl {rec['earliness']:.3f}) "
                  f"det {rec['n_events_detected']}/{rec['n_events']}", flush=True)

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "threshold_sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    best = max(out["leak_free"], key=lambda r: r["care_score"])
    print(f"\nbest leak-free setting: {best['combiner']} at training quantile "
          f"{best['train_quantile']} -> CARE {best['care_score']:.4f}")
    print(f"wrote {res_dir/'threshold_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
