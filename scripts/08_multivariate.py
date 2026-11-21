"""Option 1: score the deterministic multivariate detector on CARE to Compare.

    python scripts/08_multivariate.py

Runs the Deterministic Multivariate Bins detector (src/windmill/multivariate.py)
over all 95 CARE event datasets and scores it with the CARE score, using the
benchmark authors' own implementation so the numbers are directly comparable
with the published AE baseline of 0.66.

Both combiners are reported: `max` (largest one-sided z across channels) and
`l2` (norm of positive deviations across channels).

Outputs results/multivariate.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care, multivariate, scoring  # noqa: E402

CARE_ROOT = care.care_root(ROOT / "data")
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")

PUBLISHED_BASELINES = {"AE (published)": 0.66, "random": 0.50, "iForest": "< 0.5"}

#: The threshold rule: the quantile of the cross-fitted TRAINING score
#: distribution. Leak-free -- the threshold never sees an event it is later
#: scored on -- and the same rule the CARE paper's own baseline uses. The value
#: is the best leak-free setting found by scripts/08b_sweep_threshold.py.
TRAIN_QUANTILE = 0.90


def score_arm(flags_by_event: list[tuple[int, str, pd.Series, pd.Series]]) -> dict:
    """Score a set of per-event flag series with the authors' implementation."""
    from energy_fault_detector.evaluation.care_score import CAREScore as RefCAREScore

    ref = RefCAREScore()
    mine = scoring.CAREScore()
    mismatches = []
    for eid, label, flags, normal_index in flags_by_event:
        if label not in ("anomaly", "normal"):
            continue
        s, e = WINDOWS[eid]
        ref.evaluate_event(event_start=s, event_end=e, event_label=label,
                           predicted_anomalies=flags, normal_index=normal_index,
                           event_id=eid)
        m = mine.evaluate_event(event_start=s, event_end=e, event_label=label,
                                predicted_anomalies=flags, normal_index=normal_index,
                                event_id=eid)
        r = ref.evaluated_events.iloc[-1]
        for key in ("f_beta_score", "accuracy", "max_criticality", "weighted_score"):
            a, b = r[key], m[key]
            if pd.isna(a) and pd.isna(b):
                continue
            if abs(float(a) - float(b)) > 1e-6:
                mismatches.append({"event": eid, "metric": key,
                                   "reference": float(a), "ours": float(b)})

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
        care_score = ((avg_acc * 2.0 + avg_ws * 1.0 + avg_f * 1.0 + reliability * 1.0) / 5.0)

    return {
        "care_score": round(float(care_score), 4),
        "sub_scores": {
            "coverage": round(avg_f, 4),
            "accuracy": round(avg_acc, 4),
            "reliability": round(reliability, 4),
            "earliness": round(avg_ws, 4),
        },
        "n_events": int(len(df)),
        "n_anomaly": int(is_anom.sum()),
        "n_normal": int((~is_anom).sum()),
        "n_events_detected": int(df["anomaly_detected"].sum()),
        "n_mismatches_vs_reference": len(mismatches),
        "mismatches": mismatches[:10],
        "per_event": df.to_dict("records"),
    }


WINDOWS: dict[int, tuple[int, int]] = {}


def main() -> int:
    global WINDOWS

    events: list[dict] = []
    flags_max: list[tuple[int, str, pd.Series, pd.Series]] = []
    flags_l2: list[tuple[int, str, pd.Series, pd.Series]] = []

    for farm in FARMS:
        farm_dir = CARE_ROOT / farm
        if not farm_dir.exists():
            continue
        info = care.event_info(farm_dir)
        for _, row in info.iterrows():
            WINDOWS[int(row["event_id"])] = (int(row["event_start_id"]), int(row["event_end_id"]))

        sample = sorted((farm_dir / "datasets").glob("*.csv"))
        if not sample:
            continue
        columns = care.dataset_columns(sample[0])
        pcol, thermal, wind, ambient = care.feature_rows(farm_dir, columns)
        print(f"[{farm}] power={pcol} thermal={len(thermal)} wind={len(wind)} "
              f"ambient={ambient}")
        if not pcol:
            continue

        for path in sample:
            eid = int(path.stem)
            row = info[info["event_id"].astype(int) == eid]
            if row.empty:
                continue
            row = row.iloc[0]
            label = str(row["event_label"]).strip().lower()

            df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
            train = df[df["train_test"] == "train"]
            test = df[df["train_test"] == "prediction"]
            train = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]
            if train.empty or test.empty:
                continue

            res = multivariate.detect(train, test, thermal, pcol,
                                      ambient_col=ambient, quantile=TRAIN_QUANTILE)
            if not res.channels_used:
                continue

            normal_index = pd.Series(
                test["status_type_id"].isin(care.NORMAL_STATUS_IDS).to_numpy(),
                index=test.index)

            ids = test.index.to_numpy()
            s, e = WINDOWS[eid]
            in_event = (ids >= s) & (ids <= e)
            scorable = in_event & normal_index.to_numpy()

            events.append({
                "event_id": eid, "farm": farm, "label": label,
                "description": str(row.get("event_description", ""))[:100],
                "n_channels": len(res.channels_used),
                "threshold_max": round(res.threshold_max, 4),
                "threshold_l2": round(res.threshold_l2, 4),
                "n_flags_max": int(res.flags_max.sum()),
                "n_flags_l2": int(res.flags_l2.sum()),
                "n_scorable_event_rows": int(scorable.sum()),
                "zmax_median_event": round(float(res.zmax.to_numpy()[scorable].mean()), 4)
                if scorable.sum() else None,
                "l2_median_event": round(float(res.l2.to_numpy()[scorable].mean()), 4)
                if scorable.sum() else None,
                "zmax_median_outside": round(float(res.zmax.to_numpy()[~in_event].mean()), 4),
                "l2_median_outside": round(float(res.l2.to_numpy()[~in_event].mean()), 4),
            })
            flags_max.append((eid, label, res.flags_max, normal_index))
            flags_l2.append((eid, label, res.flags_l2, normal_index))

    if not events:
        print("no datasets processed")
        return 1

    out = {
        "arm": "Deterministic Multivariate Bins (Option 1)",
        "published_baselines": PUBLISHED_BASELINES,
        "max_combiner": score_arm(flags_max),
        "l2_combiner": score_arm(flags_l2),
        "per_dataset": events,
    }
    for key in ("max_combiner", "l2_combiner"):
        out[key]["beats_published_AE"] = bool(out[key]["care_score"] > 0.66)

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "multivariate.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

    print(f"\n=== DETERMINISTIC MULTIVARIATE BINS ({len(events)} events) ===")
    print(f"  {'combiner':<8}{'coverage':>10}{'accuracy':>10}{'reliability':>13}"
          f"{'earliness':>12}{'CARE':>9}{'events det.':>13}")
    for key in ("max_combiner", "l2_combiner"):
        v = out[key]
        s = v["sub_scores"]
        print(f"  {key.replace('_combiner',''):<8}{s['coverage']:>10.4f}{s['accuracy']:>10.4f}"
              f"{s['reliability']:>13.4f}{s['earliness']:>12.4f}{v['care_score']:>9.4f}"
              f"{v['n_events_detected']:>9d}/{v['n_events']}")
    print(f"\n  published AE baseline: 0.66 | random: 0.50")
    print(f"  mismatches vs the benchmark authors' scorer: "
          f"{out['max_combiner']['n_mismatches_vs_reference']} (max), "
          f"{out['l2_combiner']['n_mismatches_vs_reference']} (l2)")
    print(f"\nwrote {res_dir/'multivariate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
