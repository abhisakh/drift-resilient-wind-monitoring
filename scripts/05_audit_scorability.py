"""Audit: how much of each CARE anomaly event is actually scorable?

The CARE score's Coverage and Reliability sub-scores are computed only over rows
the dataset marks as NORMAL OPERATION (status-ID 0 or 2). For an anomaly event the
ground truth inside the event window is True, but if the dataset has already
flagged those rows as status 4 (down due to a fault), they are excluded from the
metric entirely -- the model cannot earn credit for them.

This script measures that, event by event, and writes results/farm_a_audit.json.

    python scripts/05_audit_scorability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care  # noqa: E402

CARE_ROOT = care.care_root(ROOT / "data")
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
SEP = ";"
NORMAL_STATUS_IDS = (0, 2)


def main() -> int:
    rows = []
    for farm in FARMS:
        farm_dir = CARE_ROOT / farm
        if not farm_dir.exists():
            continue
        info = pd.read_csv(farm_dir / "event_info.csv", sep=SEP)
        for path in sorted((farm_dir / "datasets").glob("*.csv")):
            event_id = int(path.stem)
            row = info[info["event_id"].astype(int) == event_id]
            if row.empty:
                continue
            row = row.iloc[0]
            label = str(row["event_label"]).strip().lower()

            df = pd.read_csv(path, sep=SEP, usecols=["id", "train_test", "status_type_id"],
                             low_memory=False)
            test = df[df["train_test"] == "prediction"]
            ids = test["id"].to_numpy()
            in_event = (ids >= int(row["event_start_id"])) & (ids <= int(row["event_end_id"]))
            status = test["status_type_id"].to_numpy()

            n_event = int(in_event.sum())
            n_event_normal = int((in_event & np.isin(status, NORMAL_STATUS_IDS)).sum())
            n_event_fault = int((in_event & (status == 4)).sum())
            n_pred = len(test)
            n_pred_normal = int(np.isin(status, NORMAL_STATUS_IDS).sum())

            rows.append({
                "farm": farm,
                "event_id": event_id,
                "label": label,
                "description": str(row.get("event_description", ""))[:100],
                "n_prediction_rows": n_pred,
                "n_prediction_normal_rows": n_pred_normal,
                "n_event_rows": n_event,
                "n_event_rows_normal": n_event_normal,
                "n_event_rows_status4_fault": n_event_fault,
                "frac_event_rows_normal": round(n_event_normal / n_event, 4) if n_event else None,
                "frac_event_rows_status4": round(n_event_fault / n_event, 4) if n_event else None,
                # A row inside the event window that is ALSO normal-operation is the
                # only kind that can earn Coverage credit.
                "scorable_rows_in_event": n_event_normal,
                "coverage_is_structurally_capped": bool(n_event_normal == 0),
            })

    df = pd.DataFrame(rows)
    res = ROOT / "results"
    res.mkdir(exist_ok=True)

    per_farm = {}
    for farm, g in df.groupby("farm"):
        anom = g[g["label"] == "anomaly"]
        per_farm[farm] = {
            "n_events": int(len(g)),
            "n_anomaly": int(len(anom)),
            "n_anomaly_with_zero_scorable_rows": int(anom["scorable_rows_in_event"].eq(0).sum()),
            "anomaly_events_fully_scorable": int(
                (anom["n_event_rows"] > 0).sum() - anom["scorable_rows_in_event"].eq(0).sum()),
            "mean_frac_event_rows_normal": round(float(anom["frac_event_rows_normal"].mean()), 4)
            if len(anom) else None,
            "mean_frac_event_rows_status4": round(float(anom["frac_event_rows_status4"].mean()), 4)
            if len(anom) else None,
            "total_scorable_rows_in_anomaly_events": int(anom["scorable_rows_in_event"].sum()),
            "total_event_rows_in_anomaly_events": int(anom["n_event_rows"].sum()),
        }

    out = {
        "question": (
            "Which rows inside a labelled anomaly event can a detector actually be "
            "credited for? The CARE score computes Coverage and Reliability only "
            "over rows whose status-ID marks normal operation (0 or 2)."
        ),
        "per_farm": per_farm,
        "all_anomaly_events_zero_scorable": int(
            df[(df["label"] == "anomaly")]["scorable_rows_in_event"].eq(0).sum()),
        "n_anomaly_events": int((df["label"] == "anomaly").sum()),
        "per_event": df.to_dict("records"),
    }
    (res / "farm_a_audit.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

    print("=== SCORABILITY OF LABELLED ANOMALY EVENTS ===")
    print(f"  {'farm':<14}{'anom':>6}{'zero-scorable':>16}{'partly':>9}"
          f"{'mean%normal':>14}{'mean%fault':>13}")
    for farm, s in per_farm.items():
        print(f"  {farm:<14}{s['n_anomaly']:>6}{s['n_anomaly_with_zero_scorable_rows']:>16}"
              f"{s['anomaly_events_fully_scorable']:>9}"
              f"{(s['mean_frac_event_rows_normal'] or 0)*100:>13.1f}%"
              f"{(s['mean_frac_event_rows_status4'] or 0)*100:>12.1f}%")
    print(f"\n  anomaly events with ZERO scorable rows: "
          f"{out['all_anomaly_events_zero_scorable']} / {out['n_anomaly_events']}")
    print(f"\nwrote {res/'farm_a_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
