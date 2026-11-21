"""How much thermal signal is actually inside a fault window?

    python scripts/pipeline/14_thermal_signal.py

WHY
---
This project detects faults from THERMAL channels. The CARE benchmark's
faults are mostly not thermal -- "Hydraulic group", "Pitchfailure",
"Converter Failure", "Safety chain relay open". This script measures what
that costs, so the limitation is a number rather than an excuse.

For every labelled anomaly event it compares the detector's score INSIDE the
event window against the score OUTSIDE it, on the same dataset. If the thermal
signal inside the fault were meaningful, the in-event score would be
consistently higher. If it is not, the two are indistinguishable -- and the
project's low Coverage is a property of the CHANNELS, not of the threshold.

This is the same measurement the diagnostic scripts/diagnostics/06_spike_thermal.py
prints, promoted into the pipeline because it matters: anything worth
asserting in the README has to be backed by an artefact.

Outputs results/thermal_signal.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care, multivariate  # noqa: E402

FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
#: The training quantile the scored arm uses (scripts/pipeline/08_multivariate.py).
TRAIN_QUANTILE = 0.90


def measure(data_dir: Path) -> list[dict]:
    rows = []
    for farm in FARMS:
        farm_dir = care.care_root(data_dir) / farm
        if not farm_dir.exists():
            continue
        info = care.event_info(farm_dir)
        for path in sorted((farm_dir / "datasets").glob("*.csv")):
            eid = int(path.stem)
            row = info[info["event_id"].astype(int) == eid]
            if row.empty:
                continue
            row = row.iloc[0]
            if str(row["event_label"]).strip().lower() != "anomaly":
                continue

            df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
            train = df[df["train_test"] == "train"]
            test = df[df["train_test"] == "prediction"]
            train = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]
            if train.empty or test.empty:
                continue

            columns = care.dataset_columns(path)
            pcol, thermal, _wind, ambient = care.feature_rows(farm_dir, columns)
            if not pcol or len(thermal) < 2:
                continue

            res = multivariate.detect(
                train, test, thermal, pcol, ambient_col=ambient,
                quantile=TRAIN_QUANTILE, max_channels=24,
            )
            if not res.channels_used:
                continue

            ids = test.index.to_numpy()
            s, e = int(row["event_start_id"]), int(row["event_end_id"])
            in_event = (ids >= s) & (ids <= e)
            normal = test["status_type_id"].isin(care.NORMAL_STATUS_IDS)
            scorable = in_event & normal.to_numpy()
            outside = ~in_event

            zmax = res.zmax.to_numpy()
            l2 = res.l2.to_numpy()
            rows.append({
                "farm": farm,
                "event_id": eid,
                "description": str(row.get("event_description", ""))[:80],
                "n_scorable_event_rows": int(scorable.sum()),
                "zmax_median_event": _med(zmax, scorable),
                "zmax_median_outside": _med(zmax, outside),
                "l2_median_event": _med(l2, scorable),
                "l2_median_outside": _med(l2, outside),
            })
    return rows


def _med(values: np.ndarray, mask: np.ndarray):
    if not mask.any():
        return None
    v = values[mask]
    v = v[np.isfinite(v)]
    return round(float(np.median(v)), 4) if v.size else None


def main() -> int:
    data_dir = ROOT / "data"
    if not care.care_root(data_dir).exists():
        print("CARE not extracted -- run scripts/pipeline/01_build_dataset.py")
        return 1

    rows = measure(data_dir)
    if not rows:
        print("no anomaly events were scored")
        return 1
    df = pd.DataFrame(rows)

    scorable = df[df["n_scorable_event_rows"] > 0]
    summary = {"n_anomaly_events": int(len(df)),
               "n_with_scorable_rows": int(len(scorable))}
    for stat, col in (("max_z", "zmax"), ("l2", "l2")):
        e = scorable[f"{col}_median_event"].dropna()
        o = scorable[f"{col}_median_outside"].dropna()
        both = e.index.intersection(o.index)
        e, o = e.loc[both], o.loc[both]
        summary[f"{stat}_median_in_event"] = round(float(e.median()), 4) if len(e) else None
        summary[f"{stat}_median_outside"] = round(float(o.median()), 4) if len(o) else None
        summary[f"{stat}_event_higher_fraction"] = (
            round(float((e > o).mean()), 4) if len(e) else None)

    # Per farm, because the farm's fault mix drives the result.
    per_farm = {}
    for farm, g in scorable.groupby("farm"):
        e = g["zmax_median_event"].dropna()
        o = g["zmax_median_outside"].dropna()
        both = e.index.intersection(o.index)
        e, o = e.loc[both], o.loc[both]
        per_farm[farm] = {
            "n": int(len(g)),
            "zmax_median_in_event": round(float(e.median()), 4) if len(e) else None,
            "zmax_median_outside": round(float(o.median()), 4) if len(o) else None,
            "zmax_event_higher_fraction": round(float((e > o).mean()), 4) if len(e) else None,
        }

    out = {
        "purpose": "measure how much thermal signal is actually inside a fault "
                   "window, to quantify the scope limit of a thermal-only "
                   "detector on a benchmark whose faults are mostly not thermal",
        "protocol": f"detector = windmill.multivariate.detect, threshold = "
                    f"training quantile {TRAIN_QUANTILE}, cross-fitted",
        **summary,
        "per_farm": per_farm,
        "per_event": rows,
    }
    path = ROOT / "results" / "thermal_signal.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    print("=== THERMAL SIGNAL INSIDE FAULT WINDOWS ===")
    print(f"  anomaly events: {summary['n_anomaly_events']} "
          f"({summary['n_with_scorable_rows']} with scorable rows)")
    print(f"  max-z  median in-event {summary['max_z_median_in_event']} | "
          f"outside {summary['max_z_median_outside']} | "
          f"event higher in {summary['max_z_event_higher_fraction']}")
    print(f"  L2     median in-event {summary['l2_median_in_event']} | "
          f"outside {summary['l2_median_outside']} | "
          f"event higher in {summary['l2_event_higher_fraction']}")
    print()
    for farm, v in per_farm.items():
        print(f"  {farm:<14} n={v['n']:>3}  in-event {v['zmax_median_in_event']} | "
              f"outside {v['zmax_median_outside']} | "
              f"higher in {v['zmax_event_higher_fraction']}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
