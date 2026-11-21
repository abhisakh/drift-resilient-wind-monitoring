"""Quick diagnostic: is there a thermal signal in the anomaly window at all?

Run on a handful of events per farm so the answer comes back in seconds.

    python scripts/08a_probe_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care, multivariate  # noqa: E402

CARE_ROOT = ROOT / "data" / "raw" / "care_extracted" / "CARE_To_Compare"
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")


def main() -> int:
    for farm in FARMS:
        farm_dir = CARE_ROOT / farm
        if not farm_dir.exists():
            continue
        info = care.event_info(farm_dir)
        sample = sorted((farm_dir / "datasets").glob("*.csv"))
        columns = care.dataset_columns(sample[0])
        pcol, thermal, wind, ambient = care.feature_rows(farm_dir, columns)
        print(f"\n{'='*78}\n{farm} | power={pcol} thermal={len(thermal)} ambient={ambient}")

        anom = info[info["event_label"].astype(str).str.lower() == "anomaly"]
        picked = 0
        for _, row in anom.iterrows():
            eid = int(row["event_id"])
            path = farm_dir / "datasets" / f"{eid}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
            train = df[df["train_test"] == "train"]
            test = df[df["train_test"] == "prediction"]
            train = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]
            if train.empty or test.empty:
                continue

            res = multivariate.detect(train, test, thermal, pcol, ambient_col=ambient)
            if not res.channels_used:
                print(f"  event {eid}: no channels usable")
                continue

            ids = test.index.to_numpy()
            s, e = int(row["event_start_id"]), int(row["event_end_id"])
            in_event = (ids >= s) & (ids <= e)
            status = test["status_type_id"].to_numpy()
            scorable = in_event & np.isin(status, care.NORMAL_STATUS_IDS)
            normal_idx = np.isin(status, care.NORMAL_STATUS_IDS)

            # Longest run of flagged rows, and the position of the max score.
            z = res.zmax.to_numpy()
            fl = res.flags_max.to_numpy()
            best_run = cur = 0
            for v in fl:
                cur = cur + 1 if v else 0
                best_run = max(best_run, cur)
            z_in = z[in_event]
            z_out = z[~in_event]

            print(f"  event {eid:>3} [{picked}] {str(row['event_description'])[:42]:<42} "
                  f"ch={len(res.channels_used):>2} thr={res.threshold_max:6.2f} "
                  f"flags={fl.sum():>5} maxrun={best_run:>4} "
                  f"z_in={z_in.mean() if len(z_in) else float('nan'):6.2f} "
                  f"z_out={z_out.mean():6.2f} "
                  f"scorable={int(scorable.sum()):>5}")
            picked += 1
            if picked >= 5:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
