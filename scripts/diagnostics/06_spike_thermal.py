"""Spike: does a load-conditioned thermal residual actually separate anomaly
events from normal ones?

This validates the Option 1 design before building the full pipeline:

  * reference: for each temperature channel, the median value per load bin,
    estimated on the dataset's own TRAINING portion (normal operation rows only);
  * residual: measured - reference at the observed load;
  * combined: max over channels of the one-sided (overheating) normalised z,
    and the L2 norm of the positive deviations.

If the combined score does not rise during anomaly windows, the whole arm is
pointless and must be redesigned.

    python scripts/diagnostics/06_spike_thermal.py
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

CARE_ROOT = care.care_root(ROOT / "data")
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
SEP = ";"
NORMAL_STATUS_IDS = (0, 2)
LOAD_BIN = 0.05   # fraction of rated power per bin


def feature_rows(farm_dir: Path, columns: list[str]):
    """(power channel, thermal channels, wind channels, ambient channel).

    This uses the SHARED implementation in windmill.care rather than a local
    copy. The local copy was wrong: it tested `f"{base}_avg" in ALL_COLUMNS`,
    which asks whether a COLUMN NAME is a key of the farm->columns dict, and
    so never found the power channel for two of the three farms.
    """
    return care.feature_rows(farm_dir, columns)


def combined_score(
    train: pd.DataFrame, test: pd.DataFrame, channels: list[str], pcol: str,
    ambient: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Load- and ambient-conditioned thermal z-scores, combined over channels.

    This used to be a local re-implementation of the detector's maths. That
    was wrong in a way that mattered: it had none of the spread floors, so
    a channel whose robust spread collapsed produced z-scores in the
    HUNDREDS, and the probe reported Wind Farm C's thermal channels running
    "517 sigma" hot. The shared detector in windmill.multivariate has the
    floors, the ambient correction and the cross-fitting, so the probe uses
    it and reports what the project actually measures.
    """
    res = multivariate.detect(train, test, channels, pcol, ambient_col=ambient,
                              quantile=0.90, max_channels=24)
    if not res.channels_used:
        z = pd.Series(0.0, index=test.index)
        return z, z
    return res.zmax, res.l2


def main() -> int:
    rows = []
    for farm in FARMS:
        farm_dir = CARE_ROOT / farm
        if not farm_dir.exists():
            continue
        info = pd.read_csv(farm_dir / "event_info.csv", sep=SEP)
        sample = sorted((farm_dir / "datasets").glob("*.csv"))
        if not sample:
            continue
        columns = pd.read_csv(sample[0], sep=SEP, nrows=0).columns.tolist()
        pcol, thermal, _wind, ambient = feature_rows(farm_dir, columns)
        chans = [c for c in thermal if c in columns]
        print(f"[{farm}] power={pcol} thermal channels={len(chans)}")

        for path in sample:
            eid = int(path.stem)
            row = info[info["event_id"].astype(int) == eid]
            if row.empty:
                continue
            row = row.iloc[0]
            label = str(row["event_label"]).strip().lower()
            if label != "anomaly":
                continue

            df = pd.read_csv(path, sep=SEP, low_memory=False).set_index("id")
            train = df[df["train_test"] == "train"]
            test = df[df["train_test"] == "prediction"]
            train = train[train["status_type_id"].isin(NORMAL_STATUS_IDS)]
            if train.empty or test.empty or pcol is None:
                continue
            use = [c for c in chans if c in df.columns]
            if not use:
                continue
            zmax, l2 = combined_score(train, test, use, pcol, ambient)

            ids = test.index.to_numpy()
            in_event = (ids >= int(row["event_start_id"])) & (ids <= int(row["event_end_id"]))
            status = test["status_type_id"].to_numpy()
            scorable = in_event & np.isin(status, NORMAL_STATUS_IDS)

            if scorable.sum() > 0:
                ev_z = float(np.nanmedian(zmax.to_numpy()[scorable]))
                ev_l2 = float(np.nanmedian(l2.to_numpy()[scorable]))
            else:
                ev_z = ev_l2 = np.nan
            pre = in_event & ~np.isin(status, NORMAL_STATUS_IDS)
            out_ev = ~in_event
            rows.append({
                "farm": farm, "event_id": eid,
                "description": str(row.get("event_description", ""))[:70],
                "n_scorable_event_rows": int(scorable.sum()),
                "n_channels": len(use),
                "zmax_median_event": round(ev_z, 3) if np.isfinite(ev_z) else None,
                "l2_median_event": round(ev_l2, 3) if np.isfinite(ev_l2) else None,
                "zmax_median_outside": round(float(np.nanmedian(zmax.to_numpy()[out_ev])), 3),
                "l2_median_outside": round(float(np.nanmedian(l2.to_numpy()[out_ev])), 3),
            })

    if not rows:
        print("no anomaly events were scored -- is the CARE data extracted?")
        return 1
    df = pd.DataFrame(rows)

    # A DIAGNOSTIC writes no artefact. It is a probe for exploring the data,
    # not a pipeline step: anything it produced that mattered was promoted
    # into scripts/pipeline/ and committed as a result file. The old version
    # wrote results/spike_thermal.json, which contradicted both this file's
    # home in scripts/diagnostics/ and the README's statement that the
    # diagnostics produce nothing.
    scorable = df[df["n_scorable_event_rows"] > 0]
    print(f"\n=== SPIKE: {len(df)} anomaly events, {len(scorable)} with scorable rows ===")
    if len(scorable):
        for col, name in (("zmax_median_event", "max-z"), ("l2_median_event", "L2")):
            e = scorable[col].dropna()
            o = scorable[col.replace("_event", "_outside")].dropna()
            # The two series are dropped independently, so their indices can
            # differ. Comparing them unaligned raises "Can only compare
            # identically-labeled Series objects"; the pairwise comparison
            # must be over the events where BOTH values exist.
            both = e.index.intersection(o.index)
            e, o = e.loc[both], o.loc[both]
            if len(e):
                sep = float((e > o).mean())
                print(f"  {name:>6}: median in-event {e.median():8.3f} | outside {o.median():8.3f} "
                      f"| event higher in {sep*100:.0f}% of events")
        print("\n  per event (first 12):")
        print(scorable[["farm", "event_id", "n_scorable_event_rows", "zmax_median_event",
                        "zmax_median_outside", "l2_median_event", "l2_median_outside",
                        "description"]].head(12).to_string(index=False))
    print("\n(this diagnostic writes no artefact -- it is a probe, not a "
          "pipeline step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
