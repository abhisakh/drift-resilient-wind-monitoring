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

    python scripts/06_spike_thermal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CARE_ROOT = ROOT / "data" / "raw" / "care_extracted" / "CARE_To_Compare"
FARMS = ("Wind Farm A", "Wind Farm B", "Wind Farm C")
SEP = ";"
NORMAL_STATUS_IDS = (0, 2)
LOAD_BIN = 0.05   # fraction of rated power per bin


def temp_channels(farm_dir: Path) -> list[str]:
    """Average-statistic channels whose unit is a temperature."""
    fd = pd.read_csv(farm_dir / "feature_description.csv", sep=SEP)
    fd["sensor_name"] = fd["sensor_name"].astype(str)
    fd["unit"] = fd["unit"].astype(str)
    fd["description"] = fd["description"].astype(str)
    unit_ok = fd["unit"].str.contains("Celsius|°C", case=False, na=False, regex=True)
    has_avg = fd["statistics_type"].astype(str).str.contains("average", case=False, na=False)
    out = []
    for base in fd.loc[unit_ok & has_avg, "sensor_name"]:
        for cand in (f"{base}_avg", base):
            out.append(cand)
    return out


def power_channel(farm_dir: Path) -> str | None:
    fd = pd.read_csv(farm_dir / "feature_description.csv", sep=SEP)
    fd["sensor_name"] = fd["sensor_name"].astype(str)
    fd["description"] = fd["description"].astype(str)
    pw = fd[fd["sensor_name"].str.startswith("power_")
            & fd["description"].str.contains("active power", case=False, na=False)]
    if pw.empty:
        pw = fd[fd["sensor_name"].str.startswith("power_")]
    for base in pw["sensor_name"]:
        if f"{base}_avg" in ALL_COLUMNS:
            return f"{base}_avg"
    return None


ALL_COLUMNS: dict[str, list[str]] = {}


def combined_score(
    train: pd.DataFrame, test: pd.DataFrame, channels: list[str], pcol: str
) -> tuple[pd.Series, pd.Series]:
    """Load-conditioned thermal residual, combined over channels."""
    tr = train.dropna(subset=[pcol] + [c for c in channels if c in train.columns])
    tr = tr[tr[pcol] > 0.02]
    if tr.empty:
        z = pd.Series(0.0, index=test.index)
        return z, z
    load_bin_tr = np.floor(tr[pcol] / LOAD_BIN).astype("int64")

    z_cols = {}
    for c in channels:
        if c not in tr.columns or c not in test.columns:
            continue
        t = tr.dropna(subset=[c])
        if len(t) < 200:
            continue
        lb = np.floor(t[pcol] / LOAD_BIN).astype("int64")
        g = pd.DataFrame({"lb": lb, "c": t[c].to_numpy()})
        counts = g.groupby("lb")["c"].size()
        med = g.groupby("lb")["c"].median()
        med = med[counts[counts >= 15].index]
        if len(med) < 3:
            continue
        resid = t[c].to_numpy() - med.reindex(lb).to_numpy()
        g2 = pd.DataFrame({"lb": lb, "r": resid})
        spreads = {}
        for b, gg in g2.groupby("lb"):
            r = gg["r"].to_numpy()
            if len(r) < 15:
                continue
            m = np.median(r)
            mad = np.median(np.abs(r - m))
            spreads[b] = 1.4826 * mad if mad > 0 else (np.std(r) or np.nan)
        if not spreads:
            continue

        lb_te = np.floor(test[pcol].to_numpy() / LOAD_BIN).astype("int64")
        ref = med.reindex(lb_te).to_numpy()
        sp = pd.Series(spreads).reindex(lb_te).to_numpy()
        r_te = test[c].to_numpy() - ref
        z = np.where(np.isfinite(sp) & (sp > 0), (r_te - 0.0) / sp, np.nan)
        z_cols[c] = pd.Series(z, index=test.index)

    if not z_cols:
        z = pd.Series(0.0, index=test.index)
        return z, z
    Z = pd.DataFrame(z_cols)
    # One-sided: overheating is the failure mode we care about.
    pos = Z.clip(lower=0)
    zmax = Z.max(axis=1, skipna=True)
    l2 = np.sqrt((pos ** 2).sum(axis=1, skipna=True))
    return zmax.fillna(0.0), l2.fillna(0.0)


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
        ALL_COLUMNS[farm] = pd.read_csv(sample[0], sep=SEP, nrows=0).columns.tolist()
        pcol = power_channel(farm_dir)
        chans = [c for c in temp_channels(farm_dir) if c in ALL_COLUMNS[farm]]
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
            zmax, l2 = combined_score(train, test, use, pcol)

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

    df = pd.DataFrame(rows)
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "spike_thermal.json").write_text(json.dumps(df.to_dict("records"), indent=2) + "\n")

    scorable = df[df["n_scorable_event_rows"] > 0]
    print(f"\n=== SPIKE: {len(df)} anomaly events, {len(scorable)} with scorable rows ===")
    if len(scorable):
        for col, name in (("zmax_median_event", "max-z"), ("l2_median_event", "L2")):
            e = scorable[col].dropna()
            o = scorable[col.replace("_event", "_outside")].dropna()
            if len(e):
                sep = float((e > o).mean())
                print(f"  {name:>6}: median in-event {e.median():8.3f} | outside {o.median():8.3f} "
                      f"| event higher in {sep*100:.0f}% of events")
        print("\n  per event (first 12):")
        print(scorable[["farm", "event_id", "n_scorable_event_rows", "zmax_median_event",
                        "zmax_median_outside", "l2_median_event", "l2_median_outside",
                        "description"]].head(12).to_string(index=False))
    print(f"\nwrote {res/'spike_thermal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
