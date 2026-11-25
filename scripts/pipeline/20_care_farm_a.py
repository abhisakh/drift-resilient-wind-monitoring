#!/usr/bin/env python
"""CARE Wind Farm A against EDP's own records: integrity, and alarm triage.

Writes `results/care_farm_a.json`.

    python scripts/pipeline/20_care_farm_a.py

CARE's Farm A is built from EDP's open data, and EDP publishes what CARE leaves
out: the turbines' event log (their own alarms) and a logbook of confirmed
failures. This step

1. puts CARE's Farm A back on its true dates and checks it against EDP: the
   overlapping CARE files, each turbine reading for reading, every fault
   label against the logbook, the weather against EDP's met mast;
2. triages every temperature alarm of the turbines' own alarm system
   (src/windmill/triage.py): at the alarm's hottest readings, how much is the
   component's typical temperature, how much load, how much weather, how much
   excess heat its healthy model cannot explain; and the verdict;
3. measures how long the turbines stood still while windy during and after
   the alarms.

"Real" fault, as recorded: EDP's logbook lists confirmed failures, i.e. a part
repaired or replaced. An alarm is "followed by a failure" when a logged
failure of the same component, where heat was the cause or the symptom,
follows within 30 days (this project's rule). An alarm with no failure is not
judged harmless: the logbook does not record overheating that ended without a
repair.

Parsed spreadsheets are cached in data/interim/ (not committed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import daily_check as dc  # noqa: E402
from windmill import edp, triage  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "care_farm_a.json"
#: Turbines with an event log (T09 is not in EDP's download).
LOGGED = ("T01", "T06", "T07", "T11")
WINDOW_DAYS = 30
WINDY_MS = 5.0


def integrity(raw_overlap: dict, rec: pd.DataFrame, data_dir: Path, cache: Path,
              events: pd.DataFrame, logbook: pd.DataFrame, met: pd.Series) -> dict:
    out: dict = {"overlapping_copies": raw_overlap}
    # Each CARE turbine against its EDP turbine.
    match = {}
    for t in LOGGED:
        d = edp.load_edp_scada(data_dir, t, cache)
        j = rec[rec["turbine"] == t].merge(d, on="ts")
        same = ((j["sensor_11_avg"] == j["Gear_Bear_Temp_Avg"])
                & (j["sensor_0_avg"] == j["Amb_Temp_Avg"])
                & np.isclose(j["wind_speed_3_avg"], j["Amb_WindSpeed_Avg"], atol=0.05)
                & np.isclose(j["power_30_avg"] * edp.RATED_KW, j["Grd_Prod_Pwr_Avg"], atol=1.0))
        match[t] = {"shared_stamps": int(len(j)), "identical_share": round(float(same.mean()), 5)}
    out["care_vs_edp_scada"] = match
    # Every CARE fault label against the logbook.
    labels = []
    for e in events.itertuples():
        f = logbook[logbook["turbine"] == e.turbine]
        near = f[(f["ts"] - e.event_end).abs() <= pd.Timedelta(minutes=30)]
        labels.append({"event_id": int(e.event_id), "label": e.event_label,
                       "turbine": e.turbine, "end": str(e.event_end)[:16],
                       "logbook": (f"{near.iloc[0]['component']} {str(near.iloc[0]['ts'])[:16]}"
                                   if len(near) else None)})
    out["care_events_vs_logbook"] = labels
    out["anomaly_events_matching_logbook"] = sum(1 for x in labels
                                                 if x["label"] == "anomaly" and x["logbook"])
    out["anomaly_events"] = sum(1 for x in labels if x["label"] == "anomaly")
    out["normal_events_with_logbook_failure"] = sum(1 for x in labels
                                                    if x["label"] == "normal" and x["logbook"])
    # The status column: 14-day "downtime" blocks ending at logbook failures.
    blocks = []
    for f in logbook.itertuples():
        g = rec[rec["turbine"] == f.turbine]
        if not len(g) or not (g["ts"].min() <= f.ts <= g["ts"].max()):
            continue
        s4 = g[(g["status_type_id"] == 4) & (g["ts"] > f.ts - pd.Timedelta(days=15))
               & (g["ts"] <= f.ts + pd.Timedelta(hours=1))]
        prod = float((s4["power_30_avg"] > 0.05).mean()) if len(s4) else None
        blocks.append({"turbine": f.turbine, "failure": str(f.ts)[:16],
                       "downtime_days": round((s4["ts"].max() - s4["ts"].min()).total_seconds() / 86400, 2)
                       if len(s4) else 0.0,
                       "producing_share_during_downtime": None if prod is None else round(prod, 3)})
    out["status_downtime_vs_logbook"] = blocks
    # Weather against the met mast.
    mm = {}
    for t, g in rec.groupby("turbine"):
        j = g.set_index("ts")[["sensor_0_avg"]].join(met, how="inner").dropna()
        mm[t] = {"n": int(len(j)),
                 "ambient_r": round(float(j.corr().iloc[0, 1]), 3),
                 "turbine_minus_mast_k": round(float((j["sensor_0_avg"] - j["met_ambient"]).mean()), 2)}
    out["turbine_ambient_vs_met_mast"] = mm
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    data_dir = Path(args.data)
    cache = data_dir / "interim" / "edp"
    t0 = time.time()

    rec_cache = cache / "care_farm_a_true_dates.pkl"
    if rec_cache.exists():
        rec, overlap = pd.read_pickle(rec_cache)
    else:
        parts = []
        for f in sorted((edp.care_a_dir(data_dir) / "datasets").glob("*.csv")):
            d = pd.read_csv(f, sep=";", usecols=edp.CARE_COLUMNS)
            d["event_id"] = int(f.stem)
            parts.append(d)
        raw = pd.concat(parts, ignore_index=True)
        raw["time_stamp"] = pd.to_datetime(raw["time_stamp"])
        start = raw.groupby("event_id")["time_stamp"].transform("min")
        raw["ts"] = raw["time_stamp"] - pd.to_timedelta(start.map(edp.shift_days), unit="D")
        pair = raw.merge(raw, on=["asset_id", "ts"])
        pair = pair[pair["event_id_x"] < pair["event_id_y"]]
        same = []
        for c in ("sensor_11_avg", "sensor_0_avg", "power_30_avg"):
            x, y = pair[c + "_x"], pair[c + "_y"]
            same.append((x == y) | (x.isna() & y.isna()))
        overlap = {"timestamp_pairs": int(len(pair)),
                   "identical_share": round(float(np.logical_and.reduce(same).mean()), 6)}
        rec = edp.to_true_dates(raw.drop(columns="ts"))
        cache.mkdir(parents=True, exist_ok=True)
        pd.to_pickle((rec, overlap), rec_cache)
    print(f"  CARE Farm A on true dates: {len(rec)} rows ({time.time() - t0:.0f}s)")

    events = edp.care_a_events(data_dir)
    logbook = edp.load_logbook(data_dir, cache)
    log = edp.load_event_log(data_dir, cache)
    met = edp.load_met_mast(data_dir, cache)
    print(f"  EDP records loaded ({time.time() - t0:.0f}s)")

    integ = integrity(overlap, rec, data_dir, cache, events, logbook, met)
    print(f"  integrity checked ({time.time() - t0:.0f}s)")

    # The weather: the met mast, falling back to the turbine's own sensor
    # (1 K warmer on average) where the mast has a gap.
    rec["ambient"] = met.reindex(rec["ts"]).to_numpy()
    rec["ambient"] = rec["ambient"].fillna(rec["sensor_0_avg"] - 1.0)
    loads = rec.pivot_table(index="ts", columns="turbine", values="power_30_avg")

    spells = edp.label_spells(edp.alarm_spells(log[log["turbine"].isin(LOGGED)]), logbook, WINDOW_DAYS)
    rows = []
    for fam, (ch, _) in edp.FAMILIES.items():
        temps = rec.pivot_table(index="ts", columns="turbine", values=ch)
        for t in LOGGED:
            s = spells[(spells["family"] == fam) & (spells["turbine"] == t)]
            if not len(s):
                continue
            g = rec[rec["turbine"] == t]
            frame = pd.DataFrame({"ts": g["ts"].dt.tz_localize("UTC"),
                                  "P_norm": g["power_30_avg"].clip(lower=0).to_numpy(),
                                  "era5_temp_2m": g["ambient"].to_numpy(), ch: g[ch].to_numpy()})
            repairs = logbook[(logbook["turbine"] == t) & (logbook["family"] == fam)]["ts"]
            comp = triage.Component(frame, t, ch, repairs=list(repairs),
                                    sibling_gap=triage.sibling_gap(temps, loads, t))
            for r in s.itertuples():
                rows.append({"family": fam, "turbine": t, "start": str(r.start)[:10],
                             "days": int(r.days), "entries": int(r.entries),
                             "followed_thermal": bool(r.followed_thermal),
                             "followed_any": bool(r.followed_any),
                             **comp.judge_spell(r.start, r.end)})
            print(f"  {t} {fam}: {len(s)} alarms judged ({time.time() - t0:.0f}s)")
    tri = pd.DataFrame(rows)

    # Stops while windy: during, after, and normally.
    rec["day"] = rec["ts"].dt.floor("D")
    rec["stopped"] = (rec["wind_speed_3_avg"] >= WINDY_MS) & (rec["power_30_avg"] < dc.RUNNING)
    per_day = rec.groupby(["turbine", "day"])["stopped"].sum() / 6.0

    def hours(t, lo, hi):
        s = per_day.loc[t]
        s = s[(s.index >= lo) & (s.index < hi)]
        return float(s.mean()) if len(s) else np.nan
    stop = []
    for r in tri.itertuples():
        s = pd.Timestamp(r.start); e = s + pd.Timedelta(days=r.days)
        stop.append((hours(r.turbine, s, e), hours(r.turbine, e, e + pd.Timedelta(days=30)),
                     float(per_day.loc[r.turbine].mean())))
    tri[["stop_h_during", "stop_h_next30", "stop_h_normal"]] = stop

    def stop_summary(x):
        return {k: round(float(x[k].mean()), 2) for k in ("stop_h_during", "stop_h_next30", "stop_h_normal")}

    all_spells = edp.label_spells(edp.alarm_spells(
        log[log["turbine"].isin(LOGGED)],
        {**edp.FAMILIES,
         "HYDRAULIC_GROUP": ("sensor_41_avg", r"Oil leakage in Hub|Low workingpressure|High pres offlin"),
         "GEARBOX_PUMP": ("sensor_11_avg", r"Thermoerr\. oil pump")}), logbook, WINDOW_DAYS)
    all_spells.loc[all_spells["family"] == "GEARBOX_PUMP", "followed_any"] = all_spells[
        all_spells["family"] == "GEARBOX_PUMP"].apply(
        lambda r: bool(((logbook["turbine"] == r["turbine"]) & (logbook["family"] == "GEARBOX")
                        & ((logbook["ts"] - r["start"]).dt.days.between(-1, WINDOW_DAYS))).any()), axis=1)

    out = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "seconds": round(time.time() - t0, 1),
        "protocol": {
            "data": ("CARE to Compare Wind Farm A on its true dates, and EDP's open data "
                     "(doi 10.17632/zjxjnjp3xs.2): failure logbook, event log, met mast"),
            "care_turbines": {str(k): v for k, v in edp.CARE_A_TURBINES.items()},
            "turbines_with_event_log": list(LOGGED),
            "real_fault": ("a confirmed failure in EDP's logbook (a part repaired or replaced); "
                           f"an alarm is followed by a failure when one of the same component "
                           f"follows within {WINDOW_DAYS} days"),
            "thermal_failure": "a logged failure whose remark names heat (not a failed temperature sensor)",
            "temperature_alarms": {k: v[1] for k, v in edp.FAMILIES.items()},
            "spell": "consecutive calendar days with an alarm of the family on the turbine",
            "weather": "EDP's met mast at the farm (the turbine's ambient sensor, less 1 K, in its gaps)",
            "triage": {"split": "reading = typical + load + weather + excess, at the day's hottest "
                                f"{int(triage.PEAK_SHARE * 100)} % of readings",
                       "real_excess_heat_k": triage.REAL_HEAT_K,
                       "references": ("own healthy model (the daily check's monthly model, or a "
                                      "provisional one on two weeks of healthy data after a "
                                      "recorded repair) and the siblings at the same moment"),
                       "verdict": "the worst day of the spell decides"},
            "stops": f"hours a day stopped while the turbine's wind was {WINDY_MS:g} m/s or more",
        },
        "integrity": integ,
        "turbine_alarms": {
            "by_family": {f: {"spells": int(len(x)), "followed_by_failure": int(x["followed_any"].sum())}
                          for f, x in all_spells.groupby("family")},
            "spells": int(len(all_spells)),
            "followed_by_failure": int(all_spells["followed_any"].sum()),
        },
        "triage": {
            "table": triage.summarise(tri),
            "by_family": {f: triage.summarise(x) for f, x in tri.groupby("family")},
            "mean_split_k": {v: {k: round(float(x[k].mean()), 2) for k in
                                 ("reading", "typical", "load", "weather", "excess")}
                             for v, x in tri.groupby("verdict") if x["excess"].notna().any()},
            "stops": {"followed_by_failure": stop_summary(tri[tri["followed_thermal"]]),
                      "no_failure": stop_summary(tri[~tri["followed_thermal"]])},
            "alarms": tri.replace({np.nan: None}).to_dict(orient="records"),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=str))
    t = out["triage"]["table"]
    print("  verdicts: " + ", ".join(f"{k} {v['alarms']}" for k, v in t.items()))
    print(f"wrote {out_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
