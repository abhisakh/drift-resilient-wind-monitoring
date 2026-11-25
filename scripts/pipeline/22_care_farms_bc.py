#!/usr/bin/env python
"""The daily check on CARE Wind Farms B and C (offshore Germany, anonymised).

Writes `results/care_farms_bc.json`. Per-dataset results are cached in
data/interim/care_bc/ (not committed); --fresh recomputes them.

    python scripts/pipeline/22_care_farms_bc.py [--farm B|C]

Each CARE dataset is one turbine's record: about a year of operation, then a
prediction window that either holds a labelled fault ("anomaly") or is
labelled normal. On every dataset this step runs the same policy as on ENGIE
(src/windmill/daily_check.py): a healthy model per temperature sensor, learned
from at least a month of healthy data, frozen for the month, never trained on
flagged days. A day is hot when a sensor's daily mean runs hotter than its
prediction by more than that sensor's tolerance, and a run of hot days is one
call-out. Each sensor's tolerance is learned too: median + 4 robust SDs of its own
daily differences over its last 90 healthy days, never below 2 K (on ENGIE's
gearbox bearing that floor is the tolerance). A component is hot when any of
its sensors is; sensors are not averaged, because an average hides a fault
on one sensor (Farm B's transformer cell). Farm B's transformers
lag their load by hours, so a fixed 2 K fires on ordinary days there. The site's own outside-temperature
sensor stands in for ERA5 (the farms' locations are anonymised on purpose and
are not recovered).

Maintenance. On ENGIE a stop of 3 days or more stands in for maintenance (no
log exists), and the model waits for a new healthy month after it. A farm
that records service (CARE status 3: Farm C; Farm B does not) is also run
with the recorded rule: a stop of 3 days or more is maintenance only if the
turbine logged service during it; any other stop, however long, keeps the
current model (after the usual 12-hour warm-up). Farm C logs service very
often (22 periods of 6 h or more a turbine-year), so a short service visit is
not taken as maintenance. Both rules are reported; the recorded one is the farm's
headline where it exists.

Sensors are grouped into components (gearbox, generator, transformer, rotor
bearing, pitch, converter, hydraulics, cooling); cabin, hub and outside air
are left out. A component's call-out: a run of days on which any of its
sensors ran hot. Each component runs the policy with its own first sensor as
the primary, so a component's hot days are kept out of its own training.

Compared with a FROZEN model: the same healthy model learned once, on the
first month of healthy data, and never refreshed. That is the static baseline
the project argues against; its extra call-outs are the false call-outs the
monthly refresh avoids.

What is counted:
- healthy datasets ("normal"): call-outs anywhere in the record, per
  turbine-year of checked days, and call-outs in the prediction window;
- fault datasets: whether a call-out on the fault's own component (or on any
  component) falls inside the labelled fault window, and how many days of the
  window the model could check at all.
Faults are mapped to a component from CARE's description (keyword rules
below); a fault with no temperature sensor on its component (pitch encoders,
communication, wiring) is reported as outside what a temperature model sees.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care, daily_check as dc, multivariate  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "care_farms_bc.json"
TOLERANCE = 2.0
#: Sensor tolerance: median + this many robust SDs of its past healthy daily differences.
TOL_SIGMA = 4.0
CAL_DAYS = 90
MIN_CAL_DAYS = 30
FARMS = ("B", "C")
#: CARE status 3 = service.
SERVICE_STATUS = 3
#: "standin": a stop of 3 days or more stands in for maintenance (ENGIE's rule).
#: "recorded": such a stop is maintenance only if service was logged during
#: it; any other stop keeps the current model.
MODES = ("standin", "recorded")
NEVER = 36500.0  # days: no stop is long enough

#: Sensor description -> component (first match wins). None = left out.
COMPONENT_RULES = (
    (r"outside|ambient|nacelle|hub|board|cabinet|control box|container|room|"
     r"internal consumption|tower|spinner|cooling air exit", None),
    (r"transform", "transformer"),
    (r"rotor ?bearing|main bearing|rotorbearing", "rotor bearing"),
    (r"gear|planetary|axial bearing", "gearbox"),
    (r"generator|slip ring|stator|winding", "generator"),
    (r"converter|inverter|igbt|chopper", "converter"),
    (r"hydraulic", "hydraulics"),
    (r"axis|pitch|blade", "pitch"),
    (r"water|cool", "cooling"),
)

#: Fault description -> component (first match wins).
FAULT_RULES = (
    (r"transformer", "transformer"),
    (r"rotor ?bearing|main bearing|rotorbearing", "rotor bearing"),
    (r"gear oil|gearbox", "gearbox"),
    (r"hydraulic|oil level|oil pump", "hydraulics"),
    (r"converter|umrichter", "converter"),
    (r"water cooling|cooling system", "cooling"),
    (r"generator", "generator"),
    (r"pitch|axis|blade|battery|beckhoff", "pitch"),
)


def component_of(desc: str, rules=COMPONENT_RULES):
    d = desc.lower()
    for pat, comp in rules:
        if re.search(pat, d):
            return comp
    return None


def components(farm_dir: Path, thermal: list[str]) -> dict[str, list[str]]:
    fd = pd.read_csv(farm_dir / "feature_description.csv", sep=care.SEP)
    desc = dict(zip(fd["sensor_name"].astype(str) + "_avg", fd["description"].astype(str)))
    out: dict[str, list[str]] = {}
    for c in thermal:
        comp = component_of(desc.get(c, ""))
        if comp:
            out.setdefault(comp, []).append(c)
    return out


def load(farm_dir: Path, eid: int):
    path = farm_dir / "datasets" / f"{eid}.csv"
    cols = care.dataset_columns(path)
    power, thermal, _wind, amb = care.feature_rows(farm_dir, cols)
    df = pd.read_csv(path, sep=care.SEP,
                     usecols=["time_stamp", "train_test", "status_type_id", power, amb] + thermal)
    frame = pd.DataFrame({"ts": pd.to_datetime(df["time_stamp"]).dt.tz_localize("UTC")})
    train = df["train_test"].eq("train")
    rated = float(np.nanquantile(df.loc[train, power], 0.995))
    frame["P_norm"] = (df[power] / rated).clip(lower=0).to_numpy()
    frame["era5_temp_2m"] = df[amb].to_numpy()          # the site's own sensor
    for c in thermal:
        frame[c] = df[c].to_numpy()
    frame["status"] = df["status_type_id"].to_numpy()
    pred_start = frame.loc[~train.to_numpy(), "ts"].min()
    return frame, thermal, pred_start


def recorded_service(frame: pd.DataFrame) -> list:
    """Maintenance as the turbine records it: a stop of MAINTENANCE_DAYS or
    more during which the turbine logged service (CARE status 3). Returns the
    first running reading after each such stop. A long stop with no service
    logged (a fault, a grid outage, no wind) is not maintenance; nor is a
    short service visit while the turbine keeps running."""
    f, standins = dc.mark_restarts(frame)
    svc_ts = f.loc[f["status"].eq(SERVICE_STATUS), "ts"]
    if svc_ts.empty:
        return []
    running = f.loc[f["running"], "ts"]
    out = []
    for restart in standins:
        before = running[running < restart]
        stop_start = before.max() if len(before) else f["ts"].min()
        if ((svc_ts > stop_start) & (svc_ts < restart)).any():
            out.append(restart)
    return out


def has_service_status(farm_dir: Path, eids) -> bool:
    """Whether the farm records service periods at all (Farm C does, B does not)."""
    for eid in list(eids)[:5]:
        s = pd.read_csv(farm_dir / "datasets" / f"{int(eid)}.csv", sep=care.SEP,
                        usecols=["status_type_id"])["status_type_id"]
        if s.eq(SERVICE_STATUS).any():
            return True
    return False


def sensor_diffs(sim: dict, sensors: list[str]) -> list[dict]:
    """Per day: checked or not, and each sensor's daily mean difference
    (actual minus healthy prediction)."""
    out = []
    for d in sim["days"]:
        ch = d.get("ch") or {}
        diffs = {}
        if d["status"] == "checked":
            for c in sensors:
                a, p = (ch.get(c) or (None, None))
                if a is not None and p is not None and np.isfinite(a) and np.isfinite(p):
                    diffs[c] = a - p
        out.append({"day": d["day"], "checked": bool(diffs), "diffs": diffs})
    return out


def judge(days: list[dict]) -> list[dict]:
    """Hot days for one component, each sensor against a tolerance learned
    from its own past.

    A sensor's tolerance on a day is the median plus TOL_SIGMA robust standard
    deviations (1.4826 x MAD) of its daily differences over its last CAL_DAYS
    healthy (not hot) checked days, never below 2 K. The component is hot when
    any of its sensors is. Sensors are judged one by one, not averaged: on
    Farm B a transformer-cell fault ran 12 K hot while the seven windings beside
    it stayed normal, and their average hid it. Until a sensor has
    MIN_CAL_DAYS of history it is not judged. Only days before the day judged
    are used, so nothing leaks.
    """
    hist: dict[str, list[float]] = {}
    out = []
    for d in days:
        hot, judged = False, False
        for c, x in d["diffs"].items():
            h = hist.setdefault(c, [])
            if len(h) >= MIN_CAL_DAYS:
                judged = True
                w = np.asarray(h[-CAL_DAYS:])
                med = float(np.median(w))
                mad = float(np.median(np.abs(w - med)))
                if x > max(TOLERANCE, med + TOL_SIGMA * 1.4826 * mad):
                    hot = True
                    continue
            h.append(x)
        out.append({"day": d["day"], "checked": judged, "hot": hot})
    return out


def frozen_days(frame: pd.DataFrame, sensors: list[str]) -> list[dict]:
    """The static baseline: one model, learned on the first month of healthy
    data and never refreshed; same 2 K daily test."""
    f, _ = dc.mark_restarts(frame)
    usable = f["running"] & ~f["warmup"]
    first = f[usable].head(dc.MONTH_OF_READINGS)
    if len(first) < dc.MONTH_OF_READINGS:
        return []
    end = first["ts"].max()
    refs = {c: multivariate.fit_channel_reference(first, c, "P_norm", ambient_col="era5_temp_2m")
            for c in sensors}
    refs = {c: r for c, r in refs.items() if r is not None}
    if not refs:
        return []
    rest = f[usable & (f["ts"] > end)]
    out = []
    for day, g in rest.groupby("day"):
        if len(g) < dc.MIN_DAY_ROWS:
            continue
        diffs = {}
        for c, r in refs.items():
            diff = g[c].to_numpy(dtype="float64") - dc.predict(r, g["P_norm"], g["era5_temp_2m"])
            if np.isfinite(diff).any():
                diffs[c] = float(np.nanmean(diff))
        out.append({"day": str(day)[:10], "checked": bool(diffs), "diffs": diffs})
    return judge(out)


def runs(days: list[dict], key="hot") -> list[dict]:
    out, prev = [], None
    for d in days:
        if d.get(key):
            day = pd.Timestamp(d["day"])
            if prev is not None and day - prev == pd.Timedelta(days=1):
                out[-1]["days"] += 1
            else:
                out.append({"first": d["day"], "days": 1})
            prev = day
        else:
            prev = None
    return out


def per_year(n: int, checked_days: int):
    return round(n / (checked_days / 365.25), 2) if checked_days else None


def one_dataset(farm: str, farm_dir: Path, row: pd.Series, mode: str = "standin") -> dict:
    eid = int(row["event_id"])
    frame, thermal, pred_start = load(farm_dir, eid)
    comps = components(farm_dir, thermal)
    is_fault = row["event_label"] == "anomaly"
    start = pd.Timestamp(row["event_start"], tz="UTC")
    end = pd.Timestamp(row["event_end"], tz="UTC")
    spans = {"T": [(start, end)]} if is_fault else {}
    fault_comp = component_of(str(row.get("event_description", "")), FAULT_RULES) if is_fault else None

    res = {"event_id": eid, "asset_id": int(row["asset_id"]), "label": row["event_label"],
           "description": None if pd.isna(row.get("event_description")) else str(row["event_description"]),
           "fault_component": fault_comp,
           "fault_component_has_sensors": bool(fault_comp in comps) if fault_comp else False,
           "window": [str(start)[:16], str(end)[:16]],
           "prediction_start": str(pred_start)[:16],
           "components": {k: len(v) for k, v in comps.items()},
           "per_component": {}}
    if mode == "recorded":
        sim_kw = {"maintenance_days": NEVER, "maintenance_extra": recorded_service(frame)}
    else:
        sim_kw = {}
    res["maintenance_rule"] = mode
    res["recorded_service_restarts"] = len(sim_kw.get("maintenance_extra", []))
    in_win = lambda day: start.floor("D") <= pd.Timestamp(day, tz="UTC") <= end
    for comp, sensors in comps.items():
        sim = dc.simulate(frame, "T", TOLERANCE, spans=spans, channel=sensors[0],
                          channels=sensors[1:], **sim_kw)
        days = judge(sensor_diffs(sim, sensors))
        for d in days:
            d["fault_period"] = in_win(d["day"])
        fz = frozen_days(frame, sensors)
        outside = [d for d in days if not in_win(d["day"])] if is_fault else days
        outside_fz = [d for d in fz if not in_win(d["day"])] if is_fault else fz
        co, co_fz = runs(outside), runs(outside_fz)
        n_chk = sum(d["checked"] for d in outside)
        n_chk_fz = sum(d["checked"] for d in outside_fz)
        win = [d for d in days if in_win(d["day"])]
        win_co = runs(win)
        res["per_component"][comp] = {
            "sensors": len(sensors),
            "checked_days_outside_fault": n_chk,
            "call_outs_outside_fault": len(co),
            "call_outs_per_year": per_year(len(co), n_chk),
            "frozen_checked_days": n_chk_fz,
            "frozen_call_outs": len(co_fz),
            "frozen_call_outs_per_year": per_year(len(co_fz), n_chk_fz),
            "fault_window_days": len(win),
            "fault_window_checked": sum(d["checked"] for d in win),
            "fault_window_call_outs": [c["first"] for c in win_co],
            "prediction_window_call_outs": [c["first"] for c in runs(
                [d for d in days if pd.Timestamp(d["day"], tz="UTC") >= pred_start.floor("D")])],
        }
    pc = res["per_component"]
    res["caught_any"] = bool(is_fault and any(v["fault_window_call_outs"] for v in pc.values()))
    res["caught_own"] = bool(is_fault and fault_comp in pc and pc[fault_comp]["fault_window_call_outs"])
    res["own_checked_days"] = pc[fault_comp]["fault_window_checked"] if fault_comp in pc else 0
    return res


def summarise(rows: list[dict]) -> dict:
    out = {}
    healthy = [r for r in rows if r["label"] == "normal"]
    faults = [r for r in rows if r["label"] == "anomaly"]
    def tot(rs, key):
        return sum(v[key] for r in rs for v in r["per_component"].values())
    for name, rs in (("all", rows), ("healthy", healthy)):
        ck, n = tot(rs, "checked_days_outside_fault"), tot(rs, "call_outs_outside_fault")
        ckf, nf = tot(rs, "frozen_checked_days"), tot(rs, "frozen_call_outs")
        comps = sum(len(r["per_component"]) for r in rs)
        out[name] = {
            "datasets": len(rs), "component_checks": comps,
            "monthly_call_outs": n, "monthly_checked_component_days": ck,
            "monthly_per_component_year": per_year(n, ck),
            "frozen_call_outs": nf, "frozen_checked_component_days": ckf,
            "frozen_per_component_year": per_year(nf, ckf),
        }
    out["healthy_prediction_window_with_call_out"] = sum(
        any(v["prediction_window_call_outs"] for v in r["per_component"].values()) for r in healthy)
    thermal = [r for r in faults if r["fault_component_has_sensors"]]
    out["faults"] = {
        "total": len(faults),
        "on_a_component_with_temperature_sensors": len(thermal),
        "caught_on_own_component": sum(r["caught_own"] for r in thermal),
        "own_component_window_never_checked": sum(r["own_checked_days"] == 0 for r in thermal),
        "caught_on_any_component": sum(r["caught_any"] for r in faults),
        "outside_temperature": [f'{r["event_id"]}: {r["description"]}' for r in faults
                                if not r["fault_component_has_sensors"]],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", choices=FARMS, action="append")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--rule", choices=MODES, help="run only this maintenance rule")
    ap.add_argument("--fresh", action="store_true", help="ignore the per-dataset cache")
    ap.add_argument("--budget", type=float, default=0,
                    help="stop starting new datasets after this many seconds (0 = no limit)")
    a = ap.parse_args()
    out_path = Path(a.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    root = care.care_root(Path(a.data_dir))
    for farm in a.farm or FARMS:
        t0 = time.time()
        farm_dir = root / f"Wind Farm {farm}"
        ev = care.event_info(farm_dir)
        cache = Path(a.data_dir) / "interim" / "care_bc"
        cache.mkdir(parents=True, exist_ok=True)
        modes = ["standin"] + (["recorded"] if has_service_status(farm_dir, ev["event_id"]) else [])
        if a.rule:
            modes = [m for m in modes if m == a.rule] or modes
        rows_by_mode, missing = {m: [] for m in modes}, 0
        for mode, (_, r) in [(m, x) for m in modes for x in ev.iterrows()]:
            rows = rows_by_mode[mode]
            f = cache / f"{farm}_{int(r['event_id'])}_{mode}_s.json"
            if f.exists() and not a.fresh:
                rows.append(json.loads(f.read_text()))
                continue
            if a.budget and time.time() - t0 > a.budget:
                missing += 1
                continue
            row = one_dataset(farm, farm_dir, r, mode)
            f.write_text(json.dumps(row, default=str))
            rows.append(row)
        if missing:
            print(f"{farm}: {missing} datasets left; run again to continue")
            continue
        main_mode = modes[-1]
        result[f"farm_{farm}"] = {
            "maintenance_rule": main_mode,
            "summary": summarise(rows_by_mode[main_mode]),
            "datasets": rows_by_mode[main_mode],
            "by_maintenance_rule": {m: summarise(rs) for m, rs in rows_by_mode.items()},
        }
        print(farm, json.dumps(result[f"farm_{farm}"]["by_maintenance_rule"], indent=1))
    result["protocol"] = {"tolerance_k": TOLERANCE, "ambient": "site sensor (stands in for ERA5)",
                          "generated": time.strftime("%Y-%m-%d")}
    out_path.write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
