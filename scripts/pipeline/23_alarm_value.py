#!/usr/bin/env python
"""Why a trusted temperature alarm is worth acting on (EDP = CARE Farm A).

Writes `results/alarm_value.json`.

    python scripts/pipeline/23_alarm_value.py   (after `make care-a`)

Two measurements on EDP's farm, both read back by the README:

1. The weather term. The daily check is run on each logged turbine twice:
   with the met mast's outside temperature as the weather input, and with
   a constant in its place (load only). The daily error of the healthy
   prediction on healthy days is compared, per component.
2. Lead time and the cost of waiting. For every confirmed failure in EDP's
   logbook: did this project's triage (results/care_farm_a.json) confirm
   real excess heat on the same component in the 60 days before it, how many
   days before, and how many hours the turbine then stood still while the
   wind blew (>= 5 m/s), from 2 days before to 14 days after the failure,
   against its normal rate.

What it does not show: that acting on the alarm would have prevented the
failure. The lead time is the window in which a repair could have been
planned; the standstill is what the failure cost when it came.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import daily_check as dc  # noqa: E402
from windmill import edp  # noqa: E402

OUT = ROOT / "results" / "alarm_value.json"
LOGGED = ("T01", "T06", "T07", "T11")
WINDY_MS = 5.0
LOOKBACK_DAYS = 60
BEFORE_DAYS, AFTER_DAYS = 2, 14
CONSTANT_AMBIENT = 10.0


def weather_term(rec: pd.DataFrame, met: pd.Series) -> dict:
    channels = {fam: ch for fam, (ch, _) in edp.FAMILIES.items()}
    out = {}
    for t in LOGGED:
        g = rec[rec["turbine"] == t]
        amb = met.reindex(g["ts"]).to_numpy()
        amb = pd.Series(amb).fillna(pd.Series(g["sensor_0_avg"].to_numpy() - 1.0)).to_numpy()
        res = {}
        for label, a in (("with_weather", amb), ("load_only", np.full(len(g), CONSTANT_AMBIENT))):
            f = pd.DataFrame({"ts": g["ts"].dt.tz_localize("UTC").to_numpy(),
                              "P_norm": g["power_30_avg"].clip(lower=0).to_numpy(),
                              "era5_temp_2m": a})
            for ch in channels.values():
                f[ch] = g[ch].to_numpy()
            chs = list(channels.values())
            sim = dc.simulate(f, t, 99.0, spans={}, channel=chs[0], channels=chs[1:])
            for fam, ch in channels.items():
                v = [d["ch"][ch][0] - d["ch"][ch][1] for d in sim["days"]
                     if d.get("status") == "checked" and ch in (d.get("ch") or {})
                     and d["ch"][ch][0] is not None and d["ch"][ch][1] is not None]
                res.setdefault(fam, {})[label] = {
                    "days": len(v), "daily_mae_k": round(float(np.mean(np.abs(v))), 2) if v else None}
        out[t] = res
    summary = {}
    for fam in channels:
        w = [out[t][fam]["with_weather"]["daily_mae_k"] for t in LOGGED if out[t][fam]["with_weather"]["daily_mae_k"]]
        n = [out[t][fam]["load_only"]["daily_mae_k"] for t in LOGGED if out[t][fam]["load_only"]["daily_mae_k"]]
        if w and n:
            summary[fam] = {"with_weather_k": round(float(np.median(w)), 2),
                            "load_only_k": round(float(np.median(n)), 2),
                            "error_increase_without_weather": round(float(np.median(n) / np.median(w) - 1), 3)}
    return {"per_turbine": out, "median_over_turbines": summary}


def lead_and_standstill(rec: pd.DataFrame, logbook: pd.DataFrame, alarms: pd.DataFrame) -> list[dict]:
    rec = rec.copy()
    rec["day"] = rec["ts"].dt.floor("D")
    rec["stopped"] = (rec["wind_speed_3_avg"] >= WINDY_MS) & (rec["power_30_avg"] < dc.RUNNING)
    per_day = rec.groupby(["turbine", "day"])["stopped"].sum() / 6.0
    window = BEFORE_DAYS + AFTER_DAYS
    rows = []
    for r in logbook.itertuples():
        if r.turbine not in LOGGED:
            continue
        s = per_day.loc[r.turbine]
        day = pd.Timestamp(r.ts).floor("D")
        win = s[(s.index >= day - pd.Timedelta(days=BEFORE_DAYS)) & (s.index < day + pd.Timedelta(days=AFTER_DAYS))]
        prior = alarms[(alarms["turbine"] == r.turbine) & (alarms["family"] == r.family)
                       & (alarms["start"] <= day) & (alarms["start"] >= day - pd.Timedelta(days=LOOKBACK_DAYS))]
        real = prior[prior["verdict"] == "real excess heat"]
        rows.append({
            "turbine": r.turbine, "date": str(day)[:10], "family": r.family,
            "remark": r.remark, "heat_related": bool(r.thermal),
            "windy_standstill_h": round(float(win.sum()), 1), "days_with_data": int(len(win)),
            "normal_h_same_length": round(float(s.mean() * window), 1),
            "confirmed_real_heat_lead_days": int((day - real["start"].min()).days) if len(real) else None,
        })
    return rows


def main() -> int:
    t0 = time.time()
    data_dir = ROOT / "data"
    cache = data_dir / "interim" / "edp"
    rec_cache = cache / "care_farm_a_true_dates.pkl"
    if not rec_cache.exists():
        print("run `make care-a` first: it builds", rec_cache)
        return 1
    rec, _ = pd.read_pickle(rec_cache)
    rec = rec.drop_duplicates(subset=["turbine", "ts"]).sort_values("ts")
    met = edp.load_met_mast(data_dir, cache)
    logbook = edp.load_logbook(data_dir, cache)
    tri = json.loads((ROOT / "results" / "care_farm_a.json").read_text())["triage"]["alarms"]
    alarms = pd.DataFrame(tri)
    alarms["start"] = pd.to_datetime(alarms["start"])

    wx = weather_term(rec, met)
    print("weather term:", json.dumps(wx["median_over_turbines"]), f"({time.time() - t0:.0f}s)")
    failures = lead_and_standstill(rec, logbook, alarms)
    warned = [f for f in failures if f["confirmed_real_heat_lead_days"] is not None]
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "protocol": {"windy_ms": WINDY_MS, "lookback_days": LOOKBACK_DAYS,
                     "standstill_window_days": [-BEFORE_DAYS, AFTER_DAYS],
                     "load_only_constant_ambient_c": CONSTANT_AMBIENT},
        "weather_term": wx,
        "failures": failures,
        "warned_failures": len(warned),
        "logged_failures": len(failures),
    }
    OUT.write_text(json.dumps(out, indent=1))
    for f in sorted(failures, key=lambda x: -x["windy_standstill_h"])[:8]:
        print(f)
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
