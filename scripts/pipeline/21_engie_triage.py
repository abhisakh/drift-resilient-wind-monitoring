#!/usr/bin/env python
"""ENGIE La Haute Borne: the same alarm triage as CARE Farm A.

Writes `results/engie_triage.json`.

    python scripts/pipeline/21_engie_triage.py

ENGIE publishes no alarm log, so the alarms triaged here are the fixed rule
this project uses as its baseline: the gearbox bearing (Gb2t) at 80 C or more
for an hour. Consecutive days with such an alarm make one spell, as on Farm A.
Each spell gets the accounting of src/windmill/triage.py: at the day's hottest
readings, reading = typical + load + weather + excess, by the bearing's healthy
reference (the daily check's frozen monthly model, ERA5 weather) and by its
siblings at the same moment; the verdict follows the excess.

"In the fault period": the spell lies in the documented R80711 anomaly span,
the only fault record there is. A spell outside it is not judged harmless.
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
from windmill import live, triage  # noqa: E402
from windmill.confirmation import ANOMALY_SPANS  # noqa: E402
from windmill.operating_curve import alarm_runs  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "engie_triage.json"
TURBINES = ("R80711", "R80721", "R80736", "R80790")
CHANNEL = dc.CHANNEL


def limit_spells(frame: pd.DataFrame, turbine: str) -> pd.DataFrame:
    """Days with the fixed limit held for an hour, consecutive days merged."""
    f = frame.sort_values("ts").reset_index(drop=True)
    times = f["ts"].dt.tz_convert(None).to_numpy()
    starts, lengths = alarm_runs(f[CHANNEL].to_numpy(dtype="float64") > dc.HOT_C, times)
    days = sorted({f["ts"].iloc[a + dc.LIMIT_HOLD - 1].floor("D").tz_convert(None)
                   for a, n in zip(starts, lengths) if n >= dc.LIMIT_HOLD})
    spells: list[list] = []
    for d in days:
        if spells and (d - spells[-1][1]).days <= 1:
            spells[-1][1] = d
        else:
            spells.append([d, d])
    rows = [{"turbine": turbine, "start": s, "end": e, "days": (e - s).days + 1,
             "in_fault_period": dc.in_fault(turbine, s)} for s, e in spells]
    return pd.DataFrame(rows, columns=["turbine", "start", "end", "days", "in_fault_period"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    data_dir = Path(args.data)
    t0 = time.time()

    frames = {t: dc.with_era5(live._monthly_frame(data_dir, "engie", t), data_dir) for t in TURBINES}
    temps = pd.DataFrame({t: f.set_index("ts")[CHANNEL] for t, f in frames.items()})
    loads = pd.DataFrame({t: f.set_index("ts")["P_norm"] for t, f in frames.items()})
    temps.index = temps.index.tz_convert(None)
    loads.index = loads.index.tz_convert(None)
    print(f"  frames loaded ({time.time() - t0:.0f}s)")

    rows = []
    for t in TURBINES:
        f = frames[t][["ts", "P_norm", "era5_temp_2m", CHANNEL]].copy()
        spells = limit_spells(f, t)
        if not len(spells):
            print(f"  {t}: no alarms")
            continue
        comp = triage.Component(f, t, CHANNEL, sibling_gap=triage.sibling_gap(temps, loads, t))
        for r in spells.itertuples():
            rows.append({"turbine": t, "start": str(r.start)[:10], "days": int(r.days),
                         "followed_thermal": bool(r.in_fault_period),
                         **comp.judge_spell(r.start, r.end)})
        print(f"  {t}: {len(spells)} alarms judged ({time.time() - t0:.0f}s)")
    tri = pd.DataFrame(rows)

    out = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "seconds": round(time.time() - t0, 1),
        "protocol": {
            "alarm": f"gearbox bearing ({CHANNEL}) at {dc.HOT_C:g} C or more for an hour "
                     "(the project's fixed-limit baseline; ENGIE publishes no alarm log)",
            "spell": "consecutive calendar days with such an alarm",
            "weather": "ERA5 via Open-Meteo at the site",
            "fault_record": {k: [list(s) for s in v] for k, v in ANOMALY_SPANS.items()},
            "triage": {"split": "reading = typical + load + weather + excess, at the day's hottest "
                                f"{int(triage.PEAK_SHARE * 100)} % of readings",
                       "real_excess_heat_k": triage.REAL_HEAT_K,
                       "references": "the daily check's frozen monthly model, and the siblings",
                       "verdict": "the worst day of the spell decides"},
        },
        "triage": {
            "table": triage.summarise(tri),
            "by_turbine": {t: triage.summarise(x) for t, x in tri.groupby("turbine")},
            "mean_split_k": {v: {k: round(float(x[k].mean()), 2) for k in
                                 ("reading", "typical", "load", "weather", "excess")}
                             for v, x in tri.groupby("verdict") if x["excess"].notna().any()},
            "alarms": tri.replace({np.nan: None}).to_dict(orient="records"),
        },
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    print("  verdicts: " + ", ".join(f"{k} {v['alarms']}" for k, v in out["triage"]["table"].items()))
    print(f"wrote {p} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
