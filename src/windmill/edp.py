"""CARE Wind Farm A put back on its true dates, and EDP's own records.

CARE to Compare's Wind Farm A is built from EDP's open data (an onshore farm in
Portugal, 2016-2017). EDP also publishes what CARE leaves out: the turbines'
event log (their own alarms) and a logbook of confirmed component failures.
That makes Farm A the one place in this project where an alarm can be checked
against a record of what actually broke.

What was established (docs/PROGRESS.md, step 10):

* CARE turbine -> EDP turbine, matched reading for reading: `CARE_A_TURBINES`.
* CARE moved each file by a whole number of days so that it starts on the same
  calendar date in 2022: 2,191 days, or 2,192 for a file that starts before
  1 March of its first year (its real start is before the leap day of
  29 February 2016). `shift_days` reproduces that.
* CARE's power is the turbine's power divided by `RATED_KW`.
* CARE's Farm A status column is rebuilt from the failure logbook (14-day
  "downtime" blocks ending at each failure), not recorded, so stops are found
  from power, never from status.

No location is used: EDP keeps the turbine positions restricted. The weather is
EDP's own met mast at the farm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: CARE asset id -> EDP turbine. T09 is in CARE and in EDP's logbook, but its
#: SCADA and event log are not in the published download.
CARE_A_TURBINES = {0: "T07", 10: "T09", 11: "T01", 13: "T11", 21: "T06"}
RATED_KW = 2050.0
#: The anonymised date before which a file's real start lies before 29 Feb 2016.
_LEAP_EDGE = pd.Timestamp("2022-03-01")

#: Families of the turbines' own alarms: the component, its temperature sensor
#: in CARE Farm A, and the event-log texts that are TEMPERATURE alarms on it.
#: Pump, fan, leak and pressure events are deliberately not here: a temperature
#: model has nothing to say about them.
FAMILIES = {
    "TRANSFORMER": ("sensor_38_avg", r"Hot HV trafo|High temperature T#"),
    "GENERATOR": ("sensor_15_avg",
                  r"Hot generator|High temperature Gen#|High temp\. Gen bearing"
                  r"|High Gen\.SlipRTemp"),
    "GEARBOX": ("sensor_11_avg", r"High gear temp"),
}
#: Logbook remarks where heat is the cause or the symptom of the failure.
THERMAL_FAILURE = (r"[Hh]igh tem|[Tt]emperature|bearings damaged"
                   r"|bearings replaced|[Rr]efrigeration|fan damaged")
#: ... except where the failure was the temperature SENSOR itself.
SENSOR_FAILURE = r"sensor failure|replaced sensor"
#: CARE Farm A columns used here.
CARE_COLUMNS = ["time_stamp", "asset_id", "status_type_id", "sensor_0_avg",
                "wind_speed_3_avg", "power_30_avg", "sensor_11_avg",
                "sensor_12_avg", "sensor_13_avg", "sensor_14_avg",
                "sensor_15_avg", "sensor_38_avg", "sensor_41_avg"]


def shift_days(file_start) -> int:
    """Days CARE added to a Farm A file, from the file's first (anonymised) stamp."""
    return 2192 if pd.Timestamp(file_start) < _LEAP_EDGE else 2191


def care_a_dir(data_dir: Path) -> Path:
    from .care import care_root
    return care_root(Path(data_dir)) / "Wind Farm A"


def edp_dir(data_dir: Path) -> Path:
    from .acquire import EDP, extract_dir
    root = Path(data_dir) / "raw" / extract_dir(EDP)
    hits = sorted(root.glob("*/Combined-Failure-Logbook-2016-2017.xlsx"))
    if not hits:
        raise FileNotFoundError(f"EDP data not found under {root}: run `make data`")
    return hits[0].parent


def to_true_dates(raw: pd.DataFrame) -> pd.DataFrame:
    """CARE Farm A rows (any number of event files) on their true dates, one
    row per turbine and timestamp. `raw` needs `event_id`, `asset_id` and
    `time_stamp`. Overlapping files are copies of one record, so the first
    copy is kept."""
    df = raw.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    start = df.groupby("event_id")["time_stamp"].transform("min")
    days = start.map(shift_days)
    df["ts"] = df["time_stamp"] - pd.to_timedelta(days, unit="D")
    df["turbine"] = df["asset_id"].map(CARE_A_TURBINES)
    df = (df.sort_values("event_id").drop_duplicates(["asset_id", "ts"])
          .sort_values(["turbine", "ts"]).reset_index(drop=True))
    return df


def load_care_a(data_dir: Path, columns=CARE_COLUMNS) -> pd.DataFrame:
    parts = []
    for f in sorted((care_a_dir(data_dir) / "datasets").glob("*.csv")):
        d = pd.read_csv(f, sep=";", usecols=columns)
        d["event_id"] = int(f.stem)
        parts.append(d)
    return to_true_dates(pd.concat(parts, ignore_index=True))


def care_a_events(data_dir: Path) -> pd.DataFrame:
    """CARE's Farm A events on their true dates."""
    ev = pd.read_csv(care_a_dir(data_dir) / "event_info.csv", sep=";")
    starts = {}
    for f in (care_a_dir(data_dir) / "datasets").glob("*.csv"):
        starts[int(f.stem)] = pd.read_csv(f, sep=";", usecols=["time_stamp"],
                                          nrows=1)["time_stamp"].iloc[0]
    days = ev["event_id"].map(lambda e: shift_days(starts[e]))
    for c in ("event_start", "event_end"):
        ev[c] = pd.to_datetime(ev[c]) - pd.to_timedelta(days, unit="D")
    ev["turbine"] = ev["asset"].map(CARE_A_TURBINES)
    return ev


def _cached(path: Path, cache: Path | None, read):
    if cache is not None:
        c = Path(cache) / (path.stem.replace(" ", "_") + ".pkl")
        if c.exists() and c.stat().st_mtime >= path.stat().st_mtime:
            return pd.read_pickle(c)
        df = read(path)
        c.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(c)
        return df
    return read(path)


def load_logbook(data_dir: Path, cache: Path | None = None) -> pd.DataFrame:
    """EDP's confirmed failures: turbine, component, time, remark, and whether
    heat was the cause or the symptom."""
    p = edp_dir(data_dir) / "Combined-Failure-Logbook-2016-2017.xlsx"
    lb = _cached(p, cache, lambda q: pd.read_excel(q, engine="openpyxl"))
    lb = lb.rename(columns={"Turbine_ID": "turbine", "Component": "component",
                            "Remarks": "remark"})
    lb["ts"] = pd.to_datetime(lb["Timestamp"])
    lb["family"] = lb["component"].replace({"GENERATOR_BEARING": "GENERATOR"})
    lb["thermal"] = (lb["remark"].str.contains(THERMAL_FAILURE, regex=True)
                     & ~lb["remark"].str.contains(SENSOR_FAILURE, regex=True))
    return lb[["turbine", "component", "family", "ts", "remark", "thermal"]]


def template(remark: pd.Series) -> pd.Series:
    """An event-log text with its numbers replaced by #, to group alike entries."""
    return (remark.astype(str).str.replace("Â°", "°", regex=False)
            .str.replace(r"-?\d+(\.\d+)?", "#", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def load_event_log(data_dir: Path, cache: Path | None = None) -> pd.DataFrame:
    d = edp_dir(data_dir)
    parts = [_cached(d / n, cache, lambda q: pd.read_excel(q, engine="openpyxl"))
             for n in ("Wind-Turbines-Logs-2016.xlsx", "Wind Turbines Logs 2017.xlsx")]
    log = pd.concat(parts, ignore_index=True)
    log = log[log["UnitTitle"].notna() & log["TimeDetected"].notna()]
    log["ts"] = pd.to_datetime(log["TimeDetected"], utc=True).dt.tz_convert(None)
    log["turbine"] = log["UnitTitle"]
    log["remark"] = log["Remark"].astype(str).str.replace("Â°", "°", regex=False)
    log["template"] = template(log["Remark"])
    return log[["turbine", "ts", "remark", "template"]].reset_index(drop=True)


def load_met_mast(data_dir: Path, cache: Path | None = None) -> pd.Series:
    """The farm's met mast: ambient temperature, 10-minute, true dates."""
    p = edp_dir(data_dir) / "Combined-MetMast-SCADA-2016-2017.xlsx"
    m = _cached(p, cache, lambda q: pd.read_excel(q, engine="openpyxl"))
    ts = pd.to_datetime(m["Timestamp"], utc=True).dt.tz_convert(None)
    return pd.Series(m["Avg_AmbientTemp"].to_numpy(dtype="float64"), index=ts,
                     name="met_ambient").sort_index()


def load_edp_scada(data_dir: Path, turbine: str, cache: Path | None = None) -> pd.DataFrame:
    p = edp_dir(data_dir) / f"{turbine}_F99.xlsx"
    d = _cached(p, cache, lambda q: pd.read_excel(q, engine="openpyxl"))
    d = d.copy()
    d["ts"] = pd.to_datetime(d["Timestamp"], utc=True).dt.tz_convert(None)
    return d


def alarm_spells(log: pd.DataFrame, families=FAMILIES) -> pd.DataFrame:
    """The turbines' temperature alarms as spells: consecutive calendar days
    with at least one alarm of the family on the turbine make one spell."""
    rows = []
    for fam, (_, pattern) in families.items():
        hits = log[log["template"].str.contains(pattern, regex=True)]
        for t, g in hits.groupby("turbine"):
            days = sorted(g["ts"].dt.floor("D").unique())
            spells: list[list] = []
            for d in days:
                d = pd.Timestamp(d)
                if spells and (d - spells[-1][1]).days <= 1:
                    spells[-1][1] = d
                else:
                    spells.append([d, d])
            for s, e in spells:
                n = int(((g["ts"] >= s) & (g["ts"] < e + pd.Timedelta(days=1))).sum())
                rows.append({"family": fam, "turbine": t, "start": s, "end": e,
                             "days": (e - s).days + 1, "entries": n})
    return pd.DataFrame(rows, columns=["family", "turbine", "start", "end", "days", "entries"])


def label_spells(spells: pd.DataFrame, logbook: pd.DataFrame,
                 window_days: int = 30) -> pd.DataFrame:
    """Was the spell followed by a logged failure of the same component within
    `window_days`? `followed_thermal` counts only failures where heat was the
    cause or the symptom. A spell with no failure is "false" only in the sense
    that nothing was repaired: it is not a judgement that nothing was wrong."""
    out = spells.copy()

    def follows(r, thermal):
        f = logbook[(logbook["turbine"] == r["turbine"]) & (logbook["family"] == r["family"])]
        if thermal:
            f = f[f["thermal"]]
        lead = (f["ts"] - r["start"]).dt.total_seconds() / 86400.0
        return bool(((lead >= -1) & (lead <= window_days)).any())

    out["followed_any"] = out.apply(lambda r: follows(r, False), axis=1)
    out["followed_thermal"] = out.apply(lambda r: follows(r, True), axis=1)
    return out
