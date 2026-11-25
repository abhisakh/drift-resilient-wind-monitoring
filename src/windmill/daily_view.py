"""What the dashboard's daily pages show, prepared from results/daily_series.json.

Pure functions, no Streamlit: the pages only draw what these return, so the
rules (which day is green, what a turbine's badge says) are tested here.

Every day of every turbine carries the model's state:

* checked  -- the frozen month's model predicted the sensor and the day had
  enough running readings outside the warm-up to compare;
* thin     -- the model was there, but the turbine ran too little that day;
* blank    -- no model: the start of the record, or the month of healthy data
  that has to be collected after maintenance (the maintenance hold).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

#: The traffic light, in the order of its legend.
LIGHTS = ("within", "hot", "cold", "thin", "hold", "none")
LIGHT_LABELS = {
    "within": "within the tolerance",
    "hot": "hotter than predicted",
    "cold": "colder than predicted",
    "thin": "ran too little to check",
    "hold": "maintenance hold: collecting healthy data",
    "none": "no model yet",
}
#: A square says what the SENSOR did that day; only the circles under a row
#: say whether an alarm was real, so no square is drawn in the alarm's red.
LIGHT_COLOURS = {
    "within": "#2EAD6B", "hot": "#F28C38", "cold": "#5B8DEF",
    "thin": "#6B7280", "hold": "#8B7CF6", "none": "#374151",
}
ALARM_COLOURS = {"real": "#E05263", "false": "#E5E7EB"}

DEMO = {"healthy": "R80721", "faulty": "R80711",
        "start": "2017-08-15", "end": "2017-09-14", "lead": 7}


def load(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        return None


def frame(series: dict, turbine: str, sensor: str,
          start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """One turbine and sensor, one row per day, between `start` and `end`
    inclusive. Forecast columns are `f1` ... `f7` where the series has them."""
    s = series["turbines"][turbine]
    col = s["sensors"].get(sensor, {})
    df = pd.DataFrame({k: s[k] for k in (
        "day", "status", "reason", "state", "flag", "maintenance", "collected",
        "fault_period", "gearbox_hot", "sibling", "sibling_overall")})
    for k, v in col.items():
        df[k] = v
    df["day"] = pd.to_datetime(df["day"])
    if start:
        df = df[df["day"] >= pd.Timestamp(start)]
    if end:
        df = df[df["day"] <= pd.Timestamp(end)]
    df = df.reset_index(drop=True)
    if "actual" in df and "pred" in df:
        df["diff"] = df["actual"] - df["pred"]
    # On the primary sensor at the run's own tolerance, the engine's own
    # verdict is used, so the page counts exactly what the report counts
    # (the rounded daily means could tip a day sitting on the tolerance).
    if sensor == series.get("primary") and series.get("tolerance_k") is not None:
        df["engine_flag"] = df["flag"]
        df["engine_tol"] = float(series["tolerance_k"])
    return df


def light(row, tolerance: float, engine: bool = False) -> str:
    """The traffic light for one day of one sensor."""
    status = row["status"]
    if status == "blank":
        return "hold" if row["reason"] == "maintenance" else "none"
    if status != "checked" or pd.isna(row.get("diff")):
        return "thin"
    if engine:
        f = row.get("engine_flag")
        return f if f in ("hot", "cold") else "within"
    if row["diff"] > tolerance:
        return "hot"
    if row["diff"] < -tolerance:
        return "cold"
    return "within"


def lights(df: pd.DataFrame, tolerance: float) -> pd.Series:
    if not len(df):
        return pd.Series([], dtype="object")
    engine = bool("engine_tol" in df and (df["engine_tol"] == tolerance).all())
    return df.apply(lambda r: light(r, tolerance, engine), axis=1)


def badge(series: dict, turbine: str, day: str, month_of_readings: int) -> dict:
    """The turbine's state on `day`, for its card: what the model is doing
    and, in a maintenance hold, how much healthy data has been collected."""
    s = series["turbines"][turbine]
    days = s["day"]
    i = max((k for k, d in enumerate(days) if d <= day), default=None)
    if i is None:
        return {"kind": "none", "text": "no data yet"}
    status, reason, state = s["status"][i], s["reason"][i], s["state"][i]
    if status == "blank" and reason == "maintenance":
        got = s["collected"][i] or 0
        share = min(got / month_of_readings, 1.0)
        return {"kind": "hold", "share": share,
                "since": s["maintenance"][i],
                "text": (f"Maintenance hold since {s['maintenance'][i]}: "
                         f"{share:.0%} of a month of healthy data collected. "
                         "No model, so only the fixed limit protects it.")}
    if status == "blank":
        return {"kind": "none",
                "text": "No model yet: a month of healthy data is being collected."}
    if state == "frozen":
        since = next((s["day"][k] for k in range(i, -1, -1)
                      if s["state"][k] != "frozen"), days[0])
        return {"kind": "frozen", "since": since,
                "text": ("Fault confirmed: the model is frozen on its last "
                         "healthy month until maintenance.")}
    return {"kind": "monitoring",
            "text": "Monitoring: this month's model was trained on a healthy month."}


def demo_summary(df: pd.DataFrame, tolerance: float, lead: int) -> dict:
    """The demo card for one turbine over the demo window."""
    checked = df[df["status"] == "checked"].dropna(subset=["diff"])
    hot = checked[checked["diff"] > tolerance]
    fc = f"f{lead}"
    both = checked.dropna(subset=[fc]) if fc in checked else checked.iloc[0:0]
    return {
        "days_checked": int(len(checked)),
        "days_hot": int(len(hot)),
        "first_hot": hot["day"].iloc[0].strftime("%Y-%m-%d") if len(hot) else None,
        "mae_on_day": (round(float(checked["diff"].abs().mean()), 2)
                       if len(checked) else None),
        "mae_forecast": (round(float((both[fc] - both["actual"]).abs().mean()), 2)
                         if len(both) else None),
        "max_diff": round(float(checked["diff"].max()), 2) if len(checked) else None,
    }


def alarms(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """The call-outs: the first day of each run of hot days, as dispatched.

    A run is broken by any day that is not hot, checked or not, the same rule
    as `daily_check.call_outs`. `real` is True when the day lies in the
    documented fault period (the stand-in for the inspection), else the
    call-out was a false alarm. Compute it on the whole series and then cut
    the window, so a run that started before the window is not counted again.
    """
    hot = (lights(df, tolerance) == "hot").to_numpy()
    first = hot & ~pd.Series(hot).shift(1, fill_value=False).to_numpy()
    run_days = []
    n = 0
    for h in hot[::-1]:
        n = n + 1 if h else 0
        run_days.append(n)
    out = df.loc[first, ["day", "fault_period"]].copy()
    out["days"] = [run_days[::-1][i] for i in range(len(df)) if first[i]]
    out["real"] = out["fault_period"].astype(bool)
    return out.drop(columns="fault_period").reset_index(drop=True)


def alarm_counts(df: pd.DataFrame, tolerance: float) -> dict:
    a = alarms(df, tolerance)
    return {"alarms": int(len(a)), "real": int(a["real"].sum()),
            "false": int((~a["real"]).sum())}
