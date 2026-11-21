"""Channel selection for the CARE to Compare dataset.

The dataset anonymises sensor names but keeps the physically meaningful ones
(`power_NN`, `wind_speed_NN`) and describes every sensor in
`feature_description.csv`. The thermal channels are identified by their UNIT,
not their name -- which requires care: Wind Farm C writes "Celsius", while
Wind Farms A and B write a degree-Celsius sign that the CSV stores as a
non-UTF-8 byte, so the literal string is a mojibake. Matching on "ends with C"
catches all three farms and matches no other unit in the dataset (the others are
kWh, kvarh, VArh, kNm, A, V, bar, deg, %, rpm, m/s, rad/s, mG).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SEP = ";"
NORMAL_STATUS_IDS = (0, 2)

#: Fraction of rated power per load bin.
LOAD_BIN = 0.05


def _read_feature_descriptions(farm_dir: Path) -> pd.DataFrame:
    fd = pd.read_csv(farm_dir / "feature_description.csv", sep=SEP)
    fd["sensor_name"] = fd["sensor_name"].astype(str)
    fd["unit"] = fd["unit"].astype(str)
    fd["description"] = fd["description"].astype(str)
    fd["statistics_type"] = fd["statistics_type"].astype(str)
    return fd


def _has_average(fd: pd.DataFrame) -> pd.Series:
    return fd["statistics_type"].str.contains("average", case=False, na=False)


def is_temperature_unit(unit: str) -> bool:
    """True for a temperature unit, whatever spelling the farm used."""
    u = unit.strip()
    if not u:
        return False
    return u == "Celsius" or u.endswith("C")


def feature_rows(farm_dir: Path, columns: list[str]) -> tuple[str | None, list[str], list[str], str | None]:
    """(active power column, thermal columns, wind-speed columns, ambient column).

    `columns` is the real CSV header; feature_description.csv lists base sensor
    names, so each base name is mapped onto its `_avg` column.

    The ambient column is the site's own outside-temperature sensor. It is used to
    remove the ambient component of a thermal reference -- the CARE-side
    counterpart of the ERA5 conditioning used on the ENGIE arm. It is excluded
    from the thermal channel list itself: ambient temperature is a driver, not a
    component whose overheating indicates a fault.
    """
    fd = _read_feature_descriptions(farm_dir)
    has_avg = _has_average(fd)

    # --- power ------------------------------------------------------------
    power = None
    pw = fd[fd["sensor_name"].str.startswith("power_")]
    active = pw[pw["description"].str.contains("active power", case=False, na=False)]
    for base in list(active["sensor_name"]) + list(pw["sensor_name"]):
        cand = f"{base}_avg"
        if cand in columns:
            power = cand
            break

    # --- thermal ----------------------------------------------------------
    thermal = []
    temp_rows = fd[fd["unit"].map(is_temperature_unit) & has_avg]
    for base in temp_rows["sensor_name"]:
        cand = f"{base}_avg"
        if cand in columns and cand not in thermal:
            thermal.append(cand)

    # --- ambient ----------------------------------------------------------
    ambient = None
    amb_rows = fd[fd["description"].str.contains(
        r"^ambient temperature|^outside temperature|^outdoor temperature",
        case=False, na=False, regex=True) & has_avg]
    for base in amb_rows["sensor_name"]:
        cand = f"{base}_avg"
        if cand in columns:
            ambient = cand
            break
    if ambient is None:
        amb_rows = fd[fd["description"].str.contains("ambient", case=False, na=False) & has_avg]
        for base in amb_rows["sensor_name"]:
            cand = f"{base}_avg"
            if cand in columns:
                ambient = cand
                break

    # --- wind speed --------------------------------------------------------
    wind = []
    ws_rows = fd[fd["sensor_name"].str.contains("wind_speed") & has_avg]
    for base in ws_rows["sensor_name"]:
        cand = f"{base}_avg"
        if cand in columns and cand not in wind:
            wind.append(cand)

    if ambient is not None and ambient in thermal:
        thermal = [c for c in thermal if c != ambient]

    return power, thermal, wind, ambient


def event_info(farm_dir: Path) -> pd.DataFrame:
    return pd.read_csv(farm_dir / "event_info.csv", sep=SEP)


def dataset_columns(path: Path) -> list[str]:
    return pd.read_csv(path, sep=SEP, nrows=0).columns.tolist()
