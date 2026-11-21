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

#: Name of the CARE directory inside the extracted archive.
CARE_INNER_DIR = "CARE_To_Compare"


def care_root(data_dir: Path) -> Path:
    """The root of the extracted CARE dataset.

    Single source of truth for where CARE lives on disk. `scripts/01_build_dataset.py`
    extracts to this directory (via `acquire.extract_dir`), and every script that
    reads CARE asks here, so the download path and the read path cannot drift
    apart.
    """
    from windmill import acquire

    src = next(s for s in acquire.SOURCES if s.name == "care_to_compare")
    return data_dir / "raw" / acquire.extract_dir(src) / CARE_INNER_DIR

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


def audit_subset(data_dir: Path, per_farm: int | None = None) -> list:
    """A label-stratified subset of the CARE datasets.

    Shared so that every arm measured on "the same subset" is measured on
    literally the same datasets -- the AE audit and the comparison against
    this project's detector must not drift apart.

    The subset is stratified BY LABEL because the CARE score's Accuracy
    sub-score is averaged over the NORMAL datasets: an all-anomaly subset
    leaves it undefined and the final score a NaN.

    Returns (path, farm_dir, event_info_row) triples.
    """
    out = []
    for farm in ("Wind Farm A", "Wind Farm B", "Wind Farm C"):
        farm_dir = care_root(data_dir) / farm
        if not farm_dir.exists():
            continue
        info = event_info(farm_dir)
        by_label: dict[str, list] = {"anomaly": [], "normal": []}
        for _, row in info.iterrows():
            eid = int(row["event_id"])
            path = farm_dir / "datasets" / f"{eid}.csv"
            if not path.exists():
                continue
            label = str(row.get("event_label", "")).strip().lower()
            if label in by_label:
                by_label[label].append((path, farm_dir, row))
        if per_farm is None:
            out.extend(by_label["anomaly"] + by_label["normal"])
            continue
        out.extend(by_label["anomaly"][: (per_farm + 1) // 2])
        out.extend(by_label["normal"][: per_farm // 2])
    return out
