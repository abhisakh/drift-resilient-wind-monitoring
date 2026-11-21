"""Loaders for the pinned datasets, with the measured data quirks handled explicitly.

Every quirk below was measured on the actual files (see data/README.md), not assumed:

  * ENGIE timestamps carry mixed +01:00/+02:00 offsets (CET/CEST), so parsing must be
    offset-aware and then normalised to UTC -- otherwise DST hours double-count.
  * ENGIE has ~30 duplicate timestamps per turbine and 19.6-31.8 % missing intervals.
  * ENGIE's release keeps only 16 of the 34 declared variables. Ambient temperature
    is NOT among them, so temperature must come from ERA5.
  * ERA5 is hourly; SCADA is 10-min. The join is hourly-mean -> 10-min, and the
    comparison must be done at hourly resolution to avoid fake precision.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import TURBINES

# The 16 columns the ENGIE release actually ships.
ENGIE_COLUMNS = [
    "Date_time",
    "Wind_turbine_name",
    "Q_avg",
    "S_avg",
    "P_avg",
    "Git_avg",
    "Ws_avg",
    "Ds_avg",
    "DCs_avg",
    "Gost_avg",
    "Gb2t_avg",
    "Db2t_avg",
    "Gb1t_avg",
    "Db1t_avg",
    "Dst_avg",
    "Yt_avg",
    "Rbt_avg",
    "Rt_avg",
]

F99 = "F99"
F95 = "F95"


def _find_engie_root(data_dir: Path) -> Path:
    """Locate the directory holding 'Cleaned data' inside the extracted archive."""
    for p in data_dir.rglob("Cleaned data"):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError(f"no 'Cleaned data' directory under {data_dir}")


def load_engie_static(data_dir: Path) -> pd.DataFrame:
    """Turbine metadata incl. GPS (the authoritative coordinates)."""
    root = _find_engie_root(data_dir)
    df = pd.read_csv(root / "static-information.csv", encoding="utf-8-sig", sep=";")
    df.columns = [c.strip().rstrip(",") for c in df.columns]
    lat, lon = zip(
        *(tuple(float(v) for v in s.split(",")) for s in df["GPS"])
    )
    df["lat"], df["lon"] = lat, lon
    df["hub_height_m"] = df["Hub height (m)"]
    df["rated_kw"] = df["Rated power (kW)"]
    return df[["Wind_turbine_name", "lat", "lon", "hub_height_m", "rated_kw", "Department"]]


def load_engie(
    data_dir: Path,
    variant: str = F99,
    turbines: tuple[str, ...] = TURBINES,
) -> pd.DataFrame:
    """Load ENGIE SCADA as one tidy frame.

    Returns columns: ts (UTC, tz-naive), turbine, P_avg, Ws_avg, Gb2t_avg, Gost_avg,
    Rbt_avg, Git_avg, Ds_avg, DCs_avg, plus year/month.
    """
    cleaned = _find_engie_root(data_dir) / "Cleaned data"
    frames = []
    for t in turbines:
        path = cleaned / f"Turbine_data_{t}_2013-2018_{variant}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path, encoding="utf-8-sig", usecols=ENGIE_COLUMNS)
        d["turbine"] = t
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no ENGIE files for {turbines} under {cleaned}")

    df = pd.concat(frames, ignore_index=True)
    # Offset-aware parse: the raw strings carry +01:00/+02:00, so DST is handled.
    df["ts"] = pd.to_datetime(df["Date_time"], format="mixed", utc=True)
    # Drop the duplicate timestamps (measured: ~30 per turbine).
    df = df.drop_duplicates(subset=["ts", "turbine"], keep="first")
    df = df.sort_values(["turbine", "ts"]).reset_index(drop=True)
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    return df


def load_era5(data_dir: Path) -> pd.DataFrame:
    """Load the cached ERA5 hourly series for the site."""
    path = data_dir / "raw" / "era5_openmeteo.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/01_build_dataset.py")
    payload = json.loads(path.read_text())
    h = payload["hourly"]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(h["time"], utc=True),
            "era5_ws_100m": h["wind_speed_100m"],
            "era5_wdir_100m": h["wind_direction_100m"],
            "era5_temp_2m": h["temperature_2m"],
            "era5_pressure": h["surface_pressure"],
        }
    )
    return df.dropna().reset_index(drop=True)


def load_care(data_dir: Path) -> pd.DataFrame:
    """Load CARE to Compare CSVs with their ground-truth anomaly labels.

    Each CSV holds one turbine-year of 10-min SCADA. The label columns are read
    from the dataset's own ground-truth files when present.
    """
    extracted = data_dir / "raw" / "care_extracted"
    if not extracted.exists():
        raise FileNotFoundError(f"{extracted} missing -- run scripts/01_build_dataset.py")
    frames = []
    for csv_path in sorted(extracted.rglob("*.csv")):
        try:
            d = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        if "Date_time" not in d.columns:
            continue
        d["ts"] = pd.to_datetime(d["Date_time"], format="mixed", utc=True, errors="coerce")
        d["source_file"] = csv_path.name
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no usable CARE CSVs under {extracted}")
    return pd.concat(frames, ignore_index=True)


def resample_scada_to_hourly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Aggregate 10-min SCADA to hourly means (the ERA5 join resolution)."""
    tmp = df.copy()
    tmp["hr"] = tmp["ts"].dt.floor("h")
    agg = tmp.groupby(["turbine", "hr"])[cols].mean().reset_index()
    agg = agg.rename(columns={"hr": "ts"})
    return agg


def hourly_coverage(df: pd.DataFrame) -> pd.Series:
    """Hours present per turbine across the hourly-resampled frame."""
    return df.groupby("turbine")["ts"].nunique()
