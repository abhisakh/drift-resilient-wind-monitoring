"""Weather forecasting and power prediction.

The operational question this module answers is the one a control room asks
every night: given tomorrow's weather, what power should this turbine be
producing, and is it producing it?

Three pieces, and they are kept apart deliberately:

  1. `fetch_forecast` pulls a real hourly forecast from the Open-Meteo
     forecast API. No key, no account, and it is the same reanalysis family
     the project already uses for its climatology.
  2. `power_curve_from_frame` builds the turbine's own conditioned power
     curve -- wind speed to kilowatts -- from measured SCADA, so the
     prediction is the machine's own behaviour and not a catalogue curve.
  3. `predict_power` applies that curve to a forecast, with the standard
     air-density correction, so a cold, dense forecast predicts more power
     than a warm one at the same wind speed.

The fallback matters as much as the forecast: the dashboard has to work on
Streamlit Community Cloud and in CI, where the network may be absent. So
`sample_forecast` derives a deterministic 7-day profile from the committed
sample turbine CSV, and says plainly that it is a fallback and not a
forecast.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

#: The Open-Meteo forecast endpoint. No API key, no account.
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: The La Haute Borne site: the ENGIE wind farm the project measures on.
SITE = {
    "name": "ENGIE La Haute Borne (Meuse, France)",
    "lat": 48.45,
    "lon": 5.59,
}

#: The dry-air gas constant, J/(kg K). Used for the air density.
R_DRY_AIR = 287.058

#: The reference air density the power curve was measured at, kg/m^3.
RHO_REFERENCE = 1.225

#: The exponent of the standard air-density correction: power scales with
#: the cube of the density ratio.
DENSITY_EXPONENT = 1.0 / 3.0

#: The hourly variables the dashboard needs.
HOURLY_VARS = ("wind_speed_100m", "temperature_2m", "surface_pressure")


class ForecastUnavailable(RuntimeError):
    """The forecast could not be fetched. The caller falls back."""


def fetch_forecast(lat: float, lon: float, days: int = 7,
                   timeout: float = 12.0) -> pd.DataFrame:
    """Fetch an hourly weather forecast from Open-Meteo.

    Returns a frame with `ts` (UTC), `wind_speed_100m`, `temperature_2m` and
    `surface_pressure`. Raises `ForecastUnavailable` if the network is not
    there -- the caller decides what to do about it, rather than this
    function silently inventing weather.
    """
    import requests

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": days,
        "timezone": "UTC",
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        raise ForecastUnavailable(
            f"the Open-Meteo forecast could not be reached: {exc}") from exc

    try:
        hourly = payload["hourly"]
        df = pd.DataFrame({
            "ts": pd.to_datetime(hourly["time"], utc=True),
            "wind_speed_100m": hourly["wind_speed_100m"],
            "temperature_2m": hourly["temperature_2m"],
            "surface_pressure": hourly["surface_pressure"],
        })
    except Exception as exc:
        raise ForecastUnavailable(
            f"the forecast response did not have the expected shape: {exc}") from exc

    return df.dropna().reset_index(drop=True)


def air_density(temperature_c, pressure_hpa) -> np.ndarray:
    """The air density of the forecast air, kg/m^3.

    The ideal-gas law with the dry-air gas constant: rho = p / (R T), with
    the pressure in pascals and the temperature in kelvin.
    """
    t = np.atleast_1d(np.asarray(temperature_c, dtype="float64"))
    p = np.atleast_1d(np.asarray(pressure_hpa, dtype="float64"))
    kelvin = t + 273.15
    with np.errstate(invalid="ignore", divide="ignore"):
        return (p * 100.0) / (R_DRY_AIR * kelvin)


def power_curve_from_frame(df: pd.DataFrame, wind_col: str, power_col: str,
                            bin_width: float = 1.0) -> pd.DataFrame:
    """The turbine's own conditioned power curve, from measured SCADA.

    The median power in each wind-speed bin, because the median is not
    dragged around by the rows where the machine was curtailed or where it
    was overheating. Bins with too few rows are dropped rather than
    interpolated, so the curve never claims to know the machine's behaviour
    in an operating range it was never seen in.
    """
    cols = [wind_col, power_col]
    t = df.dropna(subset=cols)
    if len(t) < 100:
        raise ValueError("not enough rows to build a power curve")
    bins = np.floor(t[wind_col].to_numpy(dtype="float64") / bin_width).astype("int64")
    curve = (pd.DataFrame({"bin": bins, "p": t[power_col].to_numpy(dtype="float64")})
             .groupby("bin")["p"].agg(["median", "size"])
             .reset_index())
    curve = curve[curve["size"] >= 10].copy()
    if curve.empty:
        raise ValueError("no wind-speed bin had enough rows to fit a power curve")
    curve["wind_speed"] = (curve["bin"] + 0.5) * bin_width
    curve = curve.rename(columns={"median": "power_kw", "size": "n"})
    return curve[["wind_speed", "power_kw", "n"]].sort_values("wind_speed").reset_index(drop=True)


def predict_power(curve: pd.DataFrame, wind_speed, air_density_ratio=1.0) -> np.ndarray:
    """The power the curve says this wind speed should produce, in kW.

    The curve is read at the wind speed, and then corrected for the air
    density: the same wind speed in denser air carries more energy, so the
    machine produces more power. The correction is the cube root of the
    density ratio, which is the standard approximation.
    """
    ws = np.asarray(wind_speed, dtype="float64")
    cw = curve["wind_speed"].to_numpy(dtype="float64")
    cp = curve["power_kw"].to_numpy(dtype="float64")
    if len(cw) < 2:
        raise ValueError("the power curve needs at least two points")
    base = np.interp(ws, cw, cp, left=cp[0], right=cp[-1])
    ratio = np.asarray(air_density_ratio, dtype="float64")
    with np.errstate(invalid="ignore"):
        corrected = base * np.power(ratio, DENSITY_EXPONENT)
    return np.where(np.isfinite(corrected), corrected, 0.0)


def forecast_frame(forecast: pd.DataFrame, curve: pd.DataFrame,
                   wind_shift: float = 0.0, temp_shift: float = 0.0,
                   pressure_hpa: float = 1013.25) -> pd.DataFrame:
    """The forecast, shifted, with the predicted healthy power beside it.

    The shifts are the what-if: "what if the wind were 2 m/s stronger, or the
    air 5 degrees colder". They are applied to the forecast before the power
    curve is applied, so the effect on the predicted power is the real one.
    """
    out = forecast.copy()
    out["wind_speed_shifted"] = out["wind_speed_100m"] + wind_shift
    out["temperature_shifted"] = out["temperature_2m"] + temp_shift
    out["surface_pressure"] = out.get("surface_pressure", pressure_hpa)
    rho = air_density(out["temperature_shifted"], out["surface_pressure"])
    out["air_density"] = rho
    out["density_ratio"] = rho / RHO_REFERENCE
    out["predicted_power_kw"] = predict_power(curve, out["wind_speed_shifted"],
                                              out["density_ratio"])
    return out


def sample_forecast(csv_path: Path, days: int = 7) -> pd.DataFrame:
    """A deterministic 7-day profile derived from the sample turbine CSV.

    This is the OFFLINE FALLBACK, not a forecast: it is the sample's own
    median wind speed and temperature for each hour of the day, repeated over
    seven days from a fixed start. It exists so the dashboard is usable
    without a network, and it is labelled as a fallback everywhere it is
    shown.
    """
    df = pd.read_csv(csv_path)
    tcol = next((c for c in df.columns
                 if "date" in c.lower() and "time" in c.lower()), None)
    wind_col = next((c for c in df.columns if c.lower() in ("ws_avg", "wind_speed")
                     or "wind" in c.lower()), None)
    temp_col = next((c for c in df.columns if "temp" in c.lower()), None)
    if not (tcol and wind_col and temp_col):
        raise ValueError("the sample CSV does not have the columns the fallback needs")

    ts = pd.to_datetime(df[tcol], format="mixed", utc=True, errors="coerce")
    work = pd.DataFrame({
        "ts": ts,
        "hour": ts.dt.hour,
        "wind_speed_100m": pd.to_numeric(df[wind_col], errors="coerce"),
        "temperature_2m": pd.to_numeric(df[temp_col], errors="coerce"),
    }).dropna()

    hourly = work.groupby("hour").agg(
        wind_speed_100m=("wind_speed_100m", "median"),
        temperature_2m=("temperature_2m", "median"),
    )

    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for day in range(days):
        for hour in range(24):
            base = start + pd.Timedelta(days=day, hours=hour)
            rows.append({
                "ts": base,
                "wind_speed_100m": float(hourly["wind_speed_100m"].get(hour, 6.0)),
                "temperature_2m": float(hourly["temperature_2m"].get(hour, 10.0)),
                "surface_pressure": 1013.25,
            })
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def sample_power_curve(csv_path: Path) -> pd.DataFrame:
    """The power curve of the turbine in the committed sample CSV."""
    df = pd.read_csv(csv_path)
    wind_col = next((c for c in df.columns if c.lower() in ("ws_avg", "wind_speed")
                     or "wind" in c.lower()), None)
    power_col = next((c for c in df.columns if c.lower() in ("p_avg", "power")
                      or "power" in c.lower()), None)
    if not (wind_col and power_col):
        raise ValueError("the sample CSV does not have the columns the curve needs")
    return power_curve_from_frame(df, wind_col, power_col)
