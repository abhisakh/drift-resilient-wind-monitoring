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

#: How many days the reference is fitted on. The rule is fixed: the
#: reference is ALWAYS the thirty days immediately before the window being
#: predicted, never a longer window and never an older one.
TRAINING_DAYS = 30

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


def sample_power_curve(csv_path: Path, days: int | None = 30) -> pd.DataFrame:
    """The power curve of the turbine in the committed sample CSV.

    By default only the TRAILING `days` days are used, so the curve obeys
    the same rule as the checkpoints: the reference is the recent past, not
    the whole record. Pass `days=None` for the whole file.
    """
    df = pd.read_csv(csv_path)
    wind_col = next((c for c in df.columns if c.lower() in ("ws_avg", "wind_speed")
                     or "wind" in c.lower()), None)
    power_col = next((c for c in df.columns if c.lower() in ("p_avg", "power")
                      or "power" in c.lower()), None)
    if not (wind_col and power_col):
        raise ValueError("the sample CSV does not have the columns the curve needs")
    if days:
        tcol = next((c for c in df.columns
                     if "date" in c.lower() and "time" in c.lower()), None)
        if tcol:
            ts = pd.to_datetime(df[tcol], format="mixed", utc=True,
                                errors="coerce")
            cut = ts.max() - pd.Timedelta(days=days)
            df = df[ts >= cut]
    return power_curve_from_frame(df, wind_col, power_col)


def trailing_window(csv_path: Path, days: int = TRAINING_DAYS) -> tuple[str, str]:
    """The trailing window of a SCADA file, as (start, end)."""
    df = pd.read_csv(csv_path, usecols=None)
    tcol = next((c for c in df.columns
                 if "date" in c.lower() and "time" in c.lower()), None)
    if not tcol:
        raise ValueError("the sample CSV does not have a timestamp column")
    ts = pd.to_datetime(df[tcol], format="mixed", utc=True, errors="coerce").dropna()
    end = ts.max()
    start = end - pd.Timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ------------------------------------------------------------- checkpoints
#: The curated presentation checkpoints.
#:
#: Each one is a seven-day window in the past with the seasonal story the
#: dashboard is meant to show, and each carries the reference windows it
#: is compared against. The windows are DERIVED, never typed: the training
#: window is always the thirty days immediately before the prediction
#: window, and the stale baselines are the same rule applied one and two
#: years earlier.
CHECKPOINTS = {
    "checkpoint_1": {
        "label": "Checkpoint 1 — 2017-09-01 → 2017-09-07",
        "note": "late summer transition",
        "prediction_start": "2017-09-01",
        "prediction_end": "2017-09-07",
    },
    "checkpoint_2": {
        "label": "Checkpoint 2 — 2017-10-01 → 2017-10-07",
        "note": "autumn wind ramp-up",
        "prediction_start": "2017-10-01",
        "prediction_end": "2017-10-07",
    },
    "checkpoint_3": {
        "label": "Checkpoint 3 — 2017-11-01 → 2017-11-07",
        "note": "high-wind winter onset",
        "prediction_start": "2017-11-01",
        "prediction_end": "2017-11-07",
    },
    "checkpoint_4": {
        "label": "Checkpoint 4 — 2016-09-01 → 2016-09-07",
        "note": "a year earlier, against the 2015 and 2014 curves",
        "prediction_start": "2016-09-01",
        "prediction_end": "2016-09-07",
    },
    "checkpoint_5": {
        "label": "Checkpoint 5 — 2016-11-01 → 2016-11-07",
        "note": "a year earlier, high-wind onset",
        "prediction_start": "2016-11-01",
        "prediction_end": "2016-11-07",
    },
}

#: How many years back the stale baselines go.
BACK_YEARS = (1, 2)


def training_window(prediction_start: str, days: int = TRAINING_DAYS) -> tuple[str, str]:
    """The reference window: exactly the days immediately BEFORE the
    prediction window.

    The interval is half-open -- `[prediction_start - days, prediction_start)`
    -- so the training data never overlaps the week being predicted. That
    is the rule, and it is enforced here rather than typed into each
    checkpoint, so a checkpoint cannot be defined with a window that leaks.
    """
    start = pd.Timestamp(prediction_start, tz="UTC") - pd.Timedelta(days=days)
    end = pd.Timestamp(prediction_start, tz="UTC")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _fit_curve(train: pd.DataFrame, wind_col: str, power_col: str) -> tuple[pd.DataFrame, float]:
    """The power curve of one training window, and its reference density.

    The density matters as much as the curve: the same wind speed in denser
    air carries more energy, so a curve fitted in one season is not valid
    in another unless the density difference is accounted for.
    """
    curve = power_curve_from_frame(train, wind_col, power_col)
    rho = float(np.nanmean(train["air_density"].to_numpy(dtype="float64")))
    return curve, rho


def checkpoint_bundle(
    scada: pd.DataFrame,
    prediction_start: str,
    prediction_end: str,
    back_years: tuple[int, ...] = BACK_YEARS,
    wind_col: str = "Ws_avg",
    power_col: str = "P_avg",
) -> dict:
    """Everything needed to score one checkpoint, from one SCADA frame.

    `scada` must carry `ts`, the wind speed, the power, the ambient
    temperature and the surface pressure -- the frame the ERA5 join
    produces.

    Three fits are built, all on a thirty-day window and all on the same
    rule:

      primary      the thirty days immediately before the prediction window
      1_year_back  the same window, one year earlier
      2_years_back the same window, two years earlier

    and the target week's actuals are carried alongside, so the
    predictions can be scored without touching the raw data again.
    """
    from windmill import loaders  # noqa: F401  (kept for symmetry)

    scada = scada.copy()
    scada["ts"] = pd.to_datetime(scada["ts"], utc=True)
    scada = scada.sort_values("ts").reset_index(drop=True)
    scada["air_density"] = air_density(scada["era5_temp_2m"],
                                        scada["surface_pressure"])

    p_start = pd.Timestamp(prediction_start, tz="UTC")
    p_end = pd.Timestamp(prediction_end, tz="UTC")

    target = scada[(scada["ts"] >= p_start) & (scada["ts"] < p_end)].copy()
    target = target.dropna(subset=[wind_col, power_col, "era5_temp_2m",
                                   "surface_pressure"])

    fits: dict[str, dict] = {}
    for label, years in (("primary", 0),) + tuple(
            (f"{y}_year_back", y) for y in back_years):
        shift = pd.DateOffset(years=years)
        win_start = (p_start - pd.Timedelta(days=TRAINING_DAYS)) - shift
        win_end = p_start - shift
        train = scada[(scada["ts"] >= win_start) & (scada["ts"] < win_end)].copy()
        train = train.dropna(subset=[wind_col, power_col, "era5_temp_2m",
                                     "surface_pressure"])
        if len(train) < 400:
            fits[label] = {
                "window_start": win_start.strftime("%Y-%m-%d"),
                "window_end": win_end.strftime("%Y-%m-%d"),
                "rows": int(len(train)),
                "curve": [],
                "rho_ref": None,
                "available": False,
            }
            continue
        t0 = time.perf_counter()
        curve, rho = _fit_curve(train, wind_col, power_col)
        elapsed = time.perf_counter() - t0
        fits[label] = {
            "window_start": win_start.strftime("%Y-%m-%d"),
            "window_end": win_end.strftime("%Y-%m-%d"),
            "rows": int(len(train)),
            "seconds": round(elapsed, 3),
            "rho_ref": rho,
            "curve": [
                {"wind_speed": float(r.wind_speed), "power_kw": float(r.power_kw),
                 "n": int(r.n)}
                for r in curve.itertuples()
            ],
            "available": True,
        }

    rows = [
        {
            "ts": int(r.ts.timestamp()),
            "wind_speed": float(getattr(r, wind_col)),
            "temperature_c": float(r.era5_temp_2m),
            "air_density": float(r.air_density),
            "actual_power_kw": float(getattr(r, power_col)),
        }
        for r in target.itertuples()
    ]

    return {
        "prediction_start": prediction_start,
        "prediction_end": prediction_end,
        "fits": fits,
        "rows": rows,
    }


def predict_from_curve(curve: list[dict], wind_speed, rho_target, rho_ref) -> np.ndarray:
    """Apply a fitted curve to a week's weather.

    The curve is read at the wind speed and then corrected for the air
    density, referenced to the density the curve was fitted at -- the
    standard way a deployed power curve is applied.
    """
    if not curve or not rho_ref:
        return np.full(len(np.atleast_1d(wind_speed)), np.nan)
    cw = np.array([c["wind_speed"] for c in curve], dtype="float64")
    cp = np.array([c["power_kw"] for c in curve], dtype="float64")
    ws = np.asarray(wind_speed, dtype="float64")
    base = np.interp(ws, cw, cp, left=cp[0], right=cp[-1])
    ratio = np.asarray(rho_target, dtype="float64") / float(rho_ref)
    with np.errstate(invalid="ignore"):
        corrected = base * np.power(ratio, DENSITY_EXPONENT)
    return np.where(np.isfinite(corrected), corrected, np.nan)


def mean_absolute_error(predicted, actual) -> float | None:
    """The tracking error in kilowatts, or None if there is nothing to
    compare."""
    p = np.asarray(predicted, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    ok = np.isfinite(p) & np.isfinite(a)
    if not ok.any():
        return None
    return float(np.mean(np.abs(p[ok] - a[ok])))


def score_checkpoint(bundle: dict, wind_col: str = "wind_speed") -> dict:
    """Predict the target week with every fit and score the tracking error.

    The key comparison is between fits of the SAME rule at different ages:
    a curve refreshed every month against one that has been left for a
    year and one that has been left for two.
    """
    rows = bundle.get("rows") or []
    if not rows:
        return {"fits": {}}
    df = pd.DataFrame(rows)
    actual = df["actual_power_kw"].to_numpy(dtype="float64")
    wind = df[wind_col].to_numpy(dtype="float64")
    rho = df["air_density"].to_numpy(dtype="float64")

    out: dict[str, dict] = {}
    for label, fit in bundle["fits"].items():
        if not fit.get("available"):
            out[label] = {"available": False}
            continue
        pred = predict_from_curve(fit["curve"], wind, rho, fit["rho_ref"])
        mae = mean_absolute_error(pred, actual)
        out[label] = {
            "available": True,
            "mae_kw": round(mae, 2) if mae is not None else None,
            "predicted_kw": [None if not np.isfinite(v) else round(float(v), 2)
                             for v in pred],
        }
    return {"fits": out}


def engie_checkpoint_frame(data_dir: Path, turbine: str = "R80711") -> pd.DataFrame:
    """One turbine's SCADA joined to the ERA5 ambient and pressure.

    The frame the checkpoint machinery needs: timestamps, wind speed, power,
    the reanalysis ambient temperature, and the surface pressure that goes
    with it.
    """
    from windmill import loaders

    df = loaders.load_engie_turbine(data_dir, turbine)
    era5 = loaders.load_era5_on_grid(data_dir)
    df = df.merge(era5[["ts", "era5_temp_2m", "era5_pressure"]], on="ts",
                  how="left")
    df = df.rename(columns={"era5_pressure": "surface_pressure"})
    df["P_norm"] = df["P_avg"] / 2050.0
    return df


# ------------------------------------------------------- the exported file
#: Where `make snapshot` writes the frozen-baseline bundles.
FROZEN_BASELINES_ARTEFACT = "demo_frozen_baselines.json"


def frozen_baselines_path() -> Path:
    """Where the exported frozen-baseline artefact lives."""
    return Path(__file__).resolve().parents[2] / "results" / FROZEN_BASELINES_ARTEFACT


def load_frozen_baselines() -> dict:
    """Read the exported bundles, or an empty dict if they are not there."""
    path = frozen_baselines_path()
    if not path.exists():
        return {}
    import json
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def export_frozen_baselines(dest: Path, data_dir: Path,
                            turbine: str = "R80711") -> Path:
    """Build every checkpoint bundle and write them to `dest`.

    This is what makes the multi-year staleness comparison work on Streamlit
    Community Cloud, where the 19 GB of raw SCADA is not present: the
    reference curves for one and two years back, and the target week's
    actuals, are computed once from the raw data and committed.
    """
    import json

    frame = engie_checkpoint_frame(data_dir, turbine)
    bundles = {}
    for key, cp in CHECKPOINTS.items():
        bundle = checkpoint_bundle(frame, cp["prediction_start"],
                                   cp["prediction_end"])
        bundle["label"] = cp["label"]
        bundle["note"] = cp["note"]
        bundles[key] = bundle
    out = {
        "generated_by": "scripts/pipeline/11_demo.py",
        "turbine": turbine,
        "source": "ENGIE La Haute Borne (Mendeley v2, CC BY 4.0)",
        "checkpoints": bundles,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    return dest
