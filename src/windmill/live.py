"""The live engine behind the dashboard's interactive explorer.

Three sources, one detector, one provenance record:

  ENGIE + ERA5    a real turbine's SCADA, joined to reanalysis weather
  CARE events     a labelled benchmark dataset, with its ground truth
  an uploaded CSV  any turbine file, self-calibrated from its own history

Every entry point returns the same shape -- a frame of the scored series and
a provenance record saying what was read, how many rows, and how long it
took. A dashboard that runs a detector on the reader's own data has to be
able to show its working, and the same record is what the tests assert on.

This module is deliberately free of Streamlit: it is domain logic, so it
belongs in src/ (repo rule 2) and it can be tested without a browser.

WHAT THE DETECTOR DOES, IN ONE PARAGRAPH
Each thermal channel gets a reference that is conditioned on the operating
point: the channel's own median at the same normalised power, with the
ambient effect removed by a per-ambient-bin median offset. A reading is
scored by how many robust deviations it sits above that reference, and the
per-channel scores are combined by taking the maximum. Two reference
strategies are available and both are reported, because the difference
between them is the project's finding: a FROZEN reference goes stale as the
turbine drifts, and a ROLLING reference follows it.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from windmill import RATED_POWER_KW, care, loaders, multivariate

#: The four turbines at La Haute Borne, in the order the dashboard offers them.
ENGIE_TURBINES = ("R80711", "R80721", "R80736", "R80790")

#: The channel the case study is about: gearbox bearing 2 temperature.
ENGIE_CHANNEL = "Gb2t_avg"

#: The windows the dashboard offers for the ENGIE source. The first one is
#: the default, and it is the SAME window the committed snapshot covers --
#: so the live run and the snapshot are visibly the same measurement, and
#: the chart is dense enough to read.
ENGIE_WINDOWS = {
    "Aug 15 – Oct 15 2017 — the R80711 anomaly": ("2017-08-15", "2017-10-15"),
    "2017 — full year": ("2017-01-01", "2017-12-31"),
    "2016–2017 — two years": ("2016-01-01", "2017-12-31"),
}


class LiveEngineError(RuntimeError):
    """The live engine could not run. The message is shown to the reader."""


# ---------------------------------------------------------------- provenance
def _provenance(rows: int, seconds: float, joined: list[str], **extra) -> dict:
    """The record of what the engine actually did.

    `joined` names the datasets that were combined, because "SCADA + ERA5" and
    "SCADA alone" are different measurements and the reader has to be able to
    tell them apart.
    """
    rec = {
        "rows": int(rows),
        "seconds": round(float(seconds), 3),
        "joined": list(joined),
    }
    rec.update(extra)
    return rec


# ------------------------------------------------------------ column mapping
#: Name fragments that mark a column as a power measurement.
_POWER_HINTS = ("power", "p_avg", "active_power", "activepower", "netpower")
#: Name fragments that mark a column as a wind speed.
_WIND_HINTS = ("windspeed", "wind_speed", "ws_avg", "windspeed_avg")
#: Name fragments that mark a column as an AMBIENT temperature -- a driver,
#: not a component whose overheating would indicate a fault.
_AMBIENT_HINTS = ("ambient", "outside", "outdoor", "airtemp", "air_temp",
                  "tamb", "temp_2m", "era5_temp")
#: Name fragments that mark a column as a COMPONENT temperature.
_THERMAL_HINTS = ("gearbox", "bearing", "generator", "gen", "hub", "nacelle",
                  "oil", "sump", "inlet", "stator", "rotor", "transformer",
                  "converter", "_t_", "temp")

#: Some releases abbreviate their variables to a few characters: the ENGIE
#: release names its thermal channels Git_avg, Gost_avg, Gb2t_avg, Gb1t_avg,
#: Db2t_avg, Db1t_avg, Dst_avg, Yt_avg, Rbt_avg and Rt_avg. A keyword search
#: finds none of them, so the names themselves are matched as well.
_TEMP_NAME_RE = re.compile(
    r"^(?:[a-z]{1,4}\d?t|temp|temperature)(?:_avg|_mean|_value)?$", re.I)


def _norm(name: str) -> str:
    """A column name reduced to letters and digits, for hint matching.

    Publishers write "Wind Speed (m/s)", "wind_speed", and "Ws_avg" for the
    same thing. Matching on the raw name would find only one of them.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


#: The hints, normalised the same way as the column names.
_N_POWER = tuple(_norm(h) for h in _POWER_HINTS)
_N_WIND = tuple(_norm(h) for h in _WIND_HINTS)
_N_AMBIENT = tuple(_norm(h) for h in _AMBIENT_HINTS)
_N_THERMAL = tuple(_norm(h) for h in _THERMAL_HINTS)


def infer_columns(df: pd.DataFrame) -> dict:
    """Work out what each column of an arbitrary turbine CSV is.

    There is no schema for an arbitrary upload, so the mapping is by name
    fragment, in this order: the timestamp, the power (the operating point the
    reference is conditioned on), the wind speed, the ambient temperature, and
    the component temperatures.

    A reader whose file uses different names is told what was found and what
    was needed, rather than being handed a silent failure -- so the function
    reports its reasoning in the returned dict.
    """
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}
    norm = {c: _norm(str(c)) for c in cols}

    def _find(hints: tuple[str, ...], exclude: set[str] | None = None) -> str | None:
        exclude = exclude or set()
        for c in cols:
            if c in exclude:
                continue
            if any(h in norm[c] for h in hints):
                return c
        return None

    # --- timestamp ---------------------------------------------------------
    ts_col = None
    for c in cols:
        nk = norm[c]
        if ("date" in nk and "time" in nk) or nk in (
                "timestamp", "timestmp", "datetime", "ts", "time"):
            ts_col = c
            break
    if ts_col is None:
        # Fall back to the first column that parses as a datetime -- but ONLY
        # a non-numeric one. `pd.to_datetime` happily turns a column of floats
        # into epoch nanoseconds, which would silently make a power column
        # the timestamp.
        for c in cols[:4]:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            try:
                parsed = pd.to_datetime(df[c], format="mixed", utc=True,
                                        errors="coerce")
            except Exception:
                continue
            if parsed.notna().mean() > 0.8:
                ts_col = c
                break

    # --- power -------------------------------------------------------------
    power = _find(_N_POWER)
    if power is None:
        # "P_avg" and friends: a bare P with a unit suffix.
        for c in cols:
            if lower[c].startswith("p_") or lower[c] in ("p", "power"):
                power = c
                break

    # --- wind --------------------------------------------------------------
    wind = _find(_N_WIND)

    # --- ambient (a driver, so it is excluded from the thermal list) -------
    ambient = _find(_N_AMBIENT)

    # --- thermal -----------------------------------------------------------
    thermal: list[str] = []
    for c in cols:
        lc = lower[c]
        if c in (ts_col, power, wind, ambient):
            continue
        # A column that looks like a temperature but is not a driver.
        if any(h in norm[c] for h in _N_AMBIENT):
            continue
        if c in loaders.ENGIE_THERMAL:
            thermal.append(c)
        elif any(h in norm[c] for h in _N_THERMAL):
            thermal.append(c)
        elif _TEMP_NAME_RE.match(lc):
            thermal.append(c)

    return {
        "ts": ts_col,
        "power": power,
        "wind": wind,
        "ambient": ambient,
        "thermal": thermal,
        "all_columns": cols,
    }


def _describe_mapping(mapping: dict) -> str:
    """A sentence describing what was found, for the reader."""
    parts = []
    for key, label in (("ts", "timestamp"), ("power", "power"),
                       ("wind", "wind speed"), ("ambient", "ambient")):
        v = mapping.get(key)
        parts.append(f"{label}: {v}" if v else f"{label}: **not found**")
    parts.append(f"thermal channels: {len(mapping.get('thermal') or [])}")
    return " · ".join(parts)


# --------------------------------------------------------------- the scoring
def _normalised_power(df: pd.DataFrame, power_col: str, rated_kw: float) -> pd.Series:
    """Power as a fraction of rated output.

    The reference is conditioned on the operating point, and the operating
    point has to be comparable across files, so power is normalised by the
    turbine's rated capacity. When the rated capacity is unknown the raw
    power is used and normalised by its own maximum, which is the same thing
    up to a scale factor that cancels inside a per-bin median.
    """
    p = pd.to_numeric(df[power_col], errors="coerce")
    if rated_kw and rated_kw > 0:
        return p / float(rated_kw)
    peak = float(np.nanmax(p.to_numpy(dtype="float64"))) if len(p) else 0.0
    return p / peak if peak > 0 else p


def score_series(
    work: pd.DataFrame,
    thermal: list[str],
    power_col: str,
    ambient_col: str | None = None,
    ref_frac: float = 0.6,
    z_threshold: float = 4.0,
    quantile: float = 0.90,
    rolling_window_days: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Score a series against a frozen reference and a rolling one.

    The frozen reference is fitted on the EARLIER part of the series and
    applied to the later part -- the protocol the CARE benchmark uses, and the
    one that shows what happens when a reference goes stale.

    The rolling reference uses each row's own recent history, so it never
    goes stale. It is also blind to a fault that has been running for weeks,
    because the trailing window fills with faulted data. Both are reported;
    the comparison between them is the point.

    `work` must carry a `ts` column, a normalised-power column named
    `P_norm`, the thermal channels, and optionally an ambient column.
    """
    work = work.sort_values("ts").reset_index(drop=True)
    thermal = [c for c in thermal if c in work.columns]
    if not thermal:
        raise LiveEngineError("no component temperature channel found")
    if "P_norm" not in work.columns:
        raise LiveEngineError("no power column found")

    n = len(work)
    split = max(1, int(n * ref_frac))
    ref = work.iloc[:split]
    test = work.iloc[split:]

    # --- the frozen, cross-fitted reference --------------------------------
    z_frozen = pd.Series(np.nan, index=work.index)
    thresholds = {}
    channels_used: list[str] = []
    if len(ref) >= 200 and len(test) >= 20:
        res = multivariate.detect(
            ref, test, thermal, "P_norm", ambient_col=ambient_col,
            z_threshold=z_threshold, quantile=quantile, max_channels=24,
        )
        channels_used = list(res.channels_used)
        thresholds = {"max": res.threshold_max, "l2": res.threshold_l2}
        # detect() scores `test`; align on the index it was given.
        z_frozen.loc[test.index] = res.zmax.reindex(test.index).to_numpy()

    # --- the rolling reference ---------------------------------------------
    z_rolling = pd.Series(np.nan, index=work.index)
    if n >= 2000:
        roll = multivariate.detect_rolling(
            work, thermal, "P_norm", ambient_col=ambient_col,
            window_days=rolling_window_days, z_threshold=z_threshold,
            max_channels=24,
        )
        z_rolling = roll.zmax.reindex(work.index)
        if not channels_used:
            channels_used = list(roll.channels_used)

    out = work.copy()
    out["z_frozen"] = z_frozen
    out["z_rolling"] = z_rolling

    info = {
        "channels": channels_used,
        "reference_rows": int(len(ref)),
        "scored_rows": int(len(test)),
        "thresholds": thresholds,
        "z_threshold": float(z_threshold),
    }
    return out, info


def _peer_median_at_wind(target: pd.DataFrame, peers: dict[str, pd.DataFrame],
                         channel: str, wind_col: str | None) -> np.ndarray:
    """The control group's median channel value at the same wind speed.

    A weather effect moves every turbine at the site together, so it cancels
    against the peers; a fault on one turbine does not. The peers EXCLUDE the
    target -- a turbine compared against a group it belongs to is pulled
    toward it and the residual disappears.
    """
    frames = []
    for name, d in peers.items():
        if channel not in d.columns:
            continue
        dd = d[[wind_col, channel]].copy() if wind_col and wind_col in d.columns else d[[channel]].copy()
        if wind_col and wind_col in d.columns:
            dd["bin"] = (pd.to_numeric(dd[wind_col], errors="coerce") // 1.0).astype("int64")
            frames.append(dd[["bin", channel]])
        else:
            frames.append(dd[[channel]])
    if not frames:
        return np.full(len(target), np.nan)
    pf = pd.concat(frames, ignore_index=True).dropna()
    if wind_col and wind_col in target.columns and "bin" in pf.columns:
        by_bin = pf.groupby("bin")[channel].median()
        bins = (pd.to_numeric(target[wind_col], errors="coerce") // 1.0).astype("int64")
        return bins.map(by_bin).to_numpy(dtype="float64")
    return np.full(len(target), float(pf[channel].median()))


# ------------------------------------------------------------------- ENGIE
def _engie_csv(data_dir: Path, turbine: str) -> Path:
    """Find one turbine's cleaned SCADA CSV."""
    base = data_dir / "raw" / "engie_la_haute_borne_extracted"
    if not base.exists():
        base = data_dir / "raw" / "engie_v2_extracted"
    if not base.exists():
        raise LiveEngineError("the ENGIE release is not downloaded")
    hits = sorted(base.rglob(f"*{turbine}*F99*.csv"))
    if not hits:
        raise LiveEngineError(f"no SCADA file for {turbine}")
    return hits[0]


def _read_engie(path: Path) -> pd.DataFrame:
    """Read one ENGIE CSV.

    The cleaned CSVs are comma-separated with a UTF-8 BOM, and the
    timestamps carry mixed +01:00/+02:00 offsets, so the parse is
    offset-aware and then normalised to UTC.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    tcol = next((c for c in df.columns
                 if "date" in c.lower() and "time" in c.lower()), None)
    if tcol is None:
        raise LiveEngineError("the file has no timestamp column")
    df["ts"] = pd.to_datetime(df[tcol], format="mixed", utc=True, errors="coerce")
    df = df[df["ts"].notna()].drop_duplicates(subset=["ts"], keep="first")
    return df.sort_values("ts").reset_index(drop=True)


def run_engie(
    data_dir: Path,
    turbine: str = "R80711",
    window: str = "2017 — full year",
    z_threshold: float = 4.0,
    with_peers: bool = True,
) -> dict:
    """Run the detector on one ENGIE turbine, with its siblings as controls.

    The reference is the turbine's OWN 2013-2015 behaviour, frozen -- the same
    protocol the false-alarm numbers use. The rolling reference is computed
    over the full history, so the two can be compared on the same window.
    """
    t0 = time.time()
    if turbine not in ENGIE_TURBINES:
        raise LiveEngineError(f"unknown turbine {turbine}")
    if window not in ENGIE_WINDOWS:
        raise LiveEngineError(f"unknown window {window!r}")
    start, end = ENGIE_WINDOWS[window]

    path = _engie_csv(data_dir, turbine)
    df = _read_engie(path)

    # ERA5 supplies the ambient temperature: the ENGIE release ships no
    # ambient sensor, and the reference needs one to remove the weather's
    # contribution to a component's temperature.
    era5 = loaders.load_era5(data_dir)
    e = era5.copy()
    e["ts"] = e["ts"].dt.floor("10min")
    e = (e.set_index("ts")[["era5_temp_2m"]]
         .resample("10min").ffill().reset_index())

    df = df.merge(e, on="ts", how="left")
    df["P_norm"] = _normalised_power(df, "P_avg", RATED_POWER_KW)
    df = df.dropna(subset=["P_norm", ENGIE_CHANNEL])
    if df.empty:
        raise LiveEngineError(f"no usable rows for {turbine}")

    peers: dict[str, pd.DataFrame] = {}
    peer_names = [t for t in ENGIE_TURBINES if t != turbine]
    if with_peers:
        for peer in peer_names:
            try:
                d = _read_engie(_engie_csv(data_dir, peer))
                d["P_norm"] = _normalised_power(d, "P_avg", RATED_POWER_KW)
                peers[peer] = d.merge(e, on="ts", how="left")
            except LiveEngineError:
                continue

    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC")
    window_df = df[(df["ts"] >= lo) & (df["ts"] <= hi)].copy()
    if window_df.empty:
        raise LiveEngineError(f"no rows in {window}")

    # The reference is the turbine's own earlier behaviour -- frozen.
    ref = df[(df["ts"] >= pd.Timestamp("2013-01-01", tz="UTC"))
             & (df["ts"] <= pd.Timestamp("2015-12-31", tz="UTC"))]
    scored = pd.concat([ref, window_df]).sort_values("ts").reset_index(drop=True)
    # Mark which rows are inside the displayed window.
    scored["in_window"] = (scored["ts"] >= lo) & (scored["ts"] <= hi)

    scored, info = score_series(
        scored, loaders.ENGIE_THERMAL, "P_norm", ambient_col="era5_temp_2m",
        ref_frac=len(ref) / max(1, len(scored)), z_threshold=z_threshold,
    )

    # The rolling reference over the full history, aligned to the window.
    # `df` already carries the ERA5 ambient column -- merging again would
    # suffix it into era5_temp_2m_x / _y.
    full = df.dropna(subset=["P_norm", ENGIE_CHANNEL])
    roll = multivariate.detect_rolling(
        full, loaders.ENGIE_THERMAL, "P_norm", ambient_col="era5_temp_2m",
        window_days=30, z_threshold=z_threshold, max_channels=24)
    z_roll = roll.zmax.copy()
    z_roll.index = full.loc[z_roll.index, "ts"]
    z_roll = z_roll[~z_roll.index.duplicated()]
    scored["z_rolling"] = z_roll.reindex(scored["ts"]).to_numpy()
    if not info["channels"]:
        info["channels"] = list(roll.channels_used)

    # The peer median, at matched wind speed, on the same rows.
    if peers:
        peer_win = {k: v[(v["ts"] >= lo) & (v["ts"] <= hi)] for k, v in peers.items()}
        peer_win = {k: v for k, v in peer_win.items() if not v.empty}
        scored["peer"] = _peer_median_at_wind(
            scored, peer_win, ENGIE_CHANNEL, "Ws_avg")

    out = scored[scored["in_window"]].copy()
    out = out.rename(columns={ENGIE_CHANNEL: "value"})
    out["value_label"] = "gearbox bearing 2 temperature (Gb2t_avg)"
    out["wind"] = out.get("Ws_avg")
    out["ambient"] = out.get("era5_temp_2m")

    # The peers' own values, on the same rows, for the control-group chart.
    for peer in ENGIE_TURBINES:
        if peer == turbine or peer not in peers:
            continue
        d = peers[peer].drop_duplicates(subset="ts", keep="first").set_index("ts")
        out[f"peer_{peer}"] = d[ENGIE_CHANNEL].reindex(out["ts"]).to_numpy()

    joined = [f"ENGIE SCADA — {path.name} ({len(df):,} rows)"]
    joined.append(f"ERA5 reanalysis (open-meteo) — ambient, {len(era5):,} hourly rows")
    if peers:
        joined.append(f"control group — {', '.join(sorted(peers))}")

    prov = _provenance(
        rows=len(out),
        seconds=time.time() - t0,
        joined=joined,
        source=f"ENGIE La Haute Borne — {turbine}",
        turbine=turbine,
        window=f"{start} to {end}",
        reference=f"frozen: {turbine}'s own 2013–2015 behaviour, load- and "
                  f"ambient-conditioned",
        channels=info["channels"],
        z_threshold=z_threshold,
        alarms=int((out["z_rolling"] > z_threshold).sum()),
    )
    return {"frame": out, "provenance": prov}


# -------------------------------------------------------------------- CARE
def care_events(data_dir: Path) -> list[dict]:
    """The CARE benchmark's labelled events, anomaly and normal alike.

    The normal datasets matter as much as the anomaly ones: they are the
    control group, and the project's claim about the AE baseline is partly a
    claim about what happens on healthy turbines.
    """
    root = care.care_root(data_dir)
    if not root.exists():
        return []
    events: list[dict] = []
    for farm in ("Wind Farm A", "Wind Farm B", "Wind Farm C"):
        farm_dir = root / farm
        if not farm_dir.exists():
            continue
        try:
            info = care.event_info(farm_dir)
        except Exception:
            continue
        for _, row in info.iterrows():
            label = str(row.get("event_label", "")).strip().lower()
            events.append({
                "farm": farm,
                "event_id": str(row["event_id"]),
                "label": label,
                "description": str(row.get("event_description", "")),
                "start": int(row.get("event_start_id", 0)),
                "end": int(row.get("event_end_id", 0)),
            })
    # Anomalies first, then the normal controls.
    events.sort(key=lambda e: (e["label"] != "anomaly", e["farm"], e["event_id"]))
    return events


def run_care(
    data_dir: Path,
    farm: str = "Wind Farm C",
    event_id: str = "",
    z_threshold: float | None = None,
    quantile: float = 0.90,
) -> dict:
    """Run the detector on one labelled CARE dataset.

    This is the benchmark's OWN protocol: the reference is fitted on the
    dataset's train split (normal-status rows only) and applied to its
    prediction split, with the threshold taken from a cross-fitted training
    quantile. The ground-truth event window is carried through, so the reader
    can see exactly when the detector fires relative to the truth.
    """
    t0 = time.time()
    root = care.care_root(data_dir)
    if not root.exists():
        raise LiveEngineError("the CARE dataset is not downloaded")
    farm_dir = root / farm
    if not farm_dir.exists():
        raise LiveEngineError(f"no {farm}")

    path = farm_dir / "datasets" / f"{event_id}.csv"
    if not path.exists():
        raise LiveEngineError(f"no dataset {event_id} in {farm}")

    df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
    columns = care.dataset_columns(path)
    pcol, thermal, wind, ambient = care.feature_rows(farm_dir, columns)
    if not pcol:
        raise LiveEngineError("no power channel in this dataset")
    if not thermal:
        raise LiveEngineError("no thermal channels in this dataset")

    train = df[df["train_test"] == "train"]
    test = df[df["train_test"] == "prediction"]
    train_n = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]
    if train_n.empty or test.empty:
        raise LiveEngineError("this dataset has no usable train/prediction split")

    res = multivariate.detect(
        train_n, test, thermal, pcol, ambient_col=ambient,
        z_threshold=z_threshold, quantile=quantile, max_channels=24,
    )

    out = test.copy()
    out["z_frozen"] = res.zmax.reindex(test.index).to_numpy()
    out["z_rolling"] = np.nan
    out["P_norm"] = pd.to_numeric(out[pcol], errors="coerce")
    out["ambient"] = out[ambient] if ambient else np.nan
    out["wind"] = out[wind[0]] if wind else np.nan
    out["status_type_id"] = out["status_type_id"]
    out["in_event"] = (out.index.to_numpy() >= _event_bounds(farm_dir, event_id)[0]) & \
                      (out.index.to_numpy() <= _event_bounds(farm_dir, event_id)[1])
    out["value_label"] = f"max-z over {len(res.channels_used)} thermal channels"

    joined = [f"CARE {farm} — dataset {event_id}.csv ({len(df):,} rows)"]
    joined.append("CARE ground truth — the labelled event window")
    if ambient:
        joined.append(f"CARE site ambient sensor — {ambient}")
    else:
        joined.append("no ambient sensor at this site — load conditioning only")

    prov = _provenance(
        rows=len(out),
        seconds=time.time() - t0,
        joined=joined,
        source=f"CARE to Compare — {farm}, dataset {event_id}",
        reference="frozen: the dataset's own train split (normal status rows "
                  "only), cross-fitted training quantile",
        channels=list(res.channels_used),
        z_threshold=float(res.threshold_max),
        alarms=int(res.flags_max.sum()),
        extra_quantile=quantile,
    )
    return {"frame": out, "provenance": prov}


def _event_bounds(farm_dir: Path, event_id: str) -> tuple[int, int]:
    """The row-id bounds of one labelled event."""
    try:
        info = care.event_info(farm_dir)
    except Exception:
        return (0, 0)
    row = info[info["event_id"].astype(str) == str(event_id)]
    if row.empty:
        return (0, 0)
    r = row.iloc[0]
    return (int(r.get("event_start_id", 0)), int(r.get("event_end_id", 0)))


# ------------------------------------------------------------------ upload
def run_upload(
    df: pd.DataFrame,
    z_threshold: float = 4.0,
    ref_frac: float = 0.6,
    quantile: float = 0.90,
    rolling_window_days: int = 30,
    display_channel: str | None = None,
) -> dict:
    """Self-calibrate on an arbitrary turbine CSV and score it.

    There is no labelled reference for an uploaded file, so the engine
    calibrates on the file's OWN earlier history: the first `ref_frac` of it
    becomes the reference, and the rest is scored. That is what a deployment
    does when it is commissioned -- fit the reference on the machine's own
    history, then watch it.

    The columns are inferred by name (see `infer_columns`). If the file does
    not carry what the detector needs, the reader is told which column is
    missing rather than being shown an empty chart.
    """
    t0 = time.time()
    if df is None or df.empty:
        raise LiveEngineError("the uploaded file is empty")

    mapping = infer_columns(df)
    if not mapping["ts"]:
        raise LiveEngineError(
            "no timestamp column found. Columns present: "
            + ", ".join(map(str, mapping["all_columns"][:20]))
        )
    if not mapping["power"]:
        raise LiveEngineError(
            "no power column found, and the detector needs the operating "
            "point to condition the reference. Columns present: "
            + ", ".join(map(str, mapping["all_columns"][:20]))
        )
    if not mapping["thermal"]:
        raise LiveEngineError(
            "no component temperature channel found. Columns present: "
            + ", ".join(map(str, mapping["all_columns"][:20]))
        )

    work = df.copy()
    work["ts"] = pd.to_datetime(work[mapping["ts"]], format="mixed", utc=True,
                                errors="coerce")
    work = work[work["ts"].notna()].sort_values("ts")
    work["P_norm"] = _normalised_power(work, mapping["power"], 0.0)
    work = work.dropna(subset=["P_norm"] + [mapping["thermal"][0]])
    if len(work) < 400:
        raise LiveEngineError(
            f"only {len(work):,} usable rows -- the detector needs at least 400"
        )

    # The DETECTOR always uses every thermal channel it found -- that is what
    # the max-z combiner is for. The channel the reader looks at is a
    # separate choice, and defaults to the first one found.
    channel = display_channel if display_channel in mapping["thermal"] else mapping["thermal"][0]
    ambient = mapping["ambient"]
    scored, info = score_series(
        work, mapping["thermal"], "P_norm", ambient_col=ambient,
        ref_frac=ref_frac, z_threshold=z_threshold, quantile=quantile,
        rolling_window_days=rolling_window_days,
    )

    # The residual against the conditioned reference, for the displayed channel.
    residual = pd.Series(np.nan, index=scored.index)
    ref = scored.iloc[:max(1, int(len(scored) * ref_frac))]
    if len(ref) >= 200:
        ref_obj = multivariate.fit_channel_reference(
            ref, channel, "P_norm", ambient_col=ambient)
        if ref_obj is not None:
            corrected, _z = ref_obj.apply(
                scored["P_norm"].to_numpy(),
                scored[ambient].to_numpy() if ambient else np.zeros(len(scored)),
                scored[channel].to_numpy(),
            )
            residual = pd.Series(corrected, index=scored.index)

    out = scored.copy()
    out["value"] = out[channel]
    out["value_label"] = channel
    out["residual"] = residual
    out["peer"] = np.nan
    out["wind"] = out[mapping["wind"]] if mapping["wind"] else np.nan
    out["ambient"] = out[ambient] if ambient else np.nan
    out["in_window"] = True

    joined = [f"uploaded CSV — {len(df):,} rows, columns: "
              f"{', '.join(map(str, mapping['all_columns'][:8]))}"
              + ("…" if len(mapping["all_columns"]) > 8 else "")]
    joined.append(
        f"self-calibration — the file's own first {ref_frac:.0%} is the "
        f"reference, the rest is scored"
    )
    if ambient:
        joined.append(f"ambient driver — {ambient}")
    else:
        joined.append("no ambient column — load conditioning only")

    prov = _provenance(
        rows=len(out),
        seconds=time.time() - t0,
        joined=joined,
        source="uploaded turbine CSV",
        reference=f"frozen: the file's own first {ref_frac:.0%}, "
                  f"load-conditioned"
                  + (" and ambient-corrected" if ambient else ""),
        channels=info["channels"],
        z_threshold=z_threshold,
        alarms=int((out["z_rolling"] > z_threshold).sum()),
        mapping=_describe_mapping(mapping),
        available_channels=list(mapping["thermal"]),
        display_channel=channel,
    )
    return {"frame": out, "provenance": prov}


# ------------------------------------------------------------ sample file
#: The window the downloadable sample covers: the anomaly and the months
#: either side of it, so a rolling reference has history to work with.
SAMPLE_START = "2017-08-01"
SAMPLE_END = "2017-11-30"


def build_sample_csv(dest: Path, data_dir: Path) -> Path:
    """Write the downloadable sample turbine CSV.

    It is a real slice of the real ENGIE release, with the ERA5 ambient
    temperature added -- the ENGIE release has no ambient sensor, so the
    ambient column is labelled `era5_temp_2m` and is the reanalysis value,
    not a measurement from the turbine. A reviewer on Streamlit Cloud has no
    19 GB of raw data, so this file is what they can plug in and watch the
    detector calibrate on.
    """
    path = _engie_csv(data_dir, "R80711")
    df = _read_engie(path)
    era5 = loaders.load_era5(data_dir)
    e = era5.copy()
    e["ts"] = e["ts"].dt.floor("10min")
    e = (e.set_index("ts")[["era5_temp_2m"]]
         .resample("10min").ffill().reset_index())
    df = df.merge(e, on="ts", how="left")

    lo = pd.Timestamp(SAMPLE_START, tz="UTC")
    hi = pd.Timestamp(SAMPLE_END, tz="UTC")
    win = df[(df["ts"] >= lo) & (df["ts"] <= hi)]

    keep = ["Date_time", "P_avg", "Ws_avg", "Git_avg", "Gost_avg",
            "Gb2t_avg", "Gb1t_avg", "Db2t_avg", "era5_temp_2m"]
    keep = [c for c in keep if c in win.columns]
    out = win[keep].copy()
    for c in out.columns:
        if c != "Date_time":
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    out = out.dropna(subset=["P_avg", "Gb2t_avg"])

    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return dest


def sample_csv_path() -> Path:
    """Where the committed sample lives."""
    return Path(__file__).resolve().parents[2] / "assets" / "sample_turbine.csv"
