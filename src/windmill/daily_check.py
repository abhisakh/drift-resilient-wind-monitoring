"""The daily check: predict the healthy sensor value each day, compare it with
what the sensor shows, and alarm on the day the two part.

THE POLICY, AS AGREED WITH THE AUTHOR
-------------------------------------
1. Each month the model is trained on HEALTHY data from before the month: at
   least a month of readings, never standstill, never the warm-up after a
   restart (a bearing that restarts cold runs up to 12 K below normal for
   hours), never a day that was flagged, and never data from before the last
   maintenance.
2. During the month the model is FROZEN: it never learns from the month's own
   sensor values, because the bearing may start to fail on any day. Each day
   it is fed only that day's power and weather.
3. Every day, the model predicts what a healthy sensor should read from that
   day's power and weather, and the day's mean difference (actual minus
   predicted) is compared with a tolerance. Hotter than the tolerance: a hot
   day. Colder than minus the tolerance: a cold day (a sensor reading low is
   misbehaving too). A run of consecutive hot days is ONE call-out,
   dispatched at the end of its first day, when the day's mean is known.
4. After a CONFIRMED fault the model stays frozen until maintenance, so it
   cannot learn the fault as normal.
5. After maintenance the old healthy reference no longer applies: the model
   is BLANK until a new month of healthy data has been collected, and only the
   fixed limit protects the turbine meanwhile.

EVERY SENSOR, AND THE WEEK AHEAD
--------------------------------
* One healthy model per temperature sensor, each from power and weather only,
  never from another sensor: a fault that heats the whole gearbox would
  otherwise be predicted as normal by the sensors next to it.
* The forecast: what each sensor should read 1 to 7 days ahead, from the
  weather forecast. Power is predicted from the forecast wind through the
  turbine's own power curve (fitted against ERA5 wind on the training window),
  and the model used is the one in force when the forecast is issued. There is
  no archive of 2017 forecasts, so the forecast is ERA5 with forecast error
  added, growing with lead time (`FORECAST_SIGMA`): an assumption, stated.

WHAT STANDS IN FOR MISSING RECORDS
----------------------------------
The public data carry no alarm, inspection or maintenance records, so two
things are stood in for, and both are stated in every result:

* maintenance: a restart after `MAINTENANCE_DAYS` or more without running;
* the inspection after an alarm: if the day lies in a documented anomaly span
  (`confirmation.ANOMALY_SPANS`) the fault is confirmed; otherwise the
  inspection finds nothing and the alarm is cleared (a false alarm), and the
  flagged day is kept out of all later training.

This module works only on the frames it is handed, so it can be tested on
synthetic data.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import live, multivariate
from .confirmation import ANOMALY_SPANS
from .operating_curve import alarm_runs

CHANNEL = "Gb2t_avg"

#: The temperature sensors a turbine is checked on, each with its own model.
SENSORS = ("Gb2t_avg", "Gb1t_avg", "Gost_avg", "Git_avg", "Db2t_avg",
           "Dst_avg", "Yt_avg", "Rbt_avg", "Rt_avg")
#: Their names, for the reader.
SENSOR_NAMES = {
    "Gb2t_avg": "gearbox bearing 2", "Gb1t_avg": "gearbox bearing 1",
    "Gost_avg": "gearbox oil sump", "Git_avg": "gearbox inlet",
    "Db2t_avg": "generator bearing 2", "Dst_avg": "generator stator",
    "Yt_avg": "nacelle", "Rbt_avg": "rotor bearing", "Rt_avg": "hub",
}
#: The gearbox sensors whose agreement is tested.
GEARBOX = ("Gb2t_avg", "Gb1t_avg", "Gost_avg", "Git_avg")
#: How many gearbox sensors must be hot on the same day for the multi-sensor rule.
GEARBOX_AGREE = 2

#: Above this share of rated power the turbine is running.
RUNNING = 0.02
#: A restart is the first running reading after this long without running.
OFF_HOURS_FOR_RESTART = 6
#: The warm-up after a restart: never trained on and never checked.
WARMUP_HOURS = 12
#: A restart after this many days without running stands in for maintenance.
MAINTENANCE_DAYS = 3
#: The tolerances tested, in kelvin, on the day's mean difference.
TOLERANCES = (1.0, 1.5, 2.0, 3.0)
#: A day is checked only with at least this many usable readings (6 hours).
MIN_DAY_ROWS = 36
#: The trimming of unhealthy training rows: the engine's own setting.
TRIM_K = 4.0
#: How far back the training may reach, and by how much at a time.
#: In a month the model is frozen after a confirmed fault, judge the turbine
#: with its own healthy model for the same month a year earlier (see
#: `simulate` and `last_year_ok`). Off by default so other callers (CARE
#: Farms B and C) keep the plain frozen model; the ENGIE daily check
#: (scripts/pipeline/19_daily_check.py) turns it on.
LAST_YEAR_WHEN_FROZEN = False
#: How close last year's model must be on this year's healthy readings, K.
LAST_YEAR_GATE_K = 1.0
MAX_LOOKBACK_DAYS = live.MAX_TRAIN_LOOKBACK_DAYS
STEP_DAYS = live.TRAIN_EXTEND_DAYS
#: A month of healthy readings.
MONTH_OF_READINGS = live.MONTH_OF_READINGS
#: The fixed limit the check is compared with, held for an hour.
HOT_C = 80.0
LIMIT_HOLD = 6

#: The forecast leads, in days.
LEADS = (1, 2, 3, 4, 5, 6, 7)
#: The assumed forecast error at a lead of L days: the standard deviation of
#: the day's bias in wind speed (m/s) and in temperature (K). Each hour adds
#: noise of half that size. 1.0 at one day ahead, 2.5 at seven.
FORECAST_SIGMA = {L: 1.0 + 0.25 * (L - 1) for L in LEADS}
#: The power curve against ERA5 wind: bin width (m/s) and the least readings a
#: bin needs.
CURVE_BIN = 0.5
CURVE_MIN = 10

#: The sibling comparison: power within this share of rated of the siblings'.
SIBLING_LOAD_MATCH = 0.10
#: A wind-direction sector needs this many comparable moments in the training
#: window for its own normal gap; otherwise the window's overall normal is used.
SECTOR_MIN = 72
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
SIBLING_MIN_READINGS = 36
SIBLING_SMOOTH_DAYS = 5
SIBLING_THRESHOLDS = (2.0, 2.5, 3.0)


def _utc(x) -> pd.Timestamp:
    x = pd.Timestamp(x)
    return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")


def with_era5(frame: pd.DataFrame, data_dir) -> pd.DataFrame:
    """The frame with ERA5 100 m wind speed and direction joined on."""
    from . import loaders

    e = loaders.load_era5(data_dir).copy()
    e["ts"] = pd.to_datetime(e["ts"], utc=True).dt.floor("10min")
    e = (e.set_index("ts")[["era5_ws_100m", "era5_wdir_100m"]]
         .resample("10min").ffill().reset_index())
    out = frame.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.merge(e, on="ts", how="left")


def predict(ref, load, ambient) -> np.ndarray:
    """What a healthy sensor should read: the reference's median at that load,
    plus its ambient offset. The same arithmetic as `ChannelReference.apply`."""
    load = np.asarray(load, dtype="float64")
    ambient = np.asarray(ambient, dtype="float64")
    with np.errstate(invalid="ignore"):
        bins = np.floor(load / multivariate.LOAD_BIN)
        abins = np.floor(ambient / multivariate.AMBIENT_BIN)
    base = ref.medians.reindex(pd.Series(bins).astype("Int64")).to_numpy(
        dtype="float64", na_value=np.nan)
    off = ref.ambient_offsets.reindex(pd.Series(abins).astype("Int64")).to_numpy(
        dtype="float64", na_value=np.nan)
    return base + np.where(np.isfinite(off), off, 0.0)


def mark_restarts(frame: pd.DataFrame,
                  off_hours: float = OFF_HOURS_FOR_RESTART,
                  warmup_hours: float = WARMUP_HOURS,
                  maintenance_days: float = MAINTENANCE_DAYS):
    """The frame, sorted, with `running`, `warmup` and `day` columns, and the
    restarts that stand in for maintenance.

    A restart is the first running reading after `off_hours` without running,
    whether the turbine was stopped, becalmed or not recording: the bearing
    has cooled either way. The first running reading of the record counts as
    one (nothing before it is known).
    """
    f = frame.sort_values("ts").reset_index(drop=True).copy()
    running = f["P_norm"].to_numpy(dtype="float64") > RUNNING
    f["running"] = running
    r_ts = f.loc[running, "ts"]
    off = r_ts.diff()
    restart = off.isna() | (off >= pd.Timedelta(hours=off_hours))
    since = r_ts - r_ts.where(restart).ffill()
    f["warmup"] = False
    f.loc[r_ts.index, "warmup"] = (
        since < pd.Timedelta(hours=warmup_hours)).to_numpy()
    f["day"] = f["ts"].dt.floor("D")
    long_off = (off >= pd.Timedelta(days=maintenance_days)).fillna(False)
    maintenance = [_utc(t) for t in r_ts[long_off.to_numpy()]]
    return f, maintenance


def in_fault(turbine: str, day, spans: dict | None = None) -> bool:
    """Is this day inside a documented anomaly span of the turbine?"""
    spans = ANOMALY_SPANS if spans is None else spans
    day = _utc(day)
    return any(_utc(lo) <= day < _utc(hi) for lo, hi in spans.get(turbine, []))


def power_curve(rows: pd.DataFrame):
    """The turbine's power against ERA5 100 m wind: (bin centres, median power),
    or None. Fitted on the training window, so it is the turbine's own."""
    if "era5_ws_100m" not in rows.columns:
        return None
    r = rows[["era5_ws_100m", "P_norm"]].dropna()
    if len(r) < 200:
        return None
    b = np.floor(r["era5_ws_100m"].to_numpy(dtype="float64") / CURVE_BIN)
    g = r.groupby(b)["P_norm"].agg(["median", "size"])
    g = g[g["size"] >= CURVE_MIN]
    if len(g) < 5:
        return None
    centres = (g.index.to_numpy(dtype="float64") + 0.5) * CURVE_BIN
    return centres, g["median"].to_numpy(dtype="float64")


def train(f: pd.DataFrame, m_start: pd.Timestamp, floor: pd.Timestamp,
          excluded_days: set, channel: str = CHANNEL, channels=None):
    """The month's models and their provenance, or (None, None) when a month
    of healthy data is not there.

    Starts with the whole previous calendar month and reaches back a week at a
    time, never before `floor` (the last maintenance, or three months back),
    until the trimmed training set holds a month of healthy readings. Returns
    one reference per sensor in `channels` (the primary `channel` must fit)
    and the power curve, all from the same window.
    """
    channels = [channel] + [c for c in (channels or []) if c != channel]
    thermal = [c for c in live.CROSS_CHECK_CHANNELS if c in f.columns]
    if channel not in thermal:
        thermal.append(channel)
    usable = f["running"] & ~f["warmup"] & ~f["day"].isin(excluded_days)
    start = m_start - pd.offsets.MonthBegin(1)
    while True:
        s = max(start, floor)
        tr = f[usable & (f["ts"] >= s) & (f["ts"] < m_start)]
        if len(tr) >= MONTH_OF_READINGS:
            worst = pd.Series(0.0, index=tr.index, dtype="float64")
            for ch in thermal:
                z = live._out_of_fold_residual(tr, ch, "P_norm", "era5_temp_2m")
                worst = worst.combine(z.fillna(0.0), np.fmax)
            clean = tr[worst <= TRIM_K]
            if len(clean) >= MONTH_OF_READINGS:
                ref = multivariate.fit_channel_reference(
                    clean, channel, "P_norm", ambient_col="era5_temp_2m")
                if ref is not None:
                    refs = {channel: ref}
                    for c in channels[1:]:
                        if c in clean.columns:
                            r = multivariate.fit_channel_reference(
                                clean, c, "P_norm", ambient_col="era5_temp_2m")
                            if r is not None:
                                refs[c] = r
                    window = f[(f["ts"] >= s) & (f["ts"] < m_start)
                               & ~f["day"].isin(excluded_days)]
                    return {"refs": refs, "curve": power_curve(window)}, {
                        "train_start": str(s)[:10],
                        "train_end": str(m_start)[:10],
                        "train_rows": int(len(tr)),
                        "clean_rows": int(len(clean))}
        if start <= floor:
            return None, None
        start = start - pd.Timedelta(days=STEP_DAYS)


def _forecast_noise(seed_key: str, month: str, lead: int, n_days: int,
                    n_hours: int):
    """The forecast error for one month and lead: a bias per day and noise per
    hour, for wind (m/s) and temperature (K). Deterministic."""
    seed = zlib.crc32(f"{seed_key}|{month}|{lead}".encode())
    rng = np.random.default_rng(seed)
    s = FORECAST_SIGMA[lead]
    return (rng.normal(0, s, n_days), rng.normal(0, 0.5 * s, n_hours),
            rng.normal(0, s, n_days), rng.normal(0, 0.5 * s, n_hours))


def _month_shift(month: str, k: int) -> str:
    y, m = int(month[:4]), int(month[5:7]) - 1 + k
    return f"{y + m // 12:04d}-{m % 12 + 1:02d}"


def last_year_gate_k(f: pd.DataFrame, trained_by_month: dict, model_month: str | None,
                     model_window, excluded: set, channel: str = CHANNEL) -> float | None:
    """Median error of last year's model for the frozen model's own month on
    the frozen model's healthy training readings, K (None: no such model)."""
    prev = trained_by_month.get(_month_shift(model_month, -12)) if model_month else None
    if prev is None or model_window is None or channel not in prev[0]["refs"]:
        return None
    lo, hi = _utc(model_window[0]), _utc(model_window[1])
    w = f[(f["ts"] >= lo) & (f["ts"] < hi) & f["running"] & ~f["warmup"]
          & ~f["day"].isin(excluded)]
    if len(w) < MIN_DAY_ROWS * 7:
        return None
    r = w[channel].to_numpy(dtype="float64") - predict(
        prev[0]["refs"][channel], w["P_norm"].to_numpy(dtype="float64"),
        w["era5_temp_2m"].to_numpy(dtype="float64"))
    r = r[np.isfinite(r)]
    return float(np.median(r)) if len(r) else None


def last_year_ok(f: pd.DataFrame, trained_by_month: dict, model_month: str | None,
                 model_window, excluded: set, channel: str = CHANNEL) -> bool:
    """May last year's models stand in for a frozen one?

    Only if the component behaves as it did a year ago: last year's model for
    the frozen model's own month must predict the frozen model's healthy
    training readings to within LAST_YEAR_GATE_K (median error). A repair
    since then that changed the component fails this, and the frozen model
    is kept."""
    g = last_year_gate_k(f, trained_by_month, model_month, model_window, excluded, channel)
    return g is not None and abs(g) <= LAST_YEAR_GATE_K


def simulate(frame: pd.DataFrame, turbine: str, tolerance: float,
             warmup_hours: float = WARMUP_HOURS, spans: dict | None = None,
             channel: str = CHANNEL,
             maintenance_days: float = MAINTENANCE_DAYS,
             channels=None, leads=(), maintenance_extra=(),
             last_year_when_frozen: bool | None = None,
             last_year_months=()) -> dict:
    """Run the policy over a turbine's whole record, month by month, day by day.

    `channels` adds a healthy model and a daily check for each further sensor
    (the policy itself follows the primary `channel`); `leads` adds the
    forecast of each sensor that many days ahead. Returns the per-day results,
    the per-month state of the model, and the restarts that stood in for
    maintenance. `maintenance_extra` adds maintenance that is RECORDED (a
    failure logbook's repairs, where one exists): the model starts over after
    it exactly as after a stand-in.

    `last_year_when_frozen` (default LAST_YEAR_WHEN_FROZEN): a model frozen
    after a confirmed fault was trained in one season and is used into the
    next, so in a frozen month the turbine is judged with its own healthy
    model for the same month a year earlier, when there is one and it passes
    `last_year_ok`. `last_year_months` does the same in the months named
    whatever the state (the control: a healthy turbine judged that way).
    """
    if last_year_when_frozen is None:
        last_year_when_frozen = LAST_YEAR_WHEN_FROZEN
    f, maintenance = mark_restarts(frame, warmup_hours=warmup_hours,
                                   maintenance_days=maintenance_days)
    if len(maintenance_extra):
        maintenance = sorted(set(maintenance) | {_utc(t) for t in maintenance_extra})
    chans = [channel] + [c for c in (channels or []) if c != channel and c in f.columns]
    usable_all = (f["running"] & ~f["warmup"]).to_numpy()
    cum_usable = np.cumsum(usable_all)
    ts_values = f["ts"].values                 # naive UTC, for searching
    months = sorted(f["ts"].dt.strftime("%Y-%m").unique())
    excluded: set = set()
    model = None
    confirmed = False
    seen_maint = None
    ever_trained = False
    trained_since_maint = False
    model_window = None
    frozen_since = None
    state_by_month: dict = {}
    trained_by_month: dict = {}
    model_month = None
    days_out: list[dict] = []
    months_out: list[dict] = []
    for month in months:
        m_start = _utc(month + "-01")
        m_end = m_start + pd.offsets.MonthBegin(1)
        before = [t for t in maintenance if t < m_start]
        latest = max(before) if before else None
        if latest is not None and latest != seen_maint:
            # Maintenance since the last month: start over.
            model, confirmed, seen_maint, frozen_since = None, False, latest, None
            trained_since_maint = False
        if confirmed and model is not None:
            state, info = "frozen", None
        else:
            floor = m_start - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
            if latest is not None:
                floor = max(floor, latest)
            model, info = train(f, m_start, floor, excluded, channel, chans)
            state = "trained" if model is not None else "blank"
            if model is not None:
                ever_trained = True
                trained_since_maint = True
                model_window = (info["train_start"], info["train_end"])
                model_month = month
                trained_by_month[month] = (model, model_window)
        used, model_from = model, None
        if model is not None and ((last_year_when_frozen and state == "frozen")
                                  or month in last_year_months):
            ly = _month_shift(month, -12)
            if ly in trained_by_month and last_year_ok(
                    f, trained_by_month, model_month, model_window, excluded, channel):
                used, model_from = trained_by_month[ly][0], ly
        if model is None:
            reason = ("maintenance"
                      if seen_maint is not None and not trained_since_maint
                      else "start" if not ever_trained else "no healthy month")
        else:
            reason = None
        months_out.append({"month": month, "model": state, "reason": reason,
                           "model_train_start": model_window[0] if model else None,
                           "model_train_end": model_window[1] if model else None,
                           "model_from": model_from,
                           **(info or {})})

        mm = f[(f["ts"] >= m_start) & (f["ts"] < m_end)]
        inside = [t for t in maintenance if m_start <= t < m_end]
        cut = min(inside) if inside else None
        state_by_month[month] = (used, cut)
        load = mm["P_norm"].to_numpy(dtype="float64")
        amb = mm["era5_temp_2m"].to_numpy(dtype="float64")
        after_cut = ((mm["ts"] >= cut).to_numpy() if cut is not None
                     else np.zeros(len(mm), dtype=bool))
        base_ok = (mm["running"] & ~mm["warmup"]).to_numpy() & ~after_cut
        actual: dict = {}
        pred: dict = {}
        for c in chans:
            actual[c] = mm[c].to_numpy(dtype="float64")
            if used is not None and c in used["refs"] and len(mm):
                pred[c] = predict(used["refs"][c], load, amb)
            else:
                pred[c] = np.full(len(mm), np.nan)
        resid = actual[channel] - pred[channel]
        usable = base_ok & np.isfinite(resid)

        # The forecast of each sensor, `lead` days ahead, by the model in force
        # when the forecast is issued, fed the weather forecast.
        fc: dict = {}
        if leads and len(mm) and "era5_ws_100m" in mm.columns:
            day_idx = ((mm["day"] - m_start).dt.days).to_numpy()
            hour_idx = ((mm["ts"] - m_start) // pd.Timedelta(hours=1)).to_numpy()
            n_days = int(day_idx.max()) + 1
            n_hours = int(hour_idx.max()) + 1
            wind = mm["era5_ws_100m"].to_numpy(dtype="float64")
            for L in leads:
                bw, nw, bt, nt = _forecast_noise(turbine, month, L, n_days, n_hours)
                wind_fc = np.clip(wind + bw[day_idx] + nw[hour_idx], 0.0, None)
                temp_fc = amb + bt[day_idx] + nt[hour_idx]
                issue = mm["day"] - pd.Timedelta(days=L)
                issue_month = issue.dt.strftime("%Y-%m").to_numpy()
                out = {c: np.full(len(mm), np.nan) for c in chans}
                for im in np.unique(issue_month):
                    sel = issue_month == im
                    m_in_force, m_cut = state_by_month.get(im, (None, None))
                    if m_in_force is None or m_in_force.get("curve") is None:
                        continue
                    if m_cut is not None:
                        sel = sel & (issue < m_cut).to_numpy()
                    centres, med = m_in_force["curve"]
                    load_fc = np.interp(wind_fc[sel], centres, med)
                    for c in chans:
                        if c in m_in_force["refs"]:
                            ref_c = m_in_force["refs"][c]
                            # A forecast load past the reference's highest
                            # bin is read at that bin, not left blank.
                            top = (ref_c.medians.index.max() + 1) * multivariate.LOAD_BIN
                            out[c][sel] = predict(ref_c, np.minimum(load_fc, top - 1e-9),
                                                  temp_fc[sel])
                fc[L] = out

        running = mm["running"].to_numpy()
        flagged_fault = False
        for day, idx in mm.groupby("day").indices.items():
            ok = usable[idx]
            n = int(ok.sum())
            fault = in_fault(turbine, day, spans)
            rec = {"day": str(day)[:10], "fault_period": fault, "n": n,
                   "running": bool(running[idx].any())}
            day_after_cut = cut is not None and day >= cut.floor("D")
            if model is None or day_after_cut:
                rec["status"] = "blank"
                rec["reason"] = "maintenance" if day_after_cut else reason
                if rec["reason"] == "maintenance":
                    since = cut if day_after_cut else seen_maint
                    end = idx.max()
                    first = int(np.searchsorted(
                        ts_values, since.tz_convert(None).to_datetime64()))
                    rec["maintenance"] = str(since)[:10]
                    rec["collected"] = int(cum_usable[mm.index[end]]
                                           - (cum_usable[first - 1] if first > 0 else 0))
            elif n < MIN_DAY_ROWS:
                rec["status"] = "thin"
                rec["state"] = state
            else:
                mean = float(np.mean(resid[idx][ok]))
                flag = ("hot" if mean > tolerance
                        else "cold" if mean < -tolerance else None)
                rec.update(status="checked", state=state, mean_k=round(mean, 3),
                           flag=flag,
                           actual_k=round(float(np.mean(actual[channel][idx][ok])), 3),
                           pred_k=round(float(np.mean(pred[channel][idx][ok])), 3))
                if state == "frozen":
                    rec["frozen_since"] = frozen_since
                if model_from:
                    rec["model_from"] = model_from
                if len(chans) > 1:
                    ch_out = {}
                    for c in chans:
                        okc = base_ok[idx] & np.isfinite(actual[c][idx]) & np.isfinite(pred[c][idx])
                        if okc.sum() >= MIN_DAY_ROWS:
                            a = float(np.mean(actual[c][idx][okc]))
                            p = float(np.mean(pred[c][idx][okc]))
                            ch_out[c] = [round(a, 2), round(p, 2)]
                    rec["ch"] = ch_out
                if fc:
                    fc_out = {}
                    for L, out in fc.items():
                        per = {}
                        for c in chans:
                            okc = (base_ok[idx] & np.isfinite(actual[c][idx])
                                   & np.isfinite(pred[c][idx]) & np.isfinite(out[c][idx]))
                            if okc.sum() >= MIN_DAY_ROWS:
                                per[c] = round(float(np.mean(out[c][idx][okc])), 2)
                        if per:
                            fc_out[str(L)] = per
                    rec["fc"] = fc_out
                if flag:
                    excluded.add(day)
                if flag == "hot" and fault:
                    flagged_fault = True
                    if frozen_since is None:
                        frozen_since = str(day)[:10]
            days_out.append(rec)
        if cut is not None:
            model, confirmed, seen_maint, frozen_since = None, False, max(inside), None
            trained_since_maint = False
        elif flagged_fault:
            confirmed = True
    return {"turbine": turbine, "tolerance": tolerance,
            "warmup_hours": warmup_hours, "days": days_out,
            "months": months_out,
            "maintenance": [str(t)[:16] for t in maintenance]}


def call_outs(days: list[dict], flag: str = "hot", key=None) -> list[dict]:
    """Runs of consecutive flagged days: one call-out each, at the end of the
    first day of the run. `key(day)` overrides how a day is judged flagged."""
    judge = key or (lambda d: d.get("flag") == flag)
    out: list[dict] = []
    prev = None
    for d in days:
        if judge(d):
            day = pd.Timestamp(d["day"])
            if prev is not None and day - prev == pd.Timedelta(days=1):
                out[-1]["days"] += 1
            else:
                out.append({"first": d["day"] + " 23:50", "days": 1,
                            "fault_period": d["fault_period"]})
            prev = day
        else:
            prev = None
    return out


def gearbox_hot(d: dict, tolerance: float) -> int:
    """How many gearbox sensors ran hotter than the tolerance on this day."""
    ch = d.get("ch") or {}
    return sum(1 for c in GEARBOX if c in ch and ch[c][0] - ch[c][1] > tolerance)


def limit_call_outs(frame: pd.DataFrame, turbine: str,
                    spans: dict | None = None,
                    channel: str = CHANNEL) -> list[dict]:
    """The fixed limit, held for an hour, on the same record: the comparison."""
    f = frame.sort_values("ts").reset_index(drop=True)
    times = f["ts"].dt.tz_convert(None).to_numpy()
    s, ln = alarm_runs(f[channel].to_numpy(dtype="float64") > HOT_C, times)
    out = []
    for a, n in zip(s, ln):
        if n >= LIMIT_HOLD:
            t = f["ts"].iloc[a + LIMIT_HOLD - 1]
            out.append({"first": str(t)[:16],
                        "fault_period": in_fault(turbine, t.floor("D"), spans)})
    return out


# ------------------------------------------------------ the sibling comparison
def sector_of(direction) -> np.ndarray:
    """The 45-degree wind-direction sector, as an index into SECTORS."""
    d = np.asarray(direction, dtype="float64")
    with np.errstate(invalid="ignore"):
        s = np.floor(((d + 22.5) % 360.0) / 45.0)
    return s


def sibling_change(temps: pd.DataFrame, loads: pd.DataFrame,
                   direction: pd.Series, turbine: str, months: list[dict],
                   by_direction: bool = True,
                   smooth_days: int = SIBLING_SMOOTH_DAYS) -> pd.Series:
    """The smoothed daily change in a turbine's gap to its siblings.

    At each moment the turbine is compared with its siblings' median, only
    where its power is within `SIBLING_LOAD_MATCH` of theirs. Its NORMAL gap
    is learned from the training window of the model in force that month,
    per wind-direction sector when `by_direction` (a turbine can run warmer
    than its siblings in some winds and that is not a fault), else overall.
    The day's change is the median over its comparable moments; it is then
    smoothed by a median over the past `smooth_days` days.
    """
    peers = [p for p in temps.columns if p != turbine]
    gap = temps[turbine] - temps[peers].median(axis=1)
    similar = (((loads[turbine] - loads[peers].median(axis=1)).abs()
                < SIBLING_LOAD_MATCH) & (loads[turbine] > RUNNING) & gap.notna())
    sector = pd.Series(sector_of(direction.reindex(gap.index)), index=gap.index)
    change = pd.Series(np.nan, index=gap.index)
    for m in months:
        if not m.get("model_train_start"):
            continue
        m_start = _utc(m["month"] + "-01")
        m_end = m_start + pd.offsets.MonthBegin(1)
        lo, hi = _utc(m["model_train_start"]), _utc(m["model_train_end"])
        win = similar & (gap.index >= lo) & (gap.index < hi)
        if win.sum() < 10 * SIBLING_MIN_READINGS:
            continue
        overall = float(gap[win].median())
        normal = pd.Series(overall, index=range(len(SECTORS)), dtype="float64")
        if by_direction:
            g = gap[win].groupby(sector[win]).agg(["median", "size"])
            for s, row in g.iterrows():
                if np.isfinite(s) and row["size"] >= SECTOR_MIN:
                    normal[int(s)] = float(row["median"])
        cur = similar & (gap.index >= m_start) & (gap.index < m_end)
        sec = sector[cur]
        norm = sec.map(normal).fillna(overall)
        change[cur] = gap[cur] - norm
    c = change.dropna()
    day = c.groupby(c.index.floor("D")).agg(["median", "size"])
    day = day[day["size"] >= SIBLING_MIN_READINGS]["median"]
    return day.rolling(f"{smooth_days}D", min_periods=2).median()


def sibling_call_outs(change: pd.Series, threshold: float, turbine: str,
                      spans: dict | None = None) -> list[dict]:
    days = [{"day": str(d)[:10], "fault_period": in_fault(turbine, d, spans),
             "flag": "hot" if v > threshold else None}
            for d, v in change.items() if np.isfinite(v)]
    return call_outs(days)


def summarise(sim: dict, limit: list[dict]) -> dict:
    """One turbine, one tolerance: the figures the report quotes."""
    days = sim["days"]
    tol = sim["tolerance"]
    healthy = [d for d in days if not d["fault_period"]]
    fault = [d for d in days if d["fault_period"]]
    checked_h = [d for d in healthy if d.get("status") == "checked"]
    run_h = [d for d in healthy if d["running"]]
    years = len(run_h) / 365.25
    calls = call_outs(days)
    false_calls = [c for c in calls if not c["fault_period"]]
    fault_calls = [c for c in calls if c["fault_period"]]
    cold_h = sum(1 for d in checked_h if d.get("flag") == "cold")
    means = np.array([d["mean_k"] for d in checked_h], dtype="float64")
    fault_checked = [d for d in fault if d.get("status") == "checked"]
    by_month: dict = {}
    for d in fault_checked:
        b = by_month.setdefault(d["day"][:7], [0, 0])
        b[1] += 1
        b[0] += 1 if d.get("flag") == "hot" else 0
    lim_false = [c for c in limit if not c["fault_period"]]
    lim_fault = [c for c in limit if c["fault_period"]]

    def per_year(n):
        return round(n / years, 3) if years > 0 else None

    running = sum(1 for d in days if d["running"])
    calls_by_year: dict = {}
    for c in false_calls:
        calls_by_year[c["first"][:4]] = calls_by_year.get(c["first"][:4], 0) + 1
    hot_by_month: dict = {}
    for d in days:
        if d.get("flag") == "hot":
            hot_by_month[d["day"][:7]] = hot_by_month.get(d["day"][:7], 0) + 1
    blank_reasons: dict = {}
    for d in days:
        if d["running"] and d.get("status") == "blank":
            r = d.get("reason") or "unknown"
            blank_reasons[r] = blank_reasons.get(r, 0) + 1
    out = {
        "turbine": sim["turbine"],
        "tolerance": tol,
        "warmup_hours": sim["warmup_hours"],
        "running_days": running,
        "healthy_running_days": len(run_h),
        "checked_share": round(sum(1 for d in days if d.get("status") == "checked")
                               / max(running, 1), 4),
        "blank_running_days": sum(blank_reasons.values()),
        "blank_running_days_by_reason": blank_reasons,
        "maintenance_restarts": sim["maintenance"],
        "months_trained": sum(1 for m in sim["months"] if m["model"] == "trained"),
        "months_frozen": sum(1 for m in sim["months"] if m["model"] == "frozen"),
        "months_blank": sum(1 for m in sim["months"] if m["model"] == "blank"),
        "healthy_daily_mean_k": {
            "sd": round(float(means.std()), 3) if means.size else None,
            "p95_abs": (round(float(np.quantile(np.abs(means), 0.95)), 3)
                        if means.size else None),
            "p99": (round(float(np.quantile(means, 0.99)), 3)
                    if means.size else None),
        },
        "false_call_outs": len(false_calls),
        "false_call_outs_per_year": per_year(len(false_calls)),
        "false_call_outs_by_year": dict(sorted(calls_by_year.items())),
        "hot_days_by_month": dict(sorted(hot_by_month.items())),
        "cold_days_per_year": per_year(cold_h),
        "fault_call_outs": len(fault_calls),
        "first_fault_call_out": fault_calls[0]["first"] if fault_calls else None,
        "fault_days_checked": len(fault_checked),
        "fault_days_hot": sum(1 for d in fault_checked if d.get("flag") == "hot"),
        "fault_hot_by_month": {k: {"hot": v[0], "checked": v[1]}
                               for k, v in sorted(by_month.items())},
        "limit_false_call_outs_per_year": per_year(len(lim_false)),
        "limit_first_fault_call_out": lim_fault[0]["first"] if lim_fault else None,
    }
    # The multi-sensor rule, where every gearbox sensor was checked.
    if any("ch" in d for d in days):
        multi = call_outs(days, key=lambda d: gearbox_hot(d, tol) >= GEARBOX_AGREE)
        mf = [c for c in multi if not c["fault_period"]]
        mt = [c for c in multi if c["fault_period"]]
        out["gearbox_rule"] = {
            "rule": f"at least {GEARBOX_AGREE} of the {len(GEARBOX)} gearbox "
                    "sensors hotter than the tolerance on the same day",
            "false_call_outs_per_year": per_year(len(mf)),
            "first_fault_call_out": mt[0]["first"] if mt else None,
            "fault_days_hot": sum(1 for d in fault_checked
                                  if gearbox_hot(d, tol) >= GEARBOX_AGREE),
        }
    # The forecast's accuracy, by lead, on healthy checked days: against the
    # same model fed the actual weather (what the weather forecast's error
    # costs) and against the sensor itself.
    if any("fc" in d for d in days):
        acc = {}
        for L in sorted({int(k) for d in days for k in (d.get("fc") or {})}):
            vs_day, vs_actual = [], []
            for d in checked_h:
                v = ((d.get("fc") or {}).get(str(L)) or {}).get(CHANNEL)
                if v is None:
                    continue
                vs_day.append(abs(v - d["pred_k"]))
                vs_actual.append(abs(v - d["actual_k"]))
            if vs_day:
                acc[str(L)] = {"days": len(vs_day),
                               "mae_vs_same_day_k": round(float(np.mean(vs_day)), 3),
                               "mae_vs_sensor_k": round(float(np.mean(vs_actual)), 3)}
        same_day = [abs(d["mean_k"]) for d in checked_h]
        out["forecast_accuracy"] = {
            "channel": CHANNEL,
            "same_day_mae_vs_sensor_k": (round(float(np.mean(same_day)), 3)
                                         if same_day else None),
            "by_lead": acc,
        }
    return out
