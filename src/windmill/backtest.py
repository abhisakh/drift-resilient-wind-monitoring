"""The evaluation harness: the measurements behind every claim.

This project asserts things -- that the weather-conditioned model beats a
naive one, that the reference matters more than the model, that a fault is
caught before it is a trip. This module is where those assertions are
MEASURED, under rules that are enforced in code rather than promised in a
plan.

Eight questions, grouped by the claim each one supports:

  A. Is the temperature prediction any good?
     A.1  the weekly backtest: five arms, one protocol
     A.2  the weather ablation: load, then ambient, then wind
     A.3  the forecast chain: forecast wind -> power -> expected temperature
  B. Do you get fewer false alarms?
     B.4  the matched-threshold comparison, and the age of the reference
     B.5  the cleaning test: does trimming stop the reference absorbing a ramp?
  C. Do you catch real defects early?
     C.6  fault injection: +2, +4 and +8 K ramps, detection rate and lead time
     C.7  the R80711 lead time against the 80 C crossing and the siblings
     C.8  the CARE thermal subset

TWO RULES, ENFORCED HERE
------------------------
**The split.** A window's training rows end strictly before its test rows
begin. `Window` refuses to be constructed otherwise, and `weekly_windows`
is the only way to make one, so a backtest that leaked cannot be written
by accident. `tests/test_backtest.py` checks it independently.

**The matched threshold.** Every arm's alarm threshold is calibrated on
its OWN out-of-fold training residuals at the same target rate. So every
arm alarms on the same share of its own training window, and any
difference in the false-alarm rate on a LATER window is a difference in
how well that arm holds its calibration across weeks -- which is the
thing being compared. Comparing arms at one shared number on one scale
would only measure which model happens to have the bigger residual.

The residual is ONE-SIDED throughout: overheating is the failure mode, so
only a positive residual alarms. A channel running colder than expected is
not evidence of a fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import multivariate, nbm
from .care import LOAD_BIN

#: Rows below this share of rated power are standstill: no load, no
#: reference, and no information about the machine.
LOAD_FRACTION = 0.02

#: Degrees Celsius per wind bin, for the wind step of the weather ablation.
WIND_BIN = 1.0

#: The false-alarm rate every arm is calibrated to on its own training
#: window. One per cent of samples.
DEFAULT_TARGET_RATE = 0.01

#: How many rows a training window needs before it is worth fitting.
MIN_TRAIN_ROWS = 500

#: How many rows a test window needs before it is worth scoring.
MIN_TEST_ROWS = 100



def _utc(ts) -> np.ndarray:
    """A numpy datetime64 array in UTC, naive, for comparisons.

    Pandas keeps the timezone; numpy does not. Comparing the two raises, so
    everything is converted once, here.
    """
    return pd.to_datetime(ts, utc=True).dt.tz_convert(None).to_numpy()


# ------------------------------------------------------------------ splits
@dataclass(frozen=True)
class Window:
    """One train/test split.

    The invariant is checked here, in the constructor, so that a leaked
    split cannot be built at all.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    label: str = ""

    def __post_init__(self) -> None:
        if self.train_end > self.test_start:
            raise ValueError(
                f"train/test overlap: training ends {self.train_end} but "
                f"testing starts {self.test_start}")
        if self.test_end <= self.test_start:
            raise ValueError("the test window is empty")
        if self.train_start >= self.train_end:
            raise ValueError("the training window is empty")

    def as_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "label": self.label,
        }


def weekly_windows(index, train_days: int = 30, test_days: int = 7,
                   step_days: int = 7) -> list[Window]:
    """Every weekly (train, test) split across a series.

    The training window is the `train_days` immediately before the test
    window, and the two never touch: `Window` enforces it.
    """
    idx = pd.DatetimeIndex(index).dropna().sort_values().unique()
    if len(idx) == 0:
        return []
    out: list[Window] = []
    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    test_start = idx[0] + train_delta
    while test_start < idx[-1]:
        train_start = test_start - train_delta
        out.append(Window(
            train_start=train_start,
            train_end=test_start,
            test_start=test_start,
            test_end=test_start + test_delta,
            label=test_start.strftime("%Y-%m-%d"),
        ))
        test_start = test_start + pd.Timedelta(days=step_days)
    return out


def slice_window(df: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training and test rows of `df` for `window`, in time order."""
    ts = pd.to_datetime(df["ts"], utc=True)
    train = df[(ts >= window.train_start) & (ts < window.train_end)]
    test = df[(ts >= window.test_start) & (ts < window.test_end)]
    return train, test



def _operating_rows(df: pd.DataFrame, power_col: str,
                    load_is_fraction: bool, rated_kw: float) -> pd.DataFrame:
    """The rows that carry information about the machine.

    Standstill (no load) has no reference: a turbine that is parked reads
    whatever the weather is doing, not what the gearbox is doing.

    The load column may be a FRACTION of rated power (the project's own
    `P_norm`) or kilowatts, and the two are not interchangeable -- getting
    that wrong silently empties the training set, so the convention is
    stated rather than inferred.
    """
    t = df.dropna(subset=[power_col])
    if load_is_fraction:
        t = t[t[power_col] > LOAD_FRACTION]
    else:
        t = t[t[power_col] > LOAD_FRACTION * rated_kw]
    return t


# -------------------------------------------------------------------- arms
class Arm:
    """One way of predicting a channel's temperature.

    Every arm is fitted on the training window and asked to predict rows it
    has never seen. The arms differ ONLY in what they condition on, so the
    difference between them is the thing being measured.
    """

    name: str = "arm"

    def fit(self, train: pd.DataFrame) -> "Arm":
        raise NotImplementedError

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ConstantArm(Arm):
    """The channel's own median. No operating point, no weather.

    This is the floor: a model that knows nothing but the machine's own
    average temperature.
    """

    channel: str
    power_col: str
    rated_kw: float = 2050.0
    load_is_fraction: bool = True
    name: str = "constant"

    def fit(self, train: pd.DataFrame) -> "ConstantArm":
        t = _operating_rows(train, self.power_col, self.load_is_fraction,
                            self.rated_kw).dropna(subset=[self.channel])
        self.value = float(t[self.channel].median()) if len(t) else np.nan
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), self.value, dtype="float64")


@dataclass
class DeterministicArm(Arm):
    """The project's own model, with the weather ablation built in.

    The reference is the median of the channel per load bin -- the operating
    point. The ambient correction removes the median residual per ambient
    bin, so a hot day does not read as dozens of channels running warm. The
    wind correction does the same per wind-speed bin, which is the third
    step of the ablation.

    Setting `use_ambient` or `use_wind` off removes that step, so the three
    settings of the same class are the three arms of the ablation.
    """

    channel: str
    power_col: str
    ambient_col: str | None = None
    wind_col: str | None = None
    use_ambient: bool = True
    use_wind: bool = False
    rated_kw: float = 2050.0
    load_is_fraction: bool = True
    name: str = "load+ambient"

    def fit(self, train: pd.DataFrame) -> "DeterministicArm":
        cols = [self.channel, self.power_col]
        if self.use_ambient and self.ambient_col:
            cols.append(self.ambient_col)
        t = _operating_rows(train, self.power_col, self.load_is_fraction,
                            self.rated_kw).dropna(subset=cols)
        self.global_median = (float(t[self.channel].median())
                              if len(t) else np.nan)
        self.ref = multivariate.fit_channel_reference(
            t, self.channel, self.power_col,
            ambient_col=(self.ambient_col if self.use_ambient else None))
        self.wind_offsets = pd.Series(dtype="float64")
        if self.use_wind and self.wind_col and self.ref is not None:
            resid = self._residual(t)
            wb = np.floor(t[self.wind_col].to_numpy(dtype="float64")
                          / WIND_BIN).astype("int64")
            frame = pd.DataFrame({"w": wb, "r": resid})
            counts = frame.groupby("w")["r"].size()
            med = frame.groupby("w")["r"].median()
            self.wind_offsets = med.loc[counts[counts >= 25].index]
        return self

    def _load_medians(self, rows: pd.DataFrame) -> np.ndarray:
        bins = np.floor(rows[self.power_col].to_numpy(dtype="float64")
                        / LOAD_BIN).astype("int64")
        med = self.ref.medians.reindex(bins).to_numpy(dtype="float64")
        return np.where(np.isfinite(med), med, self.global_median)

    def _residual(self, rows: pd.DataFrame) -> np.ndarray:
        """Actual minus what the model predicts, on `rows`."""
        pred = self._load_medians(rows)
        if self.use_ambient and self.ambient_col:
            amb = np.floor(rows[self.ambient_col].to_numpy(dtype="float64")
                           / multivariate.AMBIENT_BIN).astype("int64")
            off = self.ref.ambient_offsets.reindex(amb).to_numpy(dtype="float64")
            pred = pred + np.where(np.isfinite(off), off, 0.0)
        if self.use_wind and self.wind_col:
            wb = np.floor(rows[self.wind_col].to_numpy(dtype="float64")
                          / WIND_BIN).astype("int64")
            off = self.wind_offsets.reindex(wb).to_numpy(dtype="float64")
            pred = pred + np.where(np.isfinite(off), off, 0.0)
        return rows[self.channel].to_numpy(dtype="float64") - pred

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.ref is None:
            return np.full(len(rows), np.nan)
        pred = self._load_medians(rows)
        if self.use_ambient and self.ambient_col:
            amb = np.floor(rows[self.ambient_col].to_numpy(dtype="float64")
                           / multivariate.AMBIENT_BIN).astype("int64")
            off = self.ref.ambient_offsets.reindex(amb).to_numpy(dtype="float64")
            pred = pred + np.where(np.isfinite(off), off, 0.0)
        if self.use_wind and self.wind_col:
            wb = np.floor(rows[self.wind_col].to_numpy(dtype="float64")
                          / WIND_BIN).astype("int64")
            off = self.wind_offsets.reindex(wb).to_numpy(dtype="float64")
            pred = pred + np.where(np.isfinite(off), off, 0.0)
        return pred


@dataclass
class LightGBMArm(Arm):
    """The learned arm: LightGBM predicts the channel from the operating
    point and the weather.

    The same features the deployed NBM uses, and the same pinned seeds, so
    this arm is the project's own learned model and not a fresh baseline.
    """

    channel: str
    features: list[str]
    params: dict | None = None
    name: str = "lightgbm"

    def fit(self, train: pd.DataFrame) -> "LightGBMArm":
        import lightgbm as lgb

        t = train.dropna(subset=[self.channel] + self.features)
        if len(t) < 500:
            self.model = None
            return self
        params = dict(self.params or nbm.LGBM_PARAMS)
        params["n_jobs"] = 1
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(t[self.features].to_numpy(dtype="float64"),
                       t[self.channel].to_numpy(dtype="float64"))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        if self.model is None:
            return out
        feat = rows[self.features].to_numpy(dtype="float64")
        ok = np.isfinite(feat).all(axis=1)
        if ok.any():
            out[ok] = self.model.predict(feat[ok])
        return out


# ----------------------------------------------------------------- metrics
def out_of_fold_residuals(make_arm: Callable[[], Arm], train: pd.DataFrame,
                          channel: str, folds: int = 2) -> np.ndarray:
    """The training residuals, from models that never saw the row.

    A reference fitted on a row makes that row's residual smaller than it
    really is, so a threshold taken from in-sample residuals is too strict
    on the rows that follow. Fitting on one half of the window and applying
    to the other removes that bias.
    """
    n = len(train)
    resid = np.full(n, np.nan)
    if n < 2 * 50:
        return resid
    values = train[channel].to_numpy(dtype="float64")
    bounds = np.array_split(np.arange(n), folds)
    for i in range(folds):
        held = bounds[i]
        keep = np.concatenate([bounds[j] for j in range(folds) if j != i])
        if len(held) == 0 or len(keep) == 0:
            continue
        arm = make_arm().fit(train.iloc[keep])
        pred = arm.predict(train.iloc[held])
        resid[held] = values[held] - pred
    return resid


def matched_threshold(train_residuals: np.ndarray,
                      target_rate: float = DEFAULT_TARGET_RATE) -> float:
    """The residual that alarms on `target_rate` of its own training rows.

    One-sided: the threshold is the (1 - rate) quantile of the training
    residual distribution, so by construction `target_rate` of the training
    rows exceed it.
    """
    r = np.asarray(train_residuals, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("inf")
    return float(np.quantile(r, 1.0 - target_rate))


def false_alarm_rate(residuals: np.ndarray, threshold: float) -> float:
    """The share of rows whose residual is above the threshold."""
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    return float((r > threshold).mean())


def mae(residuals: np.ndarray) -> float:
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    return float(np.mean(np.abs(r))) if len(r) else float("nan")


def rmse(residuals: np.ndarray) -> float:
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")


def bias(residuals: np.ndarray) -> float:
    r = np.asarray(residuals, dtype="float64")
    r = r[np.isfinite(r)]
    return float(np.mean(r)) if len(r) else float("nan")


# ------------------------------------------------------------ fault injection
def inject_ramp(df: pd.DataFrame, channel: str, onset, magnitude_k: float,
                ramp_days: float = 3.0, end=None) -> pd.DataFrame:
    """Add a growing fault to `channel`: `magnitude_k` K over `ramp_days`,
    then held.

    This is a SIMULATION. It says what the detector would do if the machine
    really did this, and nothing about whether the machine ever does.
    """
    out = df.copy()
    ts = pd.to_datetime(out["ts"], utc=True)
    on = pd.Timestamp(onset)
    on = on.tz_localize("UTC") if on.tzinfo is None else on.tz_convert("UTC")
    elapsed = (ts - on).dt.total_seconds().to_numpy() / 86400.0
    frac = np.clip(elapsed / max(float(ramp_days), 1e-9), 0.0, 1.0)
    if end is not None:
        stop = pd.Timestamp(end)
        stop = stop.tz_localize("UTC") if stop.tzinfo is None else stop.tz_convert("UTC")
        frac = np.where(ts > stop, 0.0, frac)
    out[channel] = out[channel].to_numpy(dtype="float64") + magnitude_k * frac
    return out


def days_to_first_alarm(residuals: np.ndarray, threshold: float,
                        times: pd.Series) -> float:
    """How long after the onset the first alarm came. NaN if none did."""
    r = np.asarray(residuals, dtype="float64")
    alarm = np.isfinite(r) & (r > threshold)
    if not alarm.any():
        return float("nan")
    first = np.flatnonzero(alarm)[0]
    t0 = pd.to_datetime(times.iloc[0], utc=True)
    t1 = pd.to_datetime(pd.Series(times.to_numpy()[first]), utc=True)
    return float((t1 - t0).total_seconds() / 86400.0)


# ---------------------------------------------------------------- the sweep
@dataclass
class ArmResult:
    """One arm, on one window."""

    arm: str
    label: str
    mae: float
    rmse: float
    bias: float
    threshold: float
    false_alarm_rate: float
    train_rows: int
    test_rows: int

    def as_dict(self) -> dict:
        return {
            "arm": self.arm, "label": self.label,
            "mae_k": round(self.mae, 4) if np.isfinite(self.mae) else None,
            "rmse_k": round(self.rmse, 4) if np.isfinite(self.rmse) else None,
            "bias_k": round(self.bias, 4) if np.isfinite(self.bias) else None,
            "threshold_k": (round(self.threshold, 4)
                            if np.isfinite(self.threshold) else None),
            "false_alarm_rate": (round(self.false_alarm_rate, 6)
                                 if np.isfinite(self.false_alarm_rate) else None),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }


def run_weekly_backtest(df: pd.DataFrame, channel: str, power_col: str,
                        arm_factories: dict[str, Callable[[], Arm]],
                        windows: list[Window] | None = None,
                        target_rate: float = DEFAULT_TARGET_RATE,
                        min_train_rows: int = MIN_TRAIN_ROWS,
                        min_test_rows: int = MIN_TEST_ROWS,
                        ) -> list[ArmResult]:
    """The weekly backtest: every arm, every window, one protocol.

    For each window and each arm:
      * the arm is fitted on the training window and asked to predict the
        test window;
      * the error is the MAE and RMSE of its residual, in the channel's own
        unit (K for a temperature);
      * the arm's threshold is calibrated on its OWN out-of-fold training
        residuals at `target_rate`, and the false-alarm rate is then counted
        on the test window.

    The last part is the point of the matched threshold: every arm is
    calibrated to alarm on 1 % of its own training window, so a higher
    false-alarm rate on a LATER window means that arm's calibration did not
    travel, which is what a stale or weather-blind reference does.
    """
    if windows is None:
        windows = weekly_windows(df["ts"])
    results: list[ArmResult] = []
    for window in windows:
        train, test = slice_window(df, window)
        if len(train) < min_train_rows or len(test) < min_test_rows:
            continue
        values = test[channel].to_numpy(dtype="float64")
        for arm_name, factory in arm_factories.items():
            arm = factory().fit(train)
            pred = arm.predict(test)
            resid = values - pred
            if not np.isfinite(resid).any():
                continue
            oof = out_of_fold_residuals(factory, train, channel)
            thr = matched_threshold(oof, target_rate)
            results.append(ArmResult(
                arm=arm_name, label=window.label,
                mae=mae(resid), rmse=rmse(resid), bias=bias(resid),
                threshold=thr,
                false_alarm_rate=false_alarm_rate(resid, thr),
                train_rows=len(train), test_rows=len(test),
            ))
    return results


def summarise(results: list[ArmResult]) -> dict:
    """Per-arm summary of a backtest: the medians and the spread.

    The MEDIAN of the weekly MAE is the headline, because a single week
    with a power curve that leaves the operating range can move a mean by
    several kelvin.
    """

    def _med(values: np.ndarray, nd: int = 4) -> float | None:
        v = np.asarray(values, dtype="float64")
        v = v[np.isfinite(v)]
        return round(float(np.median(v)), nd) if len(v) else None

    def _mean(values: np.ndarray, nd: int = 6) -> float | None:
        v = np.asarray(values, dtype="float64")
        v = v[np.isfinite(v)]
        return round(float(np.mean(v)), nd) if len(v) else None

    out: dict[str, dict] = {}
    by_arm: dict[str, list[ArmResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)
    for arm, rows in sorted(by_arm.items()):
        maes = np.array([r.mae for r in rows], dtype="float64")
        rmses = np.array([r.rmse for r in rows], dtype="float64")
        biases = np.array([r.bias for r in rows], dtype="float64")
        fas = np.array([r.false_alarm_rate for r in rows], dtype="float64")
        p90 = maes[np.isfinite(maes)]
        out[arm] = {
            "windows": len(rows),
            "mae_k_median": _med(maes),
            "mae_k_p90": (round(float(np.quantile(p90, 0.9)), 4)
                          if len(p90) else None),
            "rmse_k_median": _med(rmses),
            "bias_k_median": _med(biases),
            "false_alarm_rate_median": _med(fas),
            "false_alarm_rate_mean": _mean(fas),
        }
    return out


def paired_delta(results: list[ArmResult], arm_a: str, arm_b: str,
                 metric: str = "mae") -> dict:
    """The week-by-week difference between two arms, on the same windows.

    A paired comparison, because the weeks differ enormously in difficulty:
    an unpaired difference of medians would be swamped by which weeks
    happened to be windy.
    """
    by_label: dict[str, dict[str, ArmResult]] = {}
    for r in results:
        by_label.setdefault(r.label, {})[r.arm] = r
    diffs: list[float] = []
    for label, arms in by_label.items():
        if arm_a not in arms or arm_b not in arms:
            continue
        va = getattr(arms[arm_a], metric)
        vb = getattr(arms[arm_b], metric)
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(va - vb)
    if not diffs:
        return {"pairs": 0}
    d = np.array(diffs, dtype="float64")
    return {
        "pairs": int(len(d)),
        "median_delta": round(float(np.median(d)), 4),
        "mean_delta": round(float(np.mean(d)), 4),
        "share_a_better": round(float((d < 0).mean()), 4),
        "share_a_much_better": round(float((d < -0.5).mean()), 4),
    }


# ------------------------------------------------------- A.3 the forecast chain
def forecast_chain_error(train: pd.DataFrame, test: pd.DataFrame, channel: str,
                         power_col: str, wind_col: str, ambient_col: str,
                         curve=None, forecast_wind: np.ndarray | None = None,
                         forecast_ambient: np.ndarray | None = None,
                         rated_kw: float = 2050.0) -> dict:
    """How much does the FORECAST cost, end to end?

    The chain a deployed system runs: forecast wind and temperature, the
    turbine's own power curve, predicted power, and then the load- and
    ambient-conditioned reference to give an expected temperature. With a
    perfect forecast the chain's error is the reference's own error; with an
    imperfect one the error grows, and the ratio between them is what the
    forecast costs.

    This is A SIMULATION of a forecast: ERA5 -- a reanalysis of what
    actually happened -- plus injected error. It says how sensitive the
    chain is to forecast error, not how good any particular forecast is.
    """
    from . import conditioning

    if curve is None:
        curve = conditioning.fit_power_curve(train, wind_col, power_col)
    arm = DeterministicArm(channel, power_col, ambient_col).fit(train)

    def _chain(wind: np.ndarray, ambient: np.ndarray) -> np.ndarray:
        rows = test.copy()
        rows[wind_col] = wind
        rows[ambient_col] = ambient
        predicted_power = conditioning.apply_power_curve(rows, curve, wind_col)
        # The power curve speaks kilowatts; the reference is conditioned on
        # a FRACTION of rated power. Mixing the two puts every row outside
        # the reference's bins, and the chain silently becomes a constant.
        rows[power_col] = predicted_power / rated_kw
        return arm.predict(rows)

    wind_true = test[wind_col].to_numpy(dtype="float64")
    ambient_true = test[ambient_col].to_numpy(dtype="float64")
    actual = test[channel].to_numpy(dtype="float64")

    perfect = _chain(wind_true, ambient_true)
    out = {
        "perfect_forecast_mae_k": round(mae(actual - perfect), 4),
        "perfect_forecast_rmse_k": round(rmse(actual - perfect), 4),
    }
    if forecast_wind is not None and forecast_ambient is not None:
        forecast = _chain(forecast_wind, forecast_ambient)
        out["forecast_mae_k"] = round(mae(actual - forecast), 4)
        out["forecast_rmse_k"] = round(rmse(actual - forecast), 4)
        if out["perfect_forecast_mae_k"] > 0:
            out["amplification"] = round(
                out["forecast_mae_k"] / out["perfect_forecast_mae_k"], 3)
    return out


# ----------------------------------------------- B.4 the age of the reference
def reference_age_sweep(df: pd.DataFrame, channel: str, power_col: str,
                         ambient_col: str, windows: list[Window],
                         ages_days: tuple[int, ...] = (30, 90, 180, 365, 730),
                         ) -> dict:
    """The false-alarm ratio against the AGE of the reference.

    One number -- "a frozen reference alarms N times more than a rolling
    one" -- depends entirely on how old the frozen reference is. A
    reference a year old and a reference three years old are not the same
    claim, and the project's headline ratio is the three-year one. So the
    ratio is measured against the age, on the same windows, with the same
    matched rule, and every arm is calibrated to 1 % of its own training
    window.
    """
    out: dict[str, dict] = {}
    for age in ages_days:
        rates: list[float] = []
        for window in windows:
            train, test = slice_window(df, window)
            if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
                continue
            shift = pd.Timedelta(days=age)
            ts_all = pd.to_datetime(df["ts"], utc=True)
            older = df[(ts_all >= window.train_start - shift)
                       & (ts_all < window.train_end - shift)]
            if len(older) < MIN_TRAIN_ROWS:
                continue
            ref = multivariate.fit_channel_reference(
                older, channel, power_col, ambient_col=ambient_col)
            if ref is None:
                continue
            ambient = test[ambient_col].to_numpy(dtype="float64")
            corrected, _ = ref.apply(
                test[power_col].to_numpy(dtype="float64"), ambient,
                test[channel].to_numpy(dtype="float64"))
            thr = matched_threshold(
                _residual_proxy(older, ref, channel, power_col, ambient_col),
                DEFAULT_TARGET_RATE)
            rates.append(false_alarm_rate(corrected, thr))
        if rates:
            arr = np.array(rates, dtype="float64")
            arr = arr[np.isfinite(arr)]
            out[f"{age}d"] = {
                "windows": int(len(rates)),
                "false_alarm_rate_median": (round(float(np.median(arr)), 6)
                                            if len(arr) else None),
                "false_alarm_rate_mean": (round(float(np.mean(arr)), 6)
                                           if len(arr) else None),
            }
    base = out.get("30d", {}).get("false_alarm_rate_median")
    for age_key, stats in out.items():
        rate = stats.get("false_alarm_rate_median")
        if base and rate:
            stats["ratio_over_30d"] = round(rate / base, 2)
    return out


def reference_age_test(df: pd.DataFrame, channel: str, power_col: str,
                        ambient_col: str, windows: list[Window],
                        train_days: int = 30, train_end_offset_days: int = 0,
                        ) -> dict:
    """Frozen versus rolling reference, under one rule.

    The project's central claim is that the age of the reference matters
    more than the model. This measures it directly: the SAME detector, the
    SAME alarm rule and the SAME windows, with only the reference's age
    changed -- a reference frozen on a window `train_end_offset_days` in
    the past, against one fitted on the thirty days immediately before the
    test week.

    Both are scored on the same test weeks, and the alarm is the same
    one-sided rule, so the difference is the reference and nothing else.
    """
    out: dict[str, dict] = {}
    # The two references are scored on the SAME windows: a window the frozen
    # reference cannot reach (because the year before it is not in the data)
    # is dropped from both, so the ratio compares like with like.
    usable: list[Window] = []
    for window in windows:
        train, _ = slice_window(df, window)
        if len(train) < MIN_TRAIN_ROWS:
            continue
        if train_end_offset_days:
            shift = pd.Timedelta(days=train_end_offset_days)
            ts_all = pd.to_datetime(df["ts"], utc=True)
            older = df[(ts_all >= window.train_start - shift)
                       & (ts_all < window.train_end - shift)]
            if len(older) < MIN_TRAIN_ROWS:
                continue
        usable.append(window)

    for label, offset in (("frozen", train_end_offset_days),
                          ("rolling", 0)):
        rates: list[float] = []
        for window in usable:
            train, test = slice_window(df, window)
            if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
                continue
            if offset:
                # Move the training window further into the past. The rows
                # have to be taken from the WHOLE series: the shifted range
                # lies entirely before the original window, so slicing the
                # already-sliced training frame would find nothing.
                shift = pd.Timedelta(days=offset)
                ts_all = pd.to_datetime(df["ts"], utc=True)
                train = df[(ts_all >= window.train_start - shift)
                           & (ts_all < window.train_end - shift)]
                if len(train) < MIN_TRAIN_ROWS:
                    continue
            ref = multivariate.fit_channel_reference(
                train, channel, power_col, ambient_col=ambient_col)
            if ref is None:
                continue
            ambient = test[ambient_col].to_numpy(dtype="float64")
            corrected, z = ref.apply(
                test[power_col].to_numpy(dtype="float64"), ambient,
                test[channel].to_numpy(dtype="float64"))
            # The threshold is calibrated on the training window, so the two
            # references are compared at the same rule.
            thr = matched_threshold(_residual_proxy(train, ref, channel,
                                                    power_col, ambient_col),
                                    DEFAULT_TARGET_RATE)
            rates.append(false_alarm_rate(corrected, thr))
        if rates:
            arr = np.array(rates, dtype="float64")
            arr = arr[np.isfinite(arr)]
            out[label] = {
                "windows": int(len(rates)),
                "false_alarm_rate_median": (round(float(np.median(arr)), 6)
                                            if len(arr) else None),
                "false_alarm_rate_mean": (round(float(np.mean(arr)), 6)
                                           if len(arr) else None),
            }
    if "frozen" in out and "rolling" in out:
        f = out["frozen"]["false_alarm_rate_median"]
        r = out["rolling"]["false_alarm_rate_median"]
        if f and r:
            out["ratio_frozen_over_rolling"] = round(f / r, 2)
    return out


def _residual_proxy(train: pd.DataFrame, ref, channel: str, power_col: str,
                    ambient_col: str | None) -> np.ndarray:
    """The training residual under a fitted reference, for calibration."""
    ambient = (train[ambient_col].to_numpy(dtype="float64")
               if ambient_col else np.zeros(len(train)))
    corrected, _ = ref.apply(train[power_col].to_numpy(dtype="float64"),
                             ambient, train[channel].to_numpy(dtype="float64"))
    return corrected


# ------------------------------------------------ B.5 the cleaning test
def trimming_test(train: pd.DataFrame, channel: str, power_col: str,
                  ambient_col: str | None, healthy: pd.DataFrame,
                  ramp_k: float = 3.0, ramp_days: float = 10.0,
                  trim_k: float = 4.0) -> dict:
    """Does trimming stop the reference absorbing a growing fault?

    A ramp of `ramp_k` K is injected into the last `ramp_days` of the
    training window. The untrimmed reference includes those rows, so it
    moves toward the fault and the fault stops looking like a fault. The
    trimmed reference drops the rows the reference itself calls anomalous
    -- decided OUT OF FOLD, so the decision is not made with the
    reference's own rows -- and refits, so the fault stays visible.

    Two numbers are reported, and they are different questions:

      * how far the REFERENCE MOVED, in kelvin, toward the fault;
      * how much the fault STILL STANDS OUT on the faulted rows, in
        kelvin. That is the number the detector actually sees.
    """
    onset = (pd.to_datetime(train["ts"], utc=True).max()
             - pd.Timedelta(days=ramp_days))
    faulted = inject_ramp(train, channel, onset, ramp_k, ramp_days=ramp_days)
    ramped = (pd.to_datetime(faulted["ts"], utc=True) >= onset).to_numpy()

    clean_ref = multivariate.fit_channel_reference(
        healthy, channel, power_col, ambient_col=ambient_col)
    if clean_ref is None:
        return {"error": "the healthy window could not be modelled"}

    def _measure(fit_rows: pd.DataFrame,
                 eval_rows: pd.DataFrame) -> tuple[float, float]:
        """(how far the reference moved, how much the fault stands out).

        The reference is fitted on `fit_rows` and then evaluated on
        `eval_rows` -- always the same faulted window, so the trimmed and
        untrimmed references are compared on the same rows.
        """
        ref = multivariate.fit_channel_reference(
            fit_rows, channel, power_col, ambient_col=ambient_col)
        if ref is None:
            return float("nan"), float("nan")
        ambient = (eval_rows[ambient_col].to_numpy(dtype="float64")
                   if ambient_col else np.zeros(len(eval_rows)))
        load = eval_rows[power_col].to_numpy(dtype="float64")
        values = eval_rows[channel].to_numpy(dtype="float64")
        corrected, _ = ref.apply(load, ambient, values)
        # How far the reference moved: its own prediction on the faulted
        # rows, against the healthy reference's prediction on the same rows.
        pred_faulted = values - corrected
        pred_clean = clean_ref.apply(load, ambient, values)[0] + values
        shift = float(np.nanmedian(pred_clean[ramped] - pred_faulted[ramped])) \
            if ramped.any() else float("nan")
        stands_out = float(np.nanmedian(corrected[ramped])) \
            if ramped.any() else float("nan")
        return shift, stands_out

    # The trimmed reference: drop the rows the reference calls anomalous on
    # the FAULTED window, judged out of fold, then refit.
    oof = _out_of_fold_abs_residual(faulted, channel, power_col, ambient_col)
    keep = np.isfinite(oof) & (oof <= trim_k)
    trimmed = faulted[keep] if keep.sum() >= MIN_TRAIN_ROWS else faulted

    shift_untrimmed, stands_untrimmed = _measure(faulted, faulted)
    shift_trimmed, stands_trimmed = _measure(trimmed, faulted)
    return {
        "ramp_k": ramp_k,
        "ramp_days": ramp_days,
        "trim_k": trim_k,
        "rows_dropped_by_trimming": int((~keep).sum()),
        "untrimmed_reference_shift_k": round(shift_untrimmed, 4),
        "trimmed_reference_shift_k": round(shift_trimmed, 4),
        "untrimmed_fault_stands_out_k": round(stands_untrimmed, 4),
        "trimmed_fault_stands_out_k": round(stands_trimmed, 4),
    }


def _out_of_fold_abs_residual(train: pd.DataFrame, channel: str,
                              power_col: str, ambient_col: str | None) -> np.ndarray:
    """The absolute out-of-fold residual of every training row, in the
    channel's own unit (kelvin), so a trim threshold is in kelvin too."""
    n = len(train)
    out = np.full(n, np.nan)
    half = n // 2
    if half < 50:
        return out
    values = train[channel].to_numpy(dtype="float64")
    for fit_part, held in ((train.iloc[half:], train.iloc[:half]),
                            (train.iloc[:half], train.iloc[half:])):
        ref = multivariate.fit_channel_reference(fit_part, channel, power_col,
                                                 ambient_col=ambient_col)
        if ref is None:
            continue
        ambient = (held[ambient_col].to_numpy(dtype="float64")
                   if ambient_col else np.zeros(len(held)))
        corrected, _ = ref.apply(held[power_col].to_numpy(dtype="float64"),
                                 ambient, held[channel].to_numpy(dtype="float64"))
        out[held.index.to_numpy() - train.index.to_numpy()[0]] = np.abs(corrected)
    return out


def absorption_days(df: pd.DataFrame, channel: str, power_col: str,
                    ambient_col: str | None, fault_k: float = 3.0,
                    window_days: int = 30, tolerance: float = 0.25) -> float:
    """How many days until a rolling reference has ABSORBED a persistent fault.

    A fault that stays is eventually part of "normal" for any reference
    that keeps re-estimating itself: the trailing window fills with
    faulted data and the reference moves to meet the fault. The number of
    days is the blind spot of the rolling reference, and it is a property
    of the window length, not of the detector.
    """
    onset = pd.to_datetime(df["ts"], utc=True).min() + pd.Timedelta(days=window_days)
    faulted = inject_ramp(df, channel, onset, fault_k, ramp_days=1.0)
    resid, _ = multivariate.rolling_reference_scores(
        faulted, channel, power_col, ambient_col=ambient_col,
        window_days=window_days)
    ts = _utc(faulted["ts"])
    after = ts >= np.datetime64(onset.tz_convert(None))
    if not after.any():
        return float("nan")
    med = float(np.nanmedian(resid[after]))
    # The reference has absorbed the fault when the residual it reports for
    # the faulted period has shrunk to a quarter of the fault.
    if abs(med) < tolerance * fault_k:
        return 0.0
    hit = np.flatnonzero(np.abs(resid[after]) <= tolerance * fault_k)
    if len(hit) == 0:
        return float("nan")
    return float((ts[after][hit[0]] - np.datetime64(onset.tz_convert(None)))
                 / np.timedelta64(1, "D"))


# ------------------------------------------------- C.6 fault injection
def fault_injection_test(df: pd.DataFrame, channel: str, power_col: str,
                         ambient_col: str | None,
                         arm_factories: dict[str, Callable[[], Arm]],
                         windows: list[Window],
                         magnitudes: tuple[float, ...] = (2.0, 4.0, 8.0),
                         onset_day: float = 7.0, ramp_days: float = 2.0,
                         target_rate: float = DEFAULT_TARGET_RATE) -> list[dict]:
    """Inject a ramp into a healthy test window and see how fast it is caught.

    A SIMULATION. It says what the detector would do if the machine really
    did this; it says nothing about whether the machine ever does.

    For every window and every ramp size, the arm is fitted on the training
    window, its threshold is calibrated on its own out-of-fold training
    residuals at `target_rate` -- the same matched threshold every other
    test uses -- and the ramp is then injected into the test window at a
    known onset. What is measured is whether the fault was caught at all,
    and how many days after the onset the first alarm came.
    """
    out: list[dict] = []
    for window in windows:
        train, test = slice_window(df, window)
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        onset = pd.to_datetime(test["ts"], utc=True).min() + pd.Timedelta(days=onset_day)
        for arm_name, factory in arm_factories.items():
            arm = factory().fit(train)
            oof = out_of_fold_residuals(factory, train, channel)
            thr = matched_threshold(oof, target_rate)
            for magnitude in magnitudes:
                faulted = inject_ramp(test, channel, onset, magnitude,
                                      ramp_days=ramp_days)
                actual = faulted[channel].to_numpy(dtype="float64")
                pred = arm.predict(faulted)
                resid = actual - pred
                after = _utc(faulted["ts"]) >= np.datetime64(
                    onset.tz_convert(None))
                detected = False
                days: float = float("nan")
                if after.any():
                    tail = resid[after]
                    hit = np.flatnonzero(np.isfinite(tail) & (tail > thr))
                    detected = len(hit) > 0
                    if detected:
                        t_first = _utc(faulted["ts"])[after][hit[0]]
                        days = float((t_first - np.datetime64(
                            onset.tz_convert(None))) / np.timedelta64(1, "D"))
                out.append({
                    "arm": arm_name,
                    "label": window.label,
                    "magnitude_k": magnitude,
                    "detected": bool(detected),
                    "days_to_first_alarm": (round(days, 3)
                                            if np.isfinite(days) else None),
                    "threshold_k": round(float(thr), 4),
                })
    return out


def summarise_fault_injection(rows: list[dict]) -> dict:
    """Per arm and per ramp size: the detection rate and the lead time."""
    out: dict[str, dict] = {}
    keys = sorted({(r["arm"], r["magnitude_k"]) for r in rows})
    for arm, magnitude in keys:
        sel = [r for r in rows if r["arm"] == arm and r["magnitude_k"] == magnitude]
        det = [r for r in sel if r["detected"]]
        days = [r["days_to_first_alarm"] for r in det
                if r["days_to_first_alarm"] is not None]
        out[f"{arm}|{magnitude:g}k"] = {
            "arm": arm,
            "magnitude_k": magnitude,
            "trials": len(sel),
            "detection_rate": round(len(det) / len(sel), 4) if sel else None,
            "median_days_to_first_alarm": (round(float(np.median(days)), 3)
                                           if days else None),
            "p90_days_to_first_alarm": (round(float(np.quantile(days, 0.9)), 3)
                                        if days else None),
        }
    return out
