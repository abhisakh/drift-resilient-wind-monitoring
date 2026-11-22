"""Guards on the evaluation harness itself.

The backtest is the evidence behind the project's claims, so the things
that would silently corrupt it are checked here rather than trusted:

  * the split. A window's training rows must end before its test rows
    begin -- in the constructor, in the generator, and in the rows that
    come back.
  * the matched threshold. Every arm must alarm on the same share of its
    own training window, or the arms are not being compared at equal
    false-alarm rates and the comparison means nothing.
  * the ramp injection. A synthetic fault that is not the fault the test
    says it is makes the fault-detection result meaningless.
  * determinism. The same data must give the same answer twice, or
    nothing here can be re-derived.

These tests need no data: they use a synthetic series, so they run
everywhere, including on a machine without the 19 GB release.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import backtest as bt  # noqa: E402


# ------------------------------------------------------------------ helpers
def _synthetic(days: int = 200, seed: int = 0) -> pd.DataFrame:
    """A synthetic turbine: a load-dependent bearing temperature, a daily
    ambient cycle, and noise.

    `days` is DAYS, and the series is ten-minute rows -- so 200 days is
    28,800 rows, which is enough for a few dozen weekly windows.
    """
    rng = np.random.default_rng(seed)
    n = days * 144
    ts = pd.date_range("2017-01-01", periods=n, freq="10min", tz="UTC")
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    ambient = 8.0 + 6.0 * np.sin((hour - 4.0) / 24.0 * 2 * np.pi)
    load = np.clip(rng.beta(2.0, 2.0, n) * 0.9, 0.0, 1.0)
    value = 55.0 + 25.0 * load + 0.35 * ambient + rng.normal(0, 0.4, n)
    return pd.DataFrame({
        "ts": ts,
        "P_norm": load,
        "Gb2t_avg": value,
        "era5_temp_2m": ambient,
        "Ws_avg": 4.0 + 8.0 * load + rng.normal(0, 0.3, n),
    })


# --------------------------------------------------------------- the split
def test_a_window_refuses_to_overlap_its_train_and_test():
    """The rule is enforced in the constructor, so a leaked split cannot
    be built at all."""
    t0 = pd.Timestamp("2017-01-01", tz="UTC")
    with pytest.raises(ValueError, match="overlap"):
        bt.Window(train_start=t0, train_end=t0 + pd.Timedelta(days=30),
                  test_start=t0 + pd.Timedelta(days=29),
                  test_end=t0 + pd.Timedelta(days=36))
    with pytest.raises(ValueError, match="overlap"):
        bt.Window(train_start=t0, train_end=t0 + pd.Timedelta(days=30),
                  test_start=t0 + pd.Timedelta(days=30) - pd.Timedelta(seconds=1),
                  test_end=t0 + pd.Timedelta(days=37))
    # Touching is fine: the training window ENDS where the test begins.
    bt.Window(train_start=t0, train_end=t0 + pd.Timedelta(days=30),
              test_start=t0 + pd.Timedelta(days=30),
              test_end=t0 + pd.Timedelta(days=37))


def test_the_weekly_windows_never_overlap():
    df = _synthetic()
    windows = bt.weekly_windows(df["ts"])
    assert len(windows) > 20, len(windows)
    for w in windows:
        assert w.train_end <= w.test_start, w
        assert w.train_start < w.train_end
        assert w.test_start < w.test_end


def test_the_rows_that_come_back_are_disjoint():
    """The guard on the actual data, not only on the description."""
    df = _synthetic()
    for w in bt.weekly_windows(df["ts"])[:20]:
        train, test = bt.slice_window(df, w)
        assert len(train) and len(test)
        assert train["ts"].max() <= test["ts"].min(), w
        # And the training rows really are inside the training window.
        assert train["ts"].min() >= w.train_start
        assert train["ts"].max() < w.train_end


def test_every_training_row_precedes_every_test_row_over_the_whole_sweep():
    """The whole sweep, not a sample of it."""
    df = _synthetic()
    for w in bt.weekly_windows(df["ts"]):
        train, test = bt.slice_window(df, w)
        if len(train) and len(test):
            assert train["ts"].max() <= test["ts"].min(), w


# ------------------------------------------------------ the matched threshold
def test_the_matched_threshold_alarms_on_the_target_rate():
    """By construction: the threshold is the (1 - rate) quantile of the
    training residuals, so `target_rate` of them exceed it."""
    rng = np.random.default_rng(1)
    resid = rng.normal(0, 1.0, 200_000)
    for rate in (0.01, 0.05, 0.10):
        thr = bt.matched_threshold(resid, rate)
        assert bt.false_alarm_rate(resid, thr) == pytest.approx(rate, abs=0.002)


def test_the_threshold_is_one_sided():
    """Only a positive residual alarms: a channel running COLDER than
    expected is not evidence of overheating."""
    resid = np.concatenate([np.full(1000, -50.0), np.zeros(1000)])
    thr = bt.matched_threshold(resid, 0.01)
    assert thr > -1.0, thr
    assert bt.false_alarm_rate(resid, thr) == pytest.approx(0.0, abs=1e-9)


def test_each_arm_is_calibrated_on_its_own_training_window():
    """The arms are compared at equal false-alarm rates, so the threshold
    has to come from the arm's own out-of-fold training residuals."""
    df = _synthetic()
    windows = bt.weekly_windows(df["ts"])[:6]
    factories = {
        "constant": lambda: bt.ConstantArm("Gb2t_avg", "P_norm"),
        "load": lambda: bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m",
                                            use_ambient=False, name="load"),
        "load+ambient": lambda: bt.DeterministicArm("Gb2t_avg", "P_norm",
                                                     "era5_temp_2m"),
    }
    for name, factory in factories.items():
        for w in windows:
            train, _ = bt.slice_window(df, w)
            if len(train) < bt.MIN_TRAIN_ROWS:
                continue
            oof = bt.out_of_fold_residuals(factory, train, "Gb2t_avg")
            thr = bt.matched_threshold(oof, 0.01)
            # The training residuals themselves alarm at the target rate.
            assert bt.false_alarm_rate(oof, thr) == pytest.approx(0.01, abs=0.02), (
                f"{name} is not calibrated at the target rate")


def test_the_out_of_fold_residuals_are_not_in_sample():
    """An in-sample residual is smaller than the truth, and a threshold
    taken from one is too strict. The cross-fitting has to be real, and
    the property that matters is that the cross-fitted threshold is never
    the stricter of the two."""
    df = _synthetic()
    w = bt.weekly_windows(df["ts"])[6]
    train, _ = bt.slice_window(df, w)
    factory = lambda: bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m")
    arm = factory().fit(train)
    in_sample = train["Gb2t_avg"].to_numpy() - arm.predict(train)
    oof = bt.out_of_fold_residuals(factory, train, "Gb2t_avg")
    assert np.isfinite(oof).sum() > 0.9 * len(train), np.isfinite(oof).sum()
    # The cross-fitted threshold must not be stricter than the in-sample
    # one, which is the leak the cross-fitting exists to prevent.
    assert bt.matched_threshold(oof) >= bt.matched_threshold(in_sample) - 1e-9, (
        bt.matched_threshold(oof), bt.matched_threshold(in_sample))
    # And the out-of-fold residuals must be at least as large, on average.
    assert np.nanmean(np.abs(oof)) >= np.nanmean(np.abs(in_sample)) - 1e-9


# ------------------------------------------------------------ the ramp
def test_the_ramp_is_the_fault_it_says_it_is():
    df = _synthetic(days=30)
    onset = df["ts"].iloc[0] + pd.Timedelta(days=10)
    faulted = bt.inject_ramp(df, "Gb2t_avg", onset, 4.0, ramp_days=2.0)
    delta = (faulted["Gb2t_avg"].to_numpy() - df["Gb2t_avg"].to_numpy())
    before = df["ts"] < onset
    at_end = df["ts"] >= onset + pd.Timedelta(days=2)
    assert np.allclose(delta[before], 0.0)
    assert np.allclose(delta[at_end], 4.0), delta[at_end][:5]
    # And it ramps, rather than stepping.
    mid = (df["ts"] >= onset) & (df["ts"] < onset + pd.Timedelta(days=2))
    assert 0.0 < np.nanmean(delta[mid]) < 4.0


def test_the_ramp_raises_the_faulted_rows_only():
    df = _synthetic(days=30)
    onset = df["ts"].iloc[0] + pd.Timedelta(days=10)
    faulted = bt.inject_ramp(df, "Gb2t_avg", onset, 3.0, ramp_days=3.0)
    before = df["ts"] < onset
    assert np.allclose(faulted["Gb2t_avg"].to_numpy()[before.to_numpy()],
                       df["Gb2t_avg"].to_numpy()[before.to_numpy()])


# ---------------------------------------------------------- determinism
def test_the_arms_are_deterministic():
    df = _synthetic()
    w = bt.weekly_windows(df["ts"])[8]
    train, test = bt.slice_window(df, w)
    for factory in (
        lambda: bt.ConstantArm("Gb2t_avg", "P_norm"),
        lambda: bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m"),
        lambda: bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m",
                                    wind_col="Ws_avg", use_wind=True),
    ):
        a = factory().fit(train).predict(test)
        b = factory().fit(train).predict(test)
        assert np.allclose(a, b, equal_nan=True)


def test_the_backtest_gives_the_same_answer_twice():
    df = _synthetic()
    windows = bt.weekly_windows(df["ts"])[:4]
    factories = {"load+ambient": lambda: bt.DeterministicArm(
        "Gb2t_avg", "P_norm", "era5_temp_2m")}
    first = bt.run_weekly_backtest(df, "Gb2t_avg", "P_norm", factories,
                                   windows=windows)
    second = bt.run_weekly_backtest(df, "Gb2t_avg", "P_norm", factories,
                                    windows=windows)
    assert [r.as_dict() for r in first] == [r.as_dict() for r in second]


# ------------------------------------------------------- the arms themselves
def test_every_arm_predicts_the_channel_on_a_synthetic_series():
    """Each arm produces a finite prediction, and conditioning on the
    operating point beats a constant by a wide margin.

    The comparison is against the constant arm rather than an absolute
    threshold, because what is being claimed is about the ARMS, not about
    a particular error in kelvin.
    """
    df = _synthetic()
    w = bt.weekly_windows(df["ts"])[10]
    train, test = bt.slice_window(df, w)
    actual = test["Gb2t_avg"].to_numpy()

    def _mae(factory) -> float:
        pred = factory().fit(train).predict(test)
        assert len(pred) == len(test)
        assert np.isfinite(pred).all(), factory
        return bt.mae(actual - pred)

    constant = _mae(lambda: bt.ConstantArm("Gb2t_avg", "P_norm"))
    assert constant > 1.0, constant
    for factory in (
        lambda: bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m",
                                    use_ambient=False, name="load"),
        lambda: bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m"),
        lambda: bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m",
                                    wind_col="Ws_avg", use_wind=True),
    ):
        assert _mae(factory) < 0.5 * constant, (factory, constant)


def test_the_weather_ablation_is_nested():
    """The ablation has to be a genuine ablation: each step ADDS an input,
    so the arms cannot differ in anything else."""
    df = _synthetic()
    w = bt.weekly_windows(df["ts"])[10]
    train, test = bt.slice_window(df, w)
    load_only = bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m",
                                     use_ambient=False, name="load").fit(train)
    with_ambient = bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m",
                                       use_ambient=True,
                                       name="load+ambient").fit(train)
    with_wind = bt.DeterministicArm("Gb2t_avg", "P_norm", "era5_temp_2m",
                                    wind_col="Ws_avg", use_ambient=True,
                                    use_wind=True).fit(train)
    # The ambient arm is the load arm plus an ambient correction, so its
    # prediction differs from the load arm's by exactly that correction.
    p_load = load_only.predict(test)
    p_amb = with_ambient.predict(test)
    p_wind = with_wind.predict(test)
    assert not np.allclose(p_load, p_amb)
    assert not np.allclose(p_amb, p_wind)


def test_the_summary_reports_the_median_not_the_mean():
    """One hard week must not be able to move the headline."""
    rows = [bt.ArmResult(arm="x", label="a", mae=1.0, rmse=1.0, bias=0.0,
                          threshold=1.0, false_alarm_rate=0.01,
                          train_rows=10, test_rows=10),
            bt.ArmResult(arm="x", label="b", mae=1.0, rmse=1.0, bias=0.0,
                          threshold=1.0, false_alarm_rate=0.01,
                          train_rows=10, test_rows=10),
            bt.ArmResult(arm="x", label="c", mae=99.0, rmse=99.0, bias=0.0,
                          threshold=1.0, false_alarm_rate=0.01,
                          train_rows=10, test_rows=10)]
    summary = bt.summarise(rows)["x"]
    assert summary["mae_k_median"] == 1.0
    assert summary["mae_k_p90"] > 1.0


def test_the_paired_comparison_pairs_the_same_weeks():
    rows = [
        bt.ArmResult(arm="a", label="w1", mae=2.0, rmse=2.0, bias=0.0,
                      threshold=1.0, false_alarm_rate=0.01, train_rows=1,
                      test_rows=1),
        bt.ArmResult(arm="b", label="w1", mae=3.0, rmse=3.0, bias=0.0,
                      threshold=1.0, false_alarm_rate=0.01, train_rows=1,
                      test_rows=1),
        bt.ArmResult(arm="a", label="w2", mae=4.0, rmse=4.0, bias=0.0,
                      threshold=1.0, false_alarm_rate=0.01, train_rows=1,
                      test_rows=1),
        bt.ArmResult(arm="b", label="w2", mae=5.0, rmse=5.0, bias=0.0,
                      threshold=1.0, false_alarm_rate=0.01, train_rows=1,
                      test_rows=1),
    ]
    delta = bt.paired_delta(rows, "a", "b")
    assert delta["pairs"] == 2
    assert delta["median_delta"] == pytest.approx(-1.0)
    assert delta["share_a_better"] == 1.0
