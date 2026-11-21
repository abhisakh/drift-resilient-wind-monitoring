"""Unit tests for the deterministic multivariate detector (Option 1).

Run with:  .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import multivariate  # noqa: E402


def _synthetic(n: int = 4000, hot_after: int | None = None,
               ambient: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A turbine whose component temperature follows load (+ ambient)."""
    rng = np.random.default_rng(0)
    load = rng.uniform(0.05, 1.0, n)
    amb = rng.uniform(-5, 30, n) if ambient else np.zeros(n)
    ts = pd.date_range("2015-01-01", periods=n, freq="10min")
    idx = pd.RangeIndex(n)

    train = pd.DataFrame({"P_norm": load, "amb": amb, "ts": ts}, index=idx)
    test = pd.DataFrame({"P_norm": load, "amb": amb, "ts": ts}, index=idx)
    # Component temperature: rises with load, with ambient, plus noise.
    for df in (train, test):
        df["temp"] = (40 + 30 * df["P_norm"] + 0.25 * df["amb"]
                      + rng.normal(0, 0.5, n))
    if hot_after is not None:
        test.loc[hot_after:, "temp"] += 8.0   # a fault: 8 K above reference
    return train, test


def test_reference_recovers_a_clean_linear_relation():
    """On clean data the residual must be centred on zero."""
    train, test = _synthetic()
    ref = multivariate.fit_channel_reference(train, "temp", "P_norm")
    assert ref is not None
    resid, z = ref.apply(test["P_norm"].to_numpy(),
                         np.zeros(len(test)),
                         test["temp"].to_numpy())
    assert abs(np.nanmean(resid)) < 0.5
    assert np.nanstd(resid) < 1.0


def test_detector_flags_a_thermal_fault():
    """A channel running 8 K above its reference must be flagged."""
    train, test = _synthetic(hot_after=2000)
    res = multivariate.detect(train, test, ["temp"], "P_norm", z_threshold=4.0)
    assert "temp" in res.channels_used
    assert res.flags_max.to_numpy()[2000:].mean() > 0.9
    assert res.flags_max.to_numpy()[:2000].mean() < 0.05


def test_detector_does_not_flag_healthy_data():
    """On clean data the flag rate must stay near zero."""
    train, test = _synthetic()
    res = multivariate.detect(train, test, ["temp"], "P_norm", z_threshold=4.0)
    assert res.flags_max.to_numpy().mean() < 0.02


def test_ambient_correction_sharpens_a_real_fault():
    """Ambient correction must make a real fault MORE detectable, not less.

    A load-only reference absorbs the ambient swing into its residual SPREAD, so
    a genuine 8 K excess is diluted (the z shrinks). Removing the ambient
    component shrinks the spread, and the same fault then stands out. So the
    corrected detector must flag at least as much of the fault, and must show a
    higher z there.
    """
    rng = np.random.default_rng(3)
    n = 4000
    load = rng.uniform(0.05, 1.0, n)
    amb = np.concatenate([np.full(n // 2, 0.0), np.full(n // 2, 25.0)])
    ts = pd.date_range("2015-01-01", periods=n, freq="10min")
    idx = pd.RangeIndex(n)
    train = pd.DataFrame({"P_norm": load, "amb": amb, "ts": ts}, index=idx)
    test = pd.DataFrame({"P_norm": load, "amb": amb, "ts": ts}, index=idx)
    for df in (train, test):
        df["temp"] = 40 + 30 * df["P_norm"] + 0.25 * df["amb"] + rng.normal(0, 0.5, n)
    # A genuine fault: the last quarter of the rows runs 8 K hot.
    fault = slice(3 * n // 4, n)
    test.loc[fault, "temp"] += 8.0

    no_amb = multivariate.detect(train, test, ["temp"], "P_norm",
                                 ambient_col=None, z_threshold=4.0)
    with_amb = multivariate.detect(train, test, ["temp"], "P_norm",
                                   ambient_col="amb", z_threshold=4.0)

    z_no = no_amb.zmax.to_numpy()[fault].mean()
    z_with = with_amb.zmax.to_numpy()[fault].mean()
    assert z_with > z_no, (z_with, z_no)
    assert with_amb.flags_max.to_numpy()[fault].mean() >= \
        no_amb.flags_max.to_numpy()[fault].mean()


def test_ambient_correction_does_not_create_false_alarms():
    """With ambient correction, a merely hot day must not read as a fault."""
    rng = np.random.default_rng(4)
    n = 4000
    load = rng.uniform(0.05, 1.0, n)
    amb = np.concatenate([np.full(n // 2, 0.0), np.full(n // 2, 25.0)])
    ts = pd.date_range("2015-01-01", periods=n, freq="10min")
    idx = pd.RangeIndex(n)
    train = pd.DataFrame({"P_norm": load, "amb": amb, "ts": ts}, index=idx)
    test = pd.DataFrame({"P_norm": load, "amb": amb, "ts": ts}, index=idx)
    for df in (train, test):
        df["temp"] = 40 + 30 * df["P_norm"] + 0.25 * df["amb"] + rng.normal(0, 0.5, n)

    res = multivariate.detect(train, test, ["temp"], "P_norm",
                              ambient_col="amb", z_threshold=4.0)
    hot_half = slice(n // 2, n)
    assert res.flags_max.to_numpy()[hot_half].mean() < 0.02
    assert res.flags_max.to_numpy().mean() < 0.02


def test_overheating_is_one_sided():
    """A channel running COLDER than its reference is not evidence of a fault."""
    train, test = _synthetic()
    test["temp"] = test["temp"] - 8.0
    res = multivariate.detect(train, test, ["temp"], "P_norm", z_threshold=4.0)
    assert res.flags_max.to_numpy().mean() < 0.01


def test_spread_floor_prevents_exploding_z():
    """A near-constant channel must not produce z values in the tens."""
    rng = np.random.default_rng(5)
    n = 3000
    load = rng.uniform(0.05, 1.0, n)
    idx = pd.RangeIndex(n)
    train = pd.DataFrame({"P_norm": load, "ts": pd.date_range("2015-01-01", periods=n, freq="10min")},
                         index=idx)
    test = train.copy()
    # A channel that is constant: the spread must be floored, so z stays finite.
    train["flat"] = 50.0
    test["flat"] = 50.0 + rng.normal(0, 0.001, n)
    res = multivariate.detect(train, test, ["flat"], "P_norm", z_threshold=4.0)
    assert np.isfinite(res.zmax.to_numpy()).all()
    assert res.zmax.to_numpy().max() < 200


def test_l2_combiner_is_at_least_the_max():
    """The L2 norm of positive deviations dominates the max z (it sums squares)."""
    train, test = _synthetic(hot_after=2000)
    res = multivariate.detect(train, test, ["temp"], "P_norm", z_threshold=4.0)
    m = res.zmax.to_numpy()
    l = res.l2.to_numpy()
    valid = np.isfinite(m) & np.isfinite(l)
    assert (l[valid] >= m[valid] - 1e-9).all()
