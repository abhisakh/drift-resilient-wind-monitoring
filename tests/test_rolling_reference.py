"""Tests for the rolling local reference.

The rolling reference is what stops a stale reference from turning slow drift
into a flood of alarms, so its behaviour has to be pinned.
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


def _drifting(n_days: int = 400, drift_per_day: float = 0.02) -> pd.DataFrame:
    """A turbine whose component temperature drifts steadily warmer."""
    n = n_days * 144
    ts = pd.date_range("2015-01-01", periods=n, freq="10min")
    rng = np.random.default_rng(0)
    load = rng.uniform(0.05, 1.0, n)
    day = np.repeat(np.arange(n_days), 144)
    temp = 40 + 30 * load + drift_per_day * day + rng.normal(0, 0.5, n)
    return pd.DataFrame({"P_norm": load, "temp": temp, "ts": ts})


def test_rolling_reference_tracks_a_drift():
    """A steady drift must NOT read as a deviation from the rolling reference."""
    df = _drifting()
    resid, z = multivariate.rolling_reference_scores(df, "temp", "P_norm",
                                                     window_days=30)
    # The residual is the drift within the trailing window, so it is bounded by
    # the window length times the drift rate -- NOT unbounded.
    assert np.abs(resid).max() < 5.0
    # And the z must stay modest: a drift is not a fault.
    assert np.abs(z).max() < 10.0


def test_rolling_reference_detects_a_fault_early_then_absorbs_it():
    """A rolling reference detects a developing fault EARLY, then masks it.

    This is the honest trade-off of a rolling reference, and it is worth
    stating plainly: as the fault persists, the trailing window fills with
    faulted data and the reference moves to meet it. So the rolling reference
    is an EARLY-warning instrument -- good at the first hours of a developing
    fault, progressively blind to a fault that has been running for weeks.

    That is why a real deployment combines a rolling reference (low false
    alarms) with a frozen, periodically re-validated reference (fault
    persistence), rather than relying on either alone.
    """
    df = _drifting()
    df.loc[df.index[-1440:], "temp"] += 10.0
    resid, z = multivariate.rolling_reference_scores(df, "temp", "P_norm",
                                                     window_days=30)
    fault = z[-1440:]
    early = fault[:200]
    late = fault[-200:]
    # Early in the fault the window is still clean, so the fault is obvious.
    assert early.mean() > 4.0
    # Later the window is mostly faulted data, so the reference has moved.
    assert late.mean() < early.mean()


def test_frozen_reference_would_miss_the_fault_entirely():
    """The contrast: a whole-series mean reference does not see the fault at all.

    A frozen reference built over the whole series treats the faulted rows as
    part of "normal", so the fault is invisible to it.
    """
    df = _drifting()
    df.loc[df.index[-1440:], "temp"] += 10.0
    whole_mean = df["temp"].mean()
    frozen_z = (df["temp"].to_numpy() - whole_mean) / df["temp"].std()
    # Below the 4-sigma threshold the rolling detector uses: the frozen
    # reference does not see the fault at all.
    assert frozen_z[-1440:].max() < 4.0


def test_detect_rolling_flags_a_fault_early():
    df = _drifting()
    df.loc[df.index[-1440:], "temp"] += 10.0
    res = multivariate.detect_rolling(df, ["temp"], "P_norm", z_threshold=4.0)
    assert "temp" in res.channels_used
    flags = res.flags_max.to_numpy()
    assert flags[-1440:][:200].mean() > 0.8
    assert flags[:-1440].mean() < 0.05
