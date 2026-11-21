"""Unit tests for the scoring, conditioning and gating logic.

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

from windmill import conditioning as cond  # noqa: E402
from windmill import detector  # noqa: E402
from windmill import scoring  # noqa: E402


# ---------------------------------------------------------------- criticality

def test_criticality_counts_consecutive_detections():
    """Criticality rises with consecutive detections during normal operation."""
    anomalies = pd.Series([True] * 5 + [False] * 3 + [True] * 2)
    normal = pd.Series(True, index=anomalies.index)
    crit = scoring.calculate_criticality(anomalies, normal)
    assert list(crit[:5]) == [1, 2, 3, 4, 5]
    assert list(crit[5:8]) == [4, 3, 2]
    assert list(crit[8:]) == [3, 4]


def test_criticality_held_constant_when_not_normal():
    """During non-normal operation the counter is frozen (paper algorithm 1)."""
    anomalies = pd.Series([True, True, True])
    normal = pd.Series([False, False, False])
    crit = scoring.calculate_criticality(anomalies, normal)
    assert list(crit) == [0, 0, 0]


def test_criticality_never_negative():
    anomalies = pd.Series([False] * 5)
    normal = pd.Series(True, index=anomalies.index)
    crit = scoring.calculate_criticality(anomalies, normal)
    assert crit.min() >= 0


def test_criticality_is_bounded():
    """The counter must be capped, as in the benchmark authors' implementation.

    A long, wholly anomalous series would otherwise drive the counter without
    limit. The cap does not change any classification (Reliability only asks
    whether the maximum reaches 72) but it does change the reported maximum.
    """
    anomalies = pd.Series([True] * 3000)
    normal = pd.Series(True, index=anomalies.index)
    crit = scoring.calculate_criticality(anomalies, normal)
    assert crit.max() == 1000
    assert crit.iloc[-1] == 1000


def test_criticality_respects_a_custom_cap():
    anomalies = pd.Series([True] * 100)
    normal = pd.Series(True, index=anomalies.index)
    crit = scoring.calculate_criticality(anomalies, normal, max_criticality=10)
    assert crit.max() == 10


# ------------------------------------------------------------- weighted score

def test_weighted_score_all_detected_is_one():
    p = pd.Series(True, index=range(100))
    assert scoring.weighted_score(p) == pytest.approx(1.0)


def test_weighted_score_none_detected_is_zero():
    p = pd.Series(False, index=range(100))
    assert scoring.weighted_score(p) == pytest.approx(0.0)


def test_weighted_score_early_detection_beats_late():
    """A detection at the start of the event must score higher than one at the end."""
    length = 100
    early = pd.Series(False, index=range(length))
    early.iloc[0] = True
    late = pd.Series(False, index=range(length))
    late.iloc[length - 1] = True
    assert scoring.weighted_score(early) > scoring.weighted_score(late)


def test_weighted_score_is_normalised():
    """Detecting the first half of the event scores well above detecting the second half.

    The weight is flat (1.0) over the first quarter of the event and decays
    linearly to 0 at the end, so detecting the first half picks up the flat part
    plus the steepest part of the decay.
    """
    p = pd.Series(False, index=range(100))
    p.iloc[:50] = True
    full = scoring.weighted_score(pd.Series(True, index=range(100)))
    assert 0.6 < scoring.weighted_score(p) / full <= 1.0


def test_weighted_score_second_half_scores_less_than_half():
    """Detecting only the tail of the event must score below half of a full hit."""
    p = pd.Series(False, index=range(100))
    p.iloc[50:] = True
    full = scoring.weighted_score(pd.Series(True, index=range(100)))
    assert scoring.weighted_score(p) / full < 0.5


# ------------------------------------------------------------- CARE score end

def _toy_events():
    """Two events: one anomaly, one normal, with clean predictions.

    The anomaly event is long enough (280 consecutive detections) to pass the
    criticality threshold of 72, which is what the Reliability sub-score needs
    before any alarm counts as a detected event.
    """
    idx = pd.RangeIndex(0, 300)
    normal = pd.Series(True, index=idx)

    # Anomaly event: the model flags exactly the event window.
    anomalies = pd.Series(False, index=idx)
    anomalies.iloc[10:290] = True

    # Normal event: the model flags nothing.
    normal_anomalies = pd.Series(False, index=idx)
    return anomalies, normal, normal_anomalies


def test_care_score_perfect_detector_is_high():
    anomalies, normal, normal_anomalies = _toy_events()
    score = scoring.CAREScore()
    score.evaluate_event(10, 290, "anomaly", anomalies, normal, event_id=0)
    score.evaluate_event(0, 300, "normal", normal_anomalies, normal, event_id=1)
    assert score.get_final_score() > 0.9


def test_care_score_needs_the_criticality_threshold():
    """A single burst shorter than 72 detections must not count as a detected event.

    This is the paper's special case: with no anomaly detected at all, the CARE
    score is 0, no matter how good the pointwise metrics look.
    """
    idx = pd.RangeIndex(0, 300)
    normal = pd.Series(True, index=idx)
    anomalies = pd.Series(False, index=idx)
    anomalies.iloc[10:20] = True  # only 10 consecutive detections

    score = scoring.CAREScore()
    score.evaluate_event(10, 290, "anomaly", anomalies, normal, event_id=0)
    score.evaluate_event(0, 300, "normal", pd.Series(False, index=idx), normal, event_id=1)
    assert score.get_final_score() == 0.0


def test_care_score_zero_when_nothing_detected():
    """With no anomaly detected at all, the score must be 0 (paper special case)."""
    _, normal, _ = _toy_events()
    score = scoring.CAREScore()
    score.evaluate_event(10, 290, "anomaly", pd.Series(False, index=normal.index),
                         normal, event_id=0)
    score.evaluate_event(0, 300, "normal", pd.Series(False, index=normal.index),
                         normal, event_id=1)
    assert score.get_final_score() == 0.0


def test_care_score_caps_at_accuracy_when_accuracy_below_half():
    """Below 0.5 accuracy the score is that accuracy (worse than random)."""
    idx = pd.RangeIndex(0, 60)
    normal = pd.Series(True, index=idx)
    score = scoring.CAREScore()
    # Anomaly event flagged nowhere -> no detection. Normal event flagged
    # everywhere -> accuracy 0.
    score.evaluate_event(10, 30, "anomaly", pd.Series(False, index=idx), normal, event_id=0)
    score.evaluate_event(0, 60, "normal", pd.Series(True, index=idx), normal, event_id=1)
    assert score.get_final_score() == pytest.approx(0.0)


def test_care_score_requires_both_event_types():
    _, normal, _ = _toy_events()
    score = scoring.CAREScore()
    score.evaluate_event(10, 30, "anomaly", pd.Series(False, index=normal.index),
                         normal, event_id=0)
    with pytest.raises(ValueError):
        score.get_final_score()


# ------------------------------------------------------------------ conditioning

def test_wind_bin_handles_missing_values():
    """Missing wind must land in the -1 bin, not crash or become NaN."""
    v = pd.Series([1.0, np.nan, 3.0, 7.5])
    b = cond.wind_bin(v)
    assert b.tolist() == [2, -1, 6, 15]


def test_power_curve_is_monotone_in_the_meaningful_range():
    """A synthetic cubic power curve must be recovered as increasing."""
    v = np.linspace(3, 20, 400)
    p = np.where(v < 12, 0.5 * v ** 3, 1200.0)
    df = pd.DataFrame({"Ws_avg": v, "P_avg": p})
    curve = cond.fit_power_curve(df)
    vals = curve.loc[curve.index >= 6]
    assert vals.is_monotonic_increasing


def test_expected_power_matches_direct_computation():
    """E[P] = sum f(v) * p(v) must equal the hand-computed sum."""
    curve = pd.Series({10: 100.0, 12: 200.0, 14: 300.0})
    clim = pd.Series({10: 0.5, 12: 0.3, 14: 0.2})
    got = cond.expected_power_from_climatology(curve, clim, rated_kw=2050.0)
    assert got == pytest.approx(0.5 * 100 + 0.3 * 200 + 0.2 * 300)


def test_climatology_weights_sum_to_one():
    df = pd.DataFrame({"Ws_avg": np.random.default_rng(0).uniform(2, 20, 5000)})
    clim = cond.climatology_weights(df)
    assert clim.sum() == pytest.approx(1.0, abs=1e-6)


def test_peer_residual_is_zero_for_an_identical_twin():
    """A turbine identical to its peers must have (near-)zero peer residual.

    All three turbines see the SAME wind sequence here, so the peer median at
    each wind bin is drawn from the same population as the turbine's own value:
    the MEAN signed residual is zero. The per-row residual is not exactly zero
    because the peer reference is a per-bin MEDIAN while the turbine's own value
    varies inside the 0.5 m/s bin (a bin spans 1.0 K in the linear relation
    50 + 2v). So the property to assert is: mean signed residual ~ 0, and the
    spread bounded by the within-bin variation.
    """
    rng = np.random.default_rng(1)
    v = rng.uniform(3, 20, 2000)
    frames = []
    for t in ("A", "B", "C"):
        g = pd.DataFrame({"turbine": t, "Ws_avg": v, "Gb2t_avg": 50 + 2 * v,
                          "year": 2014,
                          "ts": pd.date_range("2014-01-01", periods=2000, freq="10min")})
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    out = cond.peer_normalised_residual(df, target="A", peers=("B", "C"))
    a = out.loc[out["turbine"] == "A", "resid"]
    # The peer reference is a per-bin MEDIAN, while the turbine's own value varies
    # inside the 0.5 m/s bin, so the residual is not identically zero. What must
    # hold is that it is centred on zero and bounded by the within-bin variation
    # (2 K per 0.5 m/s of wind).
    assert abs(a.mean()) < 0.05
    assert a.abs().max() <= 1.0 + 1e-9


def test_peer_residual_detects_one_turbine_running_hot():
    """A turbine running 10 K hotter than its peers must show a positive residual."""
    rng = np.random.default_rng(2)
    frames = []
    for t in ("A", "B", "C"):
        v = rng.uniform(3, 20, 2000)
        offset = 10.0 if t == "A" else 0.0
        g = pd.DataFrame({"turbine": t, "Ws_avg": v, "Gb2t_avg": 50 + 2 * v + offset,
                          "year": 2014, "ts": pd.date_range("2014-01-01", periods=2000, freq="10min")})
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    out = cond.peer_normalised_residual(df, target="A", peers=("B", "C"))
    assert out.loc[out["turbine"] == "A", "resid"].mean() == pytest.approx(10.0, abs=0.5)
    assert out.loc[out["turbine"] != "A", "resid"].abs().mean() < 0.5


# --------------------------------------------------------------------- gates

def test_persistence_gate_requires_consecutive_samples():
    df = pd.DataFrame({"flag": [True, False, True, True, True, False]})
    gate = detector.persistence_gate(df, consecutive=3)
    assert gate.passed is True
    df2 = pd.DataFrame({"flag": [True, True, False, True, False, False]})
    assert detector.persistence_gate(df2, consecutive=3).passed is False


def test_alarm_budget_gate_flags_a_flood():
    """More than one alarm per 10 minutes on average must fail the budget."""
    df = pd.DataFrame({"flag": [True] * 50 + [False] * 50})
    gate = detector.alarm_budget_gate(df, period_minutes=10)
    assert gate.passed is False
    assert "rate_per_10min" in gate.evidence


def test_alarm_budget_gate_accepts_a_quiet_alarm():
    df = pd.DataFrame({"flag": [True] + [False] * 999})
    gate = detector.alarm_budget_gate(df, period_minutes=10)
    assert gate.passed is True


def test_operating_state_gate_rejects_a_stopped_turbine():
    df = pd.DataFrame({"P_avg": [0.0] * 100, "Ws_avg": [8.0] * 100})
    gate = detector.operating_state_gate(df)
    assert gate.passed is False


def test_climatology_gate_flags_an_outlier_period():
    df = pd.DataFrame({"Ws_avg": [12.0] * 100, "year": 2014})
    gate = detector.climatology_consistency_gate(df, clim_mean=6.0, clim_sd=1.0, k=3.0)
    assert gate.passed is False


def test_apply_gates_requires_all_to_pass():
    df = pd.DataFrame({"flag": [True, False]})
    ok = detector.Gate("a", True)
    bad = detector.Gate("b", False)
    out = detector.apply_gates(df, [ok, bad])
    assert out["alarm_allowed"].tolist() == [False, False]
    out2 = detector.apply_gates(df, [ok])
    assert out2["flag_final"].tolist() == [True, False]
