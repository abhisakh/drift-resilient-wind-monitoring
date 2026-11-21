"""Cross-check our re-implementation against the authors' reference implementation.

The CARE score is defined in the dataset paper, but the authors ship an
implementation in `energy-fault-detector` (Fraunhofer IEE). Our copy in
`src/windmill/scoring.py` exists so the project does not depend on that package at
runtime; these tests prove the two agree, event metric by event metric.

If `energy-fault-detector` is not installed (it requires Python < 3.12), the tests
are skipped rather than failed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import scoring  # noqa: E402

ref = pytest.importorskip(
    "energy_fault_detector.evaluation.care_score",
    reason="energy-fault-detector requires Python < 3.12",
)


def _random_predictions(seed: int, n: int = 400) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    anomalies = pd.Series(rng.random(n) < 0.15, index=pd.RangeIndex(n))
    normal = pd.Series(rng.random(n) > 0.2, index=pd.RangeIndex(n))
    return anomalies, normal


def test_criticality_matches_reference():
    """Our criticality counter must equal the authors' implementation."""
    from energy_fault_detector.utils.analysis import calculate_criticality as ref_crit

    for seed in (0, 1, 2, 3):
        anomalies, normal = _random_predictions(seed)
        ours = scoring.calculate_criticality(anomalies, normal)
        theirs = ref_crit(anomalies=anomalies, normal_idx=normal)
        assert list(ours.to_numpy()) == list(theirs.to_numpy()), f"seed {seed}"


def test_weighted_score_matches_reference():
    """Our earliness weight function must equal the authors' implementation."""
    from energy_fault_detector.evaluation.care_score import CAREScore as RefCAREScore

    ref_score = RefCAREScore()
    rng = np.random.default_rng(7)
    for n in (50, 137, 400):
        p = pd.Series(rng.random(n) < 0.4, index=pd.RangeIndex(n))
        ours = scoring.weighted_score(p)
        theirs = ref_score._calculate_weighted_score(event_prediction=p)
        assert ours == pytest.approx(theirs, abs=1e-12), f"n={n}"


def test_event_metrics_match_reference():
    """Every per-event metric must agree with the authors' implementation."""
    from energy_fault_detector.evaluation.care_score import CAREScore as RefCAREScore

    for seed in (0, 1, 2):
        anomalies, normal = _random_predictions(seed, n=400)
        event_start, event_end = 50, 300

        ours = scoring.CAREScore()
        o = ours.evaluate_event(event_start, event_end, "anomaly", anomalies, normal, event_id=0)

        theirs = RefCAREScore()
        t = theirs.evaluate_event(
            event_start=event_start, event_end=event_end, event_label="anomaly",
            predicted_anomalies=anomalies, normal_index=normal, event_id=0,
        )

        assert o["f_beta_score"] == pytest.approx(t["f_beta_score"], abs=1e-9)
        assert o["accuracy"] == pytest.approx(t["accuracy"], abs=1e-9)
        assert o["max_criticality"] == t["max_criticality"]
        assert o["weighted_score"] == pytest.approx(t["weighted_score"], abs=1e-12)
        assert (o["tp"], o["fp"], o["tn"], o["fn"]) == (t["tp"], t["fp"], t["tn"], t["fn"])


def test_final_score_matches_reference():
    """The final weighted CARE score must agree too."""
    from energy_fault_detector.evaluation.care_score import CAREScore as RefCAREScore

    ours = scoring.CAREScore()
    theirs = RefCAREScore()
    for seed in range(6):
        anomalies, normal = _random_predictions(seed, n=400)
        label = "anomaly" if seed % 2 == 0 else "normal"
        ours.evaluate_event(50, 300, label, anomalies, normal, event_id=seed)
        theirs.evaluate_event(
            event_start=50, event_end=300, event_label=label,
            predicted_anomalies=anomalies, normal_index=normal, event_id=seed,
        )
    assert ours.get_final_score() == pytest.approx(theirs.get_final_score(), abs=1e-9)
