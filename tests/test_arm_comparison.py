"""The two detectors must be compared on equal terms.

The project's headline is a comparison between a deterministic detector and a
learned one. A comparison is only meaningful if both arms use the same
statistic, the same threshold rule and the same reference strategy. It is
easy to get wrong: an earlier version scored the learned arm with an L2 norm
while the deterministic arm used max-z, and gave the two arms different
threshold rules. These tests pin the comparison shut.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"


def _load_script(name: str):
    # The scripts live in scripts/pipeline (artifacts) or scripts/diagnostics
    # (one-off probes); find the file wherever it ended up.
    path = next((ROOT / "scripts").rglob(f"{name}.py"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_engie_script_scores_both_arms_with_the_same_statistic():
    """max-z for both -- the learned arm must not be scored with an L2 norm."""
    src = (ROOT / "scripts" / "pipeline" / "08c_engie_multivariate.py").read_text()
    # No L2 combination anywhere in the ENGIE comparison.
    assert "** 2)" not in src and "sqrt((Z.clip" not in src
    assert "nbm.max_z" in src, "the ENGIE arm must score with max-z"


def test_engie_script_uses_the_same_threshold_for_both_arms():
    """The rolling comparison must use the same fixed z for both detectors."""
    src = (ROOT / "scripts" / "pipeline" / "08c_engie_multivariate.py").read_text()
    assert "Z_THRESHOLD" in src
    # The learned rolling arm must threshold on the same Z_THRESHOLD the
    # deterministic rolling arm uses, not on its own quantile.
    assert "(roll_score.to_numpy() > Z_THRESHOLD)" in src


def test_learned_arm_is_cross_fitted():
    """A threshold taken from in-sample residuals is too strict.

    The learned arm must derive its threshold from out-of-fold training
    scores, the way the deterministic arm does, or the comparison handicaps it.
    """
    # The cross-fitting lives in the shared module now.
    src = (ROOT / "src" / "windmill" / "nbm.py").read_text()
    assert "oof_parts" in src
    assert "out-of-fold" in src.lower()


@pytest.mark.skipif(
    not (RESULTS / "engie_multivariate.json").exists(),
    reason="ENGIE multivariate results not generated",
)
def test_both_arms_have_an_engie_false_alarm_number():
    """Neither arm may be reported on CARE while being absent on ENGIE."""
    d = json.loads((RESULTS / "engie_multivariate.json").read_text())
    strategies = d["strategies"]
    assert "frozen_2014" in strategies
    assert "frozen_2014_lightgbm" in strategies
    assert "rolling_30d" in strategies
    assert "rolling_30d_lightgbm" in strategies


@pytest.mark.skipif(
    not (RESULTS / "engie_multivariate.json").exists(),
    reason="ENGIE multivariate results not generated",
)
def test_every_engie_strategy_reports_per_year_rates():
    d = json.loads((RESULTS / "engie_multivariate.json").read_text())
    for name, s in d["strategies"].items():
        assert s["per_year"], name
        assert s["min_rate_pct"] <= s["max_rate_pct"], name
        assert 0.0 <= s["min_rate_pct"], name


@pytest.mark.skipif(
    not (RESULTS / "engie_multivariate.json").exists(),
    reason="ENGIE multivariate results not generated",
)
def test_rolling_reference_beats_frozen_for_both_detectors():
    """The project's central claim, checked for BOTH detectors.

    If the learned arm did not also improve with a rolling reference, the
    claim would be about the deterministic arm only -- and the README would
    have to say so.
    """
    d = json.loads((RESULTS / "engie_multivariate.json").read_text())
    s = d["strategies"]
    for det in ("", "_lightgbm"):
        frozen = s[f"frozen_2014{det}"]
        rolling = s[f"rolling_30d{det}"]
        assert rolling["mean_rate_pct"] < frozen["mean_rate_pct"], det
