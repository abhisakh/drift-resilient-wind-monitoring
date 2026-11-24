"""Guards on the goal scorecard.

The dashboard's headline cards and the README's goal section both read
`windmill.scorecard`, so the one thing that must hold is that it SELECTS
figures that the pipeline wrote rather than making any up, and that a
missing or malformed artefact degrades to "not measured" instead of
taking the dashboard down.

These tests need no raw data: the committed results/*.json are enough.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import scorecard  # noqa: E402

RESULTS = ROOT / "results"


def _load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


# ------------------------------------------------------ degrading gracefully
def test_a_missing_artefact_degrades_to_none_not_an_exception():
    card = scorecard.goal_scorecard(None, None, None)
    ages = card["false_alarm"]["ratio_by_reference_age"]
    assert set(ages) == set(scorecard.AGE_KEYS)
    assert all(v is None for v in ages.values())
    assert card["false_alarm"]["ratio_frozen_2014"] is None
    assert card["detection"] == {}
    assert card["r80711"] == {"peak_c": None, "peer_max_c": None}
    assert scorecard.lead_days(card) == (None, None)


def test_a_malformed_artefact_does_not_raise():
    """Wrong shapes in place of the expected dicts must not crash the page."""
    bad = {
        "B4_reference_age": "not a dict",
        "C6_fault_injection": {"summary": {"a": 3, "b": None}},
        "C7_lead_time": None,
    }
    card = scorecard.goal_scorecard(bad, {"strategies": []}, {"kpis": None})
    assert card["detection"] == {}
    assert scorecard.lead_days(card) == (None, None)
    assert card["r80711"]["peak_c"] is None


def test_a_rate_is_always_a_fraction():
    """A percentage stored by mistake must not be shown as '9000 %'."""
    assert scorecard.as_fraction(0.9) == pytest.approx(0.9)
    assert scorecard.as_fraction(90) == pytest.approx(0.9)
    assert scorecard.as_fraction(None) is None
    assert scorecard.as_fraction("n/a") is None


# --------------------------------------------- read from the real artefacts
@pytest.mark.skipif(
    not (RESULTS / "backtest.json").exists(),
    reason="the evaluation suite has not been run",
)
def test_the_scorecard_is_selected_from_the_committed_backtest():
    backtest = _load("backtest.json")
    card = scorecard.goal_scorecard(
        backtest, _load("engie_multivariate.json"), _load("demo_r80711.json"))

    # 1. The ratios are the artefact's own, not recomputed.
    sweep = backtest["B4_reference_age"]["sweep_over_reference_age"]
    ages = card["false_alarm"]["ratio_by_reference_age"]
    for name, key in scorecard.AGE_KEYS.items():
        assert ages[name] == pytest.approx(sweep[key]["ratio_over_30d"]), name
    # The older the reference, the more false alarms: the reason to keep it
    # fresh, and the claim the headline card makes.
    assert ages["three_months"] < ages["one_year"] < ages["two_years"], ages

    # 2. Detection is quoted for the deployed arm at the headline ramp.
    det = card["detection"][scorecard.DEPLOYED_ARM]
    assert 0.0 <= det["rate"] <= 1.0, det
    assert det["median_days"] >= 0.0, det

    # 3. Lead time exists, and a more permissive false-alarm rate can only
    #    warn earlier, never later.
    one, five = scorecard.lead_days(card)
    assert one is not None and five is not None
    assert five >= one, (one, five)


@pytest.mark.skipif(
    not (RESULTS / "demo_r80711.json").exists(),
    reason="the R80711 snapshot has not been exported",
)
def test_the_case_study_figures_come_from_the_snapshot():
    snap = _load("demo_r80711.json")
    card = scorecard.goal_scorecard(None, None, snap)
    assert card["r80711"]["peak_c"] == pytest.approx(snap["kpis"]["r80711_max_c"])
    assert card["r80711"]["peer_max_c"] == pytest.approx(snap["kpis"]["peer_max_c"])
    # The siblings stay below the line the case study is about.
    assert card["r80711"]["peer_max_c"] < 80.0
    assert card["r80711"]["peak_c"] > 80.0


# ---------------------------------------------------- the operating curve
def _curve_point(arm, fd, det, days, chance, rate=0.01, m=6):
    return {"arm": arm, "target_rate": rate, "persistence_samples": m,
            "false_dispatches_per_turbine_year": fd,
            "ramps": {"4": {"detection_rate": det,
                            "median_days_to_dispatch": days},
                      "0": {"detection_rate": chance}}}


def test_the_operating_point_is_picked_out_of_the_curve_not_recomputed():
    curve = {
        "protocol": {"arms": ["load+ambient", "constant"]},
        "points": [_curve_point("load+ambient", 17.2, 0.85, 3.6, 0.26),
                   _curve_point("load+ambient", 270.0, 0.98, 1.3, 0.85, m=1),
                   _curve_point("constant", 16.1, 0.93, 2.3, 0.29)],
    }
    card = scorecard.goal_scorecard(None, None, None, curve)
    dep = card["operating_point"]["load+ambient"]
    assert dep == {"false_dispatches_per_turbine_year": 17.2,
                   "detection_rate": 0.85, "median_days": 3.6,
                   "chance_rate": 0.26}
    # The standing point is the one-hour rule, not the single-sample one.
    assert card["operating_point"]["constant"]["detection_rate"] == 0.93
    assert scorecard.curve_reading(curve, "lightgbm") is None


def test_no_curve_means_no_operating_point():
    card = scorecard.goal_scorecard(None, None, None, None)
    assert card["operating_point"] == {}
    assert scorecard.curve_reading(None) is None
    assert scorecard.curve_budget_reading(None, 12, "load+ambient") is None


@pytest.mark.skipif(
    not (RESULTS / "operating_curve.json").exists(),
    reason="the operating curve has not been run",
)
def test_the_committed_curve_gives_a_sane_standing_point():
    curve = _load("operating_curve.json")
    card = scorecard.goal_scorecard(None, None, None, curve)
    op = card["operating_point"][scorecard.DEPLOYED_ARM]
    assert op is not None
    assert 0.0 <= op["chance_rate"] <= op["detection_rate"] <= 1.0, op
    assert op["false_dispatches_per_turbine_year"] >= 0.0


def test_the_dashboard_reads_its_headline_through_the_scorecard():
    """The hero must not carry its own selection logic, or the dashboard and
    the README could drift apart."""
    app = (ROOT / "app.py").read_text()
    assert "scorecard.goal_scorecard" in app
    report = (ROOT / "scripts" / "pipeline" / "09_report.py").read_text()
    assert "scorecard.goal_scorecard" in report


# ------------------------------------------------- the real anomaly readings
def _anomaly_curve():
    def row(rate, m, before, after, first, days_after, days_before):
        return {"arm": "load+ambient", "target_rate": rate,
                "persistence_samples": m, "dispatches_before_onset": before,
                "dispatches_after_onset": after,
                "first_dispatch_after_onset": first,
                "days_after_onset": days_after,
                "days_before_hot_spell": days_before}
    return {"protocol": {"arms": ["load+ambient"]},
            "r80711_lead_time": {
                "first_hot_spell": "2017-09-08T07:10", "onset": "2017-09-01",
                "rows": [
                    row(0.01, 6, 0, 3, "2017-09-12T21:50", 11.9, -4.6),
                    row(0.2, 6, 4, 9, "2017-09-02T00:00", 1.0, 6.0),
                    row(0.2, 144, 0, 0, None, None, None),
                    row(0.05, 36, 0, 1, "2017-09-05T00:00", 4.0, 3.0)]}}


def test_the_standing_reading_on_the_real_anomaly_is_picked_out_not_recomputed():
    r = scorecard.anomaly_reading(_anomaly_curve())
    assert r["days_before_hot_spell"] == -4.6   # negative: AFTER the hot spell
    assert r["first_dispatch"] == "2017-09-12T21:50"
    assert r["dispatches_before_onset"] == 0
    assert scorecard.anomaly_reading(_anomaly_curve(), "lightgbm") is None
    assert scorecard.anomaly_reading(None) is None


def test_a_setting_that_never_dispatched_has_no_lead_time():
    r = scorecard.anomaly_reading(_anomaly_curve(), rate=0.2, persistence=144)
    assert r["first_dispatch"] is None and r["days_before_hot_spell"] is None


def test_only_early_and_clean_settings_count_as_a_warning():
    """Early AND no dispatch in the quiet stretch: the jumpy 20 % setting
    dispatched early but also four times before the onset; the standing one
    was late; one never dispatched. Only the 5 % setting qualifies."""
    assert scorecard.early_clean_settings(_anomaly_curve()) == (1, 4)
    assert scorecard.early_clean_settings(None) is None
    assert scorecard.early_clean_settings(_anomaly_curve(), "lightgbm") is None


def test_the_scorecard_carries_the_anomaly_block():
    card = scorecard.goal_scorecard(None, None, None, _anomaly_curve())
    an = card["anomaly"]
    assert an["first_hot_spell"] == "2017-09-08T07:10"
    assert an["onset"] == "2017-09-01"
    assert an["standing"]["load+ambient"]["days_before_hot_spell"] == -4.6
    assert an["early_clean_settings"]["load+ambient"] == (1, 4)
    empty = scorecard.goal_scorecard(None, None, None, None)["anomaly"]
    assert empty["standing"] == {} and empty["first_hot_spell"] is None
