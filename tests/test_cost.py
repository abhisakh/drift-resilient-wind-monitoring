"""Guards on the cost model.

The model is arithmetic over three assumed numbers and two measured ones, so
the things worth guarding are the ones that would let it quietly mislead:

  * the break-even rates really are break-evens (net value is zero there);
  * a quieter detector is never credited with a saving it did not earn;
  * bad inputs are refused rather than priced;
  * the defaults agree with the two other places a cost figure appears, so
    the dashboard and the model cannot show different numbers for the same
    assumption.

No data is needed: everything here runs on plain numbers.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import cost, live  # noqa: E402


# ------------------------------------------------------------ arithmetic
def test_net_value_is_detection_value_less_false_dispatch_cost():
    a = cost.CostAssumptions(dispatch_cost=2_000.0,
                             early_intervention_value=185_000.0,
                             faults_per_turbine_year=0.05)
    # 0.05 faults x 0.9 caught x 185,000 = 8,325; 10 false trips x 2,000 = 20,000.
    assert cost.expected_detection_value(0.9, a) == pytest.approx(8_325.0)
    assert cost.false_dispatch_cost(10, a) == pytest.approx(20_000.0)
    assert cost.annual_net_value(10, 0.9, a) == pytest.approx(8_325.0 - 20_000.0)


def test_a_detector_that_never_alarms_and_never_catches_is_worth_nothing():
    assert cost.annual_net_value(0, 0.0) == 0.0


def test_the_saving_is_negative_when_the_candidate_is_noisier():
    assert cost.false_dispatch_saving(5, 8, 2_000.0) == pytest.approx(-6_000.0)
    assert cost.false_dispatch_saving(8, 5, 2_000.0) == pytest.approx(6_000.0)
    assert cost.false_dispatch_saving(5, 5, 2_000.0) == 0.0


# ------------------------------------------------------------ break-evens
def test_the_break_even_fault_rate_is_where_net_value_crosses_zero():
    n, p = 12.0, 0.8
    base = cost.CostAssumptions()
    lam = cost.break_even_fault_rate(n, p, base)
    assert lam is not None and lam > 0
    at = cost.CostAssumptions(dispatch_cost=base.dispatch_cost,
                              early_intervention_value=base.early_intervention_value,
                              faults_per_turbine_year=lam)
    assert cost.annual_net_value(n, p, at) == pytest.approx(0.0, abs=1e-6)
    # Faults rarer than that: the detector costs more than it saves.
    rarer = cost.CostAssumptions(faults_per_turbine_year=lam / 2)
    assert cost.annual_net_value(n, p, rarer) < 0
    commoner = cost.CostAssumptions(faults_per_turbine_year=lam * 2)
    assert cost.annual_net_value(n, p, commoner) > 0


def test_a_detector_that_catches_nothing_can_never_pay():
    assert cost.break_even_fault_rate(3, 0.0) is None


def test_the_break_even_dispatch_cost_separates_two_detectors():
    """At that dispatch cost the noisy sensitive arm and the quiet blunt arm
    are worth the same; either side of it, one of them wins."""
    n_a, p_a = 20.0, 0.95   # sensitive, noisy
    n_b, p_b = 5.0, 0.70    # quiet, misses more
    base = cost.CostAssumptions()
    d_star = cost.break_even_dispatch_cost(n_a, n_b, p_a, p_b, base)
    assert d_star is not None and d_star > 0

    def value(n, p, d):
        return cost.annual_net_value(
            n, p, cost.CostAssumptions(
                dispatch_cost=d,
                early_intervention_value=base.early_intervention_value,
                faults_per_turbine_year=base.faults_per_turbine_year))

    assert value(n_a, p_a, d_star) == pytest.approx(value(n_b, p_b, d_star))
    # A dearer false call-out favours the quiet arm; a cheaper one, the sensitive.
    assert value(n_b, p_b, d_star * 2) > value(n_a, p_a, d_star * 2)
    assert value(n_b, p_b, d_star / 2) < value(n_a, p_a, d_star / 2)


def test_no_dispatch_cost_can_separate_two_equally_noisy_detectors():
    assert cost.break_even_dispatch_cost(5, 5, 0.9, 0.8) is None


# ----------------------------------------------------------- sensitivity
def test_the_sensitivity_grid_covers_every_combination_and_matches_the_formula():
    rows = cost.sensitivity_grid(8.0, 0.9)
    expected = (len(cost.DISPATCH_COSTS) * len(cost.FAULT_RATES)
                * len(cost.EARLY_VALUES))
    assert len(rows) == expected
    for r in rows:
        a = cost.CostAssumptions(
            dispatch_cost=r["dispatch_cost"],
            early_intervention_value=r["early_intervention_value"],
            faults_per_turbine_year=r["faults_per_turbine_year"])
        assert r["net_value"] == pytest.approx(cost.annual_net_value(8.0, 0.9, a))


def test_a_costlier_false_dispatch_never_makes_a_detector_look_better():
    rows = cost.sensitivity_grid(8.0, 0.9)
    by_cell = {}
    for r in rows:
        key = (r["early_intervention_value"], r["faults_per_turbine_year"])
        by_cell.setdefault(key, []).append((r["dispatch_cost"], r["net_value"]))
    for key, cells in by_cell.items():
        cells.sort()
        values = [v for _, v in cells]
        assert values == sorted(values, reverse=True), key


def test_the_share_positive_is_a_share():
    assert cost.share_positive([]) is None
    rows = [{"net_value": 1.0}, {"net_value": -1.0}, {"net_value": 0.0}]
    assert cost.share_positive(rows) == pytest.approx(1 / 3)


# ------------------------------------------------------------ validation
@pytest.mark.parametrize("bad", [-1.0, -0.001])
def test_negative_counts_are_refused(bad):
    with pytest.raises(ValueError):
        cost.false_dispatch_cost(bad)
    with pytest.raises(ValueError):
        cost.annual_net_value(bad, 0.5)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_a_probability_outside_zero_one_is_refused(bad):
    with pytest.raises(ValueError):
        cost.expected_detection_value(bad)


def test_negative_assumptions_are_refused():
    with pytest.raises(ValueError):
        cost.CostAssumptions(dispatch_cost=-1.0)
    with pytest.raises(ValueError):
        cost.CostAssumptions(faults_per_turbine_year=-0.01)


# ---------------------------------------------- the defaults agree elsewhere
def test_the_defaults_agree_with_the_existing_cost_view():
    """`live.cost_impact` has always priced an early intervention at one
    figure; the model must use the same one, or the two views of cost would
    disagree about the same assumption."""
    params = inspect.signature(live.cost_impact).parameters
    assert (params["early_intervention_value"].default
            == cost.DEFAULT_EARLY_INTERVENTION_VALUE)


def test_the_default_dispatch_cost_is_the_dashboards_slider_default():
    """The dashboard's slider opens on 2,000 and `test_dashboard` asserts it;
    the model's default must be the same number."""
    tests_src = (ROOT / "tests" / "test_dashboard.py").read_text()
    m = re.search(r"cost_slider\[0\]\.value == (\d+)", tests_src)
    assert m, "the dashboard test no longer pins the slider default"
    assert float(m.group(1)) == cost.DEFAULT_DISPATCH_COST


def test_the_grid_axes_include_the_defaults():
    """A reader must be able to find the default case on the grid."""
    assert cost.DEFAULT_DISPATCH_COST in cost.DISPATCH_COSTS
    assert cost.DEFAULT_EARLY_INTERVENTION_VALUE in cost.EARLY_VALUES
    assert cost.DEFAULT_FAULTS_PER_TURBINE_YEAR in cost.FAULT_RATES


# ------------------------------------------- pricing the operating curve
def _pt(arm, rate, m, fd, det, days=2.0, chance=0.1):
    return {"arm": arm, "target_rate": rate, "persistence_samples": m,
            "false_dispatches_per_turbine_year": fd,
            "ramps": {"4": {"detection_rate": det,
                            "median_days_to_dispatch": days},
                      "0": {"detection_rate": chance}}}


def _curve_points():
    return [
        _pt("A", 0.01, 6, 10.0, 0.9), _pt("A", 0.001, 36, 1.0, 0.3),
        _pt("B", 0.01, 6, 2.0, 0.5), _pt("B", 0.001, 36, 0.5, 0.1),
    ]


def test_the_best_setting_is_the_most_profitable_not_the_most_sensitive():
    a = cost.CostAssumptions(2_000.0, 185_000.0, 0.05)
    best = cost.best_setting(_curve_points(), "A", a)
    # 0.05 x 0.9 x 185,000 - 10 x 2,000 = -11,675 ; 0.05 x 0.3 x 185,000 - 2,000 = 775.
    assert best["target_rate"] == 0.001 and best["persistence_samples"] == 36
    assert best["net_value"] == pytest.approx(775.0)
    assert best["false_dispatch_cost"] == pytest.approx(2_000.0)
    assert best["expected_detection_value"] == pytest.approx(2_775.0)


def test_a_setting_at_the_edge_of_the_grid_is_flagged():
    a = cost.CostAssumptions(2_000.0, 185_000.0, 0.05)
    best = cost.best_setting(_curve_points(), "A", a)
    assert "least sensitive rate swept" in best["at_grid_edge"]
    assert "longest persistence swept" in best["at_grid_edge"]
    # With these prices a loud setting wins for an arm whose quiet one catches
    # nothing, and that choice is also at an edge of the grid.
    loud = [_pt("C", 0.01, 6, 0.1, 0.9), _pt("C", 0.001, 36, 0.05, 0.0)]
    assert cost.best_setting(loud, "C", a)["target_rate"] == 0.01


def test_a_named_setting_can_be_priced_and_a_missing_one_is_none():
    a = cost.CostAssumptions(2_000.0, 185_000.0, 0.05)
    s = cost.setting_at(_curve_points(), "A", 0.01, 6, a)
    assert s["net_value"] == pytest.approx(-11_675.0)
    assert cost.setting_at(_curve_points(), "A", 0.5, 6, a) is None
    assert cost.setting_at(_curve_points(), "Z", 0.01, 6, a) is None


def test_the_net_value_curve_never_falls_as_faults_become_commoner():
    a = cost.CostAssumptions(2_000.0, 185_000.0, 0.05)
    rows = cost.net_value_curve(_curve_points(), "A", [0.01, 0.05, 0.1, 0.2], a)
    nets = [r["net_value"] for r in rows]
    assert nets == sorted(nets), nets


def test_a_bad_point_without_figures_is_skipped_not_priced():
    pts = [{"arm": "A", "target_rate": 0.01, "persistence_samples": 6,
            "false_dispatches_per_turbine_year": None, "ramps": {}}]
    assert cost.best_setting(pts, "A") is None
    assert cost.net_value_curve(pts, "A", [0.05]) == []


def test_the_shares_of_the_grid_add_up_to_one():
    pts = _curve_points()
    shares = cost.share_of_assumptions_where_best(pts, ["A", "B"])
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3), shares
    assert all(0.0 <= v <= 1.0 for v in shares.values())


def test_the_worse_case_ranking_prefers_the_arm_that_holds_up_in_both_shapes():
    """A is much better under one fault shape and worthless under the other;
    B is modest under both. The worse case must favour B."""
    flat = [_pt("A", 0.01, 6, 1.0, 0.95), _pt("B", 0.01, 6, 1.0, 0.5)]
    other = [_pt("A", 0.01, 6, 1.0, 0.0), _pt("B", 0.01, 6, 1.0, 0.5)]
    shares = cost.share_where_best_in_the_worse_case(
        {"flat": flat, "load": other}, ["A", "B"])
    assert shares["B"] == 1.0 and shares["A"] == 0.0, shares
    # While the flat shape alone favours A.
    alone = cost.share_of_assumptions_where_best(flat, ["A", "B"])
    assert alone["A"] > alone["B"], alone


def test_an_arm_missing_from_one_shape_cannot_win_the_worse_case():
    flat = [_pt("A", 0.01, 6, 1.0, 0.9), _pt("B", 0.01, 6, 1.0, 0.5)]
    other = [_pt("B", 0.01, 6, 1.0, 0.5)]
    shares = cost.share_where_best_in_the_worse_case(
        {"flat": flat, "load": other}, ["A", "B"])
    assert shares["A"] == 0.0 and shares["B"] == 1.0, shares


def test_the_committed_cost_model_prices_the_committed_curve():
    import json

    cm_path = ROOT / "results" / "cost_model.json"
    cv_path = ROOT / "results" / "operating_curve.json"
    if not (cm_path.exists() and cv_path.exists()):
        return
    cm, curve = json.loads(cm_path.read_text()), json.loads(cv_path.read_text())
    assert cm["assumptions"]["dispatch_cost"] == cost.DEFAULT_DISPATCH_COST
    assert (cm["assumptions"]["early_intervention_value"]
            == cost.DEFAULT_EARLY_INTERVENTION_VALUE)
    for arm in curve["protocol"]["arms"]:
        want = cost.best_setting(curve["points"], arm)
        got = cm["arms"][arm]["best_setting"]
        assert got["net_value"] == pytest.approx(want["net_value"], abs=0.01), arm
