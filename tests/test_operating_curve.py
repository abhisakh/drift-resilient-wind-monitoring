"""Guards on the operating curve.

The curve is the project's evidence that a false call-out buys earlier
warning, so the things worth guarding are the ones that would let it
mislead:

  * a dispatch must need PERSISTENCE, and more persistence can never mean
    more dispatches;
  * an alarm run must not be stitched together across missing data;
  * the PLACEBO (a 0 K ramp) must exist, and must show that a single-sample
    rule "detects" a ramp that is not there, which is the confound the
    curve is built to remove;
  * a real ramp must be caught, and a bigger one no less often;
  * unhealthy windows must be left out for every arm alike;
  * the comparison between arms must be made at equal false call-outs.

These tests need no raw data: they run on a synthetic turbine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import operating_curve as oc  # noqa: E402

CH = "Gb2t_avg"


def _synthetic(days: int = 170, seed: int = 0, noise: float = 0.8) -> pd.DataFrame:
    """A synthetic turbine: load-driven bearing temperature plus white noise."""
    rng = np.random.default_rng(seed)
    n = days * 144
    ts = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    day = np.arange(n) / 144.0
    hour = (ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0)
    load = np.clip(0.45 + 0.35 * np.sin(2 * np.pi * day / 3.7)
                   + 0.15 * rng.standard_normal(n), 0.05, 1.0)
    ambient = 10 + 8 * np.sin(2 * np.pi * day / 90.0) + 3 * np.sin(2 * np.pi * hour / 24)
    value = 35 + 30 * load + 0.15 * ambient + noise * rng.standard_normal(n)
    return pd.DataFrame({
        "ts": ts, "P_norm": load, "era5_temp_2m": ambient,
        "Ws_avg": 4 + 8 * load, CH: value,
    })


def _arms():
    return oc.standard_arms(CH, include_lightgbm=False)


def _curve(frames=None, exclude=None, rates=(0.01, 0.05),
           persistence=(1, 6, 36), magnitudes=(0.0, 2.0, 8.0)):
    frames = frames or {"T1": _synthetic()}
    arms = {k: v for k, v in _arms().items() if k == "load+ambient"}
    run = oc.run_operating_curve(frames, CH, arms, rates=rates,
                                 persistence=persistence,
                                 magnitudes=magnitudes, exclude=exclude)
    return run, oc.summarise_points(run["rows"], magnitudes)


# ------------------------------------------------------------ the runs
def test_alarm_runs_finds_the_runs_and_their_lengths():
    starts, lengths = oc.alarm_runs([0, 1, 1, 0, 1, 1, 1, 0])
    assert list(starts) == [1, 4]
    assert list(lengths) == [2, 3]


def test_a_run_that_reaches_the_end_is_counted():
    starts, lengths = oc.alarm_runs([0, 0, 1, 1, 1])
    assert list(starts) == [2] and list(lengths) == [3]


def test_no_alarms_means_no_runs():
    starts, lengths = oc.alarm_runs([0, 0, 0])
    assert len(starts) == 0 and len(lengths) == 0
    starts, lengths = oc.alarm_runs([])
    assert len(starts) == 0


def test_a_gap_in_the_timestamps_splits_a_run():
    """Alarm either side of missing data is two runs, not one long one."""
    t = pd.date_range("2016-01-01", periods=8, freq="10min")
    times = np.array(list(t[:4]) + list(t[4:] + pd.Timedelta(hours=5)),
                     dtype="datetime64[ns]")
    starts, lengths = oc.alarm_runs(np.ones(8, dtype=bool), times)
    assert list(lengths) == [4, 4], lengths
    # Without timestamps the same flags are one run.
    s2, l2 = oc.alarm_runs(np.ones(8, dtype=bool))
    assert list(l2) == [8]


def test_a_dispatch_needs_persistence():
    starts, lengths = np.array([1, 4]), np.array([2, 3])
    assert oc.count_dispatches(starts, lengths, 1) == 2
    assert oc.count_dispatches(starts, lengths, 3) == 1
    assert oc.count_dispatches(starts, lengths, 4) == 0
    # The dispatch fires when the persistence is MET, not when the run starts.
    assert oc.first_dispatch_index(starts, lengths, 1) == 1
    assert oc.first_dispatch_index(starts, lengths, 3) == 6
    assert oc.first_dispatch_index(starts, lengths, 4) is None


# --------------------------------------------------------- the sweep
def test_more_persistence_never_means_more_dispatches():
    _, points = _curve()
    by = {}
    for p in points:
        by.setdefault(p["target_rate"], {})[p["persistence_samples"]] = (
            p["false_dispatches"])
    for rate, d in by.items():
        counts = [d[m] for m in sorted(d)]
        assert counts == sorted(counts, reverse=True), (rate, d)


def test_every_setting_has_a_point_for_every_ramp_size():
    run, points = _curve()
    assert len(points) == 2 * 3, len(points)
    for p in points:
        assert set(p["ramps"]) == {"0", "2", "8"}, p["ramps"].keys()
        assert p["windows"] == sum(run["windows_used"].values())


def test_the_placebo_shows_why_a_single_sample_rule_misleads():
    """A 0 K ramp 'detected' by a single-sample rule is pure chance. That is
    the confound the placebo exists to expose."""
    _, points = _curve()
    single = [p for p in points
              if p["target_rate"] == 0.05 and p["persistence_samples"] == 1][0]
    assert single["ramps"]["0"]["detection_rate"] >= 0.8, single["ramps"]["0"]
    # A persistence of an hour sharply cuts chance detection. It need not
    # reach zero: a reference that has gone stale over the month produces a
    # SUSTAINED false alarm, which is the project's own central effect.
    hour = [p for p in points
            if p["target_rate"] == 0.01 and p["persistence_samples"] == 6][0]
    assert (hour["ramps"]["0"]["detection_rate"]
            < single["ramps"]["0"]["detection_rate"]), (hour["ramps"]["0"],
                                                          single["ramps"]["0"])
    assert hour["ramps"]["0"]["detection_rate"] <= 0.4, hour["ramps"]["0"]


def test_a_real_ramp_is_caught_and_a_bigger_one_no_less_often():
    _, points = _curve()
    p = [q for q in points
         if q["target_rate"] == 0.01 and q["persistence_samples"] == 6][0]
    assert p["ramps"]["8"]["detection_rate"] == 1.0, p["ramps"]
    assert (p["ramps"]["8"]["detection_rate"]
            >= p["ramps"]["2"]["detection_rate"]
            >= p["ramps"]["0"]["detection_rate"]), p["ramps"]
    days = p["ramps"]["8"]["median_days_to_dispatch"]
    assert days is not None and 0.0 <= days < 3.0, days


def test_the_exposure_is_the_sum_of_the_rows_scored():
    frame = _synthetic()
    run, points = _curve({"T1": frame})
    p = points[0]
    rows = sum(x["exposure_days"] for x in run["rows"]
               if x["rate"] == p["target_rate"]
               and x["persistence"] == p["persistence_samples"])
    assert p["exposure_turbine_years"] == pytest.approx(rows / 365.25, rel=1e-3)


def test_unhealthy_windows_are_left_out():
    frame = _synthetic()
    run_all, _ = _curve({"T1": frame})
    span = {"T1": [("2016-03-15", "2016-03-20")]}
    run_ex, _ = _curve({"T1": frame}, exclude=span)
    assert run_ex["windows_excluded_as_unhealthy"] > 0
    assert (sum(run_ex["windows_used"].values())
            < sum(run_all["windows_used"].values()))
    # And no row survives from a window that touches the span.
    bad = pd.Timestamp("2016-03-15", tz="UTC")
    for r in run_ex["rows"]:
        assert pd.Timestamp(r["label"], tz="UTC") + pd.Timedelta(days=28) <= bad \
            or pd.Timestamp(r["label"], tz="UTC") - pd.Timedelta(days=30) \
            >= pd.Timestamp("2016-03-20", tz="UTC"), r["label"]


# ------------------------------------------------------ the readings
def _pt(arm, fd, det, days, mag="4"):
    return {"arm": arm, "false_dispatches_per_turbine_year": fd,
            "ramps": {mag: {"detection_rate": det,
                            "median_days_to_dispatch": days}}}


def test_the_best_setting_under_a_budget_respects_the_budget():
    pts = [_pt("A", 5.0, 0.5, 2.0), _pt("A", 10.0, 0.8, 1.0),
           _pt("A", 20.0, 0.9, 0.5), _pt("B", 3.0, 0.99, 0.1)]
    assert oc.best_under_budget(pts, "A", 12.0)["false_dispatches_per_turbine_year"] == 10.0
    assert oc.best_under_budget(pts, "A", 100.0)["false_dispatches_per_turbine_year"] == 20.0
    assert oc.best_under_budget(pts, "A", 4.0) is None
    # Another arm's setting never leaks in.
    assert oc.best_under_budget(pts, "B", 4.0)["arm"] == "B"


def test_a_tie_on_detection_goes_to_the_earlier_warning():
    pts = [_pt("A", 8.0, 0.8, 2.0), _pt("A", 9.0, 0.8, 1.0)]
    assert oc.best_under_budget(pts, "A", 10.0)["false_dispatches_per_turbine_year"] == 9.0


def test_the_frontier_drops_settings_that_are_beaten_on_both_axes():
    pts = [_pt("A", 5.0, 0.5, 1.0), _pt("A", 8.0, 0.4, 1.0),
           _pt("A", 12.0, 0.9, 1.0), _pt("A", 15.0, 0.9, 1.0)]
    kept = [p["false_dispatches_per_turbine_year"] for p in oc.frontier(pts, "A")]
    assert kept == [5.0, 12.0], kept


def test_the_standard_arms_are_the_floor_the_operating_point_and_the_model():
    arms = oc.standard_arms(CH, include_lightgbm=False)
    assert set(arms) == {"constant", "load", "load+ambient"}
    assert "lightgbm" in oc.standard_arms(CH, include_lightgbm=True)


# ------------------------------------------ the shape of the fault
def test_a_load_shaped_ramp_scales_with_load_and_leaves_the_past_alone():
    frame = _synthetic(days=40)
    onset = pd.Timestamp("2016-01-20", tz="UTC")
    shaped = oc.inject_shaped(frame, CH, onset, 4.0, ramp_days=1.0, shape="load")
    flat = oc.inject_shaped(frame, CH, onset, 4.0, ramp_days=1.0, shape="additive")
    ts = pd.to_datetime(frame["ts"], utc=True)
    before = (ts < onset).to_numpy()
    assert np.allclose(shaped[CH].to_numpy()[before], frame[CH].to_numpy()[before])
    held = (ts >= onset + pd.Timedelta(days=2)).to_numpy()
    d_load = (shaped[CH] - frame[CH]).to_numpy()[held]
    assert np.allclose(d_load, 4.0 * frame["P_norm"].to_numpy()[held] / oc.REFERENCE_LOAD)
    assert np.allclose((flat[CH] - frame[CH]).to_numpy()[held], 4.0)


def test_an_unknown_ramp_shape_is_refused():
    with pytest.raises(ValueError):
        oc.inject_shaped(_synthetic(days=5), CH,
                         pd.Timestamp("2016-01-03", tz="UTC"), 1.0,
                         shape="sideways")


# ------------------------------------------ held-out selection
def test_the_two_halves_partition_the_windows():
    run, _ = _curve({"T1": _synthetic(days=260)})
    a = oc.split_rows(run["rows"], 0)
    b = oc.split_rows(run["rows"], 1)
    ka = {(r["turbine"], r["label"]) for r in a}
    kb = {(r["turbine"], r["label"]) for r in b}
    assert ka and kb and ka.isdisjoint(kb)
    assert len(a) + len(b) == len(run["rows"])


def test_a_held_out_reading_is_scored_on_the_half_it_was_not_chosen_on():
    run, _ = _curve({"T1": _synthetic(days=260)})
    h = oc.held_out_reading(run["rows"], "load+ambient", 1000.0,
                            magnitude=8.0, magnitudes=(0.0, 2.0, 8.0))
    assert h is not None and h["folds"] == 2, h
    assert 0.0 <= h["detection_rate"] <= 1.0
    for s in h["selected_settings"]:
        assert s["target_rate"] in (0.01, 0.05)
        assert s["persistence_samples"] in (1, 6, 36)
    # A budget nothing can meet leaves nothing to report.
    assert oc.held_out_reading(run["rows"], "load+ambient", -1.0,
                               magnitude=8.0, magnitudes=(0.0, 2.0, 8.0)) is None


# ------------------------------------------ the age of the reference
def test_an_older_reference_is_scored_on_the_same_windows():
    frames = {"T1": _synthetic(days=420)}
    arms = {k: v for k, v in _arms().items() if k == "load+ambient"}
    kw = dict(rates=(0.01,), persistence=(1,), magnitudes=(0.0,),
              require_history_days=120)
    fresh = oc.run_operating_curve(frames, CH, arms, reference_age_days=0, **kw)
    old = oc.run_operating_curve(frames, CH, arms, reference_age_days=120, **kw)
    assert fresh["windows_used"] == old["windows_used"]
    assert fresh["windows_skipped_for_history"] > 0
    assert old["reference_age_days"] == 120


def test_an_excluded_span_inside_an_older_reference_excludes_the_window():
    frames = {"T1": _synthetic(days=420)}
    arms = {k: v for k, v in _arms().items() if k == "load+ambient"}
    kw = dict(rates=(0.01,), persistence=(1,), magnitudes=(0.0,),
              require_history_days=120,
              exclude={"T1": [("2016-02-01", "2016-02-10")]})
    fresh = oc.run_operating_curve(frames, CH, arms, reference_age_days=0, **kw)
    old = oc.run_operating_curve(frames, CH, arms, reference_age_days=120, **kw)
    assert (old["windows_excluded_as_unhealthy"]
            > fresh["windows_excluded_as_unhealthy"])


# ------------------------------------------ the committed artefact
def test_the_committed_curve_has_the_shape_the_readers_expect():
    import json

    path = ROOT / "results" / "operating_curve.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    for key in ("protocol", "points", "budgets_ramp_4k"):
        assert key in d, key
    arms = d["protocol"]["arms"]
    assert "load+ambient" in arms
    for p in d["points"]:
        assert set(p["ramps"]) >= {"0", "4"}, p["ramps"].keys()
        assert p["false_dispatches_per_turbine_year"] >= 0
    # Persistence can only reduce the false dispatches of a given arm and rate.
    by = {}
    for p in d["points"]:
        by.setdefault((p["arm"], p["target_rate"]), {})[
            p["persistence_samples"]] = p["false_dispatches"]
    for key, counts in by.items():
        ordered = [counts[m] for m in sorted(counts)]
        assert ordered == sorted(ordered, reverse=True), (key, counts)


# ------------------------------------------ the real anomaly, on the same dial
def _anomaly_frame(shape: str):
    """A synthetic turbine with a fault from 2016-03-01: a steady step, or
    bursts of heat for part of each day, as the real R80711 anomaly was."""
    frame = _synthetic(days=120, seed=3)
    onset = pd.Timestamp("2016-03-01", tz="UTC")
    ts = pd.to_datetime(frame["ts"], utc=True)
    after = (ts >= onset).to_numpy()
    if shape == "step":
        add = np.where(after, 15.0, 0.0)
    else:
        hours = ts.dt.hour.to_numpy()
        add = np.where(after & (hours >= 10) & (hours < 18), 15.0, 0.0)
    frame[CH] = frame[CH] + add
    return frame


def _lead(frame, rates, persistence):
    arms = {k: v for k, v in oc.standard_arms(CH, include_lightgbm=False).items()
            if k == "load+ambient"}
    return oc.lead_time_on_anomaly(
        frame, CH, arms, "2016-01-01", "2016-01-31", "2016-04-20", "2016-03-01",
        hot_c=75.0, min_hot_samples=3, rates=rates, persistence=persistence)


def _row(res, rate, m):
    return next(r for r in res["rows"]
                if r["target_rate"] == rate and r["persistence_samples"] == m)


def test_a_steady_step_is_dispatched_soon_after_the_onset_at_any_persistence():
    res = _lead(_anomaly_frame("step"), (0.01,), (6, 288))
    for m in (6, 288):
        r = _row(res, 0.01, m)
        assert r["days_after_onset"] is not None, (m, r)
        assert r["days_after_onset"] < 5.0, (m, r)
    assert res["first_hot_spell"] is not None
    assert res["onset"] == "2016-03-01"


def test_a_bursty_fault_is_missed_by_a_long_persistence_but_caught_by_a_short_one():
    """The finding on the real anomaly in miniature: a rule that waits for a
    long unbroken alarm cannot fire on a fault that comes in bursts."""
    res = _lead(_anomaly_frame("bursts"), (0.01,), (6, 288))
    short, long_ = _row(res, 0.01, 6), _row(res, 0.01, 288)
    assert short["days_after_onset"] is not None, short
    assert long_["days_after_onset"] is None, long_
    assert long_["dispatches_after_onset"] == 0


def test_a_jumpy_setting_is_charged_for_dispatches_before_the_onset():
    """Dispatching early only because it dispatches at everything is not a
    warning: the quiet-stretch dispatches are counted against the setting."""
    res = _lead(_anomaly_frame("step"), (0.01, 0.2), (1,))
    assert (_row(res, 0.2, 1)["dispatches_before_onset"]
            > _row(res, 0.01, 1)["dispatches_before_onset"])
    assert _row(res, 0.2, 1)["dispatches_before_onset"] > 0


def test_the_load_profile_of_the_excess_is_reported_by_phase():
    res = _lead(_anomaly_frame("step"), (0.01,), (6,))
    rbl = res["residual_by_load"]
    assert rbl["arm"] == "load+ambient"
    assert set(rbl["phases"]) == {"before_onset", "hot_spell_days",
                                  "other_days_after_onset"}
    for cells in rbl["phases"].values():
        assert len(cells) == len(rbl["load_bins"])
    # A steady +15 K lifts every load bin after the onset, and none before it.
    before = [c["mean_residual_k"] for c in rbl["phases"]["before_onset"]
              if c["mean_residual_k"] is not None and c["n"] > 50]
    after = [c["mean_residual_k"] for c in rbl["phases"]["hot_spell_days"]
             if c["mean_residual_k"] is not None and c["n"] > 50]
    assert before and after
    assert min(after) > max(before) + 5.0, (before, after)


def test_too_little_data_to_fit_is_reported_not_raised():
    res = oc.lead_time_on_anomaly(
        _synthetic(days=3), CH, {k: v for k, v in oc.standard_arms(
            CH, include_lightgbm=False).items() if k == "load+ambient"},
        "2016-01-01", "2016-01-02", "2016-01-03", "2016-01-02")
    assert res["rows"] == [] and "reason" in res


def test_the_high_load_coverage_records_when_the_turbine_first_ran_hard():
    """The excess shows mainly at high load, so what limits the warning time is
    when the turbine first works hard, and how much of that regime the
    reference was shown. A calm week after the onset must show as no samples."""
    frame = _anomaly_frame("step")
    ts = pd.to_datetime(frame["ts"], utc=True)
    calm = ((ts >= pd.Timestamp("2016-03-01", tz="UTC"))
            & (ts < pd.Timestamp("2016-03-08", tz="UTC"))).to_numpy()
    frame.loc[calm, "P_norm"] = 0.2
    cov = _lead(frame, (0.01,), (6,))["high_load_coverage"]
    assert cov["high_load"] == 0.6 and cov["arm"] == "load+ambient"
    assert cov["training_high_load_samples"] > 0
    assert 0.0 < cov["training_high_load_share"] < 1.0
    assert cov["training_samples"] > cov["training_high_load_samples"]
    # Four weeks, starting a week before the onset.
    assert len(cov["daily"]) == 28 and cov["daily"][0]["day"] == "2016-02-23"
    calm_days = [d for d in cov["daily"] if "2016-03-01" <= d["day"] <= "2016-03-07"]
    assert len(calm_days) == 7
    assert all(d["n_high_load"] == 0 and d["mean_residual_k"] is None
               for d in calm_days), calm_days
    busy = [d for d in cov["daily"] if d["day"] < "2016-03-01"]
    assert any(d["n_high_load"] > 0 for d in busy)
