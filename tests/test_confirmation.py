"""Guards on the historical confirmation.

The confirmation reduces each month of `monthly_cross_check` to a few facts:
how well the healthy model predicted the month, what it dispatched, what a
fixed limit would have dispatched, and which of the fixed limit's alarms the
model rightly stayed quiet on. These tests build a month with known answers, so
the counts can be checked by hand, and then check that the committed artefact
is consistent with the function that aggregates it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import confirmation  # noqa: E402

CH = "Gb2t_avg"


def _month():
    """600 ten-minute rows of a machine whose temperature is expected to be 60 C.

      rows 100-129  the machine reaches 85 C, but the healthy model EXPECTED
                    85 C (a hard-working, hot-weather stretch): it stays quiet
      rows 300-319  the machine is 15 K above expectation at 75 C: below the
                    fixed limit, but the model alarms
      rows 450-469  the machine is 30 K above expectation at 90 C: both alarm
      rows 0-49     standstill (no load)
    """
    n = 600
    ts = pd.date_range("2017-03-01", periods=n, freq="10min", tz="UTC")
    expected = np.full(n, 60.0)
    actual = np.full(n, 60.0)
    alarm = np.zeros(n, dtype=bool)
    expected[100:130], actual[100:130] = 85.0, 85.0
    actual[300:320] = 75.0
    alarm[300:320] = True
    actual[450:470] = 90.0
    alarm[450:470] = True
    load = np.full(n, 0.5)
    load[:50] = 0.0
    # Three kelvin per unit of z: the 15 K excess is z = 5, the 30 K one z = 10,
    # and the explained stretch (actual = expected) is z = 0.
    z = (actual - expected) / 3.0
    frame = pd.DataFrame({"ts": ts, CH: actual, "predicted_weather": expected,
                          "P_norm": load, "alarm_weather": alarm,
                          "z_weather": z})
    return {"frame": frame,
            "provenance": {"train_rows": 4000, "clean_rows": 3900,
                           "dropped_rows": 100}}


def test_a_month_is_reduced_to_the_facts_that_can_be_checked_by_hand():
    s = confirmation.summarise_month(_month())
    assert s["rows"] == 600 and s["operating_rows"] == 550
    # The model alarmed twice: below the limit, and on the real overheat.
    assert s["model_dispatches"] == 2
    assert s["model_dispatches_with_hot_rows"] == 1
    # A fixed 80 C limit would have dispatched twice, and the model rightly
    # stayed quiet on one of them because it expected that temperature.
    assert s["fixed_dispatches"] == 2
    assert s["fixed_dispatches_explained"] == 1
    assert s["hot_rows"] == 50
    assert s["train_rows"] == 4000 and s["dropped_rows"] == 100
    # The first dispatches are where they should be: row 300 and row 100.
    assert s["first_model_dispatch"] == "2017-03-03 02:00"
    assert s["first_fixed_dispatch"] == "2017-03-01 16:40"


def test_the_prediction_error_is_counted_on_operating_rows_only():
    s = confirmation.summarise_month(_month())
    # 550 operating rows. Absolute residual: 0 on 475 rows, 0 on the explained
    # 30 (expected=actual=85), 15 on 20 rows, 30 on 20 rows.
    expected_mae = (20 * 15.0 + 20 * 30.0) / 550
    assert s["mae_k"] == pytest.approx(expected_mae, abs=1e-3)
    # The bias is signed: the machine ran hotter than expected on average.
    assert s["bias_k"] == pytest.approx(expected_mae, abs=1e-3)
    assert s["p95_abs_k"] >= 0.0 and s["rmse_k"] >= s["mae_k"]


def test_a_month_with_no_alarms_and_no_hot_rows_dispatches_nothing():
    m = _month()
    f = m["frame"]
    f[CH] = 60.0
    f["predicted_weather"] = 60.0
    f["alarm_weather"] = False
    s = confirmation.summarise_month(m)
    assert s["model_dispatches"] == 0 and s["fixed_dispatches"] == 0
    assert s["first_model_dispatch"] is None and s["first_fixed_dispatch"] is None
    assert s["mae_k"] == 0.0


def test_a_dispatch_needs_a_sustained_alarm_not_a_blip():
    m = _month()
    f = m["frame"]
    f["alarm_weather"] = False
    f.loc[200:203, "alarm_weather"] = True       # four samples: a blip
    assert confirmation.summarise_month(m)["model_dispatches"] == 0
    f.loc[200:205, "alarm_weather"] = True       # six samples: a dispatch
    assert confirmation.summarise_month(m)["model_dispatches"] == 1


# ------------------------------------------------------------ what is healthy
def test_the_documented_anomaly_months_are_not_assumed_healthy():
    assert confirmation.month_is_healthy("R80711", "2017-07")
    assert not confirmation.month_is_healthy("R80711", "2017-08")   # starts 15 Aug
    assert not confirmation.month_is_healthy("R80711", "2017-09")
    assert not confirmation.month_is_healthy("R80711", "2017-11")
    # The fault had not ended: R80711 ran hot against its siblings on most
    # days of December 2017 and January 2018, so they are not assumed healthy.
    assert not confirmation.month_is_healthy("R80711", "2017-12")
    assert not confirmation.month_is_healthy("R80711", "2018-01")
    # The siblings are healthy throughout, including the same months.
    assert confirmation.month_is_healthy("R80721", "2017-09")
    assert confirmation.month_is_healthy("R80790", "2017-09")


# ------------------------------------------------------------- the aggregate
def _row(turbine, healthy, model, fixed, explained, mae=2.0, bias=0.1):
    return {"turbine": turbine, "month": "2016-01", "healthy_assumed": healthy,
            "mae_k": mae, "bias_k": bias, "model_dispatches": model,
            "fixed_dispatches": fixed, "fixed_dispatches_explained": explained}


def test_false_alarm_rates_count_healthy_months_only_and_per_turbine_year():
    rows = ([_row("A", True, 1, 2, 1) for _ in range(12)]       # one turbine-year
            + [_row("A", False, 9, 9, 0) for _ in range(3)])    # anomaly: excluded
    a = confirmation.aggregate(rows)["by_turbine"]["A"]
    assert a["healthy_months"] == 12 and a["anomaly_months"] == 3
    assert a["healthy_model_dispatches"] == 12
    assert a["healthy_model_dispatches_per_turbine_year"] == pytest.approx(12.0)
    assert a["healthy_fixed_limit_dispatches_per_turbine_year"] == pytest.approx(24.0)
    assert a["healthy_fixed_limit_dispatches_explained"] == 12
    # The anomaly months are reported on their own.
    assert a["anomaly_model_dispatches"] == 27
    assert a["anomaly_fixed_limit_dispatches"] == 27


def test_the_fleet_is_the_sum_of_the_turbines():
    rows = ([_row("A", True, 1, 0, 0) for _ in range(12)]
            + [_row("B", True, 3, 6, 2) for _ in range(12)])
    agg = confirmation.aggregate(rows)
    assert agg["fleet"]["healthy_months"] == 24
    assert agg["fleet"]["healthy_model_dispatches"] == 48
    # 48 dispatches over two turbine-years.
    assert agg["fleet"]["healthy_model_dispatches_per_turbine_year"] == pytest.approx(24.0)
    assert set(agg["by_turbine"]) == {"A", "B"}


def test_an_empty_history_aggregates_to_nothing_not_an_error():
    agg = confirmation.aggregate([])
    assert agg["fleet"]["months"] == 0
    assert agg["fleet"]["healthy_model_dispatches_per_turbine_year"] is None
    assert agg["fleet"]["median_mae_k"] is None


# -------------------------------------------------------- the committed result
@pytest.mark.skipif(not (ROOT / "results" / "monthly_confirmation.json").exists(),
                    reason="the historical confirmation has not been run")
def test_the_committed_confirmation_is_what_the_aggregate_says():
    d = json.loads((ROOT / "results" / "monthly_confirmation.json").read_text())
    assert d["months"], "no months were scored"
    assert confirmation.aggregate(d["months"]) == d["summary"]
    for r in d["months"]:
        assert isinstance(r["healthy_assumed"], bool)
        assert r["model_dispatches_with_hot_rows"] <= r["model_dispatches"]
        assert r["fixed_dispatches_explained"] <= r["fixed_dispatches"]
    # Health is the documented assumption, not whatever the model said.
    for r in d["months"]:
        assert r["healthy_assumed"] == confirmation.month_is_healthy(
            r["turbine"], r["month"])


# ------------------------------------------- the same month at stricter settings
def test_the_same_month_is_rescored_at_stricter_alarm_settings_without_refitting():
    s = confirmation.summarise_month(_month())
    st = s["settings"]
    # The engine's own setting (z above 3, an hour) is one of the grid.
    assert st["z3_s6"]["dispatches"] == s["model_dispatches"] == 2
    assert st["z4_s6"]["dispatches"] == 2      # z of 5 and of 10
    assert st["z6_s6"]["dispatches"] == 1      # only the 30 K excess (z = 10)
    assert st["z8_s6"]["dispatches"] == 1
    # Twenty samples are not six hours, whatever the threshold.
    assert st["z3_s36"]["dispatches"] == 0
    assert st["z8_s36"]["dispatches"] == 0
    # Which of them sit on a genuinely hot stretch, and when the first one was.
    assert st["z3_s6"]["with_hot_rows"] == 1
    assert st["z4_s6"]["first"] == "2017-03-03 02:00"
    assert st["z6_s6"]["first"] == "2017-03-04 03:00"
    # The fixed limit is held for the same lengths of time, so it is comparable.
    assert s["fixed_settings"]["s6"]["dispatches"] == 2
    assert s["fixed_settings"]["s36"]["dispatches"] == 0


def test_a_month_with_no_z_scores_still_summarises_without_a_grid():
    m = _month()
    m["frame"] = m["frame"].drop(columns=["z_weather"])
    s = confirmation.summarise_month(m)
    assert s["settings"] == {}
    assert s["model_dispatches"] == 2


def test_the_sweep_counts_false_dispatches_in_healthy_months_and_alarmed_anomaly_months():
    def row(healthy, n):
        r = _row("A", healthy, 0, 0, 0)
        r["settings"] = {"z3_s6": {"dispatches": n}}
        r["fixed_settings"] = {"s6": {"dispatches": n + 1}}
        return r

    rows = [row(True, 2) for _ in range(12)] + [row(False, k) for k in (1, 0, 2, 0)]
    a = confirmation.aggregate(rows)["by_turbine"]["A"]
    s = a["settings"]["z3_s6"]
    assert s["healthy_dispatches"] == 24
    assert s["healthy_dispatches_per_turbine_year"] == pytest.approx(24.0)
    assert s["anomaly_months"] == 4 and s["anomaly_months_alarmed"] == 2
    assert s["anomaly_dispatches"] == 3
    assert a["fixed_settings"]["s6"]["healthy_dispatches"] == 36


def test_rows_from_an_older_run_without_settings_still_aggregate():
    rows = [_row("A", True, 1, 0, 0) for _ in range(12)]
    a = confirmation.aggregate(rows)["by_turbine"]["A"]
    assert a["settings"] == {} and a["fixed_settings"] == {}
    assert a["healthy_model_dispatches"] == 12


# ---------------------------------------- the two-stage rule: limit, then veto
def test_the_model_vetoes_a_limit_alarm_that_load_and_weather_explain():
    """The hand-built month has two limit dispatches: the 85 C stretch the model
    EXPECTED (z = 0), and the 90 C overheat it did not (z = 10). The veto must
    drop the first and keep the second, at every threshold up to z = 8."""
    s = confirmation.summarise_month(_month())
    for zv in (3, 4, 5, 6, 8):
        r = s["two_stage"][f"z{zv}_s6"]
        assert r["dispatches"] == 1, (zv, r)
        # Held an hour from row 450, and the model flagged from its first row,
        # so the dispatch is row 455: 2017-03-04 03:50.
        assert r["first"] == "2017-03-04 03:50", (zv, r)
    # Twenty samples never reach a six-hour hold.
    assert s["two_stage"]["z3_s36"]["dispatches"] == 0


def test_the_veto_never_adds_an_alarm_the_limit_did_not_raise():
    s = confirmation.summarise_month(_month())
    for key, r in s["two_stage"].items():
        hold = key.split("_s")[1]
        assert r["dispatches"] <= s["fixed_settings"][f"s{hold}"]["dispatches"], key


def test_a_dispatch_waits_for_the_model_when_it_flags_late_in_the_episode():
    m = _month()
    f = m["frame"]
    f.loc[450:459, "z_weather"] = 0.0        # the model only flags from row 460
    r = confirmation.summarise_month(m)["two_stage"]["z3_s6"]
    assert r["dispatches"] == 1
    assert r["first"] == "2017-03-04 04:40"  # row 460, not row 455


def test_the_two_stage_rule_is_aggregated_like_the_others():
    def row(healthy, n):
        r = _row("A", healthy, 0, 0, 0)
        r["two_stage"] = {"z3_s6": {"dispatches": n}}
        return r

    rows = [row(True, 1) for _ in range(12)] + [row(False, k) for k in (0, 1)]
    a = confirmation.aggregate(rows)["by_turbine"]["A"]["two_stage"]["z3_s6"]
    assert a["healthy_dispatches_per_turbine_year"] == pytest.approx(12.0)
    assert a["anomaly_months_alarmed"] == 1 and a["anomaly_months"] == 2


def test_a_model_with_no_prediction_cannot_veto_the_limit():
    """Where the model has no prediction (a load it has hardly seen), the hot
    temperature is not explained, it is unobservable: the limit alarm stands."""
    m = _month()
    f = m["frame"]
    f.loc[100:129, "z_weather"] = np.nan     # the 85 C stretch: no prediction
    r = confirmation.summarise_month(m)["two_stage"]["z3_s6"]
    assert r["dispatches"] == 2, r           # the unobservable one is kept
    assert r["kept_unobservable"] == 1, r
    # And it is dispatched once the limit has been held: row 105.
    assert r["first"] == "2017-03-01 17:30", r


def test_a_mostly_predicted_episode_can_still_be_vetoed():
    m = _month()
    f = m["frame"]
    f.loc[100:110, "z_weather"] = np.nan     # 11 of 30 rows unpredicted
    r = confirmation.summarise_month(m)["two_stage"]["z3_s6"]
    assert r["dispatches"] == 1 and r["kept_unobservable"] == 0, r


# ------------------------------------------ the sibling comparison and the strategy
def _fleet(days=3, offset_day=1, offset=4.0):
    """Four turbines at the same moments. Turbine A runs `offset` K hotter than
    its siblings on day `offset_day` only; the rest of the time all agree."""
    ts = pd.date_range("2017-03-01", periods=days * 144, freq="10min", tz="UTC")
    temps = pd.DataFrame({t: 60.0 for t in "ABCD"}, index=ts)
    loads = pd.DataFrame({t: 0.5 for t in "ABCD"}, index=ts)
    day = (ts - ts[0]).days
    temps.loc[day == offset_day, "A"] += offset
    return temps, loads


def test_the_daily_sibling_gap_is_the_median_gap_at_the_same_moment():
    temps, loads = _fleet()
    gap = confirmation.daily_sibling_gap(temps, loads)
    assert list(gap["A"]["gap"]) == [0.0, 4.0, 0.0]
    assert all(gap["B"]["gap"] == 0.0)
    assert all(gap["A"]["n"] == 144)


def test_moments_at_different_power_are_not_compared():
    temps, loads = _fleet()
    loads.loc[loads.index < "2017-03-02", "A"] = 0.9   # A works much harder on day 0
    gap = confirmation.daily_sibling_gap(temps, loads)
    assert pd.Timestamp("2017-03-01", tz="UTC") not in gap["A"].index
    assert len(gap["A"]) == 2


def test_a_day_with_too_few_comparable_moments_is_left_out():
    temps, loads = _fleet()
    loads.loc[(loads.index >= "2017-03-03") & (loads.index < "2017-03-03 20:00"), "A"] = 0.0
    gap = confirmation.daily_sibling_gap(temps, loads)   # 24 of 144 moments left
    assert pd.Timestamp("2017-03-03", tz="UTC") not in gap["A"].index


def test_the_change_is_measured_against_the_training_windows_normal_gap():
    days = pd.date_range("2017-01-01", periods=60, freq="D", tz="UTC")
    gap = pd.Series(1.0, index=days)                      # a constant 1 K offset
    gap[gap.index >= "2017-02-20"] = 4.0                  # then a 3 K shift
    daily = pd.DataFrame({"gap": gap, "n": 100})
    ch = confirmation.sibling_change(daily, "2017-01-01", "2017-01-31",
                                     "2017-02-01", "2017-03-01")
    assert ch.index.min() == pd.Timestamp("2017-02-01", tz="UTC")
    assert ch[ch.index < "2017-02-20"].abs().max() == 0.0     # the offset is normal
    assert ch.iloc[-1] == pytest.approx(3.0)                   # the shift is not
    # Five-day median: one day after the shift it is still smoothed away.
    assert ch[pd.Timestamp("2017-02-20", tz="UTC")] < 3.0
    # Too few comparable days in the training window: no normal, no signal.
    assert confirmation.sibling_change(daily.iloc[:5], "2017-01-01", "2017-01-31",
                                       "2017-02-01", "2017-03-01") is None


def _sibling_series(values):
    days = pd.date_range("2017-03-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=days, dtype="float64")


def test_sibling_call_outs_one_per_run_at_the_end_of_the_first_day():
    """The hand-built month runs 1 to 5 March; the model's z is 5 on 3 March
    and 10 on 4 March and 0 otherwise."""
    sib = _sibling_series([0.5, 3.2, 3.4, 1.0, 2.6])
    s = confirmation.summarise_month(_month(), sibling=sib)
    # +3 K: 2 and 3 March flagged, one run, dispatched at the end of 2 March.
    assert s["sibling"]["t3"] == {"dispatches": 1, "first": "2017-03-02 23:50"}
    # +2.5 K: runs 2-3 March and 5 March.
    assert s["sibling"]["t2.5"]["dispatches"] == 2


def test_a_confirmed_call_out_needs_the_model_to_find_unexplained_heat():
    sib = _sibling_series([0.5, 3.2, 3.4, 1.0, 2.6])
    s = confirmation.summarise_month(_month(), sibling=sib)
    # The 2-3 March run is confirmed on 3 March (z = 5 there); the 5 March run
    # is not (the model finds nothing that day).
    assert s["sibling_confirmed"]["t3"] == {"dispatches": 1, "first": "2017-03-03 23:50"}
    assert s["sibling_confirmed"]["t2.5"]["dispatches"] == 1


def test_the_strategy_is_the_veto_plus_the_sibling_call_outs_once_per_day():
    sib = _sibling_series([0.5, 3.2, 3.4, 1.0, 2.6])
    s = confirmation.summarise_month(_month(), sibling=sib)
    # The veto at z above 2 keeps one limit alarm, at 2017-03-04 03:50.
    assert s["two_stage"]["z2_s6"]["first"] == "2017-03-04 03:50"
    assert s["strategy"]["t3"] == {"dispatches": 2, "first": "2017-03-02 23:50"}
    assert s["strategy_confirmed"]["t3"] == {"dispatches": 2, "first": "2017-03-03 23:50"}
    # A sibling call-out on a day the veto already dispatched is not counted twice.
    same_day = _sibling_series([0.0, 0.0, 0.0, 3.5, 0.0])
    s2 = confirmation.summarise_month(_month(), sibling=same_day)
    assert s2["sibling"]["t3"]["dispatches"] == 1
    assert s2["strategy"]["t3"]["dispatches"] == 1


def test_no_sibling_signal_means_no_sibling_or_strategy_figures():
    s = confirmation.summarise_month(_month(), sibling=None)
    assert s["sibling"] == {} and s["strategy"] == {}


def test_episodes_list_the_stretches_hotter_than_usual():
    days = pd.date_range("2017-01-01", periods=80, freq="D", tz="UTC")
    gap = pd.Series(1.0, index=days)
    gap[(gap.index >= "2017-02-15") & (gap.index < "2017-02-25")] = 5.0   # ten days hot
    eps = confirmation.sibling_episodes(pd.DataFrame({"gap": gap, "n": 100}))
    assert len(eps) == 1, eps
    e = eps[0]
    assert "2017-02-15" <= e["start"] <= "2017-02-18"      # the median needs a few days
    assert e["peak_k"] == pytest.approx(4.0)
    assert e["days"] >= 7
