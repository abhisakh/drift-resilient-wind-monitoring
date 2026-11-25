"""Guards on the daily check.

Every rule of the policy is tested on a synthetic turbine whose answers are
known: the bearing temperature follows load and weather plus a little noise,
and a fault, a gap or a maintenance stop is placed exactly where the test
needs it. No raw data is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import daily_check as dc  # noqa: E402

CH = "Gb2t_avg"


def _turbine(days: int = 150, start: str = "2016-01-01", seed: int = 0):
    rng = np.random.default_rng(seed)
    n = days * 144
    ts = pd.date_range(start, periods=n, freq="10min", tz="UTC")
    d = np.arange(n) / 144.0
    load = np.clip(0.45 + 0.35 * np.sin(2 * np.pi * d / 3.7)
                   + 0.1 * rng.standard_normal(n), 0.05, 1.0)
    amb = 8 + 6 * np.sin(2 * np.pi * d / 120.0)
    temp = 35 + 30 * load + 0.15 * amb + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"ts": ts, "P_norm": load, "era5_temp_2m": amb, CH: temp})


def _drop(frame, lo, hi):
    ts = frame["ts"]
    keep = ~((ts >= pd.Timestamp(lo, tz="UTC")) & (ts < pd.Timestamp(hi, tz="UTC")))
    return frame[keep].reset_index(drop=True)


# ------------------------------------------------- restarts, warm-up, maintenance
def test_a_gap_of_six_hours_or_more_is_a_restart_and_starts_a_warm_up():
    f = _drop(_turbine(10), "2016-01-05 00:00", "2016-01-05 07:00")
    marked, maint = dc.mark_restarts(f, warmup_hours=12)
    back = marked["ts"] >= pd.Timestamp("2016-01-05 07:00", tz="UTC")
    w = marked[back & (marked["ts"] < pd.Timestamp("2016-01-05 19:00", tz="UTC"))]
    assert w["warmup"].all()
    after = marked[marked["ts"] >= pd.Timestamp("2016-01-05 19:00", tz="UTC")]
    assert not after["warmup"].any()
    assert maint == []


def test_a_short_gap_is_not_a_restart():
    f = _drop(_turbine(10), "2016-01-05 00:00", "2016-01-05 03:00")
    marked, _ = dc.mark_restarts(f)
    later = marked["ts"] >= pd.Timestamp("2016-01-05 03:00", tz="UTC")
    assert not marked[later]["warmup"].any()


def test_standstill_counts_as_not_running_for_a_restart():
    f = _turbine(10)
    calm = ((f["ts"] >= pd.Timestamp("2016-01-04", tz="UTC"))
            & (f["ts"] < pd.Timestamp("2016-01-04 08:00", tz="UTC")))
    f.loc[calm, "P_norm"] = 0.0
    marked, _ = dc.mark_restarts(f)
    first = marked[marked["running"]
                   & (marked["ts"] >= pd.Timestamp("2016-01-04 08:00", tz="UTC"))].iloc[0]
    assert first["warmup"]


def test_a_stop_of_three_days_or_more_stands_in_for_maintenance():
    f = _drop(_turbine(20), "2016-01-08", "2016-01-12")
    _, maint = dc.mark_restarts(f)
    assert maint == [pd.Timestamp("2016-01-12", tz="UTC")]


# ------------------------------------------------------------- the monthly model
def test_the_first_month_is_blank_and_later_months_are_trained_on_a_month():
    sim = dc.simulate(_turbine(120), "T", tolerance=1.0, spans={})
    models = {m["month"]: m for m in sim["months"]}
    assert models["2016-01"]["model"] == "blank"
    assert models["2016-03"]["model"] == "trained"
    assert models["2016-03"]["clean_rows"] >= dc.MONTH_OF_READINGS
    # A healthy turbine with half a kelvin of noise: no false call-outs at 1 K.
    assert dc.call_outs(sim["days"]) == []


def test_warm_up_readings_are_never_checked():
    f = _drop(_turbine(120), "2016-03-10 00:00", "2016-03-10 08:00")
    warm = ((f["ts"] >= pd.Timestamp("2016-03-10 08:00", tz="UTC"))
            & (f["ts"] < pd.Timestamp("2016-03-10 20:00", tz="UTC")))
    f.loc[warm, CH] -= 12.0                       # a cold restart
    sim = dc.simulate(f, "T", tolerance=1.0, spans={})
    day = next(d for d in sim["days"] if d["day"] == "2016-03-10")
    assert day.get("flag") is None, day
    assert day["n"] == 4 * 6                       # 20:00 to midnight only


# ------------------------------------------------------- a fault, and freezing
def _faulty(start="2016-03-15"):
    f = _turbine(150)
    hot = f["ts"] >= pd.Timestamp(start, tz="UTC")
    f.loc[hot, CH] += 3.0
    return f


def test_a_fault_is_called_out_on_its_first_day():
    spans = {"T": [("2016-03-15", "2016-06-01")]}
    sim = dc.simulate(_faulty(), "T", tolerance=1.0, spans=spans)
    calls = dc.call_outs(sim["days"])
    assert calls[0]["first"] == "2016-03-15 23:50"
    assert calls[0]["fault_period"]


def test_a_confirmed_fault_freezes_the_model_so_it_stays_alarmed():
    spans = {"T": [("2016-03-15", "2016-06-01")]}
    sim = dc.simulate(_faulty(), "T", tolerance=1.0, spans=spans)
    models = {m["month"]: m["model"] for m in sim["months"]}
    assert models["2016-04"] == "frozen" and models["2016-05"] == "frozen"
    may = [d for d in sim["days"] if d["day"].startswith("2016-05")
           and d.get("status") == "checked"]
    assert may and all(d["flag"] == "hot" for d in may)


def test_an_unconfirmed_alarm_is_cleared_and_its_days_never_trained_on():
    """The same fault outside any documented span: the inspection finds
    nothing, so the model goes on retraining, but never on the flagged days.
    It reaches back past them to the healthy data before the fault (within
    three months), so it never learns the fault and the days stay hot."""
    sim = dc.simulate(_faulty(), "T", tolerance=1.0, spans={})
    models = {m["month"]: m for m in sim["months"]}
    assert "frozen" not in [m["model"] for m in models.values()]
    assert models["2016-05"]["model"] == "trained"
    assert models["2016-05"]["train_start"] < "2016-03-15"
    may = [d for d in sim["days"] if d["day"].startswith("2016-05")
           and d.get("status") == "checked"]
    assert may and all(d["flag"] == "hot" for d in may)


# ------------------------------------------------------------------ maintenance
def test_after_maintenance_the_model_is_blank_until_a_new_month_is_collected():
    f = _drop(_turbine(150), "2016-03-01", "2016-03-05")      # 4 days off
    sim = dc.simulate(f, "T", tolerance=1.0, spans={})
    models = {m["month"]: m for m in sim["months"]}
    assert models["2016-04"]["model"] == "blank"               # not a month yet
    assert models["2016-05"]["model"] == "trained"
    assert models["2016-05"]["train_start"] >= "2016-03-05"
    march = [d for d in sim["days"] if d["day"].startswith("2016-03")]
    assert all(d["status"] == "blank" for d in march)


def test_maintenance_ends_a_confirmed_fault():
    f = _faulty("2016-02-15")
    f = _drop(f, "2016-03-20", "2016-03-25")
    f.loc[f["ts"] >= pd.Timestamp("2016-03-25", tz="UTC"), CH] -= 3.0   # repaired
    spans = {"T": [("2016-02-15", "2016-03-20")]}
    sim = dc.simulate(f, "T", tolerance=1.0, spans=spans)
    models = {m["month"]: m["model"] for m in sim["months"]}
    assert models["2016-03"] == "frozen"
    assert models["2016-04"] == "blank"
    assert "frozen" not in [models[m] for m in models if m >= "2016-04"]


# ----------------------------------------------------------- call-outs, summary
def test_consecutive_hot_days_are_one_call_out_at_the_end_of_the_first():
    days = [{"day": "2016-01-01", "flag": "hot", "fault_period": False},
            {"day": "2016-01-02", "flag": "hot", "fault_period": False},
            {"day": "2016-01-03", "flag": None, "fault_period": False},
            {"day": "2016-01-04", "flag": "hot", "fault_period": True}]
    calls = dc.call_outs(days)
    assert [c["first"] for c in calls] == ["2016-01-01 23:50", "2016-01-04 23:50"]
    assert calls[0]["days"] == 2


def test_the_summary_counts_false_and_fault_call_outs_apart():
    spans = {"T": [("2016-03-15", "2016-06-01")]}
    f = _faulty()
    sim = dc.simulate(f, "T", tolerance=1.0, spans=spans)
    s = dc.summarise(sim, dc.limit_call_outs(f, "T", spans))
    assert s["false_call_outs"] == 0
    assert s["first_fault_call_out"] == "2016-03-15 23:50"
    assert s["fault_days_hot"] == s["fault_days_checked"] > 0
    assert s["months_frozen"] >= 2
    assert 0.0 < s["checked_share"] <= 1.0


# ------------------------------------------------------------- the artefact
def test_the_committed_daily_check_is_consistent():
    import json

    path = ROOT / "results" / "daily_check.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    assert d["complete_run"]
    tols = {f"{t:g}" for t in d["protocol"]["tolerances_k"]}
    assert set(d["fleet"]) == tols
    for tol, f in d["fleet"].items():
        case = next(r for r in d["by_turbine"]
                    if r["turbine"] == "R80711" and f"{r['tolerance']:g}" == tol)
        assert f["case_false_call_outs_per_year"] == case["false_call_outs_per_year"]
        assert f["case_first_fault_call_out"] == case["first_fault_call_out"]
    for r in d["by_turbine"]:
        assert 0.0 <= r["checked_share"] <= 1.0
        assert r["fault_days_hot"] <= r["fault_days_checked"]
        assert sum(r["false_call_outs_by_year"].values()) == r["false_call_outs"]
        # Only the case turbine has a documented fault.
        if r["turbine"] != "R80711":
            assert r["fault_days_checked"] == 0


# ------------------------------------------- every sensor, and the forecast
def _rich_turbine(days: int = 120, start: str = "2016-01-01", seed: int = 1):
    """A turbine whose power follows ERA5 wind, with four gearbox sensors."""
    rng = np.random.default_rng(seed)
    n = days * 144
    ts = pd.date_range(start, periods=n, freq="10min", tz="UTC")
    d = np.arange(n) / 144.0
    wind = np.clip(9.5 + 4 * np.sin(2 * np.pi * d / 3.7)
                   + 0.8 * rng.standard_normal(n), 0.0, None)
    load = np.clip((wind - 3.0) / 9.0, 0.0, 1.0) ** 1.5
    # A daily cycle and a slow season, so every month sees the same ambients.
    amb = 8 + 4 * np.sin(2 * np.pi * d) + 1.5 * np.sin(2 * np.pi * d / 365.0)
    f = pd.DataFrame({"ts": ts, "P_norm": load, "era5_temp_2m": amb,
                      "era5_ws_100m": wind})
    for i, c in enumerate(dc.GEARBOX):
        f[c] = 35 + i + (30 - 2 * i) * load + 0.15 * amb + 0.4 * rng.standard_normal(n)
    return f


def test_predict_is_the_reference_without_the_reading():
    from windmill import multivariate
    f = _turbine(60)
    ref = multivariate.fit_channel_reference(f[f["P_norm"] > 0.02], CH, "P_norm",
                                             ambient_col="era5_temp_2m")
    load, amb, v = (f[c].to_numpy() for c in ("P_norm", "era5_temp_2m", CH))
    assert np.allclose(v - dc.predict(ref, load, amb), ref.apply(load, amb, v)[0],
                       equal_nan=True)


def test_each_sensor_has_its_own_model_and_a_fault_shows_on_its_own_sensor():
    f = _rich_turbine()
    f.loc[f["ts"] >= pd.Timestamp("2016-03-15", tz="UTC"), "Gost_avg"] += 3.0
    sim = dc.simulate(f, "T", tolerance=1.0, spans={}, channels=list(dc.GEARBOX))
    late = [d for d in sim["days"] if "2016-03-16" <= d["day"] <= "2016-03-31"
            and d.get("status") == "checked"]
    assert late
    for d in late:
        gap = {c: a - p for c, (a, p) in d["ch"].items()}
        assert gap["Gost_avg"] > 2.0
        assert all(abs(gap[c]) < 1.0 for c in gap if c != "Gost_avg")
        assert d["flag"] is None                    # the primary sensor is fine
        assert dc.gearbox_hot(d, 1.0) == 1          # one sensor: not two agree


def test_the_gearbox_rule_needs_two_sensors_to_agree():
    d = {"ch": {"Gb2t_avg": [70.0, 67.5], "Gb1t_avg": [66.0, 63.0],
                "Gost_avg": [60.0, 60.2], "Db2t_avg": [50.0, 40.0]}}
    assert dc.gearbox_hot(d, 2.0) == 2              # Db2t is not a gearbox sensor
    assert dc.gearbox_hot(d, 3.0) == 0


def test_a_perfect_forecast_gives_the_on_day_prediction(monkeypatch=None):
    saved = dict(dc.FORECAST_SIGMA)
    try:
        for L in dc.FORECAST_SIGMA:
            dc.FORECAST_SIGMA[L] = 0.0
        sim = dc.simulate(_rich_turbine(), "T", tolerance=1.0, spans={},
                          channels=list(dc.GEARBOX), leads=(1, 7))
    finally:
        dc.FORECAST_SIGMA.update(saved)
    days = [d for d in sim["days"] if d.get("fc") and "1" in d["fc"]]
    assert len(days) > 30
    err = [abs(d["fc"]["1"][c] - d["ch"][c][1]) for d in days for c in d["fc"]["1"]]
    # Only the power curve stands between them: a fraction of a kelvin.
    assert np.median(err) < 0.3 and max(err) < 1.5


def test_a_forecast_uses_the_model_in_force_when_it_was_issued():
    sim = dc.simulate(_rich_turbine(), "T", tolerance=1.0, spans={},
                      channels=list(dc.GEARBOX), leads=(1, 7))
    by_day = {d["day"]: d for d in sim["days"]}
    # January has no model: a 7-day forecast for 3 February was issued then.
    assert "7" not in by_day["2016-02-03"].get("fc", {})
    assert "1" in by_day["2016-02-03"]["fc"]
    assert "7" in by_day["2016-02-10"]["fc"]


def test_a_blank_day_says_why_and_how_much_has_been_collected():
    f = _drop(_turbine(150), "2016-03-01", "2016-03-05")
    sim = dc.simulate(f, "T", tolerance=1.0, spans={})
    jan = [d for d in sim["days"] if d["day"].startswith("2016-01")]
    assert {d["reason"] for d in jan} == {"start"}
    hold = [d for d in sim["days"] if d.get("reason") == "maintenance"]
    assert hold and all(d["maintenance"] == "2016-03-05" for d in hold)
    got = [d["collected"] for d in hold]
    assert got == sorted(got) and got[-1] > got[0] > 0
    assert got[-1] < dc.MONTH_OF_READINGS + 144 * 31


def test_the_sibling_normal_learns_a_wind_direction_offset():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2016-01-01", "2016-04-01", freq="10min", tz="UTC",
                        inclusive="left")
    n = len(idx)
    day = (idx - idx[0]).days
    direction = np.where(rng.random(day.max() + 1) < 0.25, 90.0, 250.0)[day]
    direction[(idx >= pd.Timestamp("2016-03-10", tz="UTC"))
              & (idx < pd.Timestamp("2016-03-20", tz="UTC"))] = 90.0
    base = 60 + rng.standard_normal(n) * 0.3
    temps = pd.DataFrame({t: base + 0.3 * rng.standard_normal(n)
                          for t in ("A", "B", "C", "D")}, index=idx)
    temps.loc[direction == 90.0, "A"] += 2.0        # warmer in easterly wind
    loads = pd.DataFrame(0.6, index=idx, columns=temps.columns)
    months = [{"month": "2016-03", "model_train_start": "2016-01-01",
               "model_train_end": "2016-03-01"}]
    d = pd.Series(direction, index=idx)
    aware = dc.sibling_change(temps, loads, d, "A", months, by_direction=True)
    blind = dc.sibling_change(temps, loads, d, "A", months, by_direction=False)
    day_ = pd.Timestamp("2016-03-17", tz="UTC")
    assert abs(aware[day_]) < 0.5
    assert blind[day_] > 1.5
    assert dc.sibling_call_outs(aware, 1.0, "A", {}) == []
    assert dc.sibling_call_outs(blind, 1.0, "A", {})


def test_recorded_maintenance_blanks_the_model_like_a_stand_in():
    """A repair from a failure logbook starts the model over, as a long stop does."""
    sim = dc.simulate(_turbine(150), "T", tolerance=1.0, spans={},
                      maintenance_extra=[pd.Timestamp("2016-03-05", tz="UTC")])
    models = {m["month"]: m for m in sim["months"]}
    assert models["2016-04"]["model"] == "blank"
    assert models["2016-05"]["model"] == "trained"
    assert models["2016-05"]["train_start"] >= "2016-03-05"
    hold = [d for d in sim["days"] if d.get("reason") == "maintenance"]
    assert hold and all(d["maintenance"] == "2016-03-05" for d in hold)
