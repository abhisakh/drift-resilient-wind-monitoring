"""Guards on what the dashboard's daily pages show."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import daily_view as dv  # noqa: E402

SERIES = ROOT / "results" / "daily_series.json"


def _series():
    days = ["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04",
            "2017-01-05", "2017-01-06"]
    return {"turbines": {"T": {
        "day": days,
        "status": ["blank", "checked", "checked", "thin", "checked", "blank"],
        "reason": ["start", None, None, None, None, "maintenance"],
        "state": [None, "trained", "trained", "trained", "frozen", None],
        "flag": [None, None, "hot", None, "hot", None],
        "maintenance": [None] * 5 + ["2017-01-06"],
        "collected": [None] * 5 + [2160],
        "fault_period": [False] * 6,
        "gearbox_hot": [None, 0, 3, None, 4, None],
        "sibling": [None, 0.1, 0.4, None, 3.0, None],
        "sibling_overall": [None, 0.1, 0.5, None, 3.1, None],
        "sensors": {"Gb2t_avg": {
            "actual": [None, 60.0, 66.0, None, 70.0, None],
            "pred": [None, 59.5, 62.0, None, 65.0, None],
            "f7": [None, 61.0, 61.0, None, 64.0, None]}}}}}


def test_the_traffic_light_follows_the_status_and_the_tolerance():
    df = dv.frame(_series(), "T", "Gb2t_avg")
    assert list(dv.lights(df, 2.0)) == ["none", "within", "hot", "thin", "hot", "hold"]
    assert list(dv.lights(df, 5.0))[2] == "within"


def test_the_badge_says_what_the_model_is_doing():
    s = _series()
    assert dv.badge(s, "T", "2017-01-02", 4320)["kind"] == "monitoring"
    assert dv.badge(s, "T", "2017-01-05", 4320)["kind"] == "frozen"
    hold = dv.badge(s, "T", "2017-01-06", 4320)
    assert hold["kind"] == "hold" and hold["share"] == 0.5
    assert "50%" in hold["text"] and "2017-01-06" in hold["text"]
    assert dv.badge(s, "T", "2016-12-01", 4320)["kind"] == "none"


def test_the_demo_card_counts_hot_days_and_forecast_error():
    df = dv.frame(_series(), "T", "Gb2t_avg")
    card = dv.demo_summary(df, 2.0, 7)
    assert card["days_checked"] == 3 and card["days_hot"] == 2
    assert card["first_hot"] == "2017-01-03"
    assert card["mae_forecast"] == round((1.0 + 5.0 + 6.0) / 3, 2)


def test_the_committed_series_holds_the_demo():
    s = dv.load(SERIES)
    if s is None:
        return
    assert set(s["turbines"]) == {"R80711", "R80721", "R80736", "R80790"}
    for t, col in s["turbines"].items():
        n = len(col["day"])
        assert all(len(col[k]) == n for k in ("status", "reason", "sibling"))
        for c, v in col["sensors"].items():
            assert all(len(x) == n for x in v.values()), (t, c)
    healthy = dv.frame(s, dv.DEMO["healthy"], "Gb2t_avg",
                       dv.DEMO["start"], dv.DEMO["end"])
    faulty = dv.frame(s, dv.DEMO["faulty"], "Gb2t_avg",
                      dv.DEMO["start"], dv.DEMO["end"])
    tol = s["tolerance_k"]
    h = dv.demo_summary(healthy, tol, dv.DEMO["lead"])
    f = dv.demo_summary(faulty, tol, dv.DEMO["lead"])
    assert h["days_hot"] == 0 and h["days_checked"] >= 15
    assert f["days_hot"] >= 8 and f["first_hot"] == "2017-08-15"
    assert SERIES.stat().st_size < 4e6
