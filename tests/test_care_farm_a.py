"""Guards on CARE Farm A's true dates, EDP's records, and the alarm triage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import edp, multivariate, triage  # noqa: E402

RESULT = ROOT / "results" / "care_farm_a.json"


# ------------------------------------------------------------- true dates
def test_a_file_starting_before_the_leap_day_moved_one_day_more():
    assert edp.shift_days("2022-01-01 00:00") == 2192
    assert edp.shift_days("2022-02-28 23:50") == 2192
    assert edp.shift_days("2022-03-03 14:00") == 2191
    # 2016-01-01 + 2192 days is 2022-01-01; 2016-04-24 + 2191 days is 2022-04-24.
    assert pd.Timestamp("2016-01-01") + pd.Timedelta(days=2192) == pd.Timestamp("2022-01-01")
    assert pd.Timestamp("2016-04-24") + pd.Timedelta(days=2191) == pd.Timestamp("2022-04-24")


def test_overlapping_files_become_one_record_on_true_dates():
    early = pd.DataFrame({"event_id": 1, "asset_id": 0,
                          "time_stamp": pd.date_range("2022-01-01", periods=3, freq="10min"),
                          "v": [1.0, 2.0, 3.0]})
    later = pd.DataFrame({"event_id": 2, "asset_id": 0,     # the same readings, a later file
                          "time_stamp": pd.date_range("2022-03-05", periods=2, freq="10min"),
                          "v": [9.0, 9.5]})
    rec = edp.to_true_dates(pd.concat([early, later]))
    assert rec["ts"].iloc[0] == pd.Timestamp("2016-01-01")
    assert rec["turbine"].unique().tolist() == ["T07"]
    assert len(rec) == 5 and rec["ts"].is_unique


# ------------------------------------------------------- the turbines' alarms
def test_alarm_texts_are_grouped_without_their_numbers():
    s = pd.Series(["Hot HV trafo 270Â°C      0kW", "Hot generator 146Â°C  0kW"])
    assert edp.template(s).tolist() == ["Hot HV trafo #°C #kW", "Hot generator #°C #kW"]


def _log(rows):
    df = pd.DataFrame(rows, columns=["turbine", "ts", "remark"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["template"] = edp.template(df["remark"])
    return df


def test_consecutive_alarm_days_are_one_spell_and_pumps_are_not_temperature():
    log = _log([("T01", "2016-07-01 10:00", "Hot HV trafo 270°C 0kW"),
                ("T01", "2016-07-02 09:00", "Hot HV trafo 271°C 0kW"),
                ("T01", "2016-07-05 09:00", "Hot HV trafo 271°C 0kW"),
                ("T01", "2016-07-05 10:00", "Thermoerr. oil pump F410/ 57°C")])
    sp = edp.alarm_spells(log)
    assert sp["family"].tolist() == ["TRANSFORMER", "TRANSFORMER"]
    assert sp["days"].tolist() == [2, 1] and sp["entries"].tolist() == [2, 1]


def test_a_spell_is_followed_by_a_failure_only_of_its_component_in_time():
    spells = pd.DataFrame({"family": ["TRANSFORMER"] * 3, "turbine": ["T07", "T07", "T01"],
                           "start": pd.to_datetime(["2016-07-03", "2016-05-01", "2016-07-03"]),
                           "end": pd.to_datetime(["2016-07-03", "2016-05-01", "2016-07-03"]),
                           "days": 1, "entries": 1})
    lb = pd.DataFrame({"turbine": ["T07", "T07"], "family": ["TRANSFORMER", "TRANSFORMER"],
                       "ts": pd.to_datetime(["2016-07-10", "2016-07-11"]),
                       "remark": ["High temperature transformer", "Transformer fan damaged"]})
    lb["thermal"] = lb["remark"].str.contains(edp.THERMAL_FAILURE, regex=True)
    out = edp.label_spells(spells, lb, 30)
    assert out["followed_thermal"].tolist() == [True, False, False]


def test_a_failed_temperature_sensor_is_not_a_heat_failure():
    import re
    for remark, heat in (("High temperature transformer", True),
                         ("Gearbox bearings damaged", True),
                         ("Gearbox pump damaged", False),
                         ("Oil leakage in Hub", False),
                         ("Generator temperature sensor failure", False)):
        got = bool(re.search(edp.THERMAL_FAILURE, remark)) and not re.search(edp.SENSOR_FAILURE, remark)
        assert got == heat, remark


# ------------------------------------------------------------------ triage
def _ref():
    rng = np.random.default_rng(0)
    n = 5000
    load = rng.uniform(0.05, 1.0, n)
    amb = rng.uniform(0, 30, n)
    temp = 60 + 40 * load + 0.3 * (amb - 15) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"P_norm": load, "amb": amb, "T": temp})
    return multivariate.fit_channel_reference(df, "T", "P_norm", ambient_col="amb")


def test_the_reading_splits_into_typical_load_weather_and_excess():
    ref = _ref()
    load = np.array([0.9, 0.5]); amb = np.array([28.0, 15.0])
    actual = np.array([60 + 40 * 0.9 + 0.3 * 13 + 8.0, 60 + 40 * 0.5])   # 8 K excess on the first
    p = triage.decompose(ref, load, amb, actual, typical_load=0.5)
    total = p["typical"] + p["load"] + p["weather"] + p["excess"]
    assert np.allclose(total, actual)
    assert 14 < p["load"][0] < 18          # 0.4 of rated above typical
    assert 2.5 < p["weather"][0] < 5.5     # a warm day
    assert 6.5 < p["excess"][0] < 9.5
    assert abs(p["excess"][1]) < 1.5


def test_the_verdict_follows_the_excess_and_names_the_bigger_cause():
    assert triage.verdict(8.0, None, 10.0, 1.0) == "real excess heat"
    assert triage.verdict(2.0, 9.0, 10.0, 1.0) == "real excess heat"     # siblings see it
    assert triage.verdict(2.0, 1.0, 10.0, 1.0) == "explained by load"
    assert triage.verdict(2.0, 1.0, 1.0, 4.0) == "explained by weather"
    assert triage.verdict(None, None, None, None) == "no verdict"


def test_the_summary_counts_failures_inside_each_verdict():
    s = pd.DataFrame({"verdict": ["real excess heat", "real excess heat", "explained by load"],
                      "followed_thermal": [True, False, False]})
    t = triage.summarise(s)
    assert t["real excess heat"] == {"alarms": 2, "followed_by_failure": 1, "no_failure": 1}
    assert t["all"]["alarms"] == 3 and t["no verdict"]["alarms"] == 0


# ---------------------------------------------------------------- artefact
def test_the_committed_farm_a_result_is_consistent():
    if not RESULT.exists():
        return
    d = json.loads(RESULT.read_text())
    i = d["integrity"]
    assert i["overlapping_copies"]["identical_share"] == 1.0
    assert all(v["identical_share"] > 0.999 for v in i["care_vs_edp_scada"].values())
    assert i["normal_events_with_logbook_failure"] == 0
    assert i["anomaly_events_matching_logbook"] >= i["anomaly_events"] - 1
    t = d["triage"]["table"]
    assert sum(t[v]["alarms"] for v in triage.VERDICTS) == t["all"]["alarms"]
    assert t["all"]["alarms"] == len(d["triage"]["alarms"])
    for a in d["triage"]["alarms"]:
        if a.get("excess") is not None:
            parts = a["typical"] + a["load"] + a["weather"] + a["excess"]
            assert abs(parts - a["reading"]) < 0.1, a


def test_the_committed_engie_triage_is_consistent():
    path = ROOT / "results" / "engie_triage.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())["triage"]
    t = d["table"]
    assert sum(t[v]["alarms"] for v in triage.VERDICTS) == t["all"]["alarms"] == len(d["alarms"])
    for a in d["alarms"]:
        if a.get("excess") is not None:
            assert abs(a["typical"] + a["load"] + a["weather"] + a["excess"] - a["reading"]) < 0.1
