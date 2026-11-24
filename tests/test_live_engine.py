"""The live engine behind the dashboard's explorer.

The engine is the part of the dashboard that runs the project against data
the reader chooses -- a turbine from the ENGIE site, a labelled CARE event,
or their own CSV. Three things have to hold for it to be worth having:

  * it must run the SAME detector the results were produced with, so what it
    shows is not a demo of a different thing;
  * it must report what it did -- rows, runtime, datasets joined -- so the
    reader can check the working;
  * it must refuse clearly when a file does not carry what the detector
    needs, rather than drawing an empty chart.

The upload path is the one that matters most, because it is the only one
that works on Streamlit Community Cloud, where the 19 GB of raw data is not
present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import live  # noqa: E402

SAMPLE = ROOT / "assets" / "sample_turbine.csv"


# --------------------------------------------------------------- inference
def _engie_like(n: int = 900) -> pd.DataFrame:
    """A synthetic frame with the ENGIE release's column names."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2017-01-01", periods=n, freq="10min", tz="UTC")
    return pd.DataFrame({
        "Date_time": ts.strftime("%Y-%m-%dT%H:%M:%S+01:00"),
        "P_avg": rng.uniform(0, 2000, n),
        "Ws_avg": rng.uniform(0, 25, n),
        "Git_avg": rng.uniform(40, 60, n),
        "Gost_avg": rng.uniform(45, 65, n),
        "Gb2t_avg": rng.uniform(50, 70, n),
        "era5_temp_2m": rng.uniform(5, 20, n),
    })


def test_inference_finds_the_engie_abbreviated_temperature_names():
    """The ENGIE release abbreviates its variables, and the inference must
    find them: `Gb2t_avg` is a gearbox bearing temperature, and a keyword
    search for "gearbox" or "bearing" does not match it."""
    m = live.infer_columns(_engie_like())
    assert m["ts"] == "Date_time"
    assert m["power"] == "P_avg"
    assert m["wind"] == "Ws_avg"
    assert m["ambient"] == "era5_temp_2m"
    assert "Gb2t_avg" in m["thermal"]
    assert "Gost_avg" in m["thermal"]
    # The ambient temperature is a DRIVER, not a component: it must not be
    # scored as one.
    assert "era5_temp_2m" not in m["thermal"]


def test_inference_finds_descriptive_column_names():
    """A file from a different publisher uses different words."""
    n = 600
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="10min",
                                   tz="UTC"),
        "Active Power (kW)": rng.uniform(0, 2000, n),
        "Wind Speed (m/s)": rng.uniform(0, 25, n),
        "Ambient Temperature (C)": rng.uniform(0, 20, n),
        "Gearbox Bearing Temperature (C)": rng.uniform(40, 70, n),
        "Generator Bearing Temperature (C)": rng.uniform(40, 70, n),
    })
    m = live.infer_columns(df)
    assert m["ts"] == "timestamp"
    assert m["power"] == "Active Power (kW)"
    assert m["wind"] == "Wind Speed (m/s)"
    assert m["ambient"] == "Ambient Temperature (C)"
    assert len(m["thermal"]) == 2, m["thermal"]
    assert "Ambient Temperature (C)" not in m["thermal"]


def test_inference_reports_what_it_found():
    """The reader has to be able to see how the file was read."""
    m = live.infer_columns(_engie_like())
    described = live._describe_mapping(m)
    assert "timestamp: Date_time" in described
    assert "power: P_avg" in described
    assert "ambient: era5_temp_2m" in described


# ------------------------------------------------------------- the upload
def test_the_upload_path_self_calibrates_and_reports_what_it_did():
    """An uploaded file is calibrated on its own history and scored on the
    rest, and the run reports rows, runtime and what was joined."""
    df = _engie_like(2000)
    result = live.run_upload(df)
    prov = result["provenance"]

    assert prov["rows"] == 2000
    assert prov["seconds"] >= 0.0
    assert any("uploaded CSV" in j for j in prov["joined"])
    assert any("self-calibration" in j for j in prov["joined"])
    assert prov["channels"], "no thermal channel was used"
    assert "reference" in prov and prov["reference"]

    frame = result["frame"]
    assert "z_rolling" in frame.columns
    assert "z_frozen" in frame.columns
    assert frame["value"].notna().any()


def test_the_upload_path_scores_the_later_part_of_the_file():
    """The reference is the EARLIER part of the file. If the engine scored
    the whole file against itself, every z would be near zero and the
    detector would look broken."""
    df = _engie_like(2000)
    result = live.run_upload(df, ref_frac=0.6)
    frame = result["frame"]
    assert frame["z_frozen"].notna().sum() > 0
    assert frame["z_frozen"].max() > 0.5, (
        "a self-calibrated reference scored on unseen data must produce "
        "non-trivial deviations"
    )


def test_the_upload_path_refuses_a_file_with_no_power_column():
    """Without the operating point the reference cannot be conditioned, and
    the reader must be told which column is missing."""
    df = _engie_like(600).drop(columns=["P_avg"])
    with pytest.raises(live.LiveEngineError) as exc:
        live.run_upload(df)
    assert "power" in str(exc.value).lower()


def test_the_upload_path_refuses_a_file_with_no_timestamp():
    df = _engie_like(600).drop(columns=["Date_time"])
    with pytest.raises(live.LiveEngineError) as exc:
        live.run_upload(df)
    assert "timestamp" in str(exc.value).lower()


def test_the_upload_path_refuses_a_file_with_no_temperature_channel():
    df = _engie_like(600).drop(columns=["Git_avg", "Gost_avg", "Gb2t_avg"])
    with pytest.raises(live.LiveEngineError) as exc:
        live.run_upload(df)
    assert "temperature" in str(exc.value).lower()


def test_the_upload_path_refuses_a_file_that_is_too_short():
    """The detector needs enough rows to fit a reference and score a window."""
    with pytest.raises(live.LiveEngineError) as exc:
        live.run_upload(_engie_like(50))
    assert "rows" in str(exc.value).lower()


def test_the_upload_path_refuses_an_empty_file():
    with pytest.raises(live.LiveEngineError):
        live.run_upload(pd.DataFrame())


def test_the_upload_path_uses_every_temperature_channel_but_shows_one():
    """The detector combines all of them (that is what max-z is for); the
    displayed channel is only what the reader looks at."""
    df = _engie_like(2000)
    result = live.run_upload(df, display_channel="Gost_avg")
    prov = result["provenance"]
    assert prov["display_channel"] == "Gost_avg"
    assert prov["available_channels"], "the reader cannot choose a channel"
    assert result["frame"]["value_label"].iloc[0] == "Gost_avg"


# -------------------------------------------------------------- the sample
def test_the_sample_file_is_committed():
    """A reviewer on Streamlit Cloud has no 19 GB of raw data, so the
    downloadable sample has to be in the repository."""
    assert SAMPLE.exists(), (
        "assets/sample_turbine.csv is missing -- the upload path cannot be "
        "demonstrated without it"
    )
    assert SAMPLE.stat().st_size > 100_000, "the sample file is suspiciously small"


def test_the_sample_file_is_a_real_turbine_slice():
    """The sample is a slice of the ENGIE release, so it carries that
    release's column names -- and the ERA5 ambient column, which is the
    only way a reviewer can see the weather-conditioned path work."""
    df = pd.read_csv(SAMPLE)
    assert len(df) > 5_000, "the sample has too few rows to calibrate on"
    for col in ("Date_time", "P_avg", "Ws_avg", "Gb2t_avg", "era5_temp_2m"):
        assert col in df.columns, f"the sample has no {col} column"
    assert df["P_avg"].notna().all()
    assert df["Gb2t_avg"].notna().all()
    # The ambient column is the reanalysis one, and it has to be plausible.
    assert df["era5_temp_2m"].between(-30, 55).all()


def test_the_engine_runs_on_the_committed_sample():
    """The sample must actually work: a reviewer downloads it, uploads it,
    and the detector has to score it."""
    df = pd.read_csv(SAMPLE)
    result = live.run_upload(df)
    prov = result["provenance"]
    assert prov["rows"] == len(df)
    assert prov["channels"], "no channel was used on the sample"
    assert result["frame"]["z_rolling"].notna().any()


def test_the_sample_carries_the_r80711_anomaly():
    """The sample covers September 2017, so it contains the case study's
    thermal signature -- a reader who uploads it sees the detector fire."""
    df = pd.read_csv(SAMPLE)
    sep = df[df["Date_time"].str.startswith("2017-09")]
    assert len(sep) > 1_000, "the sample does not cover September 2017"
    assert sep["Gb2t_avg"].max() > 80.0, (
        "the sample does not contain the anomaly it is supposed to show"
    )


def test_the_committed_sample_is_reproducible_from_the_raw_data():
    """The sample is a committed artefact, so it must be re-derivable.

    A committed file that only exists because someone once ran a script is
    not evidence -- it is a claim. Where the raw data is present, the file
    is regenerated and compared byte for byte.
    """
    if not ((ROOT / "data" / "raw").exists()
            and any((ROOT / "data" / "raw").rglob("*R80711*F99*.csv"))):
        pytest.skip("the raw ENGIE data is not present")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = live.build_sample_csv(Path(tmp) / "sample.csv", ROOT / "data")
        assert rebuilt.read_bytes() == SAMPLE.read_bytes(), (
            "the committed sample does not match what the pipeline produces -- "
            "run 'make snapshot' and commit the result"
        )


# --------------------------------------------------- the monthly live loop
def test_the_dispatch_episodes_group_consecutive_alarms():
    """A fault raises one phone call, not forty.

    The alarmed rows are grouped into runs, and only runs of at least six
    consecutive 10-minute samples -- one hour -- count as a dispatch.
    """
    flags = [False] * 5 + [True] * 10 + [False] * 3 + [True] * 2
    episodes = live.dispatch_episodes(flags, min_run=6)
    assert len(episodes) == 1
    assert episodes[0]["rows"] == 10
    assert episodes[0]["start"] == 5
    assert episodes[0]["end"] == 14


def test_short_alarm_runs_are_not_dispatches():
    """A blip is not a truck roll."""
    flags = [True] * 5 + [False] * 2 + [True] * 20
    episodes = live.dispatch_episodes(flags, min_run=6)
    assert len(episodes) == 1
    assert episodes[0]["rows"] == 20


def test_no_alarms_is_no_dispatches():
    assert live.dispatch_episodes([False] * 100, min_run=6) == []


def test_the_minimum_run_length_is_respected():
    flags = [True] * 5 + [False] + [True] * 5
    assert live.dispatch_episodes(flags, min_run=6) == []
    assert len(live.dispatch_episodes(flags, min_run=5)) == 2


def test_the_cost_maths_is_exact():
    cost = live.cost_impact(14, 3, 2000.0)
    assert cost["avoided_dispatches"] == 11
    assert cost["cost_saved"] == 11 * 2000.0
    assert cost["naive_episodes"] == 14
    assert cost["conditioned_episodes"] == 3


def test_the_cost_never_claims_a_negative_saving():
    """A model that alarms MORE must not show a negative saving."""
    cost = live.cost_impact(3, 14, 2000.0)
    assert cost["avoided_dispatches"] == 0
    assert cost["cost_saved"] == 0


def test_the_early_intervention_value_is_only_claimed_when_the_fault_is_caught():
    missed = live.cost_impact(10, 1, 2000.0, fault_detected=False)
    caught = live.cost_impact(10, 1, 2000.0, fault_detected=True)
    assert missed["early_intervention_value"] == 0
    assert caught["early_intervention_value"] == 185_000.0


# ------------------------------------------------------- the monthly fit
def _monthly(month="2017-09"):
    return live.monthly_cross_check(month, ROOT / "data", source="sample")


def test_the_monthly_cross_check_trains_on_the_month_before():
    """A fresh fit: the reference is the thirty days before the target month
    and nothing else. No carry-over from an older baseline."""
    result = _monthly("2017-09")
    prov = result["provenance"]
    assert prov["target_month"] == "2017-09"
    assert prov["train_month"] == "2017-08"
    assert prov["train_rows"] > 1000


def test_the_monthly_cross_check_reports_its_timing():
    """The fit is measured, not asserted."""
    result = _monthly("2017-09")
    prov = result["provenance"]
    assert prov["fit_seconds"] > 0
    assert prov["fit_seconds"] < 60
    assert prov["seconds"] >= prov["fit_seconds"]


def test_the_monthly_cross_check_excludes_anomalous_training_rows():
    """A fault that was already developing last month must not become the
    new normal, so the rows the reference itself calls anomalous are dropped
    and the reference refitted."""
    result = _monthly("2017-09")
    prov = result["provenance"]
    assert prov["dropped_rows"] > 0, (
        "nothing was excluded, so a contaminated training window would "
        "silently become the reference"
    )
    assert prov["clean_rows"] < prov["train_rows"]
    assert prov["clean_rows"] > 0


def test_the_monthly_cross_check_removes_the_weather_false_alarms():
    """The whole point: a reference that is not refreshed for the month's
    conditions raises a flood of false dispatches, and the conditioned model
    does not."""
    result = _monthly("2017-09")
    prov = result["provenance"]
    assert prov["naive_alarm_pct"] > prov["conditioned_alarm_pct"]
    assert prov["naive_false_dispatches"] > prov["conditioned_false_dispatches"]
    assert result["cost"]["avoided_dispatches"] > 0
    assert result["cost"]["cost_saved"] > 0


def test_the_monthly_cross_check_judges_every_arm_on_one_scale():
    """Two models whose residual scales differ cannot be compared on alarm
    rate, so the common scale has to be imposed."""
    result = _monthly("2017-09")
    frame = result["frame"]
    # Every arm is thresholded at the same multiple of the same spread.
    for arm in ("static", "load", "weather"):
        z = frame[f"z_{arm}"].to_numpy()
        alarms = frame[f"alarm_{arm}"].to_numpy()
        above = z > 4.0
        assert (alarms == above).all(), arm
    assert result["threshold"] == pytest.approx(
        result["provenance"]["threshold_k"], rel=1e-3)


def test_the_monthly_cross_check_splits_genuine_from_false_dispatches():
    """An episode on the real fault is worth something; one that is not is a
    wasted truck roll. They must not be counted as the same thing."""
    result = _monthly("2017-09")
    episodes = result["episodes"]
    assert len(episodes["naive"]) == (
        len(episodes["naive_false"]) + len(episodes["naive_true"]))
    assert len(episodes["conditioned"]) == (
        len(episodes["conditioned_false"]) + len(episodes["conditioned_true"]))
    # And the false ones are the ones that do not touch an overheating row.
    hot = result["frame"]["overheating"].to_numpy()
    for e in episodes["naive_false"]:
        assert not hot[e["start"]:e["end"] + 1].any()


def test_the_monthly_cross_check_refuses_a_month_with_no_predecessor():
    """There is nothing to train on for the first month in the data."""
    first = sorted(live._monthly_frame(ROOT / "data", "sample", "R80711")["month"].unique())[0]
    with pytest.raises(live.LiveEngineError):
        live.monthly_cross_check(first, ROOT / "data", source="sample")


def test_the_training_holds_at_least_a_month_of_healthy_readings():
    """The rule: at least one month of HEALTHY data, after the unhealthy rows
    are trimmed, starting from the whole previous calendar month and reaching
    further back only as far as needed, at most about three months."""
    for month in live.available_months(ROOT / "data", "sample"):
        prov = live.monthly_cross_check(month, ROOT / "data", source="sample")["provenance"]
        assert prov["clean_rows"] >= live.MONTH_OF_READINGS, (month, prov["clean_rows"])
        start = pd.Timestamp(prov["train_start"])
        m_start = pd.Timestamp(month + "-01")
        # never less than the whole previous calendar month ...
        assert start <= m_start - pd.offsets.MonthBegin(1), (month, start)
        # ... and never more than about three months back
        assert (m_start - start).days <= live.MAX_TRAIN_LOOKBACK_DAYS, (month, start)
        assert prov["train_end"] == str(m_start - pd.Timedelta(days=1))[:10]


def test_a_month_without_a_month_of_healthy_data_behind_it_is_refused():
    """The sample starts in May 2017: June has only May behind it, which is
    less than a month of readings, so June is refused, not scored on too little,
    and is not offered."""
    with pytest.raises(live.LiveEngineError, match="less than a month"):
        live.monthly_cross_check("2017-06", ROOT / "data", source="sample")
    assert "2017-06" not in live.available_months(ROOT / "data", "sample")


def test_the_monthly_cross_check_refuses_a_month_that_is_not_there():
    with pytest.raises(live.LiveEngineError):
        live.monthly_cross_check("1999-01", ROOT / "data", source="sample")


def test_every_available_month_after_the_first_can_be_cross_checked():
    months = live.available_months(ROOT / "data", "sample")
    assert len(months) >= 4, "the sample should cover several months"
    for month in months[1:]:
        result = live.monthly_cross_check(month, ROOT / "data", source="sample")
        assert result["provenance"]["rows"] > 0


def test_at_the_default_threshold_the_fault_is_still_caught():
    """The default operating point must remove the false alarms AND keep the
    genuine one.

    This is the behaviour the dashboard's cost card depends on: if the
    conditioned model simply went quiet, the saving would be worthless. At
    the default 3-sigma threshold the R80711 overheat is still raised as a
    genuine dispatch, while every false dispatch is avoided.
    """
    result = live.monthly_cross_check("2017-09", ROOT / "data", source="sample",
                                      z_threshold=3.0)
    prov = result["provenance"]
    assert prov["conditioned_true_dispatches"] > 0, (
        "the genuine fault was not raised as a dispatch at the default "
        "threshold -- a model that never alarms is worthless"
    )
    assert prov["conditioned_false_dispatches"] == 0, (
        "the conditioned model still raised false dispatches"
    )
    assert prov["naive_false_dispatches"] > 0
    assert result["cost"]["avoided_dispatches"] > 0


def test_the_false_dispatch_saving_survives_every_threshold():
    """The saving is not an artefact of one threshold choice."""
    for z in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        result = live.monthly_cross_check("2017-09", ROOT / "data",
                                          source="sample", z_threshold=z)
        prov = result["provenance"]
        assert prov["naive_false_dispatches"] > prov["conditioned_false_dispatches"], (
            f"at z={z} the conditioned model did not reduce the false "
            f"dispatches: {prov['naive_false_dispatches']} vs "
            f"{prov['conditioned_false_dispatches']}"
        )


# ------------------------------------------------ the daily aggregation
def test_the_daily_summary_has_one_row_per_day():
    """The daily view is what a planner reads: one row per day, with the
    columns the dashboard plots."""
    result = _monthly("2017-09")
    daily = live.daily_temperature_summary(result["frame"])
    assert len(daily) > 1, daily
    days = list(daily["day"])
    assert days == sorted(days), "the days are not in order"
    for col in live.DAILY_TEMPERATURE_COLUMNS:
        assert col in daily.columns, f"{col} is missing"
    assert daily["rows"].sum() == len(result["frame"]), (
        "the daily row counts do not add up to the ten-minute rows"
    )


def test_the_daily_summary_agrees_with_the_ten_minute_rows():
    """The aggregation must be the same arithmetic, not a second model."""
    result = _monthly("2017-09")
    frame = result["frame"].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    daily = live.daily_temperature_summary(frame)

    for idx in (0, 5, 13, len(daily) - 1):
        day = daily["day"].iloc[idx]
        same_day = frame[frame["ts"].dt.floor("1D") == day]
        assert len(same_day) > 0, day
        assert daily["daily_actual_temp_max"].iloc[idx] == pytest.approx(
            same_day["Gb2t_avg"].max(), abs=1e-6)
        assert daily["daily_actual_temp_mean"].iloc[idx] == pytest.approx(
            same_day["Gb2t_avg"].mean(), abs=1e-6)
        assert daily["daily_predicted_temp_max"].iloc[idx] == pytest.approx(
            same_day["predicted_weather"].max(), abs=1e-6)
        assert daily["daily_predicted_temp_mean"].iloc[idx] == pytest.approx(
            same_day["predicted_weather"].mean(), abs=1e-6)
        assert daily["daily_ambient_temp"].iloc[idx] == pytest.approx(
            same_day["era5_temp_2m"].mean(), abs=1e-6)
        assert daily["daily_residual_mean"].iloc[idx] == pytest.approx(
            same_day["residual_weather"].mean(), abs=1e-6)
        assert daily["daily_residual_max"].iloc[idx] == pytest.approx(
            same_day["residual_weather"].max(), abs=1e-6)
        assert daily["rows"].iloc[idx] == len(same_day)


def test_the_daily_summary_shows_the_september_overheat():
    """The fault has to be visible at day resolution -- that is the whole
    point of offering a daily view."""
    result = _monthly("2017-09")
    daily = live.daily_temperature_summary(result["frame"])
    threshold = result["provenance"]["threshold_k"]

    worst = daily.loc[daily["daily_residual_max"].idxmax()]
    assert worst["daily_residual_max"] > threshold, (
        f"the worst day only reaches {worst['daily_residual_max']:.2f} K "
        f"against a {threshold:.2f} K threshold"
    )
    assert worst["daily_actual_temp_max"] > worst["daily_predicted_temp_max"], (
        "on the worst day the actual temperature did not exceed the "
        "model's prediction"
    )
    # And it is not just noise: several days clear the threshold.
    over = daily[daily["daily_residual_max"] > threshold]
    assert len(over) >= 2, f"only {len(over)} day(s) cleared the threshold"


def test_the_daily_summary_of_nothing_is_empty():
    """An empty frame must give an empty summary with the same columns,
    not an exception."""
    daily = live.daily_temperature_summary(pd.DataFrame())
    assert len(daily) == 0
    for col in live.DAILY_TEMPERATURE_COLUMNS:
        assert col in daily.columns, col


# ------------------------------------- the recalibrated static baseline
def test_the_recalibrated_static_baseline_is_reported_and_partitioned():
    """The static arm judged on the weather model's scale is not a fair
    baseline, so the same arm is also calibrated on its OWN residuals. Its
    dispatches must be reported, and split into false and genuine like the
    others, or the saving cannot be quoted against it."""
    res = _monthly("2017-09")
    prov, eps = res["provenance"], res["episodes"]
    assert len(eps["recalibrated"]) == (
        len(eps["recalibrated_false"]) + len(eps["recalibrated_true"]))
    assert prov["recalibrated_static_episodes"] == len(eps["recalibrated"])
    assert (prov["recalibrated_static_false_dispatches"]
            == len(eps["recalibrated_false"]))
    assert (res["cost"]["recalibrated_static_false_dispatches"]
            == len(eps["recalibrated_false"]))


def test_calibrated_on_its_own_residuals_the_static_arm_is_far_quieter():
    """The whole reason it is reported: judged on the weather model's scale the
    static arm alarms whenever the machine runs hard, and that is a
    calibration effect, not a modelling one."""
    prov = _monthly("2017-09")["provenance"]
    assert (prov["recalibrated_static_false_dispatches"]
            < prov["naive_false_dispatches"]), prov
    assert prov["recalibrated_static_threshold_k"] is not None
