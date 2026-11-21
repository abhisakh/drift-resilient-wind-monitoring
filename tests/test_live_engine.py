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
