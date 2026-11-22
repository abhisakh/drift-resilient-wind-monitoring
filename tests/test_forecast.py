"""The forecasting module: the forecast, the power curve, and the what-if.

Everything here is offline-safe. The one function that touches the network,
`fetch_forecast`, is tested against a MOCKED response -- a test suite that
depended on Open-Meteo being reachable would fail on a train, in CI, and
whenever their API had a bad afternoon, and would tell you nothing about
whether the parsing is right.

The fallback matters as much as the forecast: the dashboard has to work on
Streamlit Community Cloud and in CI, where there may be no network, so the
sample-derived path is tested as carefully as the live one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import forecast as fc  # noqa: E402

SAMPLE = ROOT / "assets" / "sample_turbine.csv"


# ------------------------------------------------------------- the mock API
class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _payload(n=48):
    return {
        "hourly": {
            "time": [f"2026-01-0{1 + i // 24}T{i % 24:02d}:00" for i in range(n)],
            "wind_speed_100m": [5.0 + (i % 10) for i in range(n)],
            "temperature_2m": [2.0 + (i % 5) for i in range(n)],
            "surface_pressure": [1010.0 + (i % 3) for i in range(n)],
        }
    }


@pytest.fixture
def mocked_get(monkeypatch):
    """Replace requests.get with a recorder that returns a fixed payload."""
    calls = {}

    def _fake_get(url, params=None, timeout=None):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return _Response(_payload())

    import requests
    monkeypatch.setattr(requests, "get", _fake_get)
    return calls


# ------------------------------------------------------------ the forecast
def test_the_forecast_is_fetched_from_open_meteo(mocked_get):
    df = fc.fetch_forecast(48.45, 5.59, days=7)
    assert mocked_get["url"] == fc.FORECAST_URL
    assert mocked_get["params"]["latitude"] == 48.45
    assert mocked_get["params"]["longitude"] == 5.59
    assert mocked_get["params"]["forecast_days"] == 7
    assert "wind_speed_100m" in mocked_get["params"]["hourly"]
    assert len(df) == 48
    for col in ("ts", "wind_speed_100m", "temperature_2m", "surface_pressure"):
        assert col in df.columns


def test_a_network_failure_becomes_a_forecast_unavailable(monkeypatch):
    """A dead network must not be a crash, and must not be silent either."""

    def _boom(*a, **k):
        raise OSError("network unreachable")

    import requests
    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(fc.ForecastUnavailable):
        fc.fetch_forecast(48.45, 5.59)


def test_a_malformed_response_becomes_a_forecast_unavailable(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: _Response({"hourly": {"time": []}}))
    with pytest.raises(fc.ForecastUnavailable):
        fc.fetch_forecast(48.45, 5.59)


def test_an_http_error_becomes_a_forecast_unavailable(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: _Response({}, status=503))
    with pytest.raises(fc.ForecastUnavailable):
        fc.fetch_forecast(48.45, 5.59)


# ------------------------------------------------------------ air density
def test_the_air_density_at_the_reference_conditions():
    """At 15 degrees C and 1013.25 hPa the air is 1.225 kg/m3."""
    rho = fc.air_density(15.0, 1013.25)
    assert float(rho[0]) == pytest.approx(1.225, abs=0.002)


def test_cold_air_is_denser_than_warm_air():
    cold = fc.air_density(0.0, 1013.25)
    warm = fc.air_density(25.0, 1013.25)
    assert float(cold[0]) > float(warm[0])


def test_higher_pressure_is_denser_air():
    low = fc.air_density(10.0, 980.0)
    high = fc.air_density(10.0, 1030.0)
    assert float(high[0]) > float(low[0])


# ------------------------------------------------------------ power curve
def _synthetic_scada(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    ws = rng.uniform(0, 18, n)
    power = np.clip((ws ** 3) * 4.0, 0, 2050)
    return pd.DataFrame({"Ws_avg": ws, "P_avg": power})


def test_the_power_curve_is_built_from_the_measured_scada():
    curve = fc.power_curve_from_frame(_synthetic_scada(), "Ws_avg", "P_avg")
    assert len(curve) > 5
    assert curve["wind_speed"].is_monotonic_increasing
    assert curve["power_kw"].max() > 500
    assert (curve["n"] >= 10).all(), "a sparse bin must be dropped, not kept"


def test_a_power_curve_needs_enough_rows():
    with pytest.raises(ValueError):
        fc.power_curve_from_frame(_synthetic_scada(n=50), "Ws_avg", "P_avg")


def test_the_predicted_power_rises_with_the_wind():
    curve = fc.power_curve_from_frame(_synthetic_scada(), "Ws_avg", "P_avg")
    ws = np.array([2.0, 6.0, 10.0, 14.0])
    p = fc.predict_power(curve, ws)
    assert np.all(np.diff(p) >= 0), p


def test_denser_air_predicts_more_power():
    """The same wind speed in denser air carries more energy."""
    curve = fc.power_curve_from_frame(_synthetic_scada(), "Ws_avg", "P_avg")
    thin = fc.predict_power(curve, [8.0], 1.10)
    dense = fc.predict_power(curve, [8.0], 1.30)
    assert float(dense[0]) > float(thin[0])


def test_the_density_correction_is_the_cube_root():
    curve = fc.power_curve_from_frame(_synthetic_scada(), "Ws_avg", "P_avg")
    base = fc.predict_power(curve, [8.0], 1.0)
    corrected = fc.predict_power(curve, [8.0], 1.3)
    assert float(corrected[0]) / float(base[0]) == pytest.approx(1.3 ** (1 / 3), rel=1e-9)


# ------------------------------------------------------------- the what-if
def test_the_what_if_shifts_move_the_prediction_the_right_way():
    frame = fc.sample_forecast(SAMPLE, days=7)
    curve = fc.sample_power_curve(SAMPLE)
    base = fc.forecast_frame(frame, curve)
    windier = fc.forecast_frame(frame, curve, wind_shift=3.0)
    colder = fc.forecast_frame(frame, curve, temp_shift=-10.0)

    assert windier["predicted_power_kw"].mean() > base["predicted_power_kw"].mean()
    assert colder["predicted_power_kw"].mean() > base["predicted_power_kw"].mean()


def test_the_forecast_frame_carries_the_diagnostics():
    frame = fc.sample_forecast(SAMPLE, days=7)
    curve = fc.sample_power_curve(SAMPLE)
    out = fc.forecast_frame(frame, curve, wind_shift=1.0, temp_shift=-2.0)
    for col in ("wind_speed_shifted", "temperature_shifted", "air_density",
                "density_ratio", "predicted_power_kw"):
        assert col in out.columns
    assert (out["wind_speed_shifted"] == out["wind_speed_100m"] + 1.0).all()
    assert (out["temperature_shifted"] == out["temperature_2m"] - 2.0).all()


# ------------------------------------------------------- the offline path
def test_the_offline_fallback_is_deterministic():
    """It has to be: the dashboard shows it as a fixed reference, and a
    fallback that changed on every rerun would be worse than useless."""
    a = fc.sample_forecast(SAMPLE, days=7)
    b = fc.sample_forecast(SAMPLE, days=7)
    pd.testing.assert_frame_equal(a, b)


def test_the_offline_fallback_is_seven_days_of_hours():
    frame = fc.sample_forecast(SAMPLE, days=7)
    assert len(frame) == 7 * 24
    assert frame["ts"].is_monotonic_increasing
    for col in ("wind_speed_100m", "temperature_2m", "surface_pressure"):
        assert frame[col].notna().all()


def test_the_offline_fallback_comes_from_the_sample():
    """Its wind speeds have to be the sample's own, not invented."""
    frame = fc.sample_forecast(SAMPLE, days=7)
    raw = pd.read_csv(SAMPLE)
    assert frame["wind_speed_100m"].min() >= raw["Ws_avg"].min() - 0.5
    assert frame["wind_speed_100m"].max() <= raw["Ws_avg"].max() + 0.5


def test_the_committed_sample_builds_a_power_curve():
    curve = fc.sample_power_curve(SAMPLE)
    assert len(curve) > 5
    assert curve["power_kw"].max() > 1000


# ------------------------------------------------------------- checkpoints
def test_the_training_window_is_the_thirty_days_immediately_before():
    """The rule, and it is not negotiable: the reference is the thirty days
    immediately before the prediction window, and nothing else."""
    start, end = fc.training_window("2017-09-01")
    assert start == "2017-08-02"
    assert end == "2017-09-01"


def test_the_training_window_never_overlaps_the_prediction_window():
    """The interval is half-open, so the last training day is the day before
    the first predicted one."""
    for start in ("2017-09-01", "2017-10-01", "2017-11-01", "2016-09-01"):
        w_start, w_end = fc.training_window(start)
        assert w_end == start
        assert w_start < w_end
        assert (pd.Timestamp(w_end) - pd.Timestamp(w_start)).days == 30


def test_the_curated_checkpoints_are_the_ones_the_brief_names():
    assert fc.CHECKPOINTS["checkpoint_1"]["prediction_start"] == "2017-09-01"
    assert fc.CHECKPOINTS["checkpoint_1"]["prediction_end"] == "2017-09-07"
    assert fc.CHECKPOINTS["checkpoint_2"]["prediction_start"] == "2017-10-01"
    assert fc.CHECKPOINTS["checkpoint_2"]["prediction_end"] == "2017-10-07"
    assert fc.CHECKPOINTS["checkpoint_3"]["prediction_start"] == "2017-11-01"
    assert fc.CHECKPOINTS["checkpoint_3"]["prediction_end"] == "2017-11-07"


def test_the_stale_baselines_are_the_same_rule_one_and_two_years_back():
    """A stale baseline is not a different rule -- it is the same rule, with
    the window moved. So the only thing that differs between the three fits
    is the age of the reference."""
    bundle = _sample_bundle()
    p = bundle["fits"]["primary"]
    one = bundle["fits"]["1_year_back"]
    two = bundle["fits"]["2_year_back"]
    assert p["window_end"] == "2017-09-01"
    assert one["window_end"] == "2016-09-01"
    assert two["window_end"] == "2015-09-01"
    for fit in (p, one, two):
        assert (pd.Timestamp(fit["window_end"])
                - pd.Timestamp(fit["window_start"])).days == 30


def test_the_exported_frozen_baselines_are_committed():
    """The Cloud deployment has no 19 GB of raw data, so the stale-baseline
    curves have to be committed."""
    path = fc.frozen_baselines_path()
    assert path.exists(), (
        f"{path.name} is missing -- the multi-year comparison cannot work on "
        f"Streamlit Community Cloud without it"
    )
    exported = fc.load_frozen_baselines()
    assert set(exported["checkpoints"]) == set(fc.CHECKPOINTS)
    bundle = exported["checkpoints"]["checkpoint_1"]
    assert bundle["fits"]["primary"]["window_start"] == "2017-08-02"
    assert bundle["rows"], "the exported checkpoint has no target-week rows"


def test_the_exported_bundle_reproduces_the_live_computation():
    """The Cloud path and the local path must give the same numbers.

    The exported artefact is what a reviewer on Streamlit sees; if it
    disagreed with the live computation, the dashboard would be showing
    two different answers to the same question.
    """
    if not ((ROOT / "data" / "raw").exists()
            and any((ROOT / "data" / "raw").rglob("*R80711*F99*.csv"))):
        pytest.skip("the raw ENGIE data is not present")

    exported = fc.load_frozen_baselines()["checkpoints"]["checkpoint_1"]
    frame = fc.engie_checkpoint_frame(ROOT / "data", "R80711")
    live = fc.checkpoint_bundle(frame, "2017-09-01", "2017-09-07")

    assert live["fits"]["primary"]["window_start"] == \
        exported["fits"]["primary"]["window_start"]
    assert live["fits"]["primary"]["rows"] == exported["fits"]["primary"]["rows"]
    a = fc.score_checkpoint(live)["fits"]["primary"]["mae_kw"]
    b = fc.score_checkpoint(exported)["fits"]["primary"]["mae_kw"]
    assert a == pytest.approx(b, abs=0.05), (a, b)


def test_the_scoring_reports_the_tracking_error_of_every_fit():
    bundle = _sample_bundle()
    scored = fc.score_checkpoint(bundle)
    for label in ("primary", "1_year_back", "2_year_back"):
        fit = scored["fits"][label]
        assert fit["available"], label
        assert fit["mae_kw"] is not None
        assert fit["mae_kw"] >= 0
        assert len(fit["predicted_kw"]) == len(bundle["rows"])


def test_the_prediction_applies_the_air_density_correction():
    """A curve fitted in denser air must predict more power at the same
    wind speed."""
    curve = [{"wind_speed": 5.0, "power_kw": 200.0, "n": 100},
             {"wind_speed": 10.0, "power_kw": 1200.0, "n": 100}]
    thin = fc.predict_from_curve(curve, [8.0], 1.10, 1.225)
    dense = fc.predict_from_curve(curve, [8.0], 1.35, 1.225)
    assert float(dense[0]) > float(thin[0])


def _sample_bundle():
    """A checkpoint bundle built from the committed artefact."""
    exported = fc.load_frozen_baselines()
    if not exported:
        pytest.skip("results/demo_frozen_baselines.json is not generated")
    return exported["checkpoints"]["checkpoint_1"]
