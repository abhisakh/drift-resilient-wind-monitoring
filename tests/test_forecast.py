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
