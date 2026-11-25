"""The CARE Farms B and C step: component grouping and the learned tolerance."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "care_farms_bc", ROOT / "scripts" / "pipeline" / "22_care_farms_bc.py")
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def test_sensors_map_to_components_and_air_is_left_out():
    assert bc.component_of("Transformer L1 (mid-voltage) temperature") == "transformer"
    assert bc.component_of("Gearbox bearing temperature 2") == "gearbox"
    assert bc.component_of("Rotor bearing temperature 1") == "rotor bearing"
    assert bc.component_of("Outside temperature") is None
    assert bc.component_of("Hub temperature") is None
    assert bc.component_of("Nacelle outside temperature") is None


def test_faults_map_to_components():
    rules = bc.FAULT_RULES
    assert bc.component_of("high temperature in transformer cell", rules) == "transformer"
    assert bc.component_of("Rotor Bearing 2 - Damage", rules) == "rotor bearing"
    assert bc.component_of("Communication fault BK1120 in NC300", rules) is None


def _days(values):
    return [{"day": f"2023-01-{i + 1:02d}" if i < 31 else f"2023-02-{i - 30:02d}",
             "diffs": {"s": v}} for i, v in enumerate(values)]


def test_nothing_is_judged_before_enough_history():
    out = bc.judge(_days([50.0] * (bc.MIN_CAL_DAYS - 1)))
    assert not any(d["checked"] or d["hot"] for d in out)


def test_learned_tolerance_never_below_the_floor_and_flags_excess():
    rng = np.random.default_rng(0)
    quiet = list(rng.normal(0.0, 0.1, bc.MIN_CAL_DAYS))
    out = bc.judge(_days(quiet + [1.5, 2.5]))
    assert out[-2]["checked"] and not out[-2]["hot"]   # under the 2 K floor
    assert out[-1]["hot"]                               # over it


def test_a_noisy_component_gets_a_wider_tolerance():
    rng = np.random.default_rng(1)
    noisy = list(rng.normal(0.0, 3.0, bc.MIN_CAL_DAYS))
    out = bc.judge(_days(noisy + [5.0]))
    assert not out[-1]["hot"]          # 5 K is ordinary for a 3 K-spread sensor


def test_hot_days_do_not_enter_the_history():
    base = [0.0] * bc.MIN_CAL_DAYS
    out = bc.judge(_days(base + [10.0] * 5))
    assert all(d["hot"] for d in out[-5:])   # a fault cannot widen its own tolerance


def _frame_with_stop(service_during_stop: bool):
    import pandas as pd
    ts = pd.date_range("2023-01-01", periods=6 * 24 * 20, freq="10min", tz="UTC")
    f = pd.DataFrame({"ts": ts, "P_norm": 0.5, "status": 0})
    stop = (ts >= pd.Timestamp("2023-01-08", tz="UTC")) & (ts < pd.Timestamp("2023-01-12", tz="UTC"))
    f.loc[stop, "P_norm"] = 0.0
    if service_during_stop:
        f.loc[stop & (ts < pd.Timestamp("2023-01-09", tz="UTC")), "status"] = bc.SERVICE_STATUS
    return f


def test_a_long_stop_is_maintenance_only_when_service_was_logged():
    assert len(bc.recorded_service(_frame_with_stop(True))) == 1
    assert bc.recorded_service(_frame_with_stop(False)) == []


def test_short_service_while_running_is_not_maintenance():
    f = _frame_with_stop(False)
    f.loc[f.index[100:140], "status"] = bc.SERVICE_STATUS   # 6+ h, turbine running
    assert bc.recorded_service(f) == []


def test_one_hot_sensor_is_not_hidden_by_healthy_neighbours():
    rng = np.random.default_rng(2)
    days = []
    for i in range(bc.MIN_CAL_DAYS + 1):
        diffs = {f"w{k}": float(rng.normal(0, 0.5)) for k in range(7)}
        diffs["cell"] = float(rng.normal(0, 0.5)) if i < bc.MIN_CAL_DAYS else 10.0
        days.append({"day": f"d{i}", "diffs": diffs})
    assert bc.judge(days)[-1]["hot"]   # the mean of the 8 would be only ~1.25 K
