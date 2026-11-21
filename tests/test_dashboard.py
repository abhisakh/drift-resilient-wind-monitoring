"""The dashboard must import cleanly and render without the 19 GB of data.

Streamlit Community Cloud has no `data/raw/`, so the snapshot mode is the mode
that matters most -- and it is the one that is never exercised locally, because
a developer always has the data.

`streamlit.testing.v1.AppTest` runs the app headlessly, exactly as
`streamlit run app.py` would, and reports any exception the app raised. This
test therefore proves the dashboard renders in the mode a reviewer on the web
will actually see.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)


def test_app_imports_cleanly():
    """Importing the module must not execute the UI or raise."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("windmill_app", ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")


def test_app_renders_in_snapshot_mode_without_raw_data(tmp_path):
    """The CLOUD mode must work with no data/raw/ present.

    This has to be tested in a copy of the repository without the raw data,
    not in the working tree: a developer always has data/raw/ present, so
    running the app in place never exercises the mode a reviewer on
    Streamlit Community Cloud will actually see.
    """
    import shutil

    from streamlit.testing.v1 import AppTest

    cloud = tmp_path / "cloud"
    cloud.mkdir()
    shutil.copy2(ROOT / "app.py", cloud / "app.py")
    shutil.copytree(ROOT / "results", cloud / "results")
    assert not (cloud / "data").exists(), "the cloud copy must have no data/"

    at = AppTest.from_file(str(cloud / "app.py"), default_timeout=180)
    at.run()

    assert not at.exception, f"the app raised in cloud mode: {at.exception}"
    assert len(at.tabs) == 4, f"expected 4 tabs, got {len(at.tabs)}"

    text = "\n".join(m.value for m in at.sidebar.markdown)
    assert "CLOUD SNAPSHOT MODE" in text, (
        f"the sidebar did not report cloud mode: {text[:400]}"
    )
    assert "LIVE SCADA ENGINE" not in text, (
        "the sidebar claimed the live engine with no data present"
    )

    # The cloud mode must still DRAW, not just render text: the whole point
    # of the snapshot is that the charts work without the 19 GB.
    assert len(at.dataframe) >= 3, "the cloud mode rendered no tables"
    assert len(at.get("plotly_chart")) >= 3, "the cloud mode rendered no charts"

    # And the numbers must be the project's, read from the artefacts.
    body = "\n".join(m.value for m in at.markdown)
    assert "86.64" in body, "the R80711 peak is missing from the cloud render"
    assert "0.5028" in body, "the CARE score is missing from the cloud render"


def test_app_renders_in_live_mode_when_the_raw_data_is_present():
    """With the raw data present the badge must say so."""
    if not ((ROOT / "data" / "raw").exists()
            and any((ROOT / "data" / "raw").rglob("*R80711*F99*.csv"))):
        pytest.skip("the raw ENGIE data is not present")

    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    text = "\n".join(m.value for m in at.sidebar.markdown)
    assert "LIVE SCADA ENGINE" in text, (
        f"the sidebar did not report live mode: {text[:400]}"
    )


def test_app_shows_the_snapshot_badge_when_there_is_no_raw_data():
    """The sidebar must say which engine is running."""
    at = _app_test()
    at.run()
    assert not at.exception
    # The badge is raw HTML inside a markdown block, so it shows up in
    # sidebar.markdown rather than in a typed element.
    text = "\n".join(m.value for m in at.sidebar.markdown)
    assert "CLOUD SNAPSHOT MODE" in text or "LIVE SCADA ENGINE" in text, (
        f"the sidebar does not say which engine is running: {text[:400]}"
    )


def test_the_badge_matches_whether_the_raw_data_is_present():
    """The badge must not claim the live engine when there is no raw data."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    text = "\n".join(m.value for m in at.sidebar.markdown)
    has_raw = (ROOT / "data" / "raw").exists() and any(
        (ROOT / "data" / "raw").rglob("*R80711*F99*.csv"))
    if has_raw:
        assert "LIVE SCADA ENGINE" in text
    else:
        assert "CLOUD SNAPSHOT MODE" in text


def test_app_reads_its_numbers_from_results_not_from_hardcoded_values():
    """The dashboard must not carry its own copy of a headline number.

    If app.py hard-coded 0.5028 it would silently disagree with the
    artefacts the moment a result was regenerated. So every figure it shows
    must be present in results/*.json.
    """
    app_text = (ROOT / "app.py").read_text()
    # A bare CARE score or false-alarm percentage typed into the source.
    import re

    literals = re.findall(r"(?<![\w.])(0\.\d{4})(?![\w.])", app_text)
    artefacts = ""
    for p in RESULTS.glob("*.json"):
        artefacts += p.read_text()
    for lit in set(literals):
        assert lit in artefacts, (
            f"app.py contains the number {lit}, which is not in results/*.json -- "
            f"read it from the artefact instead"
        )


def test_switching_the_reference_changes_the_alarm_count():
    """The monitor must show the project's central result, not just the noisy end.

    Same window, same detector, same threshold -- only the reference differs.
    A frozen 2014 reference applied to 2017 data reads the turbine's own drift
    as overheating; a rolling 30-day reference does not. If the dashboard only
    ever showed the frozen case, a reviewer would read it as "the detector is
    broken" rather than "the reference is the problem".
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"

    radios = [r for r in at.radio if r.label == "Reference"]
    assert radios, "the monitor has no reference selector"
    radios[0].set_value("rolling 30-day").run()
    rolling = _alarm_count(at)
    radios[0].set_value("frozen 2014").run()
    frozen = _alarm_count(at)

    assert rolling is not None and frozen is not None, "no alarm count found"
    assert rolling < frozen, (
        f"the rolling reference should alarm far less than the frozen one, "
        f"got rolling={rolling} frozen={frozen}"
    )
    # And it must be the documented magnitude, not a rounding difference.
    assert frozen > 100 * max(1, rolling), (
        f"expected a two-orders-of-magnitude difference, got "
        f"rolling={rolling} frozen={frozen}"
    )


def _alarm_count(at) -> int | None:
    """Read the alarm-count KPI out of the rendered markdown."""
    for m in at.markdown:
        if "ALARMS" in m.value.upper() and "REFERENCE" in m.value.upper():
            import re
            n = re.search(r"class='v'>([\d,]+)</div>", m.value)
            if n:
                return int(n.group(1).replace(",", ""))
    return None


def test_the_snapshot_artefact_has_the_series_the_dashboard_plots():
    """The snapshot must carry everything tab 2 draws."""
    path = RESULTS / "demo_r80711.json"
    if not path.exists():
        pytest.skip("results/demo_r80711.json not generated")
    snap = json.loads(path.read_text())

    assert snap["meta"]["target_turbine"] == "R80711"
    assert len(snap["meta"]["peer_turbines"]) == 3
    for key in ("t", "series", "kpis", "meta"):
        assert key in snap, key

    series = snap["series"]
    for key in ("R80711", "peer_median", "residual", "z", "zmax",
                "zmax_rolling", "ambient", "wind_speed"):
        assert key in series, f"the snapshot has no {key} series"
        assert len(series[key]) == len(snap["t"]), key

    for peer in snap["meta"]["peer_turbines"]:
        assert peer in series, f"the snapshot has no {peer} control series"

    # The peak must be the measured one: 86.64 C on R80711.
    assert snap["kpis"]["r80711_max_c"] == 86.64
    # And the two references must differ by orders of magnitude, not by a
    # rounding difference -- that is the project's headline.
    fr = snap["kpis"]["false_alarm_pct_frozen_reference"]
    rl = snap["kpis"]["false_alarm_pct_rolling_reference"]
    assert fr > 100 * rl, f"frozen {fr} % vs rolling {rl} %"
    assert snap["kpis"]["peer_max_c"] < 80.0
    assert snap["kpis"]["peer_rows_over_80c"] == 0


def test_the_headline_ratio_is_stated_the_same_way_everywhere():
    """46.5x, not 46x and not 47x.

    The ratio is 46.54, so it rounds to 47 but truncates to 46. The README's
    hand-written headline and the generated sections disagreed over exactly
    that, so the figure is now quoted to one decimal everywhere and checked.
    """
    engie_path = RESULTS / "engie_multivariate.json"
    if not engie_path.exists():
        pytest.skip("artefacts missing")
    s = json.loads(engie_path.read_text())["strategies"]
    ratio = s["frozen_2014"]["mean_rate_pct"] / s["rolling_30d"]["mean_rate_pct"]
    stated = f"{ratio:.1f}"

    readme = (ROOT / "README.md").read_text()
    assert f"{stated}×" in readme or f"{stated}x" in readme, (
        f"the README does not state the ratio as {stated}x"
    )
    # And the wrong roundings must be gone.
    for wrong in ("46×", "46x", "47×", "47x"):
        assert wrong not in readme, (
            f"the README still says {wrong}; the ratio is {ratio:.4f}, stated "
            f"as {stated}"
        )

    app_text = (ROOT / "app.py").read_text()
    assert "_fmt(ratio, 1)" in app_text, (
        "the dashboard must quote the ratio to one decimal too"
    )


def test_dashboard_and_readme_agree_on_the_headline_numbers():
    """Both documents are generated from the same artefacts, so they must agree."""
    engie_path = RESULTS / "engie_multivariate.json"
    readme = ROOT / "README.md"
    if not engie_path.exists() or not readme.exists():
        pytest.skip("artefacts missing")
    s = json.loads(engie_path.read_text())["strategies"]
    det_frozen = s["frozen_2014"]["mean_rate_pct"]
    det_roll = s["rolling_30d"]["mean_rate_pct"]
    ratio = det_frozen / det_roll

    body = readme.read_text()
    assert f"{det_roll:.2f}" in body, "the README does not quote the rolling rate"
    assert f"{det_frozen:.2f}" in body, "the README does not quote the frozen rate"
    assert f"{ratio:.0f}" in body, "the README does not quote the ratio"
