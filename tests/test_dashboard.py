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


def _cloud_copy(tmp_path):
    """A copy of the repository with NO data/ -- what Cloud actually serves.

    It carries app.py, results/, src/ and assets/: the dashboard imports the
    engine from src/, and the downloadable sample lives in assets/, so both
    are needed for the cloud deployment to work at all.
    """
    import shutil

    cloud = tmp_path / "cloud"
    cloud.mkdir()
    shutil.copy2(ROOT / "app.py", cloud / "app.py")
    shutil.copytree(ROOT / "results", cloud / "results")
    shutil.copytree(ROOT / "src", cloud / "src")
    if (ROOT / "assets").exists():
        shutil.copytree(ROOT / "assets", cloud / "assets")
    assert not (cloud / "data").exists(), "the cloud copy must have no data/"
    return cloud


def test_app_renders_in_snapshot_mode_without_raw_data(tmp_path):
    """The CLOUD mode must work with no data/raw/ present.

    This has to be tested in a copy of the repository without the raw data,
    not in the working tree: a developer always has data/raw/ present, so
    running the app in place never exercises the mode a reviewer on
    Streamlit Community Cloud will actually see.
    """
    from streamlit.testing.v1 import AppTest

    cloud = _cloud_copy(tmp_path)

    at = AppTest.from_file(str(cloud / "app.py"), default_timeout=180)
    at.run()

    assert not at.exception, f"the app raised in cloud mode: {at.exception}"
    # Four pages, and the four sections of the evidence page inside the last.
    assert len(at.tabs) == 8, f"expected 8 tabs, got {len(at.tabs)}"

    text = "\n".join(m.value for m in at.sidebar.markdown)
    assert "CLOUD SNAPSHOT MODE" in text, (
        f"the sidebar did not report cloud mode: {text[:400]}"
    )
    assert "LIVE SCADA ENGINE" not in text, (
        "the sidebar claimed the live engine with no data present"
    )

    # The cloud mode must still DRAW, not just render text: the whole point
    # of the snapshot is that the charts work without the 19 GB. The tables
    # are st.table rather than st.dataframe, because a dataframe is drawn on
    # a canvas and its text cannot be sized from CSS.
    assert len(at.table) >= 3, "the cloud mode rendered no tables"
    assert len(at.get("plotly_chart")) >= 3, "the cloud mode rendered no charts"

    # And the numbers must be the project's, read from the artefacts.
    body = "\n".join(m.value for m in at.markdown)
    assert "86.64" in body, "the R80711 peak is missing from the cloud render"
    # The project's first claim, from results/limit_vs_model.json: the days
    # only the model flagged. (The CARE score used to be checked here; it was
    # shown in the "does and does not claim" box, which is gone.)
    lv = json.loads((ROOT / "results" / "limit_vs_model.json").read_text())
    assert "flagged by the model only" in body, (
        "the fixed-limit comparison is missing from the cloud render")
    assert str(lv["anomaly_span"]["model_only"]) in body


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

    # The ENGIE explorer needs the 19 GB release, which CI does not have.
    # The mechanism is covered everywhere by the upload-path test below.
    mod = _import_app()
    if not mod.live_available():
        pytest.skip("the raw ENGIE release is not present")

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"

    # The switch lives with the live monitor now, so the run has to happen
    # first -- the control drives that run, it does not replace it.
    run = [b for b in at.button if "run the explorer" in b.label.lower()]
    assert run, "there is no button to run the explorer"
    run[0].click().run()
    assert not at.exception, f"the app raised: {at.exception}"

    radios = [r for r in at.radio if r.label == "Reference"]
    assert radios, "the monitor has no reference selector"
    assert set(radios[0].options) == {"rolling 30-day", "frozen 2014"}, (
        radios[0].options
    )
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

    # The peak must be the measured one: 86.64 C on R80711. Cross-check it
    # against the README rather than hard-coding it here, so the dashboard and
    # the documentation cannot drift apart.
    readme = _docs_text()
    assert f"{snap['kpis']['r80711_max_c']} °C" in readme or \
           f"{snap['kpis']['r80711_max_c']} °C" in readme.replace("\u00a0", " "), (
        "the README does not quote the snapshot's peak temperature"
    )
    # And the two references must differ by orders of magnitude, not by a
    # rounding difference -- that is the project's headline.
    fr = snap["kpis"]["false_alarm_pct_frozen_reference"]
    rl = snap["kpis"]["false_alarm_pct_rolling_reference"]
    assert fr > 100 * rl, f"frozen {fr} % vs rolling {rl} %"

    # The snapshot's own false-alarm rates must agree with the ENGIE artefact
    # the dashboard quotes elsewhere -- different measurement, same order of
    # magnitude, so a mismatch means one of them is wrong.
    engie_path = RESULTS / "engie_multivariate.json"
    if engie_path.exists():
        strategies = json.loads(engie_path.read_text())["strategies"]
        engie_frozen = strategies["frozen_2014"]["mean_rate_pct"]
        engie_rolling = strategies["rolling_30d"]["mean_rate_pct"]
        assert fr > engie_frozen * 0.5, (
            f"the snapshot's frozen false-alarm rate ({fr} %) and the ENGIE "
            f"artefact's ({engie_frozen} %) disagree by more than a factor of two"
        )
        assert rl < engie_rolling * 5, (
            f"the snapshot's rolling false-alarm rate ({rl} %) and the ENGIE "
            f"artefact's ({engie_rolling} %) disagree by more than a factor of five"
        )
    assert snap["kpis"]["peer_max_c"] < 80.0
    assert snap["kpis"]["peer_rows_over_80c"] == 0


def _docs_text() -> str:
    """The README and the full report it links to, read together: the
    headline numbers may sit in either."""
    return (ROOT / "README.md").read_text() + (ROOT / "docs" / "RESULTS.md").read_text()


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

    readme = _docs_text()
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

    body = _docs_text()
    assert f"{det_roll:.2f}" in body, "the README does not quote the rolling rate"
    assert f"{det_frozen:.2f}" in body, "the README does not quote the frozen rate"
    assert f"{ratio:.0f}" in body, "the README does not quote the ratio"


# ------------------------------------------------- the live explorer
def _source_radio(at):
    """The explorer's source selector."""
    radios = [r for r in at.radio if r.label == "Source"]
    assert radios, "the explorer has no source selector"
    return radios[0]


def test_the_explorer_offers_all_three_sources():
    """ENGIE with real weather, the labelled benchmark, and the reader's
    own file -- the three things the project can be run against."""
    at = _app_test()
    at.run()
    assert not at.exception
    options = list(_source_radio(at).options)
    assert len(options) == 3, options
    assert any("ENGIE" in o for o in options)
    assert any("CARE" in o for o in options)
    assert any("Upload" in o for o in options)


def test_the_upload_source_offers_the_sample_download():
    """A reviewer on Cloud has no 19 GB of raw data, so the sample file is
    how they can try the detector at all."""
    at = _app_test()
    at.run()
    _source_radio(at).set_value("Upload a turbine CSV").run()
    assert not at.exception, f"the app raised: {at.exception}"
    downloads = [d.label for d in at.get("download_button")]
    assert any("sample" in d.lower() for d in downloads), downloads
    assert at.get("file_uploader"), "the upload source has no file uploader"


def test_running_the_upload_source_reports_rows_runtime_and_datasets():
    """The banner has to say what the engine did: rows, seconds, and which
    datasets were joined. A reader has to be able to check the working."""
    at = _app_test()
    at.run()
    _source_radio(at).set_value("Upload a turbine CSV").run()
    sample = ROOT / "assets" / "sample_turbine.csv"
    at.get("file_uploader")[0].upload("sample_turbine.csv", sample.read_bytes(),
                                      "text/csv")
    at.run()
    run = [b for b in at.button if b.label == "Run the explorer"]
    assert run, "the upload source has no run button"
    run[0].click().run()
    assert not at.exception, f"the app raised: {at.exception}"

    banners = [s.value for s in at.success]
    assert banners, "the run produced no success banner"
    banner = banners[-1]
    assert "rows" in banner.lower(), banner
    assert "s**" in banner or " s " in banner, f"no runtime in the banner: {banner}"
    assert "joined" in banner.lower(), f"no datasets named: {banner}"
    assert "uploaded CSV" in banner, banner
    # And it must have drawn the result.
    assert len(at.get("plotly_chart")) >= 4, "the explorer rendered no chart"


def test_the_explorer_runs_in_cloud_mode_with_no_raw_data(tmp_path):
    """THE CLOUD TEST.

    On Streamlit Community Cloud there is no data/raw/, so the ENGIE and CARE
    sources cannot run. The upload path must still work, or the explorer is
    decoration for a reviewer and a tool for the author.
    """
    from streamlit.testing.v1 import AppTest

    cloud = _cloud_copy(tmp_path)
    at = AppTest.from_file(str(cloud / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"the app raised in cloud mode: {at.exception}"

    # The badge says snapshot mode.
    text = "\n".join(m.value for m in at.sidebar.markdown)
    assert "CLOUD SNAPSHOT MODE" in text

    # The upload source is offered, with the sample, and it RUNS.
    _source_radio(at).set_value("Upload a turbine CSV").run()
    assert not at.exception, f"the app raised: {at.exception}"
    sample = cloud / "assets" / "sample_turbine.csv"
    assert sample.exists(), "the cloud copy has no sample file"
    at.get("file_uploader")[0].upload("sample_turbine.csv", sample.read_bytes(),
                                      "text/csv")
    at.run()
    run = [b for b in at.button if b.label == "Run the explorer"]
    assert run, "no run button in cloud mode"
    run[0].click().run()
    assert not at.exception, f"the app raised: {at.exception}"

    banners = [s.value for s in at.success]
    assert banners, "the cloud run produced no success banner"
    assert "rows" in banners[-1].lower()
    assert "joined" in banners[-1].lower()


def test_the_engie_source_offers_all_four_turbines():
    """The four La Haute Borne turbines are the case study and its controls."""
    at = _app_test()
    at.run()
    turbines = [s for s in at.selectbox if s.label == "Turbine"]
    if not turbines:
        pytest.skip("the ENGIE source is not available without the raw data")
    options = list(turbines[0].options)
    assert set(options) == {"R80711", "R80721", "R80736", "R80790"}, options


def test_running_the_engie_source_reports_what_it_joined():
    """The ENGIE source joins SCADA and ERA5 -- two different datasets, and
    the banner has to say so."""
    if not ((ROOT / "data" / "raw").exists()
            and any((ROOT / "data" / "raw").rglob("*R80711*F99*.csv"))):
        pytest.skip("the raw ENGIE data is not present")

    at = _app_test()
    at.run()
    run = [b for b in at.button if b.label == "Run the explorer"]
    assert run, "no run button"
    run[0].click().run()
    assert not at.exception, f"the app raised: {at.exception}"
    banners = [s.value for s in at.success]
    assert banners, "the ENGIE run produced no success banner"
    banner = banners[-1]
    assert "SCADA" in banner, banner
    assert "ERA5" in banner, banner


# ------------------------------------------------------- the visual polish
def _import_app():
    """Import app.py as a module, without running its UI."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("windmill_app_2", ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_sidebar_carries_the_brand_mark():
    """The turbine icon has to be beside the title, not replace it.

    It is inline SVG rather than an image file, so it stays crisp at any
    scale and needs no asset.
    """
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    text = "\n".join(m.value for m in at.sidebar.markdown)
    assert "Windmill" in text, "the sidebar has no title"
    assert "brand-icon" in text, "the sidebar title has no turbine icon"
    assert "<svg" in text, "the brand icon is not an inline SVG"


def test_the_watermark_is_a_fixed_low_opacity_layer():
    """A watermark must not eat clicks or wash out the content.

    The three properties that make it a watermark rather than an overlay:
    it is FIXED in place, it is faint (a low opacity, not one exact figure),
    and it ignores pointer events.
    """
    import re

    app_text = (ROOT / "app.py").read_text()
    assert "body::after" in app_text, "there is no watermark layer"
    assert "position: fixed" in app_text
    m = re.search(r"body::after\s*\{[^}]*?opacity:\s*([0-9.]+)", app_text, re.S)
    assert m, "the watermark layer has no opacity"
    assert 0.03 <= float(m.group(1)) <= 0.20, m.group(1)
    assert "pointer-events: none" in app_text


def test_the_watermark_is_a_valid_inline_svg():
    """The watermark is carried as a data URI, so it needs no file at all."""
    import base64

    mod = _import_app()
    uri = mod._watermark_uri()
    assert uri.startswith("data:image/svg+xml;base64,")
    svg = base64.b64decode(uri.split(",", 1)[1]).decode("utf-8")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    # A turbine: three blades, a hub and a tower.
    assert svg.count("<path") >= 4, svg
    assert "<circle" in svg


# ---------------------------------------------- the headline cards
def test_the_headline_cards_state_the_goal():
    """The first screen says what the project is FOR, then what it found.

    The cards are the goal's two halves (fewer false call-outs, earlier
    warning) plus the case study, each read from the evaluation suite
    through windmill.scorecard.
    """
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "fewer false call-outs" in body, "the goal is not stated"
    for card in ("False-alarm reduction", "Injected +4 K ramps caught",
                 "R80711 bearing overheat"):
        assert card in body, f"the headline card {card!r} is missing"
    # The third card reads the real-anomaly check when the curve carries it,
    # and the older single-sample lead time otherwise.
    assert ("R80711 dispatch vs the first hot spell" in body
            or "R80711 lead time before 80 °C" in body), \
        "the R80711 lead-time card is missing"


def test_the_headline_cards_render_in_cloud_mode(tmp_path):
    """The cards read results/backtest.json, which the cloud copy carries."""
    from streamlit.testing.v1 import AppTest

    cloud = _cloud_copy(tmp_path)
    at = AppTest.from_file(str(cloud / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"the app raised in cloud mode: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "Injected +4 K ramps caught" in body
    assert ("dispatch vs the first hot spell" in body
            or "lead time before 80" in body)


# ---------------------------------------------- the four-tab layout
def _tab_labels(at):
    return [t.label for t in at.tabs]


def test_the_four_pages_lead_with_the_demo():
    """The demo first, then free exploration, then the record's performance;
    the research behind it sits in its own last page, in four sections."""
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    labels = _tab_labels(at)
    assert labels[:4] == ["Demo", "Explore", "Performance", "Evidence & method"], labels
    assert labels[4:] == ["Weather Forecast", "Benchmark Comparison",
                          "Engineering & Forensics", "Data Explorer"], labels


def test_the_demo_shows_a_healthy_and_a_faulty_turbine_on_the_same_month():
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "R80721, healthy" in body and "R80711, developing a fault" in body
    assert "2017-08-15 to 2017-09-14" in body
    # The demo opens on the real data alone; the prediction and the forecast
    # are one tick away, and the lead-time slider comes with the forecast.
    boxes = {b.key: b for b in at.checkbox if b.key and b.key.startswith("demo_show")}
    assert set(boxes) == {"demo_show_pred", "demo_show_forecast"}
    assert not any(b.value for b in boxes.values()), "the demo must open on the sensor only"
    # By key: the Explore page has a slider with the same label, and the test
    # renders every page at once.
    assert not any(s.key == "demo_lead" for s in at.select_slider)
    boxes["demo_show_pred"].check().run()
    boxes = {b.key: b for b in at.checkbox if b.key and b.key.startswith("demo_show")}
    boxes["demo_show_forecast"].check().run()
    assert not at.exception, f"the app raised: {at.exception}"
    assert any(s.key == "demo_lead" for s in at.select_slider)


def test_the_explore_page_offers_turbines_month_and_sensor():
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    assert [m.label for m in at.multiselect] == ["Turbines"]
    labels = [s.label for s in at.selectbox]
    assert "Month" in labels and "Sensor" in labels
    body = "\n".join(m.value for m in at.markdown)
    assert "MONITORING" in body or "MAINTENANCE HOLD" in body


def test_the_forecast_tab_carries_no_benchmark_comparison():
    """The first tab tells the operational story. A model comparison table
    on it would answer a question nobody has asked yet.

    Checked on the SOURCE, because the rendered tree contains every tab at
    once and so cannot show what one tab does or does not contain.
    """
    import ast
    import inspect

    mod = _import_app()
    src = inspect.getsource(mod.tab_forecast)
    for banned in ("lightgbm", "autoencoder", "care_score", "multivariate",
                   "engie_multivariate", "ae_audit"):
        assert banned not in src.lower(), (
            f"the forecast tab mentions {banned} -- the benchmark comparison "
            f"belongs in its own tab"
        )
    # And it must not define domain logic of its own -- no nested helper
    # functions, which is where an inline calculation would hide.
    tree = ast.parse(src)
    nested = [n.name for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name != "tab_forecast"]
    assert not nested, f"the forecast tab defines its own functions: {nested}"


def test_the_forecast_tab_renders_the_prediction():
    """The live forecast is still there -- it is one dropdown click away."""
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    # The tab opens on the first historical checkpoint.
    _checkpoint_selector(at).set_value(
        "Live 7-day forecast — Open-Meteo (now)").run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "predicted power" in body.lower(), (
        "the forecast tab did not render a predicted power figure"
    )
    sliders = [s.label for s in at.slider]
    assert any("Wind speed shift" in s for s in sliders), sliders
    assert any("Temperature shift" in s for s in sliders), sliders
    assert len(at.get("plotly_chart")) >= 4, "the forecast charts are missing"






def test_the_benchmark_tab_holds_the_model_comparison():
    """The bar chart, the per-year table and the line chart all moved into
    the third tab, together with the CARE audit."""
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "Per year" in body, "the per-year table is missing"
    assert "CARE benchmark" in body or "autoencoder" in body.lower(), (
        "the CARE/AE audit is missing from the benchmark tab"
    )




# ------------------------------------------- the demo checkpoints on tab 1
def _checkpoint_selector(at):
    sel = [s for s in at.selectbox if s.label == "Demo Checkpoint"]
    assert sel, "there is no demo checkpoint selector"
    return sel[0]


def test_the_checkpoint_selector_lists_every_curated_window():
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    options = list(_checkpoint_selector(at).options)
    assert len(options) >= 5, options
    joined = " ".join(options)
    for cp in ("2017-09-01", "2017-10-01", "2017-11-01"):
        assert cp in joined, f"{cp} is not in the checkpoint list: {options}"


def test_the_stale_baseline_toggles_are_present_and_default_on():
    """The toggles belong to the checkpoint, so a checkpoint has to be
    selected before they exist."""
    at = _app_test()
    at.run()
    _checkpoint_selector(at).set_value(
        "Checkpoint 1 — 2017-09-01 → 2017-09-07 — late summer transition"
    ).run()
    assert not at.exception, f"the app raised: {at.exception}"
    boxes = {c.label: c.value for c in at.get("checkbox")}
    one = [k for k in boxes if "Frozen 1 year before" in k]
    two = [k for k in boxes if "Frozen 2 years before" in k]
    assert one, f"the 1-year-back toggle is missing: {list(boxes)}"
    assert two, f"the 2-years-back toggle is missing: {list(boxes)}"
    # The chart opens clean, so the stale baselines are OFF by default and
    # the presenter switches them on when they want the comparison.
    assert boxes[one[0]] is False, "the 1-year-back baseline is on by default"
    assert boxes[two[0]] is False, "the 2-years-back baseline is on by default"


def test_the_stale_baseline_labels_follow_the_checkpoint():
    """On a 2016 checkpoint the one-year-back curve is 2015's and the
    weather is 2016's. A hard-coded year would be a lie."""
    at = _app_test()
    at.run()
    _checkpoint_selector(at).set_value(
        "Checkpoint 4 — 2016-09-01 → 2016-09-07 — a year earlier, against "
        "the 2015 and 2014 curves"
    ).run()
    assert not at.exception, f"the app raised: {at.exception}"
    labels = [c.label for c in at.get("checkbox")]
    assert any("2015 curve + 2016 weather" in lab for lab in labels), labels
    assert any("2014 curve + 2016 weather" in lab for lab in labels), labels
    # And the 2017 wording must be gone.
    assert not any("2016 curve + 2017 weather" in lab for lab in labels), labels


def test_both_checkpoint_charts_share_one_x_axis():
    """The power chart and the weather chart below it must cover the same
    span, or a gust does not line up with the power it produced.

    Plotly auto-ranges each figure separately, and the power chart can
    zoom to a single day while the weather chart shows the whole week.
    """
    mod = _import_app()
    import pandas as pd

    week = pd.date_range("2017-09-01", periods=7 * 24, freq="h", tz="UTC")
    axis = mod._shared_axis_range(week)
    assert axis["tickformat"] == "%b %d"
    assert axis["range"][0] == week[0]
    assert axis["range"][1] == week[-1]
    # The range must cover the whole week, not one day of it.
    span = (axis["range"][1] - axis["range"][0]).days
    assert span >= 6, span

    # And both charts must be handed that same axis object.
    src = (ROOT / "app.py").read_text()
    assert "x_axis = _shared_axis_range(df[\"ts\"])" in src
    assert src.count("xaxis=x_axis") >= 2, (
        "the power chart and the weather chart must both use x_axis"
    )
    assert "xaxis=fx_axis" in src


def test_the_four_power_lines_have_the_intended_colours():
    """The actual output is the quiet neutral line; the three predictions
    are what the demo is about. Healthy power must not look like an alarm."""
    mod = _import_app()
    lines = {line[0]: line for line in mod._power_lines(True, True, True)}
    assert lines["actual"][1] == "Actual measured power (kW)"
    assert lines["actual"][3] == "#E5E7EB"
    assert lines["primary"][1] == "Predicted — last 30d fit (kW)"
    assert lines["primary"][3] == "#00D4AA"
    assert lines["1_year_back"][2] == "dash"
    assert lines["1_year_back"][3] == "#F59E0B"
    assert lines["2_year_back"][2] == "dot"
    assert lines["2_year_back"][3] == "#F43F5E"


def test_the_tracking_error_summary_is_a_scannable_box():
    """The dense paragraph became three bullets, so the key numbers have to
    be findable."""
    at = _app_test()
    at.run()
    _checkpoint_selector(at).set_value(
        "Checkpoint 1 — 2017-09-01 → 2017-09-07 — late summer transition"
    ).run()
    assert not at.exception, f"the app raised: {at.exception}"
    infos = [i.value for i in at.info]
    joined = "\n".join(infos)
    assert "43.9" in joined and "49.0" in joined, joined[:400]
    assert "19.7" in joined, joined[:400]
    assert joined.count("- ") >= 2, "the summary is not a list of bullets"


def test_switching_to_a_checkpoint_shows_the_training_dates():
    """The training window has to be stated where it cannot be missed."""
    at = _app_test()
    at.run()
    _checkpoint_selector(at).set_value(
        "Checkpoint 1 — 2017-09-01 → 2017-09-07 — late summer transition"
    ).run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "Trained on" in body, "the training callout is missing"
    assert "2017-08-02" in body, "the training start date is missing"
    assert "2017-08-31" in body or "2017-09-01" in body, (
        "the training end date is missing"
    )
    assert "Trailing 30 Days" in body, "the training rule is not stated"


def test_the_checkpoint_view_reports_the_tracking_error_of_every_fit():
    """All three MAEs, once the presenter switches the stale baselines on."""
    at = _app_test()
    at.run()
    _checkpoint_selector(at).set_value(
        "Checkpoint 1 — 2017-09-01 → 2017-09-07 — late summer transition"
    ).run()
    assert not at.exception, f"the app raised: {at.exception}"

    boxes = {c.label: c for c in at.get("checkbox")}
    for label_fragment in ("Frozen 1 year before", "Frozen 2 years before"):
        box = [c for k, c in boxes.items() if label_fragment in k][0]
        box.set_value(True).run()
    assert not at.exception, f"the app raised: {at.exception}"

    body = "\n".join(m.value for m in at.markdown)
    for label in ("last 30-day fit", "frozen 1-year-back", "frozen 2-years-back"):
        assert label in body, f"the {label} MAE is missing"
    assert "MAE" in body


def test_the_checkpoint_view_works_in_cloud_mode(tmp_path):
    """The checkpoints have to work on Streamlit Community Cloud, where the
    stale-baseline curves come from the committed artefact rather than from
    the raw data."""
    from streamlit.testing.v1 import AppTest

    cloud = _cloud_copy(tmp_path)
    at = AppTest.from_file(str(cloud / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"the app raised in cloud mode: {at.exception}"
    sel = [s for s in at.selectbox if s.label == "Demo Checkpoint"]
    assert sel, "the cloud copy has no checkpoint selector"
    sel[0].set_value(
        "Checkpoint 1 — 2017-09-01 → 2017-09-07 — late summer transition"
    ).run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "Trained on" in body, "the cloud checkpoint has no training callout"
    assert "2017-08-02" in body
    assert "last 30-day fit" in body, "the cloud checkpoint has no MAEs"

    # The stale baselines are on demand in the cloud too.
    boxes = {c.label: c for c in at.get("checkbox")}
    for fragment in ("Frozen 1 year before", "Frozen 2 years before"):
        box = [c for k, c in boxes.items() if fragment in k][0]
        box.set_value(True).run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "frozen 1-year-back" in body, "the cloud checkpoint has no stale MAEs"
    assert "frozen 2-years-back" in body


def test_the_chart_opens_with_two_lines_and_grows_to_four():
    """The default view is Actual + Last 30d Fit; the stale curves are on
    demand."""
    mod = _import_app()

    clean = mod._power_lines(True, False, False)
    assert len(clean) == 2, clean
    assert clean[0][0] == "actual"
    assert clean[1][0] == "primary"

    full = mod._power_lines(True, True, True)
    assert len(full) == 4, full
    assert [line[0] for line in full] == [
        "actual", "primary", "1_year_back", "2_year_back"]

    # And each toggle adds exactly one line.
    assert len(mod._power_lines(True, True, False)) == 3
    assert len(mod._power_lines(True, False, True)) == 3
    assert len(mod._power_lines(False, False, False)) == 1


def test_the_default_view_shows_two_mae_cards_and_the_expanded_view_four():
    """Each card follows its toggle, so the KPI row grows with the chart."""
    at = _app_test()
    at.run()
    _checkpoint_selector(at).set_value(
        "Checkpoint 1 — 2017-09-01 → 2017-09-07 — late summer transition"
    ).run()
    assert not at.exception, f"the app raised: {at.exception}"

    body = "\n".join(m.value for m in at.markdown)
    assert "MAE — last 30-day fit" in body
    assert "actual mean power" in body
    assert "MAE — frozen 1-year-back" not in body, (
        "the stale-baseline card is showing with its toggle off"
    )
    assert "MAE — frozen 2-years-back" not in body

    # Switch the stale baselines on: the cards appear with them.
    boxes = {c.label: c for c in at.get("checkbox")}
    one = [c for k, c in boxes.items() if "Frozen 1 year before" in k][0]
    two = [c for k, c in boxes.items() if "Frozen 2 years before" in k][0]
    one.set_value(True).run()
    two.set_value(True).run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "MAE — frozen 1-year-back" in body
    assert "MAE — frozen 2-years-back" in body


def test_the_last_30d_fit_can_be_switched_off_too():
    """A presenter may want only the actual output against the stale
    curves -- that is the picture that shows the stale ones are worse."""
    mod = _import_app()
    lines = mod._power_lines(False, True, True)
    assert len(lines) == 3
    assert [line[0] for line in lines] == [
        "actual", "1_year_back", "2_year_back"]


def test_the_app_opens_on_the_first_historical_checkpoint():
    """The comparison controls must be on screen immediately, so the
    dropdown opens on Checkpoint 1 rather than on the live forecast."""
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    sel = _checkpoint_selector(at)
    # The value is the option key; format_func only changes what is drawn.
    assert sel.value == "checkpoint_1", sel.value
    assert any("Checkpoint 1" in opt for opt in sel.options), sel.options

    # And the three prediction checkboxes are therefore there on load.
    labels = [c.label for c in at.get("checkbox")]
    assert any("Last 30 days rolling" in lab for lab in labels), labels
    assert any("Frozen 1 year before" in lab for lab in labels), labels
    assert any("Frozen 2 years before" in lab for lab in labels), labels

    # And the training callout is on screen without any interaction.
    body = "\n".join(m.value for m in at.markdown)
    assert "Trained on" in body, "the training callout is not on screen"
    assert "2017-08-02" in body


def test_the_prediction_checkboxes_default_to_one_curve():
    """The 30-day fit is on; the two stale baselines are off."""
    at = _app_test()
    at.run()
    boxes = {c.label: c.value for c in at.get("checkbox")}
    fresh = [v for k, v in boxes.items() if "Last 30 days rolling" in k]
    one = [v for k, v in boxes.items() if "Frozen 1 year before" in k]
    two = [v for k, v in boxes.items() if "Frozen 2 years before" in k]
    assert fresh and fresh[0] is True, "the 30-day fit is off by default"
    assert one and one[0] is False, "the 1-year-back baseline is on by default"
    assert two and two[0] is False, "the 2-years-back baseline is on by default"


def _prediction_boxes(at):
    """The three prediction checkboxes, by fit."""
    boxes = {c.label: c for c in at.get("checkbox")}
    fresh = [c for k, c in boxes.items() if "Last 30 days rolling" in k][0]
    one = [c for k, c in boxes.items() if "Frozen 1 year before" in k][0]
    two = [c for k, c in boxes.items() if "Frozen 2 years before" in k][0]
    return fresh, one, two


def test_each_prediction_can_be_plotted_on_its_own():
    """Any single prediction curve, against the actual measured power."""
    cases = {
        "Last 30 days rolling": "MAE — last 30-day fit",
        "Frozen 1 year before": "MAE — frozen 1-year-back",
        "Frozen 2 years before": "MAE — frozen 2-years-back",
    }
    for fragment, card in cases.items():
        at = _app_test()
        at.run()
        fresh, one, two = _prediction_boxes(at)
        for box in (fresh, one, two):
            box.set_value(box.label.startswith(fragment)).run()
        assert not at.exception, f"the app raised for {fragment}: {at.exception}"
        body = "\n".join(m.value for m in at.markdown)
        assert card in body, f"{fragment} did not produce {card}"
        # The other two must NOT have cards.
        for other_fragment, other_card in cases.items():
            if other_fragment != fragment:
                assert other_card not in body, (
                    f"{other_card} is showing although only {fragment} is on"
                )


def test_all_three_predictions_can_be_plotted_together():
    at = _app_test()
    at.run()
    fresh, one, two = _prediction_boxes(at)
    for box in (one, two):
        box.set_value(True).run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    for card in ("MAE — last 30-day fit", "MAE — frozen 1-year-back",
                 "MAE — frozen 2-years-back"):
        assert card in body, f"{card} is missing"


def test_switching_everything_off_still_plots_the_actual_output():
    """With no prediction selected there is nothing to compare against, so
    the dashboard says so instead of showing a bare line."""
    at = _app_test()
    at.run()
    fresh, one, two = _prediction_boxes(at)
    fresh.set_value(False).run()
    assert not at.exception, f"the app raised: {at.exception}"
    # st.info renders as an alert element, not as markdown.
    infos = "\n".join(i.value for i in at.info)
    assert "Select at least one prediction baseline" in infos, (
        "there is no prompt when every prediction is switched off"
    )
    # And no MAE card, because no prediction is being scored.
    body = "\n".join(m.value for m in at.markdown)
    assert "MAE — last 30-day fit" not in body
    assert "MAE — frozen 1-year-back" not in body






















def test_the_reference_switch_works_without_the_raw_release():
    """The same control, on the path that works everywhere.

    The dramatic magnitude needs the full ENGIE release; the mechanism --
    switching the reference changes the number of alarms -- has to hold on
    the upload path too, because that is the path a reviewer on Streamlit
    Community Cloud can actually run.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"

    _source_radio(at).set_value("Upload a turbine CSV").run()
    assert not at.exception, f"the app raised: {at.exception}"
    sample = ROOT / "assets" / "sample_turbine.csv"
    at.get("file_uploader")[0].upload("sample_turbine.csv", sample.read_bytes(),
                                      "text/csv")
    at.run()
    run = [b for b in at.button if "run the explorer" in b.label.lower()]
    assert run, "there is no button to run the explorer"
    run[0].click().run()
    assert not at.exception, f"the app raised: {at.exception}"

    radios = [r for r in at.radio if r.label == "Reference"]
    assert radios, "the live monitor has no reference selector"
    assert set(radios[0].options) == {"rolling 30-day", "frozen 2014"}, (
        radios[0].options
    )
    radios[0].set_value("rolling 30-day").run()
    rolling = _alarm_count(at)
    radios[0].set_value("frozen 2014").run()
    frozen = _alarm_count(at)

    assert rolling is not None and frozen is not None, "no alarm count found"
    assert rolling != frozen, (
        f"switching the reference changed nothing: rolling={rolling} "
        f"frozen={frozen}"
    )
    assert frozen > rolling, (
        f"the frozen reference should not alarm less than the rolling one, "
        f"got rolling={rolling} frozen={frozen}"
    )


def test_the_typography_is_sized_for_reading():
    """The reviewer could not read the dashboard, so the sizes are stated
    in one table and the CSS is built from it.

    This is a guard on the intent, not on the exact pixels: the point is
    that the reading text is large, and that every size in the table has a
    CSS rule that uses it.
    """
    mod = _import_app()
    sizes = mod.FONT_SIZES
    assert sizes["body"] >= 30, sizes
    assert sizes["note"] >= 26, sizes
    assert sizes["callout"] >= 28, sizes
    assert sizes["alert"] >= 26, sizes
    assert sizes["kpi_value"] >= 40, sizes
    assert sizes["kpi_label"] >= 22, sizes
    assert sizes["kpi_delta"] >= 20, sizes
    assert sizes["table"] >= 18, sizes
    assert sizes["slider"] >= 18, sizes

    src = (ROOT / "app.py").read_text()
    for key in sizes:
        assert f"__{key}__" in src, f"{key} has no CSS rule"

    # And the chart text is set on the figure, not on the page: Plotly
    # lays a chart out for the font it is given, so a CSS override would
    # leave the layout computed for a smaller font.
    assert mod.CHART_FONT_PX >= 16, mod.CHART_FONT_PX
    assert "font=dict(size=CHART_FONT_PX)" in src
    assert ".js-plotly-plot text" not in src, (
        "the chart text is being overridden from CSS, which breaks the "
        "chart's own layout"
    )




def test_the_tabs_are_a_navigation_bar_not_body_text():
    """The tab headers sat in the same size as the prose with a hairline
    under them, so they read as part of the page. They are now smaller than
    the body text, set in capitals, held in their own bar, with one clear
    active state and no emoji."""
    import re

    mod = _import_app()
    assert mod.FONT_SIZES["tab"] < mod.FONT_SIZES["body"], mod.FONT_SIZES
    src = (ROOT / "app.py").read_text()
    assert 'aria-selected="true"' in src, "the active tab has no style"
    assert 'data-baseweb="tab-highlight"' in src, (
        "Streamlit's own tab highlight is not hidden, so there would be "
        "two indicators")
    assert "text-transform: uppercase" in src

    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    for label in _tab_labels(at):
        assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", label), label




def test_the_performance_page_shows_what_only_the_model_sees():
    """The project's first claim, on screen: the days the fixed 80 C limit
    missed and the model flagged, read from results/limit_vs_model.json."""
    lv = RESULTS / "limit_vs_model.json"
    if not lv.exists():
        pytest.skip("artefact missing")
    at = _app_test()
    at.run()
    assert not at.exception, f"the app raised: {at.exception}"
    body = "\n".join(m.value for m in at.markdown)
    assert "The fixed 80 °C limit against the model, day by day" in body
    assert "flagged by the model only" in body


def test_the_demo_charts_draw_the_fixed_limit_as_a_dashed_line():
    """The old rule on the same axis: a dashed horizontal line at 80 C, so a
    viewer sees the days the bearing ran hot while still under the limit."""
    import pandas as pd

    mod = _import_app()
    df = pd.DataFrame({"day": ["2017-08-15", "2017-08-16"], "status": ["checked", "checked"],
                       "actual": [72.0, 75.0], "pred": [70.0, 71.0], "f1": [70.5, 71.5],
                       "fault_period": [True, True]})
    fig = mod._three_line_figure(df, 2.0, 1, "t", y_range=[60, 82], limit_c=80.0,
                                 show_pred=False, show_forecast=False)
    lines = [s for s in fig.layout.shapes if s.y0 == 80.0 and s.y1 == 80.0]
    assert lines, "no horizontal line at the fixed limit"
    assert lines[0].line.dash == "dash"
    assert any("fixed alarm limit 80" in (a.text or "") for a in fig.layout.annotations)
    # Without the boxes ticked only the sensor is drawn (plus its markers).
    names = {t.name for t in fig.data}
    assert "sensor (daily mean)" in names
    assert "predicted on the day (actual weather)" not in names


def test_the_explore_strip_keeps_names_and_alarm_counts_apart():
    """The turbine name is the row label; the alarm count is its own label
    right of the row. As one two-line tick label they were drawn on top of
    each other."""
    import pandas as pd

    mod = _import_app()
    dv = mod._daily_view()
    df = pd.DataFrame({"day": ["2017-09-01", "2017-09-02"], "status": ["checked", "checked"],
                       "reason": [None, None], "diff": [0.0, 4.0],
                       "actual": [70.0, 75.0], "pred": [70.0, 71.0],
                       "fault_period": [False, False]})
    alarms = {"R80711": pd.DataFrame({"day": ["2017-09-02"], "real": [True]})}
    fig = mod._strip_figure([("R80711", df), ("R80721", df)], 2.0, dv, alarms=alarms)
    ticks = list(fig.layout.yaxis.ticktext)
    assert all("<br>" not in t and "alarm" not in t for t in ticks), ticks
    texts = [a.text for a in fig.layout.annotations]
    assert any("1 alarm" in t and "1 in the abnormal period" in t for t in texts), texts
    # One line each: two lines at the chart font ran into the next row.
    assert all("<br>" not in t for t in texts), texts
