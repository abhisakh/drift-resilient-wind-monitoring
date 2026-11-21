"""The repository's own maintenance rules, enforced rather than just documented.

Rule 1  Single source of truth: README.md is GENERATED from results/*.json.
        No number in it may be typed by hand.
Rule 2  Logic lives in src/windmill/, orchestration in scripts/.
        A script may load data, call src/windmill/, and write JSON -- it may
        not contain reusable maths of its own.
Rule 3  Determinism is protected by CI.
Rule 4  Local cache artefacts never reach git.

These are cheap to check and expensive to lose, so they are checked.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PIPELINE = SCRIPTS / "pipeline"
DIAGNOSTICS = SCRIPTS / "diagnostics"
RESULTS = ROOT / "results"


# --- Rule 1: README.md is generated ----------------------------------------
def test_readme_has_the_generated_markers():
    readme = (ROOT / "README.md").read_text()
    assert "<!-- RESULTS:START -->" in readme
    assert "<!-- RESULTS:END -->" in readme
    assert "<!-- METHOD:START -->" in readme
    assert "<!-- METHOD:END -->" in readme


def test_readme_numbers_come_from_results():
    """Every CARE score and false-alarm rate in the README must exist in results/.

    A hand-typed number that is not backed by an artefact is the failure mode
    this rule exists to prevent.
    """
    readme = (ROOT / "README.md").read_text()
    # Only the GENERATED section is under this rule. The prose above it is
    # written by hand and is checked by eye; the numbers in it that come
    # from the literature live in data/README.md, which is included below.
    body = readme.split("<!-- RESULTS:START -->", 1)[1].split("<!-- RESULTS:END -->", 1)[0]
    body += readme.split("<!-- METHOD:START -->", 1)[1].split("<!-- METHOD:END -->", 1)[0]

    artefacts = {}
    for p in (ROOT / "results").glob("*.json"):
        artefacts[p.name] = p.read_text()
    # Values quoted from the published papers (the documented ENGIE fault peak,
    # for instance) live in the data provenance notes.
    artefacts["data/README.md"] = (ROOT / "data" / "README.md").read_text()

    # The README prints 4 decimal places; the JSON artefacts store full
    # precision. So compare NUMBERS, not strings: a README figure must match
    # some artefact value once both are rounded to the README's precision.
    def _numbers(blob: str) -> list[float]:
        out = []
        for m in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", blob):
            try:
                out.append(float(m))
            except ValueError:
                pass
        return out

    # The README prints a figure at some precision; the JSON stores full
    # precision. A README number is "backed" if it is a rounding of an
    # artefact value at ANY display precision -- 14.89 is the 2-dp form of
    # 14.8883.
    backed: set[float] = set()
    for blob in artefacts.values():
        for v in _numbers(blob):
            for k in range(1, 7):
                backed.add(round(v, k))
            # An artefact may store a FRACTION where the README shows a
            # PERCENTAGE (mean_frac_event_rows_normal 0.9504 -> "95.0 %").
            # That is the same quantity in different units, so it counts as
            # backed.
            for k in range(1, 7):
                backed.add(round(v * 100, k))

    missing = sorted({n for n in _numbers(body) if not any(
        round(n, k) in backed for k in range(1, 7))})
    assert not missing, (
        f"these README numbers are not backed by any results/*.json: {missing}"
    )


def test_report_generator_owns_the_results_section():
    """09_report.py must be the only thing that writes the README's results."""
    src = (PIPELINE / "09_report.py").read_text()
    assert "RESULTS:START" in src
    assert "RESULTS:END" in src
    assert "METHOD:START" in src


# --- Rule 2: logic in src/, orchestration in scripts/ ------------------------
def test_every_referenced_script_path_exists():
    """A path mentioned in a docstring or an error message must be real.

    The scripts moved from `scripts/` into `scripts/pipeline/` and
    `scripts/diagnostics/`, and the references in the docstrings and in the
    user-facing error messages ("run scripts/01_build_dataset.py") were left
    behind. A reader who followed one was told to run a file that is not
    there.
    """
    import re

    pattern = re.compile(r"scripts/([0-9]{2}[a-z]?_[A-Za-z0-9_]*\.py)")
    stale = []
    for pat in ("*.py", "*.md", "*.yml"):
        for p in ROOT.rglob(pat):
            if ".venv" in p.parts or ".git" in p.parts:
                continue
            for name in pattern.findall(p.read_text(errors="ignore")):
                if not ((PIPELINE / name).exists() or (DIAGNOSTICS / name).exists()):
                    stale.append(f"{p.relative_to(ROOT)} -> scripts/.../{name}")
    assert not stale, f"references to scripts that do not exist: {stale}"


def test_diagnostics_write_no_artefacts():
    """scripts/diagnostics/ must not write into results/.

    The split between pipeline/ and diagnostics/ is only useful if it is
    true: a reader must be able to trust that results/*.json comes from
    scripts/pipeline/ and from nowhere else. One diagnostic used to write
    results/spike_thermal.json.
    """
    import re

    pattern = re.compile(r"""results["']?\s*/\s*["']?(\w+\.json)|["']results["']""")
    offenders = []
    for p in sorted(DIAGNOSTICS.glob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "results" not in line:
                continue
            # a mention in a comment or docstring is fine; a write is not
            if line.lstrip().startswith("#"):
                continue
            if re.search(r"""(write_text|to_csv|open|json\.dump|mkdir)\s*\(""",
                         line) and "results" in line:
                offenders.append(f"{p.name}:{i}: {line.strip()[:70]}")
    assert not offenders, (
        f"diagnostics write into results/: {offenders}"
    )


def test_scripts_are_split_into_pipeline_and_diagnostics():
    """The split tells a reader which scripts produce results and which do not."""
    assert PIPELINE.is_dir(), "scripts/pipeline/ is missing"
    assert DIAGNOSTICS.is_dir(), "scripts/diagnostics/ is missing"
    stray = [p.name for p in SCRIPTS.glob("*.py")]
    assert not stray, f"scripts left in the top level: {stray}"


def test_pipeline_scripts_write_a_result_artifact():
    """Everything in pipeline/ must produce a result file."""
    writers = {
        "01_build_dataset.py": "MANIFEST.json",
        "02_condition.py": "conditioning.json",
        "02b_decompose.py": "decomposition.json",
        "02c_per_turbine_curve.py": "per_turbine_curve.json",
        "03_detect.py": "detection.json",
        "03b_residual_attribution.py": "residual_attribution.json",
        "04_score.py": "care_score.json",
        "05_audit_scorability.py": "farm_a_audit.json",
        "07_lightgbm_nbm.py": "lightgbm_nbm.json",
        "08_multivariate.py": "multivariate.json",
        "08b_sweep_threshold.py": "threshold_sweep.json",
        "08c_engie_multivariate.py": "engie_multivariate.json",
        "09_report.py": "README.md",
        "11_demo.py": "demo_r80711.json",
        "10_summary.py": "summary.json",
        "12_ae_audit.py": "ae_audit.json",
        "13_subset_comparison.py": "subset_comparison.json",
        "14_thermal_signal.py": "thermal_signal.json",
    }
    for script, artefact in writers.items():
        p = PIPELINE / script
        assert p.exists(), f"{script} is missing from scripts/pipeline/"
        assert artefact in p.read_text(), (
            f"{script} does not mention {artefact} -- it must produce it"
        )


def test_pipeline_scripts_use_the_shared_module():
    """A pipeline script must import from src/windmill, not reimplement it.

    The two report generators are exempt: they read results/*.json and format
    them, so they have no domain logic to share.
    """
    exempt = {"09_report.py", "10_summary.py"}
    for p in sorted(PIPELINE.glob("*.py")):
        if p.name in exempt:
            continue
        text = p.read_text()
        assert "from windmill import" in text or "from windmill." in text, (
            f"{p.name} does not import from src/windmill"
        )


def test_reusable_math_lives_in_src():
    """The NBM helpers are used by two pipeline scripts, so they must be shared.

    They were originally defined inside a script and imported by path from
    another script, which is the pattern this rule forbids.
    """
    src = ROOT / "src" / "windmill"
    assert (src / "nbm.py").exists(), (
        "the learned-model helpers belong in src/windmill/nbm.py, not inside a "
        "pipeline script"
    )
    # EVERY script -- pipeline and diagnostics alike. A diagnostic that
    # imports a pipeline script by path is the same violation.
    for p in sorted(PIPELINE.glob("*.py")) + sorted(DIAGNOSTICS.glob("*.py")):
        text = p.read_text()
        assert "importlib.util.spec_from_file_location" not in text, (
            f"{p.name} imports another script by path -- shared code belongs in "
            f"src/windmill/"
        )


@pytest.mark.skipif(
    not (RESULTS / "subset_comparison.json").exists(),
    reason="subset comparison not generated",
)
def test_the_two_audits_cover_the_same_datasets():
    """The AE comparison is only meaningful if both arms saw the same events.

    The AE audit and the subset comparison each pick a subset. If they drift
    apart the "like-for-like" claim becomes false while still looking true in
    the README.
    """
    sc = json.loads((RESULTS / "subset_comparison.json").read_text())
    ae = json.loads((RESULTS / "ae_audit.json").read_text())
    subset_ids = set(sc.get("dataset_ids") or [])
    ae_ids = {r.get("event_id") for r in ae.get("per_dataset", [])
              if r.get("event_id") is not None}
    assert subset_ids, "the subset comparison recorded no dataset ids"
    assert subset_ids <= ae_ids, (
        f"the subset comparison scored events the AE audit never ran: "
        f"{sorted(subset_ids - ae_ids)}"
    )


@pytest.mark.skipif(
    not (RESULTS / "subset_comparison.json").exists(),
    reason="subset comparison not generated",
)
def test_the_subset_comparison_scores_both_labels():
    """Accuracy is averaged over the NORMAL datasets, so both labels are needed."""
    sc = json.loads((RESULTS / "subset_comparison.json").read_text())
    for key in ("this_project", "published_ae_baseline_measured_here"):
        v = sc.get(key, {})
        assert v.get("n_anomaly", 0) > 0, key
        assert v.get("n_normal", 0) > 0, key
        assert v.get("accuracy") is not None, key
        assert v.get("care_score") is not None, key


# --- Rule 3: CI protects determinism ----------------------------------------
def test_ci_workflow_exists_and_runs_the_tests():
    wf = ROOT / ".github" / "workflows" / "ci.yml"
    assert wf.exists(), "no CI workflow -- the test suite must run on every push"
    text = wf.read_text()
    assert "pytest" in text
    assert "3.11" in text, "CI must test the supported Python version"
    assert "on:" in text or "True" in text


def test_makefile_exists_and_exposes_the_pipeline():
    mk = ROOT / "Makefile"
    assert mk.exists(), "no Makefile -- the script order must not have to be remembered"
    text = mk.read_text()
    for target in ("pipeline:", "report:", "test:", "data:", "demo:"):
        assert target in text, f"Makefile has no {target} target"
    # Every pipeline script must be reachable from the Makefile. Some have
    # their own target (01 -> data, 11 -> demo, 09 -> report, 10 -> summary),
    # so the check is that the script is named SOMEWHERE, not that it is in
    # the `pipeline` target specifically.
    for p in sorted(PIPELINE.glob("*.py")):
        assert p.name in text, (
            f"{p.name} is not mentioned in the Makefile -- it must be runnable "
            "from a target, so the execution order never has to be remembered"
        )


# --- Rule 4: local cache artefacts never reach git --------------------------
def test_gitignore_covers_python_caches():
    text = (ROOT / ".gitignore").read_text()
    assert "__pycache__/" in text
    assert "*.pyc" in text


def test_no_python_cache_is_tracked():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    bad = [f for f in tracked if "__pycache__" in f or f.endswith(".pyc")]
    assert not bad, f"cache artefacts are tracked in git: {bad}"


def test_working_tree_has_no_cache_artifacts():
    """A stray global-python invocation must not dirty git status."""
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    bad = [line for line in out.splitlines() if "__pycache__" in line or ".pyc" in line]
    assert not bad, f"cache artefacts show up in git status: {bad}"
