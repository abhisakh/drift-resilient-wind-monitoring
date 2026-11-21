"""The download path and the read path must agree.

`scripts/pipeline/01_build_dataset.py` extracts each archive into a directory, and every
script that reads the data looks in a directory. If those two disagree, a fresh
clone unpacks the data somewhere the scripts never look and every script fails
at run time -- with an error that points at the data, not at the path bug.

That is exactly the bug this file exists to prevent: CARE was extracted to
`care_extracted/` while the download script wrote `care_to_compare_extracted/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import acquire, care  # noqa: E402


def test_care_extraction_directory_is_well_known():
    """The CARE extraction directory is defined once, and is the one on disk."""
    src = next(s for s in acquire.SOURCES if s.name == "care_to_compare")
    assert acquire.extract_dir(src) == "care_extracted"


def test_care_root_points_inside_the_extraction_directory():
    """care_root() must live under the directory the download script writes."""
    src = next(s for s in acquire.SOURCES if s.name == "care_to_compare")
    extract = acquire.extract_dir(src)
    root = care.care_root(ROOT / "data")
    assert root.parent.name == extract
    assert root.name == "CARE_To_Compare"


def test_download_script_uses_the_shared_definition():
    """01_build_dataset.py must not derive the directory name on its own."""
    src = (ROOT / "scripts" / "pipeline" / "01_build_dataset.py").read_text()
    assert "acquire.extract_dir(src)" in src
    # the old, divergent derivation must be gone
    assert 'f"{src.name}_extracted"' not in src


def test_no_script_hardcodes_the_care_path():
    """No script may spell the CARE path out by hand."""
    offenders = []
    for p in list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "src").glob("**/*.py")):
        text = p.read_text()
        if "care_extracted" in text or '"CARE_To_Compare"' in text:
            # allowed only inside the shared definitions
            if p.name in {"care.py", "acquire.py"}:
                continue
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"hardcoded CARE path in: {offenders}"


@pytest.mark.skipif(
    not (ROOT / "data" / "raw" / "care_extracted").exists(),
    reason="CARE not downloaded",
)
def test_extracted_care_actually_lives_where_the_code_expects():
    """On a real checkout, the data is where the code says it is."""
    root = care.care_root(ROOT / "data")
    assert root.exists(), f"expected CARE at {root}"
    assert root.name == "CARE_To_Compare"
    for farm in ("Wind Farm A", "Wind Farm B", "Wind Farm C"):
        assert (root / farm).exists(), f"{farm} missing under {root}"
