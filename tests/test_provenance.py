"""Provenance: every dataset must carry a checksum a reviewer can check.

A reviewer who downloads the data independently should be able to confirm they
fetched the same bytes we used. That requires the expected SHA-256 to be
recorded in the source code, not just computed locally at run time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import acquire  # noqa: E402


def test_every_source_records_an_expected_checksum():
    """Without a recorded hash, a reviewer cannot verify their download."""
    missing = [s.name for s in acquire.SOURCES if not s.sha256]
    assert not missing, f"no expected sha256 for: {missing}"


def test_expected_checksums_are_well_formed():
    for s in acquire.SOURCES:
        assert s.sha256 is not None, s.name
        assert len(s.sha256) == 64, s.name
        assert all(ch in "0123456789abcdef" for ch in s.sha256), s.name


def test_manifest_reports_expected_hash_even_when_absent(tmp_path):
    """The manifest is a provenance record, not just a local inventory."""
    m = acquire.manifest(tmp_path)
    assert m, "manifest is empty"
    for name, entry in m.items():
        assert entry["expected_sha256"], name
        assert entry["url"].startswith("http"), name
        assert entry["licence"], name


def test_manifest_marks_verification_status(tmp_path):
    """A present file is checked against the expected hash."""
    src = acquire.CARE
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / src.filename).write_bytes(b"not the real archive")

    m = acquire.manifest(tmp_path)
    entry = m["care_to_compare"]
    assert entry["present"] is True
    assert entry["sha256_verified"] is False  # wrong bytes -> not verified


def test_verify_returns_none_when_no_checksum_is_recorded(tmp_path):
    src = acquire.Source(
        name="x", url="http://example.invalid/x", filename="x.bin", licence="none"
    )
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert acquire.verify(src, p) is None


def test_verify_detects_a_matching_file(tmp_path):
    # SHA-256 of b"hello"
    src = acquire.Source(
        name="x", url="http://example.invalid/x", filename="x.bin", licence="none",
        sha256="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
    )
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert acquire.verify(src, p) is True
