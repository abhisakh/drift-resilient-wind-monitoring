"""Download and cache the pinned public datasets.

Every source is public and key-free. Each download is recorded with its SHA-256 so
a reviewer can verify they fetched the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

_USER_AGENT = "windmill-portfolio/0.1 (+reproducible research project)"


@dataclass
class Source:
    """A pinned remote file."""

    name: str
    url: str
    filename: str
    licence: str
    note: str = ""
    unpack: str | None = None  # "zip" | "tar" | None
    #: Expected SHA-256 of the downloaded file, when known. A reviewer can
    #: verify their own download against this without trusting ours.
    sha256: str | None = None


ENGIE = Source(
    name="engie_la_haute_borne",
    url="https://data.mendeley.com/public-api/zip/vmyg4yp3s8/download/2",
    filename="engie_v2.zip",
    licence="CC BY 4.0",
    note=(
        "Mendeley Data v2, DOI 10.17632/vmyg4yp3s8.2. Contains one nested zip of "
        "'Cleaned data' per turbine plus static-information.csv and "
        "data_description.csv."
    ),
    unpack="zip",
    sha256="8f6ae60a6351dc504209e9cf6d345b067be0dfa9036ab625e5c190e60a5f17c4",
)

ERA5 = Source(
    name="era5_openmeteo",
    url=(
        "https://archive-api.open-meteo.com/v1/era5"
        "?latitude=48.4569&longitude=5.5847"
        "&hourly=wind_speed_100m,wind_direction_100m,temperature_2m,surface_pressure"
        "&start_date=2012-12-01&end_date=2018-02-01"
        "&wind_speed_unit=ms&timezone=UTC"
    ),
    filename="era5_openmeteo.json",
    licence="Copernicus/ECMWF ERA5 (public)",
    note="ERA5 hourly reanalysis at the site; no API key required.",
    sha256="dbe8facbdc74cee115a4921335b0b2433500c10a16843c75936ce0c9aeee3371",
)

CARE = Source(
    name="care_to_compare",
    url="https://zenodo.org/api/records/15846963/files/CARE_To_Compare.zip/content",
    filename="CARE_To_Compare.zip",
    licence="CC BY-SA 4.0",
    note="Zenodo v6, DOI 10.5281/zenodo.15846963. 5.5 GB.",
    unpack="zip",
    sha256="ca61379e98956d891041ad45c885109bd8a14199fde0688d0184a11c2d4194f1",
)

SOURCES = (ENGIE, ERA5, CARE)

#: Where each source's archive is extracted, relative to `data/raw/`.
#:
#: This is the single source of truth. The download script and every reader
#: must agree on it: a mismatch means a fresh clone unpacks the data somewhere
#: the scripts do not look, and everything fails at run time.
EXTRACT_DIRS = {
    "care_to_compare": "care_extracted",
}


def extract_dir(src: Source) -> str:
    """The directory, relative to `data/raw/`, that `src` is extracted into."""
    return EXTRACT_DIRS.get(src.name, f"{src.name}_extracted")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(src: Source, dest_dir: Path, force: bool = False) -> Path:
    """Fetch src into dest_dir (if absent) and return its local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.filename
    if target.exists() and not force:
        return target

    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(src.url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=1800) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(target)
    return target


def verify(src: Source, path: Path) -> bool | None:
    """Check a downloaded file against the source's expected SHA-256.

    Returns True/False, or None when the source has no recorded checksum.
    """
    if src.sha256 is None:
        return None
    return sha256(path) == src.sha256


def unpack(src: Source, archive: Path, out_dir: Path) -> Path:
    """Extract src's archive into out_dir, returning the extracted root."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if src.unpack == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
            names = zf.namelist()
    elif src.unpack == "tar":
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
            names = tf.getnames()
    else:
        return out_dir

    # Mendeley's ENGIE archive nests a second zip; unwrap it too.
    for name in names:
        inner = out_dir / name
        if inner.suffix.lower() == ".zip":
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(inner.parent / inner.stem)
    return out_dir


def manifest(data_dir: Path) -> dict:
    """Build the provenance manifest for whatever is present in data_dir.

    Every source carries its URL, licence and expected SHA-256 whether or not
    the file is present locally, so the manifest doubles as the provenance
    record a reviewer checks their own download against. `sha256_verified` says
    whether the local copy actually matched that expectation.
    """
    entries = {}
    for src in SOURCES:
        path = data_dir / "raw" / src.filename
        entry = {
            "url": src.url,
            "licence": src.licence,
            "note": src.note,
            "extract_dir": extract_dir(src) if src.unpack else None,
            "expected_sha256": src.sha256,
            "present": path.exists(),
        }
        if path.exists():
            entry["bytes"] = path.stat().st_size
            actual = sha256(path)
            entry["sha256"] = actual
            entry["sha256_verified"] = (
                None if src.sha256 is None else actual == src.sha256
            )
        entries[src.name] = entry
    return entries


def write_manifest(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "MANIFEST.json"
    out.write_text(json.dumps(manifest(data_dir), indent=2) + "\n")
    return out
