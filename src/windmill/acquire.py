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
)

CARE = Source(
    name="care_to_compare",
    url="https://zenodo.org/api/records/15846963/files/CARE_To_Compare.zip/content",
    filename="CARE_To_Compare.zip",
    licence="CC BY-SA 4.0",
    note="Zenodo v6, DOI 10.5281/zenodo.15846963. 5.5 GB.",
    unpack="zip",
)

SOURCES = (ENGIE, ERA5, CARE)


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
    """Build the provenance manifest for whatever is present in data_dir."""
    entries = {}
    for src in SOURCES:
        path = data_dir / "raw" / src.filename
        entry = {
            "url": src.url,
            "licence": src.licence,
            "note": src.note,
            "present": path.exists(),
        }
        if path.exists():
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = sha256(path)
        entries[src.name] = entry
    return entries


def write_manifest(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "MANIFEST.json"
    out.write_text(json.dumps(manifest(data_dir), indent=2) + "\n")
    return out
