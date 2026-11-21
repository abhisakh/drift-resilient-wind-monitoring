"""Download and cache the pinned public datasets, then record their provenance.

    python scripts/01_build_dataset.py [--skip-care]

Downloads to data/raw/, extracts, and writes data/MANIFEST.json with the SHA-256 of
every file so a reviewer can verify they fetched identical bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from windmill import acquire  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-care", action="store_true",
                    help="skip the 5.5 GB CARE download")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    raw = ROOT / "data" / "raw"
    sources = list(acquire.SOURCES)
    if args.skip_care:
        sources = [s for s in sources if s.name != "care_to_compare"]

    for src in sources:
        print(f"[{src.name}] {src.url}")
        path = acquire.download(src, raw, force=args.force)
        print(f"  -> {path.name}  {path.stat().st_size/1e6:.1f} MB")
        ok = acquire.verify(src, path)
        if ok is True:
            print("  -> sha256 verified against the expected value")
        elif ok is False:
            print(f"  -> WARNING: sha256 does NOT match the expected value "
                  f"({(src.sha256 or '')[:16]}...) -- the download may be corrupt")
        if src.unpack:
            # The extraction directory is defined once, in acquire.extract_dir,
            # and every reader uses that same definition. Deriving it from the
            # source name here instead would silently disagree with the
            # scripts that read the data.
            out_dir = raw / acquire.extract_dir(src)
            acquire.unpack(src, path, out_dir)
            print(f"  -> extracted to {out_dir}")

    manifest_path = acquire.write_manifest(ROOT / "data")
    print(f"\nmanifest: {manifest_path}")
    for name, entry in acquire.manifest(ROOT / "data").items():
        if entry.get("present"):
            print(f"  {name}: {entry['bytes']/1e6:.1f} MB  sha256={entry['sha256'][:16]}...")
        else:
            print(f"  {name}: not present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
