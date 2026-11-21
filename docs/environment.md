# Environment

## Why Python 3.11

The CARE score is defined in the CARE to Compare paper, but the benchmark's
authors ship an implementation in `energy-fault-detector` (Fraunhofer IEE). That
package declares `Requires-Python < 3.12`, so this project is pinned to
**Python 3.11**.

The project does not *depend* on that package at runtime: `src/windmill/scoring.py`
is a self-contained re-implementation, and `tests/test_scoring_matches_reference.py`
proves the two agree event by event (criticality counter, weighted score, every
per-event metric, and the final weighted score). The package is used in the
scoring scripts so the published numbers are directly comparable with the
mini-benchmark in the paper (AE 0.66, random 0.5, iForest < 0.5, trivial 0).

If the package is unavailable, `scripts/pipeline/04_score.py` and
`scripts/pipeline/08_multivariate.py` refuse to run rather than silently
producing incomparable numbers.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

`scripts/pipeline/01_build_dataset.py` downloads everything into `data/raw/`
(about 19 GB once CARE is extracted). **No dataset is committed** — `.gitignore` excludes
`data/raw/`, and `data/README.md` records the URL, licence and SHA-256 of each
file so a reviewer can verify they fetched identical bytes.

| dataset | size | note |
|---|---|---|
| ENGIE La Haute Borne (Mendeley v2) | 77 MB | 74 MB zip + extracted CSVs |
| ERA5 via open-meteo | 1.7 MB | JSON, no API key needed |
| CARE to Compare (Zenodo v6) | 5.5 GB zip | ~19 GB extracted; the slow step |

The CARE download is the reason `make data` takes a while. `--skip-care`
skips it if you only want the ENGIE arm.

## Reproducing the numbers

```bash
make data        # download + verify the pinned datasets
make pipeline    # every artifact-producing step, in order
make test        # the test suite
.venv/bin/python scripts/pipeline/09_report.py     # regenerate README from results/
```

Every script writes a JSON file into `results/`, and `09_report.py` builds the
README's Results section from those files — so the README cannot drift from the
artefacts.
