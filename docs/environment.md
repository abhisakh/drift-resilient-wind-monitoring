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

If the package is unavailable, `scripts/04_score.py` and `scripts/08_multivariate.py`
refuse to run rather than silently producing incomparable numbers.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

`scripts/01_build_dataset.py` downloads everything into `data/raw/` (19 GB
uncompressed for CARE alone). **No dataset is committed** — `.gitignore` excludes
`data/raw/`, and `data/README.md` records the URL, licence and SHA-256 of each
file so a reviewer can verify they fetched identical bytes.

| dataset | size | note |
|---|---|---|
| ENGIE La Haute Borne (Mendeley v2) | 77 MB | 74 MB zip + extracted CSVs |
| ERA5 via open-meteo | 1.7 MB | JSON, no API key needed |
| CARE to Compare (Zenodo v6) | 5.5 GB zip | ~19 GB extracted; the slow step |

The CARE download is the reason `01_build_dataset.py` takes a while. `--skip-care`
skips it if you only want the ENGIE arm.

## Reproducing the numbers

```bash
.venv/bin/python scripts/01_build_dataset.py
.venv/bin/python scripts/02_condition.py            # climatology + ERA5 agreement
.venv/bin/python scripts/03_detect.py              # ENGIE scheme comparison + gates
.venv/bin/python scripts/04_score.py               # CARE score, power-only baseline
.venv/bin/python scripts/05_audit_scorability.py   # what the benchmark can score
.venv/bin/python scripts/08_multivariate.py        # Option 1: multivariate bins
.venv/bin/python scripts/08b_sweep_threshold.py    # threshold sweep
.venv/bin/python scripts/08c_engie_multivariate.py # ENGIE, multivariate arm
.venv/bin/python scripts/09_report.py              # regenerate README from results/
```

Every script writes a JSON file into `results/`, and `09_report.py` builds the
README's Results section from those files — so the README cannot drift from the
artefacts.
