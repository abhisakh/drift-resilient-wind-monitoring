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

## The test count must not depend on the environment

The README states how many tests pytest collects, and `09_report.py` asks
pytest for that number so the count cannot go stale. That makes the count a
thing the CI documentation check depends on: the committed README has to be
reproducible **byte for byte** on every job in the matrix.

So the collected count has to be the same on Python 3.11 and 3.12, and the
same whether or not the optional packages are installed. The rule that
achieves this is:

> A test module may not skip itself at import time.

`pytest.importorskip` at module level skips during **collection**, so pytest
never collects that module's tests at all -- and the count then depends on
the Python version and the installed packages. `energy-fault-detector`
requires Python < 3.12, so the reference cross-check module skipped itself
on 3.12 and the count fell from 142 to 135. The 3.12 CI job then failed to
reproduce the committed README, by a single line, on a machine where
nothing about this project's own code was wrong.

The fix is to skip at **run** time -- inside the test, or in a fixture -- so
the tests are always collected and only their execution is skipped.
`tests/test_repo_rules.py::test_no_test_module_skips_at_import_time` enforces
this, so the mistake cannot be reintroduced silently.

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
