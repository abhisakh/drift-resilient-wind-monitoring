# Windmill — Drift-Resilient Wind Turbine Condition Monitoring

Weather-conditioned residuals, deterministic gates, and auditability.

**Status:** active development. Numbers in `results/` are produced by the scripts in
`src/windmill/` against the pinned public datasets listed in `data/README.md`.

## The problem

A wind turbine's power depends on the weather, and the weather changes from year to
year and from season to season. A condition-monitoring baseline calibrated on one
period therefore reports "the turbine is degrading" when the wind was simply
different, and stays silent when a real fault appears during an unusual weather
regime. Operators lose trust in the alarm and stop acting on it.

We test one falsifiable claim:

> On healthy turbines, a baseline calibrated on a single period produces a false
> degradation signal whose size tracks the weather shift; conditioning the baseline on
> the site's own wind climatology removes that signal, while a genuine degradation
> still surfaces.

## The data (all public, all pinned)

| Dataset | What it gives | Licence |
|---|---|---|
| ENGIE La Haute Borne SCADA (Mendeley v2) | 4 Senvion MM82 turbines, 10-min, 2013-01→2018-01, GPS coordinates, one documented gearbox bearing failure | CC BY 4.0 |
| Copernicus ERA5 via open-meteo archive | Hourly reanalysis wind speed @100 m and 2 m temperature at the site | free, no key |
| WeDoWind CARE to Compare (Zenodo v6) | 3 farms, 36 turbines, 10-min, labelled anomaly windows, CARE score | CC BY-SA 4.0 |

ENGIE gives the **year-over-year weather shift** test bed. CARE gives the **labelled
fault-detection benchmark** with a published score to beat (AE baseline 0.66).

## Layout

```
data/            pinned datasets + provenance (README.md records URL, licence, checksum)
src/windmill/    loaders, conditioning, detectors, gates, scoring
scripts/         one command per analysis step
results/         machine-readable outputs (JSON) — every number in the README comes from here
tests/           unit tests for the conditioning and gating logic
```

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_build_dataset.py     # download + verify + cache
.venv/bin/python scripts/02_condition.py          # build climatology-conditioned baseline
.venv/bin/python scripts/03_detect.py            # residual + deterministic gates
.venv/bin/python scripts/04_score.py              # CARE score vs published baselines
.venv/bin/python scripts/05_report.py            # write results/report.md
```

## Licence

Code: MIT. Data: see `data/README.md` — each dataset keeps its own licence.
