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
| ENGIE La Haute Borne SCADA (Mendeley v2) | 4 Senvion MM82 turbines, 10-min, 2013-01→2018-01, GPS coordinates, multi-year healthy operation | CC BY 4.0 |
| Copernicus ERA5 via open-meteo archive | Hourly reanalysis wind speed @100 m and 2 m temperature at the site | free, no key |
| WeDoWind CARE to Compare (Zenodo v6) | 3 farms, 36 turbines, 10-min, labelled anomaly windows, CARE score | CC BY-SA 4.0 |

ENGIE gives the **year-over-year weather shift** test bed and the **false-alarm**
measurement — but **no usable fault label**: the one documented gearbox bearing
failure (R80721, 2015) is *absent* from the released cleaned CSVs, whose maximum
`Gb2t_avg` over the whole period is 78.85 °C against a documented 84.12 °C peak.
The cluster-based outlier filter removed it. CARE gives the **labelled
fault-detection benchmark** with a published score to beat (AE baseline 0.66).

One consequence is worth stating plainly: the abnormal turbine we *do* find in
ENGIE (R80711, September 2017) is **not** a labelled fault. It is an abnormality
discovered in unlabelled data with its three siblings as the control group.

Provenance, checksums and the measured data limitations are in `data/README.md`.

<!-- METHOD:START -->

## What this is, and what it is not

The method is **deterministic**: a per-turbine reference conditioned on load and
ambient temperature, a robust (MAD) spread, a one-sided z-score, and a fixed
threshold rule. There is no learned model in the primary arm.

A learned arm (LightGBM normal-behaviour model) is included as a **comparison**,
measured with the same statistic (max-z), the same threshold rule and the same
reference strategies, so only the model differs. On the CARE benchmark it
detects more of the true anomalies (coverage 0.2157 vs 0.1513) and pays for that with
false alarms on normal behaviour (accuracy 0.8517 vs 0.9512).

On the ENGIE site the comparison is more interesting than a clean sweep,
and the honest reading is this: **with a stale reference the learned arm is
the better one** (15.79 % false alarms against the
deterministic arm's 26.37 %), because a smooth model of
the operating point drifts less sharply than a set of fixed per-bin
medians. **With a current reference the deterministic arm wins by a wide
margin** (0.57 % against 3.94 %, about 7x fewer false alarms), and
it does so with no model to train and no thresholds to re-validate.

The project's claim is therefore not 'the deterministic method is better'
but the sharper one: **the reference matters more than the model.**
Re-estimating the reference reduced false alarms by 47x for the deterministic
detector and 4x for the learned one.

We also do **not** claim operational weather forecasting. ERA5 conditions a
long-term reference; it is not a forecast engine, and the residual is always
measured against the turbine's own anemometer.

<!-- METHOD:END -->

<!-- RESULTS:START -->

## Results

### The weather shift is real and measurable

At this site the capacity factor moves **3.28 pp** between years (CV 5.99 %), and the ERA5 annual wind deviation tracks the measured capacity-factor deviation with **r = 0.9546**. A wind-speed shift of about 2.16x is amplified into power — the cube law at work. The seasonal spread is larger still.

ERA5 against the turbine's own anemometer: r = 0.8103, bias -0.004 m/s. The intercept is large because the nacelle anemometer has a low-wind floor that ERA5 does not share — so ERA5 conditions the long-term reference, it does not replace the turbine's own sensor.

### Conditioning removes the false alarms

False-alarm rate on healthy turbines, per year (2013-2017), by scheme:

| scheme | min % | max % | spread (pp) | alarms/10 min | corr with weather shift |
|---|---|---|---|---|---|
| A_pooled_static | 3.083 | 7.590 | 4.507 | 0.05501 | 0.5136 |
| B_perturbine_static | 2.860 | 7.694 | 4.834 | 0.05533 | 0.5152 |
| C_perturbine_density | 2.498 | 6.970 | 4.472 | 0.04976 | 0.4048 |
| D_perturbine_density_band | 0.381 | 1.373 | 0.993 | 0.00942 | 0.2743 |
| E_perturbine_density_band_season | 0.373 | 2.719 | 2.346 | 0.01728 | 0.4786 |

A single-year, site-pooled baseline spans 3.08–7.59 % across years; the per-turbine, density-normalised, band-conditioned detector spans 0.38–1.37 % — the year-to-year spread falls from 4.51 pp to 0.99 pp, and the alarm rate stops tracking the weather shift (r 0.5136 → 0.2743).

### Most of the residual is a stale reference, not a bad turbine

Decomposing the static-baseline error into the part explained by the wind distribution and the part explained by the curve itself moving:

| year | total (kW) | distribution (kW) | curve (kW) | curve share |
|---|---|---|---|---|
| 2013 | 27.94 | 26.55 | 1.39 | 5.0 % |
| 2014 | 0.00 | 0.00 | 0.00 | nan % |
| 2015 | 64.74 | 57.27 | 7.47 | 11.5 % |
| 2016 | 11.69 | 4.87 | 6.82 | 58.4 % |
| 2017 | 42.92 | 39.67 | 3.25 | 7.6 % |

Refitting the reference on each year removes most of the bias -- so what a single-period baseline reports as degradation is mostly a stale reference, not a bad turbine:

| year | mean residual, 2014 reference (kW) | year-matched reference (kW) |
|---|---|---|
| 2013 | 2.724 | 1.145 |
| 2014 | 1.135 | 1.135 |
| 2015 | 6.949 | 0.589 |
| 2016 | 5.152 | 0.318 |
| 2017 | 4.500 | 0.396 |

### ENGIE: the reference strategy decides the false-alarm rate

The same multivariate thermal method on the ENGIE site, with every turbine healthy throughout -- so **every alarm is a false alarm**. Two detectors (the deterministic bins and a learned LightGBM normal-behaviour model) under two reference strategies:

| detector | reference | year | false-alarm rate % |
|---|---|---|---|
| deterministic | frozen_2014 | 2013 | 37.7254 |
| deterministic | frozen_2014 | 2015 | 17.5334 |
| deterministic | frozen_2014 | 2016 | 14.8883 |
| deterministic | frozen_2014 | 2017 | 35.3203 |
| learned (LightGBM) | frozen_2014 | 2013 | 19.2181 |
| learned (LightGBM) | frozen_2014 | 2015 | 7.1026 |
| learned (LightGBM) | frozen_2014 | 2016 | 9.6738 |
| learned (LightGBM) | frozen_2014 | 2017 | 27.1700 |
| deterministic | rolling_30d | 2013 | 0.9485 |
| deterministic | rolling_30d | 2014 | 0.3427 |
| deterministic | rolling_30d | 2015 | 0.5014 |
| deterministic | rolling_30d | 2016 | 0.5633 |
| deterministic | rolling_30d | 2017 | 0.4766 |
| learned (LightGBM) | rolling_30d | 2013 | 5.7270 |
| learned (LightGBM) | rolling_30d | 2014 | 2.5609 |
| learned (LightGBM) | rolling_30d | 2015 | 3.7721 |
| learned (LightGBM) | rolling_30d | 2016 | 4.1114 |
| learned (LightGBM) | rolling_30d | 2017 | 3.5296 |

A **frozen 2014 reference** -- what a naive deployment does -- gives 14.89–37.73 % false alarms (mean 26.37 %, spread 22.84 pp). Each turbine's thermal channels move by 5-9 K between years, so the frozen reference leaves a systematic offset that reads as overheating on every channel at once.

A **rolling local reference** -- each row compared against its own recent history at the same load bin, re-estimated continuously -- gives 0.34–0.95 % (mean 0.57 %, spread 0.61 pp). That is a **47x** reduction in false alarms from changing nothing but the reference.

The **learned** arm was given the same two reference strategies, so the comparison is like for like: its model is refitted weekly on the trailing 30 days, exactly as the deterministic arm's reference is re-estimated continuously. With a frozen reference the learned arm is 15.79 % (mean false-alarm rate) against the deterministic arm's 26.37 % -- so the model is *not* the deciding factor; the reference is. With a rolling reference the learned arm gives 3.94 % against the deterministic arm's 0.57 %.

This is the project's central result on the weather-shift side: the model matters far less than the reference it is compared against, and a reference that is not re-estimated turns slow drift into a flood of alarms.

### What the benchmark can and cannot score

The CARE score evaluates Coverage and Reliability only over rows the dataset marks as normal operation (status-ID 0 or 2). Rows already flagged as fault (status 4) inside an event window are excluded from the metric entirely, so no detector can earn credit for them.

| farm | anomaly events | zero scorable rows | mean % of event window normal | mean % status 4 |
|---|---|---|---|---|
| Wind Farm A | 12 | 11 | 4.7 % | 95.0 % |
| Wind Farm B | 6 | 0 | 95.1 % | 0.3 % |
| Wind Farm C | 27 | 0 | 72.1 % | 1.8 % |

**11 of 45** labelled anomaly events contain no scorable rows at all. This is a property of the published labels, not of any detector, and it caps what Coverage can be.

### CARE benchmark (95 labelled events)

Scored with the benchmark authors' own implementation (`energy-fault-detector` 0.8.1), cross-checked against our re-implementation (0 mismatches over 95 events).

| sub-score | value |
|---|---|
| coverage | 0.1263 |
| accuracy | 0.9606 |
| reliability | 0.2459 |
| earliness | 0.1347 |
| **CARE score** | **0.4856** |

Published baselines: AE 0.66, random 0.5, iForest < 0.5. Beats published AE: **False**.

### Option 1: deterministic multivariate bins

| combiner | coverage | accuracy | reliability | earliness | CARE score | events detected |
|---|---|---|---|---|---|---|
| max z | 0.1513 | 0.9512 | 0.3810 | 0.0795 | 0.5028 | 15/95 |
| L2 norm | 0.1342 | 0.9538 | 0.3608 | 0.0685 | 0.4942 | 13/95 |

### Threshold sweep for the multivariate arm

The threshold is the arm's one free parameter, so it was swept rather than chosen by taste. Two rules are swept, and they are not equivalent:

**Leak-free (training quantile)** — the threshold is the quantile of the cross-fitted TRAINING score distribution, so it never sees an event it is later scored on. This is the protocol the CARE paper's own baseline uses, and it is the row to quote:

| combiner | training quantile | CARE score | coverage | accuracy | reliability | earliness | events detected |
|---|---|---|---|---|---|---|---|
| zmax | 0.9 | 0.5028 | 0.1513 | 0.9512 | 0.3810 | 0.0795 | 15/95 |
| zmax | 0.95 | 0.4545 | 0.0808 | 0.9774 | 0.2055 | 0.0313 | 7/95 |
| zmax | 0.98 | 0.4263 | 0.0412 | 0.9900 | 0.0943 | 0.0161 | 2/95 |
| zmax | 0.99 | 0.4266 | 0.0269 | 0.9961 | 0.1020 | 0.0119 | 1/95 |
| zmax | 0.995 | 0.0000 | 0.0089 | 0.9986 | 0.0000 | 0.0042 | 0/95 |
| zmax | 0.999 | 0.0000 | 0.0025 | 0.9996 | 0.0000 | 0.0022 | 0/95 |
| l2 | 0.9 | 0.4942 | 0.1342 | 0.9538 | 0.3608 | 0.0685 | 13/95 |
| l2 | 0.95 | 0.4651 | 0.0786 | 0.9777 | 0.2597 | 0.0318 | 8/95 |
| l2 | 0.98 | 0.4243 | 0.0349 | 0.9913 | 0.0877 | 0.0161 | 3/95 |
| l2 | 0.99 | 0.4240 | 0.0226 | 0.9957 | 0.0943 | 0.0115 | 2/95 |
| l2 | 0.995 | 0.0000 | 0.0074 | 0.9978 | 0.0000 | 0.0028 | 0/95 |
| l2 | 0.999 | 0.0000 | 0.0018 | 0.9994 | 0.0000 | 0.0006 | 0/95 |

**Upper bound (fixed z, selected on the scored events)** — reported for completeness only. Choosing the threshold on the same 95 events that are then scored is leakage, so these numbers are optimistic and are NOT the result:

| combiner | z | CARE score | coverage | accuracy | reliability | earliness | events detected |
|---|---|---|---|---|---|---|---|
| zmax | 2.0 | 0.5285 | 0.3416 | 0.8146 | 0.4595 | 0.2123 | 35/95 |
| zmax | 2.5 | 0.5320 | 0.2874 | 0.8722 | 0.4698 | 0.1582 | 26/95 |
| zmax | 3.0 | 0.5234 | 0.2420 | 0.9063 | 0.4380 | 0.1241 | 23/95 |
| zmax | 4.0 | 0.5150 | 0.1766 | 0.9431 | 0.4286 | 0.0838 | 15/95 |
| l2 | 2.0 | 0.4857 | 0.4681 | 0.5538 | 0.4325 | 0.4205 | 61/95 |
| l2 | 2.5 | 0.4961 | 0.4397 | 0.6403 | 0.4151 | 0.3452 | 55/95 |
| l2 | 3.0 | 0.5076 | 0.4093 | 0.7117 | 0.4222 | 0.2830 | 45/95 |
| l2 | 4.0 | 0.5118 | 0.3476 | 0.8127 | 0.3886 | 0.1973 | 37/95 |

### Arm comparison

| arm | coverage | accuracy | reliability | earliness | CARE score |
|---|---|---|---|---|---|
| power-only residual (baseline) | 0.1263 | 0.9606 | 0.2459 | 0.1347 | 0.4856 |
| multivariate bins (max z) | 0.1513 | 0.9512 | 0.3810 | 0.0795 | 0.5028 |
| multivariate bins (L2) | 0.1342 | 0.9538 | 0.3608 | 0.0685 | 0.4942 |
| LightGBM thermal NBM | 0.2157 | 0.8517 | 0.3356 | 0.1785 | 0.4866 |

All four arms are scored on the same 95 events (45 anomaly, 50 normal). The learned arm has the **best coverage** of the three, and pays for it in accuracy: it raises alarms on more of the true anomalies and also on more normal behaviour.

Both the deterministic and the learned arm use the same threshold rule: the quantile of the TRAINING score distribution, so the threshold never sees an event it is later scored on. The learned arm's seeds are pinned explicitly (`random_state`, `seed`) and verified bit-identical across runs by `tests/test_lightgbm_determinism.py` -- without that, its score would not have been re-derivable, which is the whole point of the exercise.

### Is the published AE baseline (0.66) comparable with ours?

Read out of the benchmark authors' own implementation (`energy-fault-detector` 0.8.1):

- autoencoder layers [200, 100, 50], code 20, 10 epochs, anomaly score rmse, threshold selector quantile 0.95, fit_on_val=true
- the threshold is fitted on a 20% validation split of the TRAINING data -- so the threshold RULE itself does not use the scored events
- but the split and the Keras training are **not** seeded, so the published number is one stochastic run

Measured, by running the authors' own autoencoder on a subset of the datasets and scoring it with their own CARE scorer:

| configuration | datasets | coverage | accuracy | reliability | earliness | CARE |
|---|---|---|---|---|---|---|
| published | 3 | 0.4073 | not measured | 0.7143 | 0.3995 | nan |

This audit runs on a stratified SUBSET of the CARE datasets, not all 95: the authors' autoencoder takes roughly 5-8 minutes per dataset, so the full benchmark is an 8-13 hour job. The subset is stated here rather than presented as the full benchmark.

**What we claim:** that our 0.5028 is a leak-free number under a protocol we can re-derive and defend, and that the published 0.66 is not that. We do **not** claim to beat it -- the honest statement is that the two numbers are not the same kind of number, and that closing the gap honestly would require re-running the published baseline's own tuning loop under a held-out protocol, which is a separate piece of work.

The same two arms on ENGIE, where every alarm is a false alarm:

| arm | min % | max % | mean % | spread (pp) |
|---|---|---|---|---|
| deterministic (rolling ref) | 0.34 | 0.95 | 0.57 | 0.61 |
| learned (rolling ref) | 2.56 | 5.73 | 3.94 | 3.17 |

<!-- RESULTS:END -->

## Layout

```
data/                    pinned datasets + provenance (README.md records URL, licence, checksum)
src/windmill/            all the reusable logic: loaders, conditioning, detectors, scoring
scripts/pipeline/        the reproducible pipeline — one script per artefact, in order
scripts/diagnostics/     one-off probes used while designing the detector; they write nothing
results/                 machine-readable outputs (JSON) — every number in the README comes from here
tests/                   unit tests, including the repo's own maintenance rules
Makefile                 the execution order, so it never has to be remembered
```

The split between `pipeline/` and `diagnostics/` is the point: a reader can tell
at a glance which scripts produce `results/*.json` and which were just tools
for exploring the data. `tests/test_repo_rules.py` enforces it.

## Running it

Python **3.11** is required — the benchmark authors' CARE scorer
(`energy-fault-detector`) declares `Python < 3.12`. See `docs/environment.md`.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

make demo        # <-- START HERE: watch the detector work on real SCADA
make setup       # create the venv and install dependencies
make data        # download + verify the pinned datasets (19 GB, once)
make pipeline    # run every artefact-producing step, in order
make report      # regenerate README.md from results/
make test        # run the test suite
make help        # list every target
```

`make pipeline` runs the whole chain in the right order and ends by
regenerating this document, so no number in it can drift from the artefacts.
The individual steps, if you want to run one on its own:

```bash
.venv/bin/python scripts/pipeline/02_condition.py        # climatology, ERA5 agreement
.venv/bin/python scripts/pipeline/03_detect.py           # ENGIE scheme comparison + gates
.venv/bin/python scripts/pipeline/04_score.py            # CARE score, power-only baseline
.venv/bin/python scripts/pipeline/05_audit_scorability.py# what the benchmark can and cannot score
.venv/bin/python scripts/pipeline/08_multivariate.py     # Option 1: multivariate bins
.venv/bin/python scripts/pipeline/08b_sweep_threshold.py # leak-free threshold sweep
.venv/bin/python scripts/pipeline/08c_engie_multivariate.py # ENGIE: frozen vs rolling
.venv/bin/python scripts/pipeline/07_lightgbm_nbm.py     # learned-model arm
.venv/bin/python scripts/pipeline/12_ae_audit.py         # AE-comparability audit
.venv/bin/python scripts/pipeline/09_report.py           # regenerate this README
.venv/bin/python scripts/pipeline/10_summary.py          # one table across all arms
```

`scripts/diagnostics/` holds the one-off probes used while designing the
detector. They write nothing and are not part of the pipeline.

## The demo

`make demo` (i.e. `scripts/pipeline/11_demo.py`) runs the real detector on real SCADA and prints what it
sees. Two demonstrations:

```bash
.venv/bin/python scripts/pipeline/11_demo.py                       # ENGIE R80711, Sept 2017
.venv/bin/python scripts/pipeline/11_demo.py --dataset care # a labelled CARE event
.venv/bin/python scripts/pipeline/11_demo.py --list         # list the CARE events
```

The ENGIE one is the strongest single piece of evidence in the project: R80711
runs hot on one gearbox bearing channel while its three siblings, on the same
channel and the same weather, stay inside their range — and the peer residual
grows month over month instead of returning to normal. Nothing about it is
simulated; it is the same code path the CARE scoring uses.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

75 tests, including `tests/test_scoring_matches_reference.py`, which proves our
CARE score is identical to the benchmark authors' implementation — the
criticality counter, the weighted score, every per-event metric and the final
weighted score.

Also pinned:

- `tests/test_data_paths.py` — the download path and the read path cannot
  disagree (this caught a real bug: CARE used to be extracted somewhere no
  script looked).
- `tests/test_provenance.py` — every dataset carries a checksum a reviewer can
  verify their own download against.
- `tests/test_arm_comparison.py` — the deterministic and learned arms must be
  scored with the same statistic, the same threshold rule and the same
  reference strategies, so the comparison stays meaningful.
- `tests/test_lightgbm_determinism.py` — the learned arm's score must be
  re-derivable from a clean run.

## Licence

Code: MIT. Data: see `data/README.md` — each dataset keeps its own licence.
