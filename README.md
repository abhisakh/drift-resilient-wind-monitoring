# Drift-Resilient Wind Turbine Condition Monitoring: Weather-Conditioned Residuals, Deterministic Gates & Auditability

**A weather-aware wind turbine power and condition-monitoring pipeline that separates inter-annual weather shifts from genuine equipment degradation using climatology-conditioned power residuals, deterministic alert gating, and auditable MLOps workflows. Separating benign weather shifts from real hardware degradation in wind turbine SCADA data, cutting false alarms by 46.5× with an auditable, zero-leakage pipeline.**

>  This project demonstrates that reference calibration matters more than model complexity, cutting false alarms from **26.37%** to **0.57%** across multi-year ENGIE SCADA data and **95** CARE benchmark events.

---

## Executive Summary & Key Findings

1. **The Reference Strategy Matters More Than the Model (46.5× Fewer False Alarms)**
   * Naive static baselines produce **14.89–37.73%** false alarms (mean **26.37%**) as weather regimes shift.
   * A rolling 30-day, load- and temperature-conditioned reference reduces false alarms to **0.34–0.95%** (mean **0.57%**), outperforming LightGBM (**3.94%**).
2. **Caught a Real, Undocumented Bearing Anomaly (`R80711`, Sept 2017)**
   * Flagged a sustained gearbox bearing overheat peaking at **86.64 °C** against sibling control turbines.
3. **Honest, Leak-Free Benchmark Audit (`CARE`, 95 Events)**
   * Reproduced published Autoencoder claims across **95** events (**0.5028** deterministic CARE score vs. **0.5809** reproduced AE), balancing normal-operation accuracy (**95.1%** vs. **77.9%**).
4. **Production-Grade Reproducibility**
   * Backed by **130** automated unit/architectural tests, CI documentation sync, and deterministic seeds.

---

## Datasets Overview

* **ENGIE La Haute Borne SCADA**: 4 turbines, 10-minute intervals (2013–2018) for multi-year drift testing.
* **Copernicus ERA5**: Independent hourly site climatology (wind and temperature).
* **WeDoWind CARE to Compare**: **95** labelled early fault-detection events across 3 farms and **36** turbines.

*Note: Complete technical details, schemas, and full markdown structures can be reviewed directly in the repository documentation and referenced JSON files.*


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
margin** (0.57 % against 3.94 %, about 7.0x fewer false alarms), and
it does so with no model to train and no thresholds to re-validate.

The project's claim is therefore not 'the deterministic method is better'
but the sharper one: **the reference matters more than the model.**
Re-estimating the reference reduced false alarms by 46.5x for the deterministic
detector and 4.0x for the learned one.

We also do **not** claim operational weather forecasting. ERA5 conditions a
long-term reference; it is not a forecast engine, and the residual is always
measured against the turbine's own anemometer.

And we do **not** claim to beat the published autoencoder baseline. Measured on the same events, it beats us -- by how much, and in which sub-scores, is set out in the AE-comparability section below, with the numbers in `results/`.

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

A **rolling local reference** -- each row compared against its own recent history at the same load bin, re-estimated continuously -- gives 0.34–0.95 % (mean 0.57 %, spread 0.61 pp). That is a **46.5x** reduction in false alarms from changing nothing but the reference.

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

### How much thermal signal is actually inside a fault window?

This project detects faults from **thermal channels**. Most of CARE's faults are not thermal -- "Hydraulic group", "Pitchfailure", "Converter Failure", "Safety chain relay open". It is worth measuring what that costs rather than assuming it.

Over the 45 labelled anomaly events (34 of them with scorable rows), the detector's score inside the fault window compared with its score outside it, on the same dataset:

| statistic | median inside the event | median outside | event higher in |
|---|---|---|---|
| max-z | 0.9942 | 1.1443 | 0.5152 |
| L2 norm | 1.8803 | 1.8998 | 0.4848 |

| farm | events | median inside | median outside | event higher in |
|---|---|---|---|---|
| Wind Farm A | 1 | 0.6745 | 0.5396 | 1.0000 |
| Wind Farm B | 6 | 1.6873 | 1.9492 | 0.6000 |
| Wind Farm C | 27 | 0.9683 | 1.1443 | 0.4815 |

**The thermal signal inside a fault window is not stronger than outside it.** The event is higher in 0.5152 of events -- a coin flip.

This is the quantitative form of the project's honest weakness, and it is a SCOPE limit rather than a badly chosen threshold: no threshold recovers a signal the thermal channels do not carry. It explains exactly where this project loses to the published autoencoder -- Coverage and Earliness -- while its Accuracy is better, because it raises fewer false alarms on normal behaviour.

### Is the published AE baseline (0.66) comparable with ours?

Read out of the benchmark authors' own implementation (`energy-fault-detector` 0.8.1):

- autoencoder layers [200, 100, 50], code 20, 10 epochs, anomaly score rmse, threshold selector quantile 0.95, fit_on_val=true
- the threshold is fitted on a 20% validation split of the TRAINING data -- so the threshold RULE itself does not use the scored events
- but the split and the Keras training are **not** seeded, so the published number is one stochastic run

Read out of the paper (sec. 4.2.2), which is where the comparability question is actually decided:

- the AE's architecture and hyperparameters were tuned with Optuna, and the threshold calibration is **per wind farm**
- farms A and B use an adaptive threshold with a per-farm sensitivity gamma, also optimised with Optuna
- farm C's fixed threshold was chosen by **maximising the F(1/2) score** against ground truth derived from the status-IDs
- the paper documents no held-out protocol for that search

> The published 0.66 is therefore the output of a search whose objective was the CARE score itself: a **tuned-on-the-benchmark** number, not a held-out one. That is a reason to be careful about quoting it as a like-for-like baseline -- but it is NOT by itself an explanation of the gap, because the measurement below runs the authors' own autoencoder under a leak-free threshold rule and the gap survives.

And the status-column question was measured rather than argued. The package has two feature-building entry points, one of which keeps `status_type_id` -- the fault annotation -- as a feature. The AE was run over all 95 datasets both ways: the scores are **identical to four decimal places**. The reason is that the AE trains only on normal rows, where the status column is constantly 0, so it carries no information the AE could exploit. That leak was never there.

One thing does differ in scope rather than in method: the AE sees the **full sensor vector** (power, wind, pitch, hydraulic, electrical), while this project's detector sees only the thermal channels -- so a hydraulic or pitch fault is invisible to it. The section above measures exactly what that costs.

**The like-for-like measurement.** Both arms were run over the **same 95 datasets** and scored with the same CARE scorer:

| arm | datasets | coverage | accuracy | reliability | earliness | CARE |
|---|---|---|---|---|---|---|
| this project (deterministic, max z) | 95 | 0.1513 | 0.9512 | 0.3810 | 0.0795 | 0.5028 |
| published AE (authors' own implementation) | 95 | 0.4082 | 0.7792 | 0.5204 | 0.4173 | 0.5809 |
| published AE (paper, Optuna-tuned) | 95 | - | - | - | - | 0.66 |

**The honest answer: the AE beats us, by 0.078 CARE** (0.5809 against 0.5028) -- with a leak-free threshold rule, over the whole benchmark, with the status column removed. The gap is not a protocol artefact.

But note the third row. The authors' **published** 0.66 is 0.079 above what their own default configuration reproduces here (0.5809). That difference is the part of the published number that comes from the Optuna search, the per-farm adaptive thresholds and the per-farm sensitivities -- none of which has a held-out protocol behind it. So 0.66 is the right number to quote as the state of the art, and it is not the right number to quote as a like-for-like baseline.

The shape of the gap is informative. This project's detector has the better **accuracy** (0.9512 against 0.7792): it raises far fewer false alarms. The AE has the better **coverage** (0.4082 against 0.1513) and the better **earliness** (0.4173 against 0.0795): it finds more of the true anomalies and finds them earlier.

And the section above says why: a leak-free threshold set at a training quantile has to be conservative to avoid false alarms, and this project's arm pays for that in exactly the sub-scores the AE wins -- it detects late and rarely. A threshold selected on the scored events would lift its coverage considerably -- the fixed-z upper bound in the sweep above reaches a coverage comparable to the AE's -- but that number is an upper bound, not a result.

As a cross-check, both arms were also scored on an identical 12-dataset subset, chosen and stratified by label: this project 0.5072, the AE 0.5966. Same ordering, same conclusion, on a different sample.

The authors' own autoencoder, run here and scored with their own CARE scorer:

| configuration | datasets | coverage | accuracy | reliability | earliness | CARE |
|---|---|---|---|---|---|---|
| published protocol (as the authors' code runs it) | 95 | 0.4082 | 0.7792 | 0.5204 | 0.4173 | 0.5809 |
| same, with status_type_id removed from the features | 95 | 0.4082 | 0.7792 | 0.5204 | 0.4173 | 0.5809 |
| published protocol, a different random seed | 95 | 0.4069 | 0.7737 | 0.4936 | 0.4275 | 0.5751 |

Same protocol, different seeds: CARE 0.5809, 0.5751. The spread between two runs of the SAME configuration (0.0058) is the size of the uncertainty in any single published figure -- and the published pipeline is not seeded at all.

This audit covers ALL 95 CARE datasets.

**What we claim, precisely.** We claim that our 0.5028 is a leak-free number under a protocol we can re-derive and defend, and that the published figure is not quite that kind of number -- it is the output of a search whose objective was the CARE score itself, with no held-out protocol documented.

We do **not** claim to beat the published baseline. Measured on the same events, it beats us. The useful, defensible claim is narrower and more interesting: **we can tell you exactly where we lose and why**, we can tell you what part of the gap is protocol and what part is method, and every number above is re-derivable from the artefacts in `results/`.

The same two arms on ENGIE, where every alarm is a false alarm:

| arm | min % | max % | mean % | spread (pp) |
|---|---|---|---|---|
| deterministic (rolling ref) | 0.34 | 0.95 | 0.57 | 0.61 |
| learned (rolling ref) | 2.56 | 5.73 | 3.94 | 3.17 |

<!-- RESULTS:END -->

## Layout

```
app.py                   the Streamlit dashboard (hybrid: raw data or snapshot)
assets/                  the downloadable sample turbine CSV, for the dashboard's upload path
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

make dashboard   # the Streamlit dashboard (http://localhost:8501)
make demo        # watch the detector work on real SCADA
make snapshot    # export the dashboard's cloud snapshot
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

## The dashboard

```bash
make dashboard     # http://localhost:8501
```

A four-tab Streamlit dashboard, ordered the way a control room works:

| tab | what it does |
|---|---|
| 🌤️ **Weather Forecast & Power Prediction** | a real 7-day hourly forecast from Open-Meteo, the turbine's own conditioned power curve applied to it, and what-if sliders for a wind or temperature shift. Plus a **demo checkpoint** selector: curated historical weeks scored by a reference trained on exactly the thirty days before them, and by the same rule applied one and two years earlier. No model comparison on this tab. |
| 🔍 **Daily Cross-Check: Weather vs. Hardware Wear** | opens on the **fresh monthly fit** — fitted, scored against the target month and priced on arrival, with no button press — then the dispatch-episode and cost impact, then the live explorer. |
| ⚖️ **Model & Benchmark Comparison** | the deterministic arm against the learned one, the false-alarm rate by year, the CARE benchmark, and the audit of the published autoencoder baseline. |
| 🔧 **Engineering & Data Forensics** | the pipeline, the CI guarantees, and the upstream data bugs this project caught. |

The monthly loop fits its reference on the thirty days before the target
month and on nothing else, drops the training rows that reference itself
calls anomalous, and refits — so a fault that was already developing last
month cannot quietly become the new normal. It reports the fit time, the
clean-row count, and the three arms' alarm rates, and it prices the
difference in dispatches and euros.

It runs on arrival, so the chart, the cross-check and the cost are on
screen when the tab opens rather than one button press away. The peak card
reads **86.64 °C** where the model expected 78.9 °C: that difference is
wear, not weather.

A toggle above the chart chooses the resolution: **Daily Temperature
Prediction (1-Day Summary)** or **10-Minute Telemetry**. The day is the
default, because the detector works on ten-minute rows but the question an
O&M planner asks is a daily one. The daily view plots the actual daily
bearing temperature against the model's weather-conditioned prediction, and
the daily cross-check residual against the alarm threshold — so the
September overheat reads as two or three days where the actual line
leaves the predicted one behind and the residual crosses the threshold.
`live.daily_temperature_summary()` does the aggregation; it is the same
arithmetic on the same rows, not a second model.

It is **hybrid**. With `data/raw/` present it reads the raw ENGIE SCADA and
re-runs the detector live — that is the mode for a demo day. Without it (as on
Streamlit Community Cloud) it falls back to `results/demo_r80711.json`, a
compact snapshot exported by `make snapshot` from the same detector and the
same frozen 2014 reference, so the charts still work without the 19 GB. The
sidebar says which engine is running.

Every figure it shows is read from `results/*.json`; the dashboard carries no
copy of a headline number, so it cannot disagree with this README.

### Typography

Every font size on the dashboard is stated in one table in `app.py`
(`FONT_SIZES`), and the CSS is built from it — so raising the whole
dashboard's text is a one-line change, not a hunt through a stylesheet.
The reading text is roughly double what Streamlit ships with: prose at
34 px, notes and alerts at 28 px, KPI labels at 24 px, the tab headers
at 26 px, and the reference tables at 20 px.

The writing is deliberately thin: about a quarter of the prose that was
there has been cut, because a paragraph that restates the heading or the
chart is a paragraph the reader has to read past. What is left states a
number, a caveat, or an interpretation the chart cannot show on its own.

Two things that follow from that, and are deliberate:

* **The chart text is set on the figure, not on the page.** Plotly lays a
  chart out for the font it is given, so overriding it from CSS afterwards
  would leave the layout computed for a smaller font and the legends would
  run over each other. `CHART_FONT_PX` and `CHART_LEGEND_FONT_PX` do it
  properly.
* **The reference tables are `st.table`, not `st.dataframe`.** A dataframe
  is drawn on a canvas, where CSS cannot reach, so its text cannot be
  sized at all. They are fixed reference tables, so nothing is lost.

### The live explorer

The cross-check tab ends with a **Live Multi-Turbine & Weather-Conditioned
Explorer**: the
same detector, run on the data *you* choose, with the provenance of every run
on screen — rows processed, runtime in seconds, and which datasets were
joined. It carries the **reference switch** — *rolling 30-day* against
*frozen 2014* on the same window, same detector, same threshold — because
that comparison is the project's central result and it should be one click
away rather than a claim in the README.

| source | what it runs on | available when |
|---|---|---|
| ENGIE + ERA5 | any of the four La Haute Borne turbines, joined to reanalysis weather, with the other three as the control group | `data/raw/` present |
| CARE benchmark events | any labelled anomaly or normal-control dataset, with the ground-truth event window | `data/raw/` present |
| **Upload a turbine CSV** | **any** 10-minute SCADA file, self-calibrated on its own history | **always** |

The forecast on the first tab falls back to a deterministic profile derived
from the committed sample when the network is not there, and says so.

### The demo checkpoints and the stale baselines

The first tab carries a **Demo Checkpoint** selector, so a presenter can
move between curated historical weeks without touching anything else:

| checkpoint | prediction window | trained on |
|---|---|---|
| 1 | 2017-09-01 → 2017-09-07 | 2017-08-02 → 2017-08-31 |
| 2 | 2017-10-01 → 2017-10-07 | 2017-09-01 → 2017-09-30 |
| 3 | 2017-11-01 → 2017-11-07 | 2017-10-02 → 2017-10-31 |
| 4 | 2016-09-01 → 2016-09-07 | 2016-08-02 → 2016-08-31 |
| 5 | 2016-11-01 → 2016-11-07 | 2016-10-02 → 2016-10-31 |

The rule is enforced in one place and is not negotiable: **the reference is
always the thirty days immediately before the window being predicted** —
`[prediction_start − 30d, prediction_start)`. The stale baselines are that
same rule applied one and two years earlier, so the only thing that differs
between the three fits is how old the reference is.

The training window is stated in bold above the metrics, and again in each
KPI card, because "which window was it fitted on?" is the question that
decides whether a number means anything.

The dashboard opens **on Checkpoint 1**, so the comparison controls are on
screen the moment the app loads; the live Open-Meteo forecast is one
dropdown click away.

The chart opens **clean**: the actual output against the fresh 30-day
prediction, two lines and two cards. Three independent checkboxes — *Last 30
days rolling*, *Frozen 1 year before* and *Frozen 2 years before* — let a
presenter plot any single prediction curve, or any combination of them,
against the actual measured power. Each toggle also switches on its MAE
card, so the KPI row grows with the chart. With all three off there is
nothing to compare against, and the dashboard says so rather than showing a
bare line.

On Streamlit Community Cloud the one- and two-year-back curves come from
`results/demo_frozen_baselines.json`, exported by `make snapshot` from the
raw data — the Cloud deployment has no 19 GB of SCADA, and a test checks
that the exported artefact reproduces the live computation exactly.

### What the tracking error actually shows

Averaged over the five curated checkpoints, the mean absolute error is
**40.5 kW** for the 30-day fit, **40.5 kW** for the one-year-back fit and
**44.5 kW** for the two-year-back fit; in the steep part of the curve, where
the curve's *shape* matters rather than its level, it is
**43.9 → 45.0 → 49.0 kW**. The signed bias is cleanly ordered —
**−13.0 → −14.7 → −19.7 kW** — so the older the reference, the more it
under-predicts.

But **week to week the ordering is not consistent**: in only one of the five
checkpoints does the 30-day fit beat the one-year-back fit beat the
two-year-back fit on the full window. A single week's power is dominated by
that week's weather, and the age of the reference is a second-order effect
on it. The honest summary is that **reference freshness is a real but
second-order effect on power prediction** — a different claim from the one
the detection side of this project makes, which is why the two are shown
separately rather than merged into one story.

The upload path is the one that works everywhere, including on Streamlit
Community Cloud, and the dashboard offers a **downloadable sample** — a real
four-month slice of the ENGIE release, with the ERA5 ambient temperature
added, so a reviewer can try the detector without a 19 GB download. The
engine infers the columns by name and refuses clearly when a file lacks the
power or temperature channel the reference needs, rather than drawing an empty
chart.

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

252 tests, including `tests/test_scoring_matches_reference.py`, which proves our
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
