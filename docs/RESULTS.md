# Windmill: full results, method and evidence

*The short overview is the [README](../README.md). This is the full technical report: every measurement, the method, and the negative results. The marked sections are generated from `results/*.json` by `scripts/pipeline/09_report.py`; do not edit them by hand.*

**Condition monitoring for wind-turbine gearbox bearings, built around one goal: cut the cost of unnecessary maintenance call-outs by raising fewer false alarms, while still warning early enough of a developing fault that the turbine can be stopped on purpose instead of tripping on its own.**

>  The central finding is that *the reference a detector compares against matters more than the model it uses.* A reference that is calibrated once and left to age turns slow, harmless drift into a stream of false alarms; one that is kept fresh and conditioned on the machine's operating point does not. The size of that effect, and what it costs in early warning, is measured below at matched false-alarm rates, including the results that do not support the project.

<!-- GOAL:START -->

## The goal

**Cut the cost of unnecessary maintenance call-outs by raising fewer false alarms, while still warning early enough of a developing bearing fault that the turbine can be stopped on purpose instead of tripping on its own.**

The two halves pull against each other: a detector can always be made quieter by making it less sensitive, and made to warn earlier by making it more sensitive. So no figure below is quoted alone -- each is stated at a false-alarm rate, and the evaluation suite compares every arm at the same one.

| the question | the answer | measured by |
|---|---|---|
| Fewer false alarms? | 2.2x fewer than with a one-year-old reference and 4.5x fewer than with a two-year-old one, at the same threshold rule | B.4, the age of the reference |
| Is a developing fault caught? | 85 % of injected +4 K ramps, a median 3.6 days after onset, with an hour of sustained alarm at a 1 % rate; a 0 K placebo is "caught" 26 % of the time; 17.2 false call-outs per turbine-year; a constant-median arm recalibrated the same way: 93 %, 2.3 days, chance 29 %, 16.1 false call-outs; the learned arm: 94 %, 3.0 days, chance 26 %, 13.6 false call-outs | the operating curve, a simulation |
| How much warning on the one real case? | R80711, September 2017: the deployed model at the standing 1 %, 1 h setting dispatched 4.6 days after the first sustained hot spell (2017-09-08); 0 of 40 settings of its dial warned early without a false dispatch in the quiet stretch before the onset | the real-anomaly check, unlabelled |

**What this does not show.**

- **The savings are modelled, not measured.** The public data has no maintenance log, so the cost of a false call-out is an assumption (a flat figure per false dispatch, adjustable in the dashboard), not an observation.
- **The faults are simulated, and the one real case is unlabelled.** The ENGIE release carries no fault label for the R80711 anomaly, so its lead time says how early a detector would have fired, not that a fault was confirmed.
- **On ENGIE's gearbox bearing the operating point does the work and the weather adds little.** Conditioning on load cuts the median prediction error by 2.49 K against a constant. Adding ambient temperature changes it by 0.007 K and wind speed by -0.012 K on this channel, which its oil cooling holds near its operating temperature. On EDP's transformer, generator and gearbox the weather term does matter (the EDP section), and it is what lets an alarm be split into load, weather and real excess heat.
- **The learned model is not the worse predictor.** Its median error is 2.02 K against 2.10 K for the deterministic model; the deterministic one is kept for being transparent, not for being more accurate.
- **The 46.5x figure is the extreme case, not the like-for-like one.** It compares a reference frozen in 2014 with a rolling one, each arm keeping its own threshold rule; the matched-threshold figures above are the ones to quote.
- **The simulated fault and the real one disagree.** The settings the economics prefer for a flat offset were also run on the one real anomaly: 4 of 4 arms never dispatched during it. The real excess grows with load and comes in spells, not as a steady step, so a rule that waits for a long unbroken alarm can miss it. The cost ranking should be read with that in mind.
- **Early warning at equal false call-outs.** At 12 false call-outs per turbine-year the best setting of a constant-median arm catches 90 % of +4 K ramps in 2.9 days, against 100 % in 2.1 days for the deployed model. The operating curve gives every budget, a held-out check and a load-proportional fault.
- **The weather is hindsight, not a forecast.** The expected temperature is computed from the turbine's measured power and reanalysis ambient temperature; the forecast chain (A.3) is a simulation with injected forecast error.

<!-- GOAL:END -->


---

## Executive Summary & Key Findings

1. **A fresh reference means fewer false alarms, by a factor of roughly two for a one-year-old reference.**
   * A reference that is calibrated once and left to age raises more false alarms the older it gets; the matched-threshold figures are in the goal table above, and the operating-curve section measures it in false call-outs per turbine-year for every arm. The extreme case, a reference frozen in 2014, gives a mean false-alarm rate of **26.37%** against **0.57%** for a rolling 30-day reference (**46.5×**, across multi-year ENGIE SCADA data). That comparison lets each arm keep its own threshold rule, so quote the matched figures first.
2. **Early warning is real, and it is paid for in false alarms.**
   * On `R80711` (Sept 2017) a sustained gearbox-bearing overheat peaks at **86.64 °C** while its sibling turbines stay below 80 °C. The case is unlabelled (no fault label, no maintenance record), and the days of warning a detector gives depend on the false-alarm rate it is allowed.
3. **Conditioning on the operating point improves prediction a great deal, but on this data it does not buy earlier warning at equal false call-outs.**
   * It cuts the temperature prediction error by more than half. But a constant median recalibrated the same way catches simulated ramps about as well at the same number of false call-outs a year, so the weather-conditioning idea is not what the early-warning result rests on. This held when the setting was chosen on half the weeks and scored on the other half, and for a fault whose heat scales with load. The learned (LightGBM) arm is not yet part of that comparison.
4. **The model class matters little.**
   * A transparent lookup table predicts the bearing temperature about as well as LightGBM. The deterministic model is the primary arm because it is auditable and has nothing to retrain, not because it is more accurate.
5. **An honest benchmark audit (`CARE`, 95 events), as external validity.**
   * The published autoencoder beats this project's thermal-only detector (**0.5809** reproduced against **0.5028**), because most CARE faults are not thermal; the audit shows where it loses and why.
6. **Reproducibility.**
   * An automated test suite (the count is generated in the Tests section), CI documentation sync, and deterministic seeds.

---

## Datasets Overview

* **ENGIE La Haute Borne SCADA**: 4 turbines, 10-minute intervals (2013–2018) for multi-year drift testing.
* **Copernicus ERA5**: Independent hourly site climatology (wind and temperature).
* **EDP onshore wind farm, Portugal** (Mendeley Data v2): the original of CARE's Farm A, with the turbines' own alarm log, a failure logbook and a met mast; the ground truth for the alarm triage.
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
medians. **With a current reference the deterministic arm has far fewer
false alarms in that comparison** (0.57 % against 3.94 %, about 7.0x fewer false alarms), and
it does so with no model to train and no thresholds to re-validate.

The project's claim is therefore not 'the deterministic method is better'
but the sharper one: **the reference matters more than the model.**
Re-estimating the reference reduced false alarms by 46.5x for the deterministic
detector and 4.0x for the learned one.

**The like-for-like reading.** The two ENGIE comparisons above let each arm keep its own threshold rule and its own spread estimate, so they measure more than the model. The evaluation suite further down fixes the protocol -- a strict train-then-test split, every arm calibrated to the same 1 % rate on its own training residuals -- and there the two models predict the bearing temperature almost equally well: a median error of 2.02 K for the learned arm and 2.10 K for the deterministic one. The deterministic model is the primary arm because it is transparent and has nothing to retrain, not because it is the more accurate predictor.

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

### The evaluation suite: every claim, measured

Eight tests, one protocol. The split is enforced in code (a window's
training rows end strictly before its test rows begin) and every arm's
alarm threshold is calibrated on its OWN out-of-fold training residuals
at the same 1 % rate, so the arms are compared at equal false-alarm
rates rather than at one shared number on one scale.

**A.1 — the weekly backtest.** 997 test weeks across the four ENGIE turbines; each predicted by a reference fitted on the thirty days immediately before it. The error is the median over weeks of the MAE, in kelvin.

| arm | MAE (K) | MAE p90 | RMSE (K) | bias (K) | false-alarm rate |
|---|---|---|---|---|---|
| constant median (no operating point) | 4.699 | 7.034 | 6.4307 | -2.1101 | 0.0 |
| load-conditioned (the operating point) | 2.088 | 3.2926 | 3.2623 | -0.6143 | 0.0015 |
| load + ambient (**the deployed model**) | 2.1023 | 3.3011 | 3.2819 | -0.5514 | 0.0014 |
| load + ambient + wind | 2.0912 | 3.275 | 3.2452 | -0.518 | 0.0021 |
| LightGBM (the learned arm) | 2.019 | 3.1266 | 2.9869 | 0.14 | 0.0065 |

**What this says, stated both ways.** Conditioning on the operating point is worth everything: the paired median difference against a constant median is **-2.493 K**, better in **100 %** of weeks. The learned arm is the more accurate PREDICTOR: its median MAE is 2.019 K against the deployed model's 2.1023 K, and the deployed model is better in only **37 %** of weeks. But at the matched threshold the deployed model's false-alarm rate is **0.0014** against the learned arm's **0.0065**: its calibration holds across weeks and the learned arm's does not. The reference matters more than the model -- but it is not the better predictor.

**A.2 — the weather ablation.** The paired median difference from adding the ambient temperature is **0.0069 K** (better in 37 % of weeks), and from adding wind speed on top of that is **-0.0118 K** (better in 69 % of weeks). **On this channel the weather inputs do not measurably cut the prediction error.** The operating point does; the weather does not. That is a negative result for the ablation as posed, and it is reported as one.

**A.3 — the forecast chain.** Forecast wind and temperature → the turbine's own power curve → predicted power → the load- and ambient-conditioned reference → expected temperature, with ERA5 plus injected forecast error standing in for a real forecast.

| forecast | chain MAE (K) | amplification |
|---|---|---|
| perfect | 2.2913 | 1.0 |
| good (+-0.5 m/s, +-1.0 K) | 2.71 | 1.169 |
| typical (+-1.0 m/s, +-1.5 K) | 3.6491 | 1.5 |
| poor (+-2.0 m/s, +-2.5 K) | 5.1071 | 1.976 |

The chain's error is already the reference's own error plus the power curve's; the forecast error multiplies it. At a typical forecast the chain's error stays well inside the alarm band, so a forecast does NOT blow the band up -- but the chain is a worse predictor than conditioning on the measured power.

**B.4 — the age of the reference.** The same detector, the same rule, the same weeks, with only the reference's age changed. Every arm is calibrated to 1 % of its own training window.

| reference age | false-alarm rate | ratio vs 30 days |
|---|---|---|
| 30 days | 0.012727 | 1.0 |
| 90 days | 0.023412 | 1.84 |
| 180 days | 0.022442 | 1.76 |
| 365 days | 0.027933 | 2.19 |
| 730 days | 0.056831 | 4.47 |

The ratio grows with the age of the reference, which is the project's headline claim measured rather than asserted: 1.8× at three months, 2.2× at a year, 4.5× at two years, on the same windows and the same rule.

**B.5 — the cleaning test.** A +3 K ramp is injected into the last ten days of the training window, and the reference is fitted with and without trimming.

* how far the reference moved toward the fault: 1.9832 K untrimmed, 2.0491 K trimmed;
* how much the fault still stands out on the faulted rows: 0.6771 K untrimmed, 0.7157 K trimmed.

**Trimming does not stop the reference absorbing the ramp** at this ramp size: the median residual per load bin is a robust statistic, and a ramp over a third of the window moves it whether or not the ramped rows are dropped. Trimming protects against a fault that is large relative to the spread; it does not protect against one that is a slow bias.

**C.6 — fault injection.** A ramp injected into a healthy test window, scored at the matched 1 % threshold. A simulation: it says what the detector would do if the machine did this, and nothing about whether the machine ever does.

| arm | ramp | detection rate | median days to first alarm |
|---|---|---|---|
| lightgbm | 2 K | 1.0 | 1.278 |
| lightgbm | 4 K | 1.0 | 0.923 |
| lightgbm | 8 K | 1.0 | 0.667 |
| load+ambient | 2 K | 0.825 | 2.076 |
| load+ambient | 4 K | 0.9 | 1.469 |
| load+ambient | 8 K | 1.0 | 0.927 |

**C.7 — the R80711 lead time.** The September 2017 anomaly, unlabelled, with the reference re-estimated every day.

* the temperature first crosses 80 °C on **2017-09-08**;
* it first diverges from its three siblings by 5 K on **2017-09-01**;
* at the matched 1 % rate the arms first alarm on **{'constant': '2017-09-08', 'load': '2017-09-08', 'load+ambient': '2017-09-08', 'load+ambient+wind': '2017-09-08', 'lightgbm': '2017-09-02'}**;
* at a 5 % rate, **{'constant': '2017-09-05', 'load': '2017-09-02', 'load+ambient': '2017-09-02', 'load+ambient+wind': '2017-09-02', 'lightgbm': '2017-09-02'}**.

**At 1 % the detector gives no lead over the 80 °C crossing; at 5 % it gives six days.** The lead time is bought with false alarms, and the price is measurable.

**C.8 — the CARE thermal subset.** The events whose fault windows contain rows in normal operation are the ones where the score is even defined.

| set | anomaly events | coverage | earliness |
|---|---|---|---|
| all anomaly events | 45 | 0.1513 | 0.0795 |
| thermal-observable | 34 | 0.2003 | 0.0951 |

Coverage improves by **0.049** and earliness by **0.0156** once the non-thermal faults are excluded — which is the argument for a thermal detector being the right instrument for this benchmark.
### The daily check: predict the healthy value, compare it every day

Each month a model is trained on at least a month of healthy readings from before the month, leaving out standstill, the 12-hour warm-up after every restart (a bearing that restarts cold runs up to 12 K below normal), every day that was flagged, and everything before the last maintenance. It is then frozen for the month: it never learns from the month's own sensor values. Every day it predicts what a healthy sensor should read from that day's power and weather, and the day's mean difference from what the sensor shows is compared with a tolerance. A run of hot days is one call-out at the end of its first day. After a confirmed fault the model stays frozen until maintenance; after maintenance it is blank until a new month of healthy data exists.

**Stood in for, because the data have no records:** maintenance is a restart after three days or more without running; what an inspection would find is the documented anomaly span (R80711 from 15 August 2017). A hot day outside it counts as a false alarm.

**False call-outs a year and the real fault, by tolerance:**

| tolerance | healthy siblings, false call-outs a year | R80711, false call-outs a year | R80711, first call-out on the fault | fault days hot, of those checked | siblings, cold days a year |
|---|---|---|---|---|---|
| 1 K | 1.5 | 9.0 | 2017-08-15 23:50 | 30 of 60 | 35.5 |
| 1.5 K | 0.1 | 6.4 | 2017-08-15 23:50 | 64 of 114 | 24.4 |
| 2 K | 0.1 | 4.6 | 2017-08-15 23:50 | 62 of 114 | 16.6 |
| 3 K | 0.1 | 1.8 | 2017-08-15 23:50 | 45 of 115 | 7.5 |
| plain 80 °C limit, held 1 h | 0.0 / 0.0 / 0.0 | 2.9 | 2017-08-15 11:10 | | |

On healthy days the day's mean difference has a standard deviation of 1.00 to 1.19 K on the siblings, so a tolerance of 1 K sits inside the model's own day-to-day noise. Too tight a tolerance also starves the training: every flagged day is left out, and when too few healthy days remain the model goes blank.

**R80711 before its fault (2 K).** Hot call-outs outside the fault period, by year: 2013: 1, 2015: 4, 2016: 1, 2017: 15. Hot days by month from 2017: 2017-05: 2, 2017-06: 6, 2017-07: 9, 2017-08: 7, 2017-09: 10, 2017-10: 14, 2017-11: 13, 2017-12: 15, 2018-01: 7. The bearing began running hotter than its own healthy reference from May 2017, months before the first sustained hot spell on 8 September. With no fault labels this is a candidate early warning, not a proven one: those days count as false alarms here.

**What it costs in coverage (2 K):**

| turbine | running days | days blank (start, maintenance) | running days checked | maintenance restarts |
|---|---|---|---|---|
| R80711 | 1812 | 174 | 72 % | 2 |
| R80721 | 1804 | 366 | 60 % | 4 |
| R80736 | 1817 | 246 | 65 % | 2 |
| R80790 | 1559 | 357 | 61 % | 6 |

**The 7-day forecast (2 K, healthy days, the four turbines averaged).** The month's frozen model is fed a weather forecast issued days before; the power comes from the turbine's own power curve against wind. The forecast is simulated: ERA5 plus a stated error, a bias per day and noise per hour, whose sigma is 1 a day ahead rising to 2.5 a week ahead (m/s for wind, K for temperature); power from the turbine's own power curve against ERA5 wind; the model in force when the forecast is issued.

| issued, days before | forecast vs sensor (K) | forecast vs on-day prediction (K) |
|---|---|---|
| 1 | 1.68 | 1.44 |
| 2 | 1.87 | 1.63 |
| 3 | 2.02 | 1.82 |
| 4 | 2.16 | 1.96 |
| 5 | 2.34 | 2.15 |
| 6 | 2.51 | 2.31 |
| 7 | 2.55 | 2.34 |
| on the day, actual weather | 0.65 | |

The model fed the day's actual weather is within 0.65 K of the sensor on average; fed a forecast, the error grows from 1.7 K a day ahead to 2.6 K a week ahead. That is the weather's error, not the bearing's, so the call-out is made on the day and the forecast is for planning.

**Second opinions (2 K): every gearbox sensor, and the siblings.** Each sensor has its own model. The siblings are compared at the same moment, power within a tenth of rated, the normal gap learned per wind sector or overall, and a call-out when the 5-day median change passes +2 K.

| turbine | bearing model alone | 2 of 4 gearbox sensors | siblings, per wind sector | siblings, one normal |
|---|---|---|---|---|
| R80711 | 4.6 / yr, fault 2017-08-15 | 5.9 / yr, fault 2017-08-15 | 3.6 / yr, fault 2017-09-09 | 4.4 / yr, fault 2017-08-18 |
| R80721 | 0.0 / yr | 1.8 / yr | 0.5 / yr | 0.0 / yr |
| R80736 | 0.4 / yr | 0.8 / yr | 0.3 / yr | 0.0 / yr |
| R80790 | 0.0 / yr | 0.5 / yr | 0.3 / yr | 0.3 / yr |

False call-outs a year, and the day of the first call-out on the fault. Neither second opinion beats the bearing's own model: asking two gearbox sensors to agree adds false call-outs, and learning the sibling gap per wind sector did not lower them on the healthy turbines and called the fault later. They stay what the technician looks at, not the alarm.

**What this does not show.** Maintenance and inspections are stood in for, not recorded. There is one documented fault. Days with too few usable readings are not checked. The tolerance is chosen on the same history it is scored on.

### CARE Farm A against the operator's own records

CARE's Wind Farm A is built from EDP's open data (an onshore farm in Portugal), and EDP also publishes what CARE leaves out: the turbines' event log (their own alarms) and a logbook of confirmed failures. That makes it the one place where an alarm can be checked against what actually broke.

**The data hold together.** CARE's overlapping files are identical on 1,424,201 shared timestamps; each CARE turbine matches its EDP turbine reading for reading (T01 100 %, T06 100 %, T07 100 %, T11 100 %); 11 of CARE's 12 fault labels end within 30 minutes of a logged failure (the other is a second label of the same failure) and 0 normal windows contain one. CARE shifted each file by whole days to hide the year; the true dates are 2016-2017. CARE's Farm A status column is rebuilt from the logbook (14-day "downtime" blocks ending at each failure, with the turbine producing during much of them), so stops are found from power.

**What counts as a real fault.** The logbook lists confirmed failures: a part repaired or replaced. An alarm is *followed by a failure* when a logged failure of the same component, where heat was the cause or the symptom, follows within 30 days. The turbines' alarm system raised 56 alarm spells on the four logged turbines; 15 were followed by a failure of the component. An alarm with no failure is not judged harmless: overheating that ended without a repair is not in the logbook.

**Triage of the temperature alarms.** At each alarm's hottest readings the temperature is split into the component's typical value, what the load adds, what the weather adds, and the excess its healthy model cannot explain; the verdict follows the excess, by the component's own healthy model and by its siblings at the same moment.

| verdict | alarms | followed by a failure | no failure |
|---|---|---|---|
| real excess heat | 34 | 7 | 27 |
| explained by load | 5 | 1 | 4 |
| explained by weather | 0 | 0 | 0 |
| no verdict | 2 | 0 | 2 |
| all | 41 | 8 | 33 |

On the real-excess-heat alarms the hottest readings averaged 113 C: typical 74 C, load +30 K, weather 1.5 K, excess +7.8 K.

**They cost production.** While windy, the turbines stood still 3.1 h a day during alarms with no later failure and 1.2 h a day in the 30 days after, against 0.5 h normally (alarms followed by a failure: 4.3 and 2.1 h). The turbines' own entries record 0 kW at these alarms: protective trips.

**The same triage on ENGIE**, of the fixed rule this project uses as its baseline (gearbox bearing at 80 C for an hour; ENGIE publishes no alarm log): 27 alarm spells, all on R80711; 26 real excess heat, 1 explained by load, 0 by weather. 19 of the real-heat spells lie in the documented fault period; the others came in the months before it. At their hottest readings: typical 70 C, load +9.4 K, weather -0.1 K, excess +3.5 K.

**Reading.** The model, trained on healthy data with the weather, confirms real excess heat where the turbines alarmed; it does not suppress those alarms, nor should it. It cannot judge faults that are not heat (pumps, fans, leaks, pressure): Farm A has no such signals.

### Why a trusted alarm is worth acting on (EDP)

**The weather term is needed to tell weather heat from fault heat.** The daily check was run on the four logged EDP turbines with the met mast's outside temperature as its weather input, and again with a constant in its place. Without the weather the daily error of the healthy prediction grows (median over the turbines):

| component | with weather | load only | error increase |
|---|---|---|---|
| transformer | 3.44 K | 3.99 K | +16 % |
| generator | 2.41 K | 3.63 K | +51 % |
| gearbox | 0.72 K | 1.03 K | +43 % |

On ENGIE's gearbox bearing, held near its operating temperature by its oil cooling, the weather adds almost nothing (see the goal section); on EDP's components it does. Measured with the weather in, the weather explained little of the alarms that fired (the triage above), so those alarms were mostly real heat.

**Confirmed alarms came well before costly failures.** For each of the 21 logged failures on these turbines: did the triage confirm real excess heat on the same component in the 60 days before, and how long did the turbine then stand still while the wind blew (from 2 days before to 14 days after)?

| turbine | failure | logged | confirmed heat, days before | windy standstill | normally |
|---|---|---|---|---|---|
| T06 | Generator replaced | 2016-10-27 | 54 | 113.2 h | 7.7 h |
| T07 | High temperature transformer. Transformer refrigeration repaired | 2016-08-23 | 51 | 91.8 h | 11.2 h |
| T06 | High temperature generator error | 2016-09-04 | 43 | 16.3 h | 7.7 h |

A month or more of warning is time to plan a repair for calm weather. What this does not show is that acting on the alarm would have prevented the failure; that is the plausible benefit, not a measured one. Nor does every failure come with a warning: T07 generator bearings damaged (2017-08-20, 136.8 h) had no confirmed alarm before it.

### The daily check on CARE Farms B and C (offshore, anonymised)

Every dataset of CARE's Farms B and C runs the same monthly policy as on ENGIE, with the site's own outside-temperature sensor in place of ERA5 (the farms stay anonymised). Sensors are grouped into components; each sensor is judged against a tolerance learned from its own last 90 healthy days (median + 4 robust SDs, never below 2 K), and a component is hot when any of its sensors is. A fixed 2 K does not carry over: Farm B's transformers lag their load by hours and miss their prediction by several kelvin on ordinary days. Farm C logs service, so there a long stop is maintenance only when service was logged during it; Farm B logs none and keeps the 3-day stand-in.

| | Farm B | Farm C |
|---|---|---|
| healthy datasets | 9 | 31 |
| false call-outs per component-year, monthly model | 6.66 | 6.24 |
| the same, model frozen once | 6.97 | 7.23 |
| labelled faults | 6 | 27 |
| on a component with temperature sensors | 6 | 19 |
| caught on their own component | 2 | 2 |
| fault window never checked | 1 | 11 |

**Reading.** Caught: 53 (Rotor Bearing 2 - Damage); 19 (high temperature in transformer cell); 47 (Failure due to Rotorbrake and Hydraulic problemes - Hydraulic pump A disabeld, 2h later turbine back in production - Analysis indicates a longer anomaly in temp_hydraulic_oil_tank_1_average until 15.01.2024 which is neither confirmed nor refuted.); 28 (P20_spinner_carbonbrush defekt + P20_Accumulators_hydraulic system). Most of Farm C's faults are pitch, electrical or communication faults that temperature does not see; most unchecked fault windows follow a long stop with service logged, after which the model waits for a new healthy month. The false call-outs come mainly from a few noisy sensors (Farm B's generator, Farm C's pitch and hydraulics); gearbox and rotor-bearing sensors stay low. Refreshing the model every month helps less here than on ENGIE, because a learned tolerance partly adapts the frozen model too.

### The historical confirmation: train on last month, predict this month

The project in one paragraph. Every month a fresh reference is fitted on at least one month of healthy readings (the whole previous month, reached further back only as far as needed), so it knows how each sensor behaves when nothing is wrong, including how load and weather move it. It is then asked what a healthy machine would show in the next month, given that month's load and weather, and an alarm is raised when the sensor runs hotter than that expectation, not merely above a fixed number. The data are historical, so the next month is already known and the prediction can be confirmed. This runs the dashboard's monthly loop over every month of every turbine (223 turbine-months). Health is an assumption: every month of every turbine, except the documented R80711 anomaly.

**Was the month predicted?** The gearbox bearing's temperature, predicted from the month before:

| turbine | healthy months | median error | 90th-percentile error | months within 1 K of unbiased |
|---|---|---|---|---|
| R80711 | 52 | 1.67 K | 2.06 K | 92 % |
| R80721 | 58 | 1.81 K | 2.26 K | 90 % |
| R80736 | 58 | 1.90 K | 2.42 K | 81 % |
| R80790 | 49 | 1.71 K | 2.23 K | 90 % |
| all four | 217 | 1.80 K | 2.27 K | 88 % |

**False alarms in healthy months: the model against a plain 80 °C limit.** A dispatch is an alarm sustained for an hour. In a healthy month every dispatch is a false one. The last column is the fixed limit's alarms on which the model stayed quiet, because load and weather explained that temperature.

| turbine | model dispatches | per turbine-year | 80 °C limit dispatches | per turbine-year | limit alarms the model stayed quiet on |
|---|---|---|---|---|---|
| R80711 | 60 | 13.8 | 12 | 2.8 | 0 of 12 |
| R80721 | 22 | 4.6 | 0 | 0.0 | 0 of 0 |
| R80736 | 29 | 6.0 | 0 | 0.0 | 0 of 0 |
| R80790 | 17 | 4.2 | 0 | 0.0 | 0 of 0 |
| all four | 128 | 7.1 | 12 | 0.7 | 0 of 12 |

**The anomaly months (R80711).** Here an alarm is what is wanted:

| month | prediction error | model dispatches | first model dispatch | 80 °C limit dispatches | first limit dispatch |
|---|---|---|---|---|---|
| 2017-08 | 2.06 K | 0 | none | 3 | 2017-08-03 13:50 |
| 2017-09 | 2.29 K | 1 | 2017-09-13 16:20 | 9 | 2017-09-08 09:10 |
| 2017-10 | 2.07 K | 0 | none | 8 | 2017-10-05 01:30 |
| 2017-11 | 2.47 K | 1 | 2017-11-11 17:30 | 11 | 2017-11-02 04:30 |
| 2017-12 | 2.35 K | 7 | 2017-12-08 01:00 | 25 | 2017-12-07 16:20 |
| 2018-01 | 1.97 K | 0 | none | 8 | 2018-01-01 00:00 |

**Can the false alarms be turned down, and what does it cost?** The same history re-scored at stricter alarm settings, with nothing refitted. False dispatches per turbine-year in healthy months, then how many of R80711's anomaly months were alarmed at all, and when the first dispatch of September 2017 came (the real hot spell began on 2017-09-08).

| rule | all four | R80711 | R80721 | R80736 | R80790 | anomaly months alarmed | first dispatch, Sep 2017 |
|---|---|---|---|---|---|---|---|
| model, z above 2, held 1 h | 19.7 | 33.7 | 10.6 | 20.1 | 15.2 | 6 of 6 | 2017-09-13 16:20 |
| model, z above 2.5, held 1 h | 11.7 | 22.4 | 6.8 | 10.8 | 7.3 | 6 of 6 | 2017-09-13 16:20 |
| model, z above 3, held 1 h | 7.1 | 13.8 | 4.6 | 6.0 | 4.2 | 3 of 6 | 2017-09-13 16:20 |
| model, z above 4, held 1 h | 3.8 | 7.4 | 2.9 | 2.3 | 2.9 | 2 of 6 | 2017-09-13 16:20 |
| model, z above 5, held 1 h | 2.3 | 5.3 | 1.7 | 0.8 | 1.5 | 1 of 6 | none |
| model, z above 6, held 1 h | 1.4 | 4.2 | 1.0 | 0.2 | 0.5 | 1 of 6 | none |
| model, z above 8, held 1 h | 0.8 | 2.1 | 0.8 | 0.0 | 0.2 | 0 of 6 | none |
| **plain 80 °C limit, held 1 h** | 0.7 | 2.8 | 0.0 | 0.0 | 0.0 | 6 of 6 | 2017-09-08 09:10 |
| **80 °C limit, vetoed by the model at z above 2, held 1 h** | 0.7 | 2.8 | 0.0 | 0.0 | 0.0 | 5 of 6 | 2017-09-08 11:50 |
| **80 °C limit, vetoed by the model at z above 2.5, held 1 h** | 0.7 | 2.8 | 0.0 | 0.0 | 0.0 | 5 of 6 | 2017-09-13 07:30 |
| **80 °C limit, vetoed by the model at z above 3, held 1 h** | 0.7 | 2.8 | 0.0 | 0.0 | 0.0 | 4 of 6 | 2017-09-13 07:30 |
| **80 °C limit, vetoed by the model at z above 4, held 1 h** | 0.7 | 2.8 | 0.0 | 0.0 | 0.0 | 4 of 6 | 2017-09-13 08:40 |
| **80 °C limit, vetoed by the model at z above 5, held 1 h** | 0.7 | 2.8 | 0.0 | 0.0 | 0.0 | 4 of 6 | 2017-09-12 21:20 |
| **80 °C limit, vetoed by the model at z above 6, held 1 h** | 0.7 | 2.8 | 0.0 | 0.0 | 0.0 | 3 of 6 | 2017-09-12 21:20 |
| **80 °C limit, vetoed by the model at z above 8, held 1 h** | 0.3 | 1.2 | 0.0 | 0.0 | 0.0 | 3 of 6 | 2017-09-12 21:20 |
| model, z above 2, held 6 h | 1.6 | 4.4 | 1.0 | 0.2 | 1.0 | 1 of 6 | none |
| model, z above 2.5, held 6 h | 0.9 | 2.5 | 0.8 | 0.2 | 0.2 | 1 of 6 | none |
| model, z above 3, held 6 h | 0.8 | 1.8 | 0.8 | 0.2 | 0.2 | 0 of 6 | none |
| model, z above 4, held 6 h | 0.3 | 0.7 | 0.6 | 0.0 | 0.0 | 0 of 6 | none |
| model, z above 5, held 6 h | 0.2 | 0.2 | 0.4 | 0.0 | 0.0 | 0 of 6 | none |
| model, z above 6, held 6 h | 0.1 | 0.0 | 0.4 | 0.0 | 0.0 | 0 of 6 | none |
| model, z above 8, held 6 h | 0.1 | 0.0 | 0.2 | 0.0 | 0.0 | 0 of 6 | none |
| **plain 80 °C limit, held 6 h** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5 of 6 | 2017-09-08 17:10 |
| **80 °C limit, vetoed by the model at z above 2, held 6 h** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4 of 6 | 2017-09-08 23:00 |
| **80 °C limit, vetoed by the model at z above 2.5, held 6 h** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4 of 6 | 2017-09-13 07:30 |
| **80 °C limit, vetoed by the model at z above 3, held 6 h** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3 of 6 | 2017-09-13 07:30 |
| **80 °C limit, vetoed by the model at z above 4, held 6 h** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3 of 6 | 2017-09-13 08:40 |
| **80 °C limit, vetoed by the model at z above 5, held 6 h** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3 of 6 | 2017-09-13 02:20 |
| **80 °C limit, vetoed by the model at z above 6, held 6 h** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3 of 6 | 2017-09-13 02:20 |
| **80 °C limit, vetoed by the model at z above 8, held 6 h** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3 of 6 | 2017-09-13 02:20 |

The rows in bold that say *vetoed* are the two-stage rule: the 80 °C limit triggers, and a dispatch stands only if the healthy model also finds the temperature unexpected inside it. A limit alarm that load and weather explain is dropped; one the model cannot predict (a load it has hardly seen) is NOT explained, so it stands. Note that the August row counts as an anomaly month because the anomaly span starts on 15 August, so a limit alarm early in August is counted as caught although it came before the anomaly.

**The sibling warning and the full strategy.** The sibling warning compares a turbine with its siblings at the same moment and similar power, against its normal gap over the model's training window, smoothed by a 5-day median; a run of flagged days is one call-out at the end of its first day. *Confirmed* means the model also found unexplained heat that day. The strategy is the veto (z above 2) plus the sibling call-outs, counted once per day. False call-outs per turbine-year in healthy months:

| rule | R80711 | R80721 | R80736 | R80790 | all four | R80711 fault months alarmed | first call-out, Sep 2017 |
|---|---|---|---|---|---|---|---|
| plain 80 °C limit, held 1 h | 2.8 | 0.0 | 0.0 | 0.0 | 0.7 | 6 of 6 | 2017-09-08 09:10 |
| limit, vetoed by the model at z above 2 | 2.8 | 0.0 | 0.0 | 0.0 | 0.7 | 5 of 6 | 2017-09-08 11:50 |
| sibling warning, +2 K | 4.2 | 0.0 | 0.0 | 0.2 | 1.1 | 5 of 6 | 2017-09-09 23:50 |
| sibling warning, +2 K, confirmed | 3.7 | 0.0 | 0.0 | 0.2 | 0.9 | 5 of 6 | 2017-09-09 23:50 |
| **strategy: veto + sibling, +2 K** | 6.9 | 0.0 | 0.0 | 0.2 | 1.7 | 6 of 6 | 2017-09-08 11:50 |
| **strategy: veto + confirmed sibling, +2 K** | 6.5 | 0.0 | 0.0 | 0.2 | 1.6 | 6 of 6 | 2017-09-08 11:50 |
| sibling warning, +2.5 K | 3.2 | 0.0 | 0.0 | 0.2 | 0.8 | 5 of 6 | 2017-09-11 23:50 |
| sibling warning, +2.5 K, confirmed | 2.3 | 0.0 | 0.0 | 0.0 | 0.6 | 5 of 6 | 2017-09-11 23:50 |
| **strategy: veto + sibling, +2.5 K** | 6.0 | 0.0 | 0.0 | 0.2 | 1.5 | 6 of 6 | 2017-09-08 11:50 |
| **strategy: veto + confirmed sibling, +2.5 K** | 5.1 | 0.0 | 0.0 | 0.0 | 1.2 | 6 of 6 | 2017-09-08 11:50 |
| sibling warning, +3 K | 1.2 | 0.0 | 0.0 | 0.0 | 0.3 | 5 of 6 | 2017-09-12 23:50 |
| sibling warning, +3 K, confirmed | 0.9 | 0.0 | 0.0 | 0.0 | 0.2 | 5 of 6 | 2017-09-12 23:50 |
| **strategy: veto + sibling, +3 K** | 3.9 | 0.0 | 0.0 | 0.0 | 0.9 | 6 of 6 | 2017-09-08 11:50 |
| **strategy: veto + confirmed sibling, +3 K** | 3.7 | 0.0 | 0.0 | 0.0 | 0.9 | 6 of 6 | 2017-09-08 11:50 |

**Where the veto's removals were.** In R80711's healthy months the limit raised 12 dispatches and the veto kept 12 of them at z above 2 and 12 at z above 3. In its fault months the limit raised 64 and the veto kept 46 and 29. So the alarms the veto removes are, on this record, alarms raised DURING the fault, most likely because the monthly reference re-learns a fault that has lasted for weeks (its training window then holds it) and so explains its heat away; that mechanism is not measured directly here.

**R80711's stretches hotter than its siblings than usual** (more than 2 K above its gap over the previous 30 days, 5-day median), for judging what they were:

| from | to | days | peak |
|---|---|---|---|
| 2013-08-15 | 2013-08-18 | 4 | +2.25 K |
| 2013-09-24 | 2013-09-28 | 5 | +3.55 K |
| 2013-09-30 | 2013-10-04 | 5 | +2.81 K |
| 2013-10-12 | 2013-10-12 | 1 | +2.03 K |
| 2013-11-29 | 2013-12-01 | 3 | +2.82 K |
| 2013-12-11 | 2013-12-11 | 1 | +4.48 K |
| 2013-12-14 | 2013-12-15 | 2 | +4.48 K |
| 2014-10-29 | 2014-10-29 | 1 | +2.03 K |
| 2015-01-21 | 2015-01-23 | 3 | +2.33 K |
| 2015-02-11 | 2015-02-15 | 5 | +4.01 K |
| 2015-10-18 | 2015-10-20 | 3 | +2.81 K |
| 2015-11-02 | 2015-11-05 | 4 | +3.85 K |
| 2016-01-21 | 2016-01-22 | 2 | +2.49 K |
| 2016-12-21 | 2016-12-21 | 1 | +2.33 K |
| 2017-07-31 | 2017-07-31 | 1 | +2.17 K |
| 2017-08-03 | 2017-08-04 | 2 | +2.17 K |
| 2017-09-09 | 2017-09-16 | 8 | +5.82 K |
| 2017-10-22 | 2017-10-24 | 3 | +4.49 K |
| 2017-11-12 | 2017-11-14 | 3 | +2.81 K |
| 2017-11-24 | 2017-11-27 | 4 | +4.51 K |
| 2017-12-10 | 2017-12-17 | 8 | +6.96 K |
| 2017-12-27 | 2017-12-27 | 1 | +2.32 K |
| 2017-12-29 | 2018-01-08 | 11 | +7.66 K |

**What this does not show.** Health is assumed, not labelled. There is one real anomaly. In the later anomaly months the fresh reference has been fitted partly on the anomaly itself, which is its documented blind spot. The alarm settings are scored on the same history they are read from, so none of them is a forecast of how a setting would do on unseen months.

### The operating curve: what a false call-out buys

The two halves of the goal pull against each other, so no figure is quoted alone. A detector has two dials: its **sensitivity** (the share of its own training window it is calibrated to alarm on) and its **persistence** (how long an alarm must be sustained before it is worth sending someone). At every setting this measures the false dispatches per turbine-year on healthy turbine-weeks, and how often and how fast an injected +4 K ramp is caught. A 0 K **placebo** ramp is run through the same code: whatever it "detects" is a false dispatch that happened to land after the onset, so it is the chance level a real detection has to be read against.

Exposure: 14.5 turbine-years of healthy operation over 241 non-overlapping windows. Windows touching the R80711 anomaly are left out for every arm alike.

**At the standing sensitivity (a 1 % rate), with and without persistence:**

| arm | sustained for | false dispatches per turbine-year | +4 K ramps caught | median days to dispatch | 0 K placebo "caught" |
|---|---|---|---|---|---|
| constant median (no operating point) | 10 min | 191.6 | 97 % | 1.1 | 67 % |
| constant median (no operating point) | 1 h | 16.1 | 93 % | 2.3 | 29 % |
| load-conditioned | 10 min | 250.2 | 98 % | 1.2 | 83 % |
| load-conditioned | 1 h | 10.3 | 84 % | 4.0 | 23 % |
| load + ambient (the deployed model) | 10 min | 271.5 | 98 % | 1.3 | 85 % |
| load + ambient (the deployed model) | 1 h | 17.2 | 85 % | 3.6 | 26 % |
| LightGBM (the learned arm) | 10 min | 447.5 | 100 % | 1.1 | 97 % |
| LightGBM (the learned arm) | 1 h | 13.6 | 94 % | 3.0 | 26 % |

With no persistence requirement the deployed arm "catches" 98 % of the +4 K ramps, but it also "catches" 85 % of the 0 K placebo ramps: most of that detection is chance. With an hour of sustained alarm it catches 85 % against a chance level of 26 %.

**At equal false call-outs per turbine-year, each arm at its own best setting:**

| false call-out budget | arm | sensitivity | sustained for | false dispatches | +4 K caught | median days | chance (0 K) |
|---|---|---|---|---|---|---|---|
| 6 | constant median (no operating point) | 2.00 % | 6 h | 3.1 | 72 % | 5.1 | 9 % |
| 6 | load-conditioned | 20.00 % | 24 h | 0.4 | 67 % | 7.1 | 1 % |
| 6 | load + ambient (the deployed model) | 20.00 % | 24 h | 0.4 | 60 % | 7.2 | 1 % |
| 6 | LightGBM (the learned arm) | 10.00 % | 6 h | 2.5 | 91 % | 3.5 | 8 % |
| 12 | constant median (no operating point) | 0.25 % | 1 h | 9.2 | 90 % | 2.9 | 21 % |
| 12 | load-conditioned | 20.00 % | 6 h | 7.5 | 99 % | 2.1 | 18 % |
| 12 | load + ambient (the deployed model) | 20.00 % | 6 h | 9.6 | 100 % | 2.1 | 20 % |
| 12 | LightGBM (the learned arm) | 20.00 % | 6 h | 8.3 | 100 % | 2.1 | 22 % |
| 26 | constant median (no operating point) | 1.00 % | 1 h | 16.1 | 93 % | 2.3 | 29 % |
| 26 | load-conditioned | 20.00 % | 6 h | 7.5 | 99 % | 2.1 | 18 % |
| 26 | load + ambient (the deployed model) | 20.00 % | 6 h | 9.6 | 100 % | 2.1 | 20 % |
| 26 | LightGBM (the learned arm) | 20.00 % | 6 h | 8.3 | 100 % | 2.1 | 22 % |
| 52 | constant median (no operating point) | 20.00 % | 6 h | 37.5 | 98 % | 3.0 | 51 % |
| 52 | load-conditioned | 20.00 % | 6 h | 7.5 | 99 % | 2.1 | 18 % |
| 52 | load + ambient (the deployed model) | 20.00 % | 6 h | 9.6 | 100 % | 2.1 | 20 % |
| 52 | LightGBM (the learned arm) | 20.00 % | 6 h | 8.3 | 100 % | 2.1 | 22 % |

* At 6 false call-outs per turbine-year: LightGBM (the learned arm) 91 % in 3.5 days; constant median (no operating point) 72 % in 5.1 days; load-conditioned 67 % in 7.1 days; load + ambient (the deployed model) 60 % in 7.2 days.
* At 12 false call-outs per turbine-year: load + ambient (the deployed model) 100 % in 2.1 days; LightGBM (the learned arm) 100 % in 2.1 days; load-conditioned 99 % in 2.1 days; constant median (no operating point) 90 % in 2.9 days.
* At 26 false call-outs per turbine-year: load + ambient (the deployed model) 100 % in 2.1 days; LightGBM (the learned arm) 100 % in 2.1 days; load-conditioned 99 % in 2.1 days; constant median (no operating point) 93 % in 2.3 days.
* At 52 false call-outs per turbine-year: load + ambient (the deployed model) 100 % in 2.1 days; LightGBM (the learned arm) 100 % in 2.1 days; load-conditioned 99 % in 2.1 days; constant median (no operating point) 98 % in 3.0 days.

**Setting chosen on half the windows, scored on the other half** (this removes the optimism of picking the best of a grid on the same weeks it is scored on):

| false call-out budget | arm | false dispatches | +4 K caught | median days | chance (0 K) |
|---|---|---|---|---|---|
| 6 | constant median (no operating point) | 5.2 | 77 % | 5.3 | 14 % |
| 6 | load-conditioned | 3.7 | 80 % | 5.4 | 9 % |
| 6 | load + ambient (the deployed model) | 2.4 | 56 % | 6.5 | 6 % |
| 6 | LightGBM (the learned arm) | 2.5 | 91 % | 3.5 | 8 % |
| 12 | constant median (no operating point) | 10.9 | 91 % | 2.7 | 21 % |
| 12 | load-conditioned | 7.5 | 99 % | 2.1 | 18 % |
| 12 | load + ambient (the deployed model) | 9.6 | 100 % | 2.2 | 20 % |
| 12 | LightGBM (the learned arm) | 8.3 | 100 % | 2.2 | 22 % |
| 26 | constant median (no operating point) | 23.3 | 93 % | 2.4 | 34 % |
| 26 | load-conditioned | 7.5 | 99 % | 2.1 | 18 % |
| 26 | load + ambient (the deployed model) | 9.6 | 100 % | 2.2 | 20 % |
| 26 | LightGBM (the learned arm) | 8.3 | 100 % | 2.2 | 22 % |
| 52 | constant median (no operating point) | 37.5 | 98 % | 2.9 | 51 % |
| 52 | load-conditioned | 7.5 | 99 % | 2.1 | 18 % |
| 52 | load + ambient (the deployed model) | 9.6 | 100 % | 2.2 | 20 % |
| 52 | LightGBM (the learned arm) | 8.3 | 100 % | 2.2 | 22 % |

**A fault whose heat scales with load** (the offset is +4 K at half of rated power and proportional to load from there), best setting within each budget:

| false call-out budget | arm | false dispatches | +4 K caught | median days | chance (0 K) |
|---|---|---|---|---|---|
| 6 | constant median (no operating point) | 3.1 | 63 % | 5.5 | 9 % |
| 6 | load-conditioned | 4.4 | 56 % | 6.5 | 13 % |
| 6 | load + ambient (the deployed model) | 5.3 | 53 % | 5.7 | 12 % |
| 6 | LightGBM (the learned arm) | 2.5 | 59 % | 7.2 | 8 % |
| 12 | constant median (no operating point) | 9.2 | 95 % | 3.0 | 21 % |
| 12 | load-conditioned | 7.5 | 78 % | 5.1 | 18 % |
| 12 | load + ambient (the deployed model) | 9.6 | 83 % | 5.4 | 20 % |
| 12 | LightGBM (the learned arm) | 8.3 | 78 % | 5.4 | 22 % |
| 26 | constant median (no operating point) | 16.1 | 97 % | 2.8 | 29 % |
| 26 | load-conditioned | 24.1 | 80 % | 4.3 | 37 % |
| 26 | load + ambient (the deployed model) | 9.6 | 83 % | 5.4 | 20 % |
| 26 | LightGBM (the learned arm) | 8.3 | 78 % | 5.4 | 22 % |
| 52 | constant median (no operating point) | 27.9 | 98 % | 2.2 | 39 % |
| 52 | load-conditioned | 44.4 | 90 % | 2.9 | 46 % |
| 52 | load + ambient (the deployed model) | 9.6 | 83 % | 5.4 | 20 % |
| 52 | LightGBM (the learned arm) | 8.3 | 78 % | 5.4 | 22 % |

**The age of the reference.** The same windows for every age, at a 1 % rate with an hour of sustained alarm; false dispatches per turbine-year:

| arm | reference 0 days old | reference 90 days old | reference 365 days old | oldest over freshest |
|---|---|---|---|---|
| constant median (no operating point) | 18.5 | 18.6 | 32.2 | 1.74x |
| load-conditioned | 11.6 | 23.7 | 23.2 | 2.00x |
| load + ambient (the deployed model) | 20.0 | 27.0 | 28.7 | 1.44x |
| LightGBM (the learned arm) | 13.0 | 38.7 | 30.7 | 2.35x |

**The one real anomaly, on the same dial.** Nothing is injected here: each arm is fitted on the month before 2017-08-15 and scored on R80711 up to 2017-10-15. The first sustained hot spell (a day with at least 30 samples above 80 °C) began on 2017-09-08; the onset, when R80711 first ran hotter than its siblings, is 2017-09-01. A dispatch before the onset is a false alarm, not a warning, so it is counted against the setting.

| arm | standing 1 %, 1 h | most profitable setting, flat offset | most profitable setting, load-proportional fault | settings that warned early with no false dispatch |
|---|---|---|---|---|
| constant median (no operating point) | 4.7 d late (0 false before onset) | 0.10 % · 6 h: never dispatched (0 false before onset) | 0.10 % · 6 h: never dispatched (0 false before onset) | 0 of 40 |
| load-conditioned | 4.6 d late (0 false before onset) | 20.00 % · 24 h: never dispatched (0 false before onset) | 0.25 % · 6 h: never dispatched (0 false before onset) | 0 of 40 |
| load + ambient (the deployed model) | 4.6 d late (0 false before onset) | 20.00 % · 24 h: never dispatched (0 false before onset) | 0.25 % · 6 h: never dispatched (0 false before onset) | 0 of 40 |
| LightGBM (the learned arm) | 4.9 d late (0 false before onset) | 20.00 % · 24 h: never dispatched (0 false before onset) | 5.00 % · 6 h: never dispatched (0 false before onset) | 0 of 40 |

**Where the excess is.** The mean temperature above what the load + ambient (the deployed model) model expected, by load (samples in brackets):

| load | before the onset | hot-spell days | other days after the onset |
|---|---|---|---|
| 0–20 % | -0.5 K (1216) | 2.3 K (121) | -1.6 K (2751) |
| 20–40 % | -0.0 K (350) | 2.9 K (169) | 0.3 K (600) |
| 40–60 % | 0.2 K (70) | 2.6 K (246) | -0.0 K (232) |
| 60–80 % | 0.3 K (19) | 6.4 K (181) | 1.8 K (56) |
| 80–100 % | 9.4 K (1) | 13.9 K (90) | 12.5 K (11) |

**What this does not show.** The ramp is a simulation of an offset on one channel; a real bearing fault may involve several channels and develop differently. The two shapes above bracket how its heat might depend on load, and the held-out table removes the optimism of choosing a setting on the weeks it is scored on, but none of the figures is an out-of-sample forecast of operations. Detection is measured on a 28-day window held to a reference fitted on the thirty days before it, which is the dashboard's monthly loop; a weekly refit would change the false-dispatch counts.

### What the operating curve is worth in money

**Three numbers here are assumed, not measured**, because the public data has no maintenance log: a false dispatch costs 2,000, catching a fault early is worth 185,000 (a planned repair against a failure that trips the turbine), and a fault turns up 0.05 times per turbine-year. What is measured is the false dispatches per turbine-year and the share of simulated faults caught, at every setting of every arm. Each arm is shown at its own most profitable setting, so it is judged at the dial that suits it, not at one setting shared by all. Net value is against not monitoring at all.

**A flat temperature offset:**

| arm | most profitable setting (sensitivity, sustained for) | false dispatches per turbine-year | +4 K caught | net value per turbine-year | break-even fault rate | net value at the standing 1 %, 1 h setting |
|---|---|---|---|---|---|---|
| constant median (no operating point) | 0.10 % · 6 h | 0.8 | 46 % | 2,743 | 0.018 | -23,723 |
| load-conditioned | 20.00 % · 24 h | 0.4 | 67 % | 5,351 | 0.007 | -12,765 |
| load + ambient (the deployed model) | 20.00 % · 24 h | 0.4 | 60 % | 4,738 | 0.007 | -26,520 |
| LightGBM (the learned arm) | 20.00 % · 24 h | 0.0 | 54 % | 5,028 | 0.000 | -18,503 |

**A fault whose heat scales with load:**

| arm | most profitable setting (sensitivity, sustained for) | false dispatches per turbine-year | +4 K caught | net value per turbine-year | break-even fault rate | net value at the standing 1 %, 1 h setting |
|---|---|---|---|---|---|---|
| constant median (no operating point) | 0.10 % · 6 h | 0.8 | 50 % | 3,088 | 0.016 | -23,300 |
| load-conditioned | 0.25 % · 6 h | 0.4 | 20 % | 1,015 | 0.022 | -14,185 |
| load + ambient (the deployed model) | 0.25 % · 6 h | 0.5 | 20 % | 838 | 0.027 | -27,864 |
| LightGBM (the learned arm) | 5.00 % · 6 h | 0.7 | 40 % | 2,343 | 0.019 | -21,421 |

* The most profitable arm for a flat offset, by share: constant median (no operating point) 0 %; load-conditioned 28 %; load + ambient (the deployed model) 0 %; LightGBM (the learned arm) 72 % of 75 combinations of the three assumptions.
* The most profitable arm for a load-proportional fault: constant median (no operating point) 61 %; load-conditioned 0 %; load + ambient (the deployed model) 0 %; LightGBM (the learned arm) 39 % of 75 combinations of the three assumptions.
* The arm with the best worse case across the two fault shapes: constant median (no operating point) 61 %; load-conditioned 0 %; load + ambient (the deployed model) 0 %; LightGBM (the learned arm) 39 % of 75 combinations of the three assumptions.

_The most profitable setting of constant median (no operating point) (least sensitive rate swept); load-conditioned (most sensitive rate swept); load + ambient (the deployed model) (most sensitive rate swept); LightGBM (the learned arm) (most sensitive rate swept) sits at the edge of the swept grid, so a wider sweep might find a better one._

**What this does not show.** The prices are placeholders: change them in the dashboard and the ranking and the break-even rates move. Each arm's setting is chosen on the same weeks it is scored on, so every figure is optimistic, equally for every arm. The detection rate is a simulated ramp, not a recorded fault. And a false dispatch is priced as one flat cost, whereas a real operator would look at the data remotely before sending anyone, which is cheaper.

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
.venv/bin/python scripts/pipeline/15_backtest.py          # the evaluation suite (8 tests)
.venv/bin/python scripts/pipeline/09_report.py           # regenerate this README
.venv/bin/python scripts/pipeline/10_summary.py          # one table across all arms
```

`make backtest` runs the evaluation suite: the weekly backtest and the
weather ablation, the forecast chain, the matched-threshold comparison
and the age of the reference, the cleaning test, fault injection, the
R80711 lead time, and the CARE thermal subset. It needs the raw release
(it reads the four ENGIE turbines), it takes about ninety seconds, and it
writes `results/backtest.json` — the source of the evaluation section
above. The train/test split is enforced in code by `Window`, which refuses
to be constructed with an overlap, and every arm's threshold is calibrated
on its own out-of-fold training residuals at the same 1 % rate.

`scripts/diagnostics/` holds the one-off probes used while designing the
detector. They write nothing and are not part of the pipeline.

## The dashboard

```bash
make dashboard     # http://localhost:8501
```

Four pages, the demonstration first and the research last:

| page | what it shows |
|---|---|
| **Demo** | one month, 15 August to 14 September 2017, a healthy turbine (R80721) above the faulty one (R80711). Three lines per day: the bearing temperature the frozen monthly model predicted from a weather forecast issued days before, what it predicted from the day's actual weather, and what the sensor read, with the tolerance as a band. The hot days are the call-outs. |
| **Explore** | any month, any of the nine temperature sensors, the four turbines: a traffic light for every day, a status card per turbine (monitoring, fault confirmed and model frozen, or maintenance hold with how much of the new healthy month has been collected), the three lines for one turbine, and the comparison with its siblings. |
| **Performance** | the daily check over the whole record: false call-outs a year by tolerance against the plain 80 °C limit, how the forecast error grows with lead time, the second opinions (every gearbox sensor, the siblings), R80711's year, and the live monthly loop with its cost. |
| **Evidence & method** | the headline measurements and the research behind them: the live weather forecast and power prediction, the model and CARE benchmark comparison, the engineering and data forensics, and the data explorer. |

Everything on the first three pages is read from `results/daily_check.json`
and `results/daily_series.json`, written by `make daily`.

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

516 tests, including `tests/test_scoring_matches_reference.py`, which proves our
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
