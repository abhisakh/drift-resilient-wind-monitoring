# Windmill: catching slow turbine degradation that fixed alarm limits miss

**A healthy-turbine model, retrained every month, that flags a component running hot for its conditions before it reaches the alarm limit, and tells the crew whether an alarm is load, weather or a part going wrong.**

Built on public data: ENGIE La Haute Borne SCADA (France), EDP's wind farm with its alarm log and failure logbook (Portugal), CARE to Compare, and Copernicus ERA5 weather. Every number in the generated sections below comes from `results/*.json` and is checked by the test suite.

**Live dashboard:** [drift-resilient-wind-monitoring.streamlit.app](https://drift-resilient-wind-monitoring.streamlit.app)

## The problem

Wind turbines raise temperature alarms on fixed limits, such as 80 °C for a gearbox bearing. That has two weaknesses:

- **It is late.** A bearing can run several kelvin hotter than it should for weeks while staying under the limit.
- **It cannot explain itself.** "Over 80 °C" does not say whether the heat comes from full power on a hot day or from a part wearing out.

A learned model can fix both, but only if it stays current: a model trained once and left to age mistakes normal drift for faults.

## Which sensors raise turbine alarms

A turbine's control system watches many signals and raises an alarm when one crosses a fixed setting. In EDP's own event log (four turbines, 2016–2017) they include:

- **Component temperatures:** the high-voltage transformer, the generator windings, bearings and slip ring, the gearbox oil and bearings, and the hydraulic oil.
- **Pitch and hydraulics:** accumulator faults on the blade pitch axes, hydraulic pressure.
- **Yaw:** the nacelle turning faster than allowed.
- **Safety and grid:** an open emergency circuit, power cut-in and cut-out.

CARE's offshore fault records add converter, cooling-water, communication and pitch-electronics faults.

**This project focuses on component temperatures**, above all the **gearbox bearing temperature** on ENGIE's turbines, where a slowly developing problem shows as heat. On EDP's farm the same method is applied to the transformer, generator and gearbox temperature alarms.

<!-- CONCLUSION:START -->

## What the project shows

### 1. A model sees slow degradation that a fixed limit misses

On ENGIE's R80711, whose gearbox bearing ran abnormally hot from August 2017, the fixed limit (80 °C held for an hour) and the model (a day more than 2 K above the healthy prediction) were compared day by day:

| | days |
|---|---|
| abnormal period, days checked | 115 |
| flagged by both | 36 |
| **flagged by the model only** | **12** |
| flagged by the limit only | 6 |
| ten weeks before: model hot days (of them, limit days) | 18 (9) |

On the model-only days the bearing averaged 73.8 °C, 3.0 K hotter than a healthy bearing at that day's power and weather: too cool for the limit, too hot for the conditions. On the three healthy siblings the model raised 0.14 false call-outs per turbine-year, about as quiet as the limit. So the model does not replace the limit; it adds what the limit cannot see.

### 2. The model must be retrained every month

Turbines drift: each ENGIE turbine's temperatures move by several kelvin between years. A reference frozen in 2014 alarmed on **26.37 %** of readings; one refreshed on the last 30 days, on **0.57 %** (46.5× fewer). A stale model buries the real signal in false alarms. The monthly model never learns from flagged days, so a fault cannot become the new normal.

### 3. Every alarm can be explained

Each alarm is split into typical temperature + load + weather + **excess heat**. On EDP's farm in Portugal, checked against the operator's own alarm log and failure logbook, 34 of 41 temperature alarms were real excess heat and 5 were load; on ENGIE's 80 °C limit, 26 of 27. The weather added 1.5 K at those alarms, and the weather term is still needed to say so: without it the model's error grows by 16-51 %.

### Why it matters

A trusted alarm is one a crew acts on. On T06 the model confirmed real heat 54 days before "Generator replaced", which then cost 113.2 h of standstill in wind (normally 7.7 h). On T07 the model confirmed real heat 51 days before "High temperature transformer. Transformer refrigeration repaired", which then cost 91.8 h of standstill in wind (normally 11.2 h). Detecting the problem that early gives time to act before it degrades further, and to plan the repair for calm weather with parts and crew ready.

### Scope

- **Temperature-related faults.** The model watches heat. On CARE's offshore farms, where most labelled faults are pitch, electrical or mechanical, it caught 2 of 6 (Farm B) and 2 of 27 (Farm C) on their own component; those faults need their own signals.
- **The ENGIE abnormal period** used in claim 1 is defined in this project from the sibling turbines; the ENGIE release has no fault label.

**Where it sits.** Normal-behaviour monitoring exists commercially and in research (Vervlimmeren et al., *Wind Energy Science*, 2025). This project adds the explicit load / weather / excess split, the monthly retraining with rules against learning a fault, and a check against an operator's own records, on public data anyone can rerun.

<!-- CONCLUSION:END -->

## How it works

1. **Learn the healthy turbine.** For each temperature sensor, learn its normal reading at the turbine's power and the outside temperature, from at least a month of healthy data.
2. **Freeze it for the month, check every day.** Predict today's healthy value from today's power and weather. A day more than 2 K hotter is a hot day; a run of hot days is one call-out.
3. **Never learn a fault.** Flagged days are never used for training; after maintenance the model waits for a new healthy month; neighbouring turbines in the same wind give a second opinion.
4. **Explain the alarm.** Split the reading into typical temperature, load, weather and the excess heat nothing explains.

The model is a learned lookup table (healthy temperature by power band and outside-temperature band): interpretable, retrained monthly, and about as accurate as LightGBM in this project's comparison.

## Try it

```bash
make setup       # Python 3.11 venv + dependencies
make data        # download and verify the pinned datasets (19 GB, once)
make pipeline    # every result, in order; ends by regenerating the docs
make dashboard   # http://localhost:8501
make test        # the test suite
```

The dashboard opens on **Demo**: one month, two neighbouring turbines in the same wind, the healthy prediction against the sensor, with the hot days marked. **Performance** has the daily check over the whole record against the plain 80 °C limit.

## Read more

| | |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | the full technical report: every measurement, the method, the benchmark audit, the negative results |
| [docs/PROGRESS.md](docs/PROGRESS.md) | the research log, step by step, including what did not work |
| [data/README.md](data/README.md) | datasets, licences, checksums |

## Layout

```
app.py                Streamlit dashboard
src/windmill/         all reusable logic: loaders, healthy model, daily check, triage, scoring
scripts/pipeline/     one script per result, in order (the Makefile runs them)
results/              machine-readable outputs; every number in the docs comes from here
tests/                unit tests, including the repository's own rules
```

## Tests

362 tests, including a check that the CARE score matches the benchmark authors' implementation, and a rule that every number in the generated sections of these documents is backed by a results file.

## Licence

Code: MIT. Data: see `data/README.md`; each dataset keeps its own licence.
