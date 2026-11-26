# Windmill: catching slow turbine degradation that fixed alarm limits miss

**A healthy-turbine model, retrained every month, that flags a component running hot for its conditions before it reaches the alarm limit, and tells the crew whether an alarm is load, weather or a part going wrong.**

Built on public data: ENGIE La Haute Borne SCADA (France), EDP's wind farm with its alarm log and failure logbook (Portugal), CARE to Compare, and Copernicus ERA5 weather. Every number in the generated sections below comes from `results/*.json` and is checked by the test suite.

**Live dashboard:** [drift-resilient-wind-monitoring.streamlit.app](https://drift-resilient-wind-monitoring.streamlit.app)

## The problem

A wind turbine protects itself with alarms. Its control system watches many signals and raises an alarm when one crosses a fixed setting. In EDP's own event log (four turbines, 2016–2017) these include:

- **Component temperatures**, each with a fixed setting:
    - generator: 145 °C
    - transformer windings: 125 °C
    - generator bearings: alarms logged at 154 °C and above
    - generator slip ring: 64–70 °C
    - high-voltage transformer: one fixed setting in every entry
    - gearbox: the control system switches the gear-oil cooler and heater on the gearbox temperature. For ENGIE's gearbox bearing, this project uses 80 °C held for an hour as the fixed-limit baseline, because ENGIE publishes no alarm settings.
- **Pitch:** accumulator faults on the blade pitch axes.
- **Yaw:** the nacelle turning faster than allowed.
- **Safety and grid:** an open emergency circuit, power cut-in and cut-out.

CARE's offshore fault records add converter, cooling-water, hydraulic, communication and pitch-electronics faults.

A fixed setting only fires once the signal crosses it. A problem that develops below the setting goes undetected, so the part keeps degrading until it fails, and a small repair becomes a long, unplanned shutdown with the turbine producing no power for the grid. And when the alarm does fire, "over the setting" does not say whether the cause is the load, the weather or a part wearing out.

**This project focuses on component temperatures**, above all the **gearbox bearing temperature** on ENGIE's turbines, where a slowly developing problem shows as heat long before it crosses a limit such as 80 °C. On EDP's farm the same method is applied to the transformer, generator and gearbox temperature alarms. A learned model can see that heat early, but only if it stays current: a model trained once and left to age mistakes normal drift for faults.

<!-- CONCLUSION:START -->

## What the project shows

### 1. A model sees slow degradation that a fixed limit misses

The model ran over ENGIE's whole record, from 31 December 2012 to 12 January 2018, on all four turbines, and checked 1,313 days on R80711 and 3,204 on its three healthy siblings (8.8 turbine-years); the other days had too little running data or fell in the wait for a new healthy month after a long stop. R80711's gearbox bearing ran abnormally hot from 15 August 2017 to the end of the record (12 January 2018). The fixed limit is part of the model's rule: a day on which the bearing held 80 °C for an hour is always a hot day, and so is a day more than 2 K above the healthy prediction on average. In that period:

| 15 August 2017 to 12 January 2018 | days |
|---|---|
| flagged by the 80 °C limit | 42 |
| **flagged by the model** | **63 (+50 %)** |
| of them below the limit | 21 |
| ten weeks before: model hot days / limit days | 18 / 9 |

On the days below the limit the bearing averaged 72.6 °C, 2.9 K hotter than a healthy bearing at that day's power and weather: too cool for the limit, too hot for the conditions. Over the 8.8 turbine-years checked on the three healthy siblings the model raised 0.14 false call-outs per turbine-year. The limit is part of the rule, so nothing it catches is lost; the model adds what the limit cannot see.

### 2. The model must be retrained every month

Turbines drift: each ENGIE turbine's temperatures move by several kelvin between years. A reference frozen in 2014 alarmed on **26.37 %** of readings; one refreshed on the last 30 days, on **0.57 %** (46.5× fewer). A stale model buries the real signal in false alarms. The monthly model never learns from flagged days, so a fault cannot become the new normal.

The same holds for a model frozen after a confirmed fault. R80711's, trained in summer, judged the winter, and its 2 K daily average alone missed 6 days the limit caught: it had no healthy reference for the high power of windy winter days, when the bearing is hottest. Each frozen month is now judged with the turbine's own healthy model for the same month a year earlier: R80711's hot days from September to January rose from 43 to 57 (+33 %), 5 of the 6 missed limit days were caught, and the three healthy turbines judged the same way over the same months had 0 false hot days in 271.

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
2. **Freeze it for the month, check every day.** Predict today's healthy value from today's power and weather. A day more than 2 K hotter on average is a hot day, and so is any day the sensor held the fixed 80 °C limit for an hour: the limit is part of the rule, so the model flags every day the limit does. A run of hot days is one call-out.
3. **Never learn a fault.** Flagged days are never used for training; after maintenance the model waits for a new healthy month; neighbouring turbines in the same wind give a second opinion. Once a fault is confirmed the model stays frozen, and each later month is judged with the turbine's own healthy model for the same month a year earlier, so a summer model never judges winter.
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

The dashboard ([live version](https://drift-resilient-wind-monitoring.streamlit.app)) has four pages:

| Page | What it shows |
|---|---|
| **Demo** | One month (15 August to 14 September 2017), two neighbouring ENGIE turbines in the same wind: healthy R80721 and R80711, whose gearbox bearing ran abnormally hot. It opens on the sensor and the fixed 80 °C limit; two boxes add the model's daily prediction (with its ±2 K band, the hot days and the model's call-outs) and the forecast up to 7 days ahead. |
| **Explore** | Any month, any of the nine temperature sensors, the four turbines: a day-by-day strip (within tolerance, hot, too little running, maintenance hold) with the model's call-outs, a status card per turbine, the prediction against the sensor for one turbine, and the comparison with its siblings at the same moment. |
| **Performance** | R80711's abnormal period: every 80 °C limit day flagged, and the days the model adds below the limit. Why a frozen summer model's daily average missed limit days in winter, and last year's same-month model as the fix, with a month chart that opens on the sensor and adds each model with a tick box. False call-outs over the whole record, and R80711's timeline from May 2017. |
| **Evidence & method** | The research behind the pages before it: a live 7-day weather forecast and power prediction (Open-Meteo), the model comparison on the CARE benchmark, the engineering and data forensics (upstream data errors found, how the results are reproduced, the sibling and multi-sensor comparisons), forecast accuracy by lead time, and a data explorer that runs the detector on ENGIE, a CARE event or an uploaded turbine CSV. |

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

516 tests, including a check that the CARE score matches the benchmark authors' implementation, and a rule that every number in the generated sections of these documents is backed by a results file.

## Licence

Code: MIT. Data: see `data/README.md`; each dataset keeps its own licence.
