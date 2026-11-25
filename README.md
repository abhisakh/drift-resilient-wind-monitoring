# Windmill: wind-turbine temperature alarms you can trust

**A healthy-turbine model that tells a maintenance crew whether a temperature alarm is load, weather, or a part going wrong, and stays honest as the turbines age.**

Built on public SCADA data (ENGIE, EDP, CARE to Compare) and Copernicus ERA5 weather. Every number below is generated from `results/*.json` and checked by the test suite.

## The problem

Wind turbines raise temperature alarms on fixed limits: 80 °C here, 145 °C there. A fixed limit cannot tell a hot, high-power summer day from a bearing that is failing. So operators either send crews for alarms that turn out to be nothing, or learn to ignore alarms, until a part fails and the turbine stands still for days in good wind.

## What it does

1. **Learn the healthy turbine.** For each temperature sensor, learn its normal reading for the turbine's power output and the outside temperature, from at least a month of healthy data only.
2. **Freeze it for the month, compare every day.** Predict today's healthy value from today's power and weather; a sensor running hotter than its learned tolerance is a hot day, and a run of hot days is one call-out.
3. **Explain every alarm.** Split the reading into typical temperature + load + weather + **excess heat nothing explains**. Send a crew for excess heat; hold back for load or weather.
4. **Guard against learning a fault.** Never train on flagged days; after maintenance, wait for a new healthy month; neighbouring turbines in the same wind give a second opinion.

The model is a learned lookup table (healthy temperature by power band and outside-temperature band): interpretable, retrained monthly, and about as accurate as LightGBM in this project's comparison (median error 2.10 K against 2.02 K).

<!-- CONCLUSION:START -->

## Results at a glance

| question | answer |
|---|---|
| Does keeping the model current cut false alarms? | Yes. On ENGIE, a reference frozen in 2014 alarmed on **26.37 %** of readings, a rolling one on **0.57 %** (46.5×). The daily check at a 2 K tolerance raised 0.14 false call-outs per healthy turbine-year. |
| Is a real overheat caught? | R80711's gearbox bearing peaked at **86.64 °C** in 2017 while its three siblings in the same wind stayed below 80 °C. The first call-out came on 2017-08-15, the first day of the anomaly span; the fixed 80 °C limit fired that day too. |
| Can an alarm be explained? | On EDP's farm, **34 of 41** temperature alarms were real excess heat and 5 were load; on ENGIE's 80 °C limit, 26 of 27. Weather added 1.5 K at those alarms. The weather term is still needed to say so: without it the model's error grows by 16-51 % on EDP. |
| Is a trusted alarm worth acting on? | T06: real heat confirmed **54 days** before "Generator replaced", which then cost **113.2 h** of standstill in wind (normally 7.7 h); T07: real heat confirmed **51 days** before "High temperature transformer. Transformer refrigeration repaired", which then cost **91.8 h** of standstill in wind (normally 11.2 h). |
| Does it work offshore? | Partly. Faults caught on their own component: 2 of 6 on CARE Farm B, 2 of 27 on Farm C, where 8 faults have no temperature sensor and 11 came right after logged maintenance. |
| Can the data be trusted? | CARE's Farm A matches EDP's original data reading for reading, and each fault label lines up with a failure in EDP's logbook. |

**What it does not show.**

- It does not halve the operators' false alarms: most of their fixed-limit temperature alarms were real heat. The saving is against a model that is not kept current, and in the alarms that can be held back.
- It is not a general fault detector: pitch, electrical and mechanical faults need their own signals.
- The savings are not measured: the data has no costs, and a lead time shows when a repair could have been planned, not that it would have prevented the failure.
- A few noisy sensors raise most false call-outs offshore (6.66 and 6.24 per component-year on Farms B and C). Telling a sensor problem from a component problem is the next step.

**Where it sits.** Normal-behaviour monitoring with smart alarms exists commercially and in research (Vervlimmeren et al., *Wind Energy Science*, 2025). This project adds an explicit load / weather / excess split of every alarm, a model kept current with rules against learning a fault, and a check against an operator's own alarm log and failure logbook, on public data anyone can rerun.

<!-- CONCLUSION:END -->

## Try it

```bash
make setup       # Python 3.11 venv + dependencies
make data        # download and verify the pinned datasets (19 GB, once)
make pipeline    # every result, in order; ends by regenerating the docs
make dashboard   # http://localhost:8501
make test        # the test suite
```

The dashboard opens on **Demo**: one month, two neighbouring turbines, same weather. Each day shows the forecast-based prediction, the prediction from the day's actual weather, and what the sensor read. Then **Explore** (any month, sensor, turbine), **Performance** (false call-outs a year, forecast error, second opinions) and **Evidence & method**.

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

505 tests, including a check that the CARE score matches the benchmark authors' implementation, and a rule that every number in the generated sections of these documents is backed by a results file.

## Licence

Code: MIT. Data: see `data/README.md`; each dataset keeps its own licence.
