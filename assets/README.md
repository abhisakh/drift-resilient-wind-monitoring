# assets/

## `sample_turbine.csv`

A real slice of the real ENGIE release, committed so that a reviewer on
Streamlit Community Cloud can run the detector without downloading 19 GB.

| | |
|---|---|
| source | ENGIE La Haute Borne, Mendeley Data **v2**, CC BY 4.0 |
| turbine | `R80711` (2 050 kW Senvion MM82, Meuse, France) |
| window | 1 August – 30 November 2017, 10-minute resolution |
| rows | ~13 200 |
| columns | `Date_time`, `P_avg`, `Ws_avg`, `Git_avg`, `Gost_avg`, `Gb2t_avg`, `Gb1t_avg`, `Db2t_avg`, `era5_temp_2m` |

It covers September 2017, so it contains the case study's thermal
signature: `Gb2t_avg` goes above 80 °C on R80711 while the ambient is
around 15 °C.

### Two things to know about it

**`era5_temp_2m` is reanalysis, not a measurement.** The ENGIE release ships
no ambient sensor, so the ambient temperature in the live ENGIE path comes
from ERA5 via open-meteo. The sample carries the same column so the
weather-conditioned path works on an uploaded file. It is labelled
`era5_temp_2m` for that reason.

**The other three turbines are not in it.** The file is one turbine, so the
control group has to be the reader's own — or, for the multi-turbine view,
the ENGIE source of the dashboard, which needs the raw release.

### Provenance

Regenerate it from the raw data with:

```bash
make snapshot
```

which runs `scripts/pipeline/11_demo.py --export` and writes both
`results/demo_r80711.json` and this file from the same detector and the same
frozen 2014 reference.

The byte-for-byte reproducibility of the committed file is asserted by
`tests/test_live_engine.py`, wherever the raw data is present.
