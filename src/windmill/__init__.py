"""Windmill: drift-resilient wind turbine condition monitoring.

Public, pinned data only. Every module is import-safe without side effects.
"""

__version__ = "0.1.0"

SITE_LAT = 48.4569
SITE_LON = 5.5847
SITE_ELEV_M = 411
RATED_POWER_KW = 2050.0
HUB_HEIGHT_M = 80

TURBINES = ("R80711", "R80721", "R80736", "R80790")

# The turbine with the documented gearbox bearing failure (context only -- the
# cleaned release does not contain the fault peak).
DOCUMENTED_FAILURE_TURBINE = "R80721"
DOCUMENTED_FAILURE_UTC = "2015-01-02 08:50"
