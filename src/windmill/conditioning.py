"""Power-curve construction, weather conditioning, and residuals.

Method
------
1. **Reference power curve** P_ref(v): the IEC 61400-12-1 "method of bins" -- median
   active power in 0.5 m/s wind-speed bins, fitted on a reference period.

2. **Climatology conditioning.** The physics: the turbine's power curve P = f(v) is a
   property of the machine and is stable year to year; what changes year to year is
   the *distribution* of v. So the expected power for a period is

       E[P] = sum over wind bins v_i of  f(v_i) * p_clim(v_i)

   where p_clim is the *climatological* wind-speed distribution for that period. A
   baseline that ignores this attributes the whole difference to the turbine.

3. **Residual** = measured P - f(v) at the observed wind. The residual is what
   separates "the weather was different" from "the turbine is degrading" -- and it
   must be computed against the *observed* wind, not against a climatological wind.

4. **Peer normalisation.** With 4 turbines on one site, a turbine's own reading can be
   compared to its peers at the same wind speed. This removes weather and ambient
   effects that are common to the site, and isolates what is specific to the machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BIN_WIDTH = 0.5  # m/s, per IEC 61400-12-1 method of bins


def wind_bin(v: pd.Series, width: float = BIN_WIDTH) -> pd.Series:
    """Bin index for wind speed.

    Returns a plain int64 (missing wind -> -1) rather than a nullable dtype: the
    nullable Int64/Float64 pandas 3.x dtypes propagate through groupby-map and
    break in-place assignment to float64 columns downstream.
    """
    vals = pd.to_numeric(v, errors="coerce").fillna(-1.0).to_numpy(dtype="float64")
    out = np.floor(vals / width).astype("int64")
    out[vals < 0] = -1
    return pd.Series(out, index=v.index)


def fit_power_curve(
    df: pd.DataFrame,
    wind_col: str = "Ws_avg",
    power_col: str = "P_avg",
    width: float = BIN_WIDTH,
    min_bin_count: int = 30,
) -> pd.Series:
    """Median power per wind bin (method of bins).

    Returns a Series indexed by bin index with median power (kW).
    """
    b = wind_bin(df[wind_col], width)
    grp = df.groupby(b)[power_col]
    med = grp.median()
    cnt = grp.size()
    med = med[cnt >= min_bin_count]
    return med.astype("float64")


def apply_power_curve(
    df: pd.DataFrame, curve: pd.Series, wind_col: str = "Ws_avg", width: float = BIN_WIDTH
) -> pd.Series:
    """Expected power for each row under the given curve (NaN outside its range)."""
    b = wind_bin(df[wind_col], width)
    return b.map(curve).astype("float64")


def wind_distribution(
    df: pd.DataFrame, wind_col: str = "Ws_avg", width: float = BIN_WIDTH
) -> pd.Series:
    """Normalised wind-speed distribution (fraction of samples per bin)."""
    b = wind_bin(df[wind_col], width)
    counts = b.value_counts(normalize=True)
    return counts.sort_index()


def climatology_weights(
    reference: pd.DataFrame,
    wind_col: str = "Ws_avg",
    width: float = BIN_WIDTH,
    min_bin_count: int = 30,
) -> pd.Series:
    """Climatological wind distribution used as the weighting for expected power.

    Uses only bins that the reference period actually populated, so the weighting is
    never dominated by tails with almost no data.
    """
    b = wind_bin(reference[wind_col], width)
    counts = b.value_counts()
    keep = counts[counts >= min_bin_count].index
    dist = wind_distribution(reference, wind_col, width)
    return dist[dist.index.isin(keep)]


def expected_power_from_climatology(
    curve: pd.Series,
    clim: pd.Series,
    rated_kw: float,
    width: float = BIN_WIDTH,
) -> float:
    """E[P] = sum_i f(v_i) * p_clim(v_i), in kW.

    The sum runs over the wind bins where BOTH the reference curve and the
    climatological distribution have support. It is an expectation of power, so
    it is bounded by the rated power naturally; it is clipped only as a guard
    against numerical noise, not as a normalisation.

    Note: the curve must be a *power* curve (kW per wind bin) and `clim` a
    normalised distribution. Passing anything else silently produces garbage.
    """
    common = curve.index.intersection(clim.index)
    if common.empty:
        return float("nan")
    e = float((curve.loc[common] * clim.loc[common]).sum())
    return float(min(e, rated_kw))


def static_baseline_bias(
    df: pd.DataFrame,
    reference_year: int,
    rated_kw: float,
    wind_col: str = "Ws_avg",
    power_col: str = "P_avg",
    width: float = BIN_WIDTH,
) -> pd.DataFrame:
    """What a single-period baseline reports per year.

    The curve is fitted on `reference_year` only, then applied to every year. The
    resulting mean residual is the false degradation signal the baseline invents.
    """
    ref = df[df["year"] == reference_year]
    curve = fit_power_curve(ref, wind_col, power_col, width)
    out = df.copy()
    out["p_ref"] = apply_power_curve(out, curve, wind_col, width)
    out["resid"] = out[power_col] - out["p_ref"]
    valid = out.dropna(subset=["p_ref"])
    g = valid.groupby("year").agg(
        n=("resid", "size"),
        bias_kw=("resid", "mean"),
        mae_kw=("resid", lambda s: s.abs().mean()),
        p_mean=("P_avg", "mean"),
    )
    g["cf_pct"] = g["p_mean"] / rated_kw * 100
    g["cf_err_pp"] = g["bias_kw"] / rated_kw * 100
    return g.round(4)


def peer_normalised_residual(
    df: pd.DataFrame,
    target: str,
    value_col: str = "Gb2t_avg",
    peers: tuple[str, ...] | None = None,
    wind_col: str = "Ws_avg",
    width: float = BIN_WIDTH,
    by_year: bool = True,
) -> pd.DataFrame:
    """Residual of one turbine against the median of its peers, at matched wind.

    residual = own value - median(peer values at the same wind bin)

    Matching on wind speed (and optionally year) removes the site-wide weather and
    ambient effects, so what remains is specific to the machine.
    """
    if peers is None:
        peers = tuple(t for t in df["turbine"].unique() if t != target)

    out = df.copy()
    out["bin"] = wind_bin(out[wind_col], width)
    keys = ["bin", "year"] if by_year else ["bin"]

    peer_df = out[out["turbine"].isin(peers)]
    tab = peer_df.groupby(keys)[value_col].median().rename("peer_med").reset_index()
    out = out.merge(tab, on=keys, how="left")
    out["resid"] = out[value_col] - out["peer_med"]
    return out


def residual_zscore(
    df: pd.DataFrame,
    resid_col: str = "resid",
    group_cols: tuple[str, ...] = ("turbine",),
    window: int | None = None,
    min_periods: int = 30,
) -> pd.Series:
    """Robust z-score of a residual within each group (rolling window optional)."""
    def _one(g: pd.Series) -> pd.Series:
        med = g.median()
        mad = (g - med).abs().median()
        if mad == 0 or pd.isna(mad):
            sd = g.std()
            if sd == 0 or pd.isna(sd):
                return pd.Series(np.nan, index=g.index)
            return (g - med) / sd
        return 0.6745 * (g - med) / mad

    if window is None:
        return df.groupby(list(group_cols))[resid_col].transform(_one)
    return (
        df.sort_values("ts")
        .groupby(list(group_cols))[resid_col]
        .transform(lambda s: s.rolling(window, min_periods=min_periods).apply(_one, raw=False))
    )
