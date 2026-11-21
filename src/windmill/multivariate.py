"""Option 1: Deterministic Multivariate Bins.

A multivariate, deterministic normal-behaviour model for wind turbine SCADA.

Why this exists
---------------
The CARE to Compare faults are thermal: generator bearing damage, gearbox failure,
transformer over-temperature, hydraulic group, rotor bearing damage. A power-curve
residual cannot see any of them until the turbine derates -- by which point the
dataset has already flagged the row as status 4 (down due to fault) and excluded
it from the metric. The instrument has to be thermal.

Method
------
For each dataset (one turbine-year) and each thermal channel c:

  1. **Reference.** The median of c per load bin (active power in 5 %-of-rated
     steps), estimated on the dataset's own TRAINING portion, restricted to rows
     the dataset marks as normal operation. Load is the primary conditioning
     variable because component temperature is driven by load.

  2. **Ambient correction.** Component temperature is driven by ambient temperature
     as well as load. A load-only reference therefore reports a hot day as dozens
     of channels all running warm -- the *same* failure mode this project exists
     to remove, one level down. So the load-residual is corrected for ambient:
     the median residual per ambient bin is removed. This is the multivariate
     counterpart of the ERA5 conditioning used on the ENGIE arm, and it uses the
     site's own ambient sensor rather than a reanalysis.

  3. **Spread.** A robust spread (1.4826 x MAD) of the corrected training
     residual per load bin, falling back to the standard deviation and then to
     the channel's global spread, so no channel is left without a scale.

  4. **Normalised deviation.** z = corrected residual / spread. ONE-SIDED:
     overheating is the failure mode, so only positive deviations count. A channel
     running *colder* than its reference is not evidence of a fault.

  5. **Combination.** Two combiners, both reported:
       - `max`: the largest one-sided z over channels (most sensitive, noisiest);
       - `l2` : the Euclidean norm of the positive deviations (aggregates weak
                evidence across many channels, the right shape for a fault that
                shows as a small excess in several sensors at once).

  6. **Threshold, cross-fitted.** The reference is fitted on one half of the
     training rows and applied to the other, and vice versa, so the score
     distribution the threshold is taken from is OUT OF SAMPLE. A threshold
     computed on the rows the reference was fitted to understates the spread and
     yields a threshold that is far too strict on the test rows.

Everything here is deterministic: no random seeds, no learned parameters, and a
row's score depends only on that row and the reference built from training data.
That is what makes an alarm defensible to a duty engineer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .care import LOAD_BIN

AMBIENT_BIN = 2.0   # degrees Celsius per ambient bin

#: Floor on a channel's robust spread, in the channel's own unit. A temperature
#: sensor that is near-constant over a load bin has a spread of ~0, which makes
#: the normalised deviation meaningless (z in the tens) and the threshold
#: meaningless with it.
ABSOLUTE_SPREAD_FLOOR = 0.05


@dataclass
class ChannelReference:
    """One channel's load-conditioned, ambient-corrected reference."""

    name: str
    medians: pd.Series            # load bin -> median value
    ambient_offsets: pd.Series    # ambient bin -> median residual offset
    spreads: dict[int, float]     # load bin -> robust spread
    fallback_spread: float
    n_train: int = 0
    coverage: float = 0.0

    def apply(self, load: np.ndarray, ambient: np.ndarray,
              values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (corrected residual, normalised deviation) for the given rows."""
        bins = np.floor(load / LOAD_BIN).astype("int64")
        ref = self.medians.reindex(bins).to_numpy()
        residual = values - ref

        abins = np.floor(ambient / AMBIENT_BIN).astype("int64")
        offset = self.ambient_offsets.reindex(abins).to_numpy()
        offset = np.where(np.isfinite(offset), offset, 0.0)
        corrected = residual - offset

        spread = np.array([self.spreads.get(int(b), np.nan) for b in bins], dtype="float64")
        fallback = np.where(np.isfinite(spread) & (spread > 0), spread, self.fallback_spread)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = corrected / fallback
        return corrected, z


def _ambient_offsets(resid: np.ndarray, ambient: np.ndarray,
                     min_count: int = 25) -> pd.Series:
    """Median residual per ambient bin (the ambient effect to remove)."""
    abins = np.floor(ambient / AMBIENT_BIN).astype("int64")
    df = pd.DataFrame({"a": abins, "r": resid})
    counts = df.groupby("a")["r"].size()
    med = df.groupby("a")["r"].median()
    return med.loc[counts[counts >= min_count].index]


def fit_channel_reference(
    train: pd.DataFrame,
    channel: str,
    power_col: str,
    ambient_col: str | None = None,
    min_bin_count: int = 15,
) -> ChannelReference | None:
    """Build the load-conditioned, ambient-corrected reference for one channel."""
    cols = [channel, power_col] + ([ambient_col] if ambient_col else [])
    t = train.dropna(subset=cols)
    t = t[t[power_col] > 0.02]          # exclude standstill: no load, no reference
    if len(t) < 200:
        return None

    load = t[power_col].to_numpy(dtype="float64")
    values = t[channel].to_numpy(dtype="float64")
    ambient = (t[ambient_col].to_numpy(dtype="float64")
               if ambient_col else np.zeros(len(t)))
    bins = np.floor(load / LOAD_BIN).astype("int64")

    df = pd.DataFrame({"bin": bins, "v": values, "a": ambient})
    counts = df.groupby("bin")["v"].size()
    medians = df.groupby("bin")["v"].median()
    keep = counts[counts >= min_bin_count].index
    medians = medians.loc[keep]
    if len(medians) < 3:
        return None

    df["ref"] = df["bin"].map(medians).to_numpy()
    df["r"] = df["v"] - df["ref"]
    if ambient_col:
        offsets = _ambient_offsets(df["r"].to_numpy(), df["a"].to_numpy())
        df["r"] = df["r"] - df["a"].map(offsets).fillna(0.0).to_numpy()
    else:
        offsets = pd.Series(dtype="float64")

    # A scale floor. Some temperature sensors are near-constant over a load bin,
    # which makes the robust spread ~0 and the normalised deviation explode
    # (z values in the tens). Such a channel carries no information, so its spread
    # is floored at the absolute floor and at 1 % of the channel's own IQR.
    iqr = float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) else 0.0
    scale_floor = max(ABSOLUTE_SPREAD_FLOOR, 0.01 * iqr)

    spreads: dict[int, float] = {}
    for b, g in df.groupby("bin"):
        r = g["r"].to_numpy()
        if len(r) < min_bin_count:
            continue
        m = float(np.median(r))
        mad = float(np.median(np.abs(r - m)))
        sd = float(np.std(r))
        sp = 1.4826 * mad if mad > 0 else sd
        if np.isfinite(sp) and sp > 0:
            spreads[int(b)] = float(max(sp, scale_floor))

    resid_all = df["r"].to_numpy()
    med = float(np.median(resid_all))
    mad = float(np.median(np.abs(resid_all - med)))
    fallback = 1.4826 * mad if mad > 0 else float(np.std(resid_all))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = scale_floor
    fallback = float(max(fallback, scale_floor))

    coverage = float(medians.reindex(np.unique(bins)).notna().mean())
    return ChannelReference(
        name=channel,
        medians=medians.astype("float64"),
        ambient_offsets=offsets.astype("float64"),
        spreads=spreads,
        fallback_spread=float(fallback),
        n_train=int(len(t)),
        coverage=coverage,
    )


def rolling_reference_scores(
    df: pd.DataFrame,
    channel: str,
    power_col: str,
    ambient_col: str | None = None,
    window_days: int = 30,
    n_load_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling local reference: the channel's own recent history, per load bin.

    A frozen reference goes stale. On the ENGIE site each turbine's thermal
    channels move by 5-9 K between years (measured in
    results/per_turbine_curve.json), so a reference fitted on 2014 and applied
    to 2017 leaves a systematic offset that reads as overheating on every
    channel at once -- which is exactly the failure mode this project exists to
    remove.

    The operational answer is a reference that is re-estimated continuously. This
    computes, for every row, the mean of the channel over the preceding
    `window_days` days at the SAME load bin, so the reference tracks the
    turbine's own slow drift while still holding load and ambient fixed.

    Returns (corrected residual, normalised deviation), both over `df`'s index.
    """
    n = len(df)
    if n < 100:
        return np.zeros(n), np.zeros(n)

    load = df[power_col].to_numpy(dtype="float64")
    values = df[channel].to_numpy(dtype="float64")
    ambient = (df[ambient_col].to_numpy(dtype="float64")
               if ambient_col else np.zeros(n))
    ok = np.isfinite(load) & np.isfinite(values) & (load > 0.02)
    if ok.sum() < 100:
        return np.zeros(n), np.zeros(n)

    # Coarse load bins: 10 bins keeps enough rows per bin for a stable mean
    # while the trailing window still holds thousands of samples.
    lb = np.clip((load * n_load_bins).astype("int64"), 0, n_load_bins - 1)
    lb[~ok] = -1

    ts = pd.to_datetime(df["ts"]).to_numpy(dtype="datetime64[s]")
    tsec = (ts - ts[0]).astype("float64")

    resid = np.zeros(n)
    spread = np.zeros(n)
    for b in range(n_load_bins):
        m = lb == b
        if m.sum() < 50:
            continue
        idx = np.flatnonzero(m)
        # Rows of this bin, in time order, with their timestamps.
        bt = tsec[idx]
        bv = values[idx]
        # Trailing window: cumulative sums over the bin's own row sequence.
        cs = np.concatenate([[0.0], np.cumsum(bv)])
        cs2 = np.concatenate([[0.0], np.cumsum(bv ** 2)])
        # For each row, the range of bin rows within the trailing window.
        lo = np.searchsorted(bt, bt - window_days * 86400.0, side="left")
        hi = np.arange(len(idx))
        cnt = np.maximum(hi - lo + 1, 1)
        mean = (cs[hi + 1] - cs[lo]) / cnt
        var = np.maximum((cs2[hi + 1] - cs2[lo]) / cnt - mean ** 2, 0.0)
        sd = np.sqrt(var)
        resid[idx] = bv - mean
        spread[idx] = sd

    # Floor the spread: a near-constant channel must not produce huge z.
    resid_all = resid[ok]
    med = float(np.median(resid_all)) if len(resid_all) else 0.0
    mad = float(np.median(np.abs(resid_all - med))) if len(resid_all) else 0.0
    global_spread = 1.4826 * mad if mad > 0 else float(np.std(resid_all) or 1.0)
    scale_floor = max(ABSOLUTE_SPREAD_FLOOR, 0.01 * global_spread)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, scale_floor)
    spread = np.maximum(spread, scale_floor)

    # Ambient correction: the trailing window may span a temperature change.
    if ambient_col:
        abins = np.floor(ambient / AMBIENT_BIN).astype("int64")
        df_a = pd.DataFrame({"a": abins, "r": resid})
        counts = df_a.groupby("a")["r"].size()
        med_a = df_a.groupby("a")["r"].median()
        keep = counts[counts >= 25].index
        med_a = med_a.loc[keep]
        if len(med_a):
            resid = resid - med_a.reindex(abins).fillna(0.0).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        z = resid / spread
    return np.where(ok, resid, 0.0), np.where(ok, z, 0.0)


def detect_rolling(
    df: pd.DataFrame,
    channels: list[str],
    power_col: str,
    ambient_col: str | None = None,
    window_days: int = 30,
    z_threshold: float = 4.0,
    max_channels: int = 24,
) -> MultivariateResult:
    """Multivariate detection with a rolling local reference.

    The reference for every row comes from that row's own recent history, so the
    detector does not go stale as the turbine or the season drifts.

    LIMITATION, stated plainly: a rolling reference absorbs a developing fault.
    As the fault persists, the trailing window fills with faulted data and the
    reference moves to meet it. So this is an EARLY-warning instrument -- sharp
    in the first hours of a developing fault, progressively blind to one that has
    been running for weeks. A real deployment pairs it with a frozen reference
    that is re-validated on a schedule; neither alone is sufficient.
    """
    use = [c for c in channels if c in df.columns][:max_channels]
    z_by_channel: dict[str, np.ndarray] = {}
    for c in use:
        _, z = rolling_reference_scores(df, c, power_col, ambient_col, window_days)
        if np.isfinite(z).any() and np.abs(z).max() > 0:
            z_by_channel[c] = z
    if not z_by_channel:
        z = pd.Series(0.0, index=df.index)
        empty = pd.Series(False, index=df.index)
        return MultivariateResult(z, z.copy(), empty, empty.copy(), z_threshold,
                                  z_threshold, [])
    zmax, l2 = combine_scores(z_by_channel, df.index)
    return MultivariateResult(
        zmax=zmax, l2=l2,
        flags_max=zmax > z_threshold, flags_l2=l2 > z_threshold,
        threshold_max=z_threshold, threshold_l2=z_threshold,
        channels_used=list(z_by_channel.keys()),
    )


@dataclass
class MultivariateResult:
    """Per-row combined scores and the flags derived from them."""

    zmax: pd.Series
    l2: pd.Series
    flags_max: pd.Series
    flags_l2: pd.Series
    threshold_max: float
    threshold_l2: float
    channels_used: list[str] = field(default_factory=list)
    train_scores: dict[str, pd.Series] = field(default_factory=dict)
    quantile_thresholds: dict[str, float] = field(default_factory=dict)


def combine_scores(z_by_channel: dict[str, np.ndarray], index) -> tuple[pd.Series, pd.Series]:
    """Max one-sided z and L2 norm of positive deviations across channels."""
    if not z_by_channel:
        z = pd.Series(0.0, index=index)
        return z, z.copy()
    Z = pd.DataFrame(z_by_channel, index=index)
    Z = Z.replace([np.inf, -np.inf], np.nan)
    pos = Z.clip(lower=0.0)
    zmax = Z.max(axis=1, skipna=True).fillna(0.0)
    l2 = np.sqrt((pos ** 2).sum(axis=1, skipna=True)).fillna(0.0)
    return zmax, l2


def _score_rows(
    train: pd.DataFrame, rows: pd.DataFrame, channels: list[str],
    power_col: str, ambient_col: str | None, refs: dict[str, ChannelReference]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply the references to `rows`, returning z for test and train rows."""
    load = rows[power_col].to_numpy(dtype="float64")
    ambient = (rows[ambient_col].to_numpy(dtype="float64")
               if ambient_col else np.zeros(len(rows)))
    z_out: dict[str, np.ndarray] = {}
    for c, ref in refs.items():
        v = rows[c].to_numpy(dtype="float64")
        _, z = ref.apply(load, ambient, v)
        z_out[c] = z
    return z_out, {}


def detect(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channels: list[str],
    power_col: str,
    ambient_col: str | None = None,
    z_threshold: float | None = None,
    quantile: float = 0.999,
    max_channels: int = 24,
    cross_fit: bool = True,
) -> MultivariateResult:
    """Fit the references on `train`, score `test`.

    Two ways to set the threshold:

    * `z_threshold` given: a FIXED z threshold. Interpretable ("a component four
      robust deviations above its load- and ambient-conditioned reference"), but
      it is a free parameter that has to be chosen by evidence.

    * `z_threshold=None`: the threshold is the `quantile` of the TRAINING score
      distribution, computed from CROSS-FITTED scores (the reference is fitted on
      one half of the training rows and applied to the other, and vice versa).
      This is the protocol the CARE paper's own baseline uses -- the threshold is
      calibrated on training data, never on the events being scored -- and
      cross-fitting is what makes it honest, because a threshold taken from the
      rows the reference was fitted to understates the spread.

    Cross-fitting also prevents a subtler leak: the training rows are the
    reference's own rows, so their residuals are smaller than the test rows' and
    any threshold derived from them is too strict.
    """
    use = [c for c in channels if c in train.columns and c in test.columns][:max_channels]

    # --- fit the references on the full training portion --------------------
    refs: dict[str, ChannelReference] = {}
    for c in use:
        ref = fit_channel_reference(train, c, power_col, ambient_col)
        if ref is not None:
            refs[c] = ref
    if not refs:
        z = pd.Series(0.0, index=test.index)
        empty = pd.Series(False, index=test.index)
        return MultivariateResult(z, z.copy(), empty, empty.copy(), z_threshold,
                                  z_threshold, [])

    z_test, _ = _score_rows(train, test, use, power_col, ambient_col, refs)
    zmax_te, l2_te = combine_scores(z_test, test.index)

    # --- reference thresholds, from out-of-fold training scores --------------
    if cross_fit and len(train) >= 400:
        half = len(train) // 2
        folds = [train.iloc[:half], train.iloc[half:]]
        zmax_parts, l2_parts = [], []
        for i in (0, 1):
            fit_part = folds[1 - i]
            held = folds[i]
            fold_refs = {}
            for c in refs:
                r2 = fit_channel_reference(fit_part, c, power_col, ambient_col)
                if r2 is not None:
                    fold_refs[c] = r2
            if not fold_refs:
                continue
            zf, _ = _score_rows(fit_part, held, use, power_col, ambient_col, fold_refs)
            Z = pd.DataFrame(zf)
            pos = Z.clip(lower=0.0)
            zmax_parts.append(Z.max(axis=1, skipna=True).fillna(0.0))
            l2_parts.append(np.sqrt((pos ** 2).sum(axis=1, skipna=True)).fillna(0.0))
        if zmax_parts:
            zmax_tr = pd.concat(zmax_parts).sort_index()
            l2_tr = pd.concat(l2_parts).sort_index()
        else:
            z_tr, _ = _score_rows(train, train, use, power_col, ambient_col, refs)
            zmax_tr, l2_tr = combine_scores(z_tr, train.index)
    else:
        z_tr, _ = _score_rows(train, train, use, power_col, ambient_col, refs)
        zmax_tr, l2_tr = combine_scores(z_tr, train.index)

    q_thresholds = {
        "zmax": float(zmax_tr.quantile(quantile)),
        "l2": float(l2_tr.quantile(quantile)),
    }

    if z_threshold is None:
        # Calibrate on the training data only -- the same protocol the CARE
        # paper's baseline uses, and the reason the threshold never sees an
        # event it is later scored on.
        thr_max = q_thresholds["zmax"]
        thr_l2 = q_thresholds["l2"]
    else:
        thr_max = thr_l2 = float(z_threshold)

    flags_max = zmax_te > thr_max
    flags_l2 = l2_te > thr_l2

    return MultivariateResult(
        zmax=zmax_te, l2=l2_te, flags_max=flags_max, flags_l2=flags_l2,
        threshold_max=thr_max, threshold_l2=thr_l2,
        channels_used=list(refs.keys()),
        train_scores={"zmax": zmax_tr, "l2": l2_tr},
        quantile_thresholds=q_thresholds,
    )
