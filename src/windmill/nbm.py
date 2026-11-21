"""The learned normal-behaviour model (NBM), shared by both pipeline scripts.

This module holds the reusable maths of the learned arm: the LightGBM
hyper-parameters, the robust spread, the per-channel residual model, and the
combination of per-channel z-scores into one anomaly score.

It lives in src/windmill/ rather than inside a script because TWO pipeline
scripts need it -- the CARE arm (scripts/pipeline/07_lightgbm_nbm.py) and the
ENGIE arm (scripts/pipeline/08c_engie_multivariate.py) -- and because
importing one script from another by path is not a dependency anyone can see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: LightGBM hyper-parameters for the per-channel normal-behaviour models.
#:
#: The seeds are pinned EXPLICITLY. `subsample` and `colsample_bytree` draw
#: randomness, and without a fixed seed the arm's CARE score would not be
#: reproducible -- fatal for a portfolio project whose whole claim is that its
#: numbers can be re-derived. Both the sklearn-level `random_state` and
#: LightGBM's own `seed` are pinned. Verified by
#: tests/test_lightgbm_determinism.py.
LGBM_PARAMS = {
    "n_estimators": 60,
    "learning_rate": 0.12,
    "num_leaves": 15,
    "min_child_samples": 80,
    "subsample": 0.6,
    "colsample_bytree": 0.7,
    "reg_lambda": 2.0,
    "verbose": -1,
    "n_jobs": 4,
    "random_state": 0,
    "seed": 0,
}

#: How many thermal channels to model per dataset. The full set is 24-76
#: depending on the farm; capping it keeps the fit tractable.
MAX_CHANNELS = 12

#: Cap on the training rows used to fit one channel model.
MAX_TRAIN_ROWS = 12_000


def l2norm_positive(z: pd.DataFrame) -> pd.Series:
    """L2 norm of the positive part of each row's z-scores.

    Negative deviations (a component running COLDER than expected) are clipped
    away: overheating is one-sided, and a symmetric norm would let a cold
    component mask a hot one.
    """
    if z.empty:
        return pd.Series(0.0, index=z.index)
    pos = z.clip(lower=0.0)
    return np.sqrt((pos ** 2).sum(axis=1, skipna=True)).fillna(0.0)


def max_z(z: pd.DataFrame) -> pd.Series:
    """The most deviant channel's z-score.

    This is the statistic the deterministic arm uses, so the learned arm is
    scored with exactly the same quantity and the comparison is like for like.
    """
    if z.empty:
        return pd.Series(0.0, index=z.index)
    return z.max(axis=1, skipna=True).fillna(0.0)


def robust_spread(residuals: np.ndarray) -> tuple[float, float] | None:
    """The median and the robust spread (1.4826 * MAD) of some residuals.

    Returns None when the spread is not usable -- a degenerate channel (a
    constant sensor, say) has no meaningful spread and must be dropped rather
    than allowed to produce infinite z-scores.
    """
    if residuals.size == 0:
        return None
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    spread = 1.4826 * mad if mad > 0 else float(np.std(residuals))
    if not np.isfinite(spread) or spread <= 0:
        return None
    return med, spread


def fit_channel_model(train: pd.DataFrame, test: pd.DataFrame,
                      channel: str, features: list[str],
                      max_train_rows: int = MAX_TRAIN_ROWS,
                      params: dict | None = None):
    """Fit one normal-behaviour model for one thermal channel.

    The channel is predicted from the operating point (power, wind speed,
    ambient temperature); the residual, scaled by its own robust spread, is the
    anomaly signal for that channel.

    Returns (train_z, test_z) as Series, or None when the channel cannot be
    modelled.
    """
    import lightgbm as lgb

    params = params or LGBM_PARAMS
    t = train.dropna(subset=[channel])
    if len(t) < 500:
        return None
    if len(t) > max_train_rows:
        t = t.sample(n=max_train_rows, random_state=0).sort_index()

    model = lgb.LGBMRegressor(**params)
    model.fit(t[features], t[channel].to_numpy())
    r_tr = t[channel].to_numpy() - model.predict(t[features])
    stats = robust_spread(r_tr)
    if stats is None:
        return None
    med, spread = stats

    te = test.dropna(subset=[channel])
    if te.empty:
        return None
    r_te = te[channel].to_numpy() - model.predict(te[features])

    train_z = pd.Series((r_tr - med) / spread, index=t.index)
    test_z = pd.Series((r_te - med) / spread, index=te.index)
    return train_z, test_z


def cross_fitted_scores(train: pd.DataFrame, test: pd.DataFrame,
                        channels: list[str], features: list[str],
                        max_train_rows: int = MAX_TRAIN_ROWS,
                        params: dict | None = None):
    """Fit the NBM and return (test_score, train_score) as max-z series.

    The TRAIN score comes from OUT-OF-FOLD predictions: the reference is
    fitted on one half of the training rows and applied to the other, and vice
    versa. Without this, the threshold would be derived from in-sample
    residuals, which are smaller than out-of-sample ones, and the threshold
    would be too strict -- the same leak the deterministic arm avoids.

    The returned statistic is max-z, matching the deterministic arm.
    """
    params = params or LGBM_PARAMS
    half = len(train) // 2
    folds = [train.iloc[:half], train.iloc[half:]] if len(train) >= 400 else []

    oof_parts: list[pd.Series] = []
    for i in (0, 1):
        fit_part, held = folds[1 - i], folds[i]
        if fit_part.empty or held.empty:
            continue
        fold_z = []
        for ch in channels:
            out = fit_channel_model(fit_part, held, ch, features,
                                    max_train_rows, params)
            if out is None:
                continue
            _, held_z = out
            fold_z.append(held_z)
        if fold_z:
            oof_parts.append(max_z(pd.concat(fold_z, axis=1)))

    test_z, train_z = [], []
    for ch in channels:
        out = fit_channel_model(train, test, ch, features, max_train_rows, params)
        if out is None:
            continue
        tr_z, te_z = out
        train_z.append(tr_z)
        test_z.append(te_z)

    if not test_z:
        return None, None
    test_score = max_z(pd.concat(test_z, axis=1))
    if oof_parts:
        train_score = pd.concat(oof_parts).sort_index()
    else:
        train_score = max_z(pd.concat(train_z, axis=1))
    return test_score, train_score
