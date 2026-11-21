"""The learned-model arm must be reproducible.

A portfolio project whose central claim is "these numbers can be re-derived"
cannot have an arm whose score moves when you re-run it. LightGBM's
`subsample` and `colsample_bytree` draw randomness, so the seeds are pinned
explicitly in LGBM_PARAMS and this test proves the fits are identical.

Skipped if LightGBM or the CARE data are not present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from windmill import care  # noqa: E402

@pytest.fixture(scope="module")
def lgb():
    """The LightGBM module, or a skip.

    A fixture, not a module-level `pytest.importorskip`. The module-level
    form skips during COLLECTION, so on a machine without LightGBM pytest
    never collected these three tests -- and the README's generated test
    count, which is the number pytest collects, changed with the Python
    version and the set of optional packages installed.

    Skipping at run time keeps the collected count identical everywhere, so
    the documentation reproducibility check holds on every CI job.
    """
    return pytest.importorskip("lightgbm", reason="lightgbm not installed")

CARE_ROOT = care.care_root(ROOT / "data")
pytestmark = pytest.mark.skipif(
    not CARE_ROOT.exists(), reason="CARE dataset not downloaded"
)


def _load_params():
    """The LightGBM parameters live in src/windmill/nbm.py, shared by both arms."""
    from windmill import nbm
    return nbm.LGBM_PARAMS


def test_lightgbm_params_pin_their_seeds():
    """The seeds must be explicit, not left to an implementation default."""
    params = _load_params()
    assert params.get("random_state") is not None
    assert params.get("seed") is not None


def test_two_fits_on_the_same_data_are_identical(lgb):
    """The same data and params must give the same predictions, bit for bit."""
    params = _load_params()
    farm_dir = CARE_ROOT / "Wind Farm C"
    path = sorted((farm_dir / "datasets").glob("*.csv"))[0]
    df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
    train = df[df["train_test"] == "train"]
    train = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]

    columns = care.dataset_columns(path)
    pcol, thermal, wind, _ambient = care.feature_rows(farm_dir, columns)
    feats = [pcol] + [w for w in wind if w in df.columns]
    channel = thermal[1]
    t = train.dropna(subset=feats + [channel]).sample(n=3000, random_state=0).sort_index()
    X = t[feats]
    y = t[channel].to_numpy()

    preds = []
    for _ in range(2):
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        preds.append(model.predict(X))

    assert np.array_equal(preds[0], preds[1])
    assert float(np.abs(preds[0] - preds[1]).max()) == 0.0


def test_row_subsampling_is_seeded():
    """The row subsample used to cap training rows must also be seeded.

    `train.sample(...)` without a seed would draw a different subset each run,
    which would break reproducibility even with fixed model seeds.
    """
    src = (ROOT / "scripts" / "pipeline" / "07_lightgbm_nbm.py").read_text()
    assert "random_state=0" in src
