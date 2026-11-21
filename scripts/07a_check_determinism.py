"""The learned-model arm must be reproducible.

A portfolio project whose claim is "these numbers can be re-derived" cannot
have an arm whose score moves when you re-run it. LightGBM's `subsample` and
`colsample_bytree` draw randomness, so the seeds are pinned explicitly and this
test proves the fits are identical.

    python scripts/07a_check_determinism.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("lgbm", ROOT / "scripts" / "07_lightgbm_nbm.py")
lgbm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lgbm)

from windmill import care  # noqa: E402

CARE_ROOT = care.care_root(ROOT / "data")


def _sample_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, str, str]:
    """One CARE dataset, reduced to a single thermal channel."""
    farm_dir = CARE_ROOT / "Wind Farm C"
    info = care.event_info(farm_dir)
    path = sorted((farm_dir / "datasets").glob("*.csv"))[0]
    df = pd.read_csv(path, sep=";", low_memory=False).set_index("id")
    train = df[df["train_test"] == "train"]
    train = train[train["status_type_id"].isin(care.NORMAL_STATUS_IDS)]

    columns = care.dataset_columns(path)
    pcol, thermal, wind, _ambient = care.feature_rows(farm_dir, columns)
    feats = [pcol] + [w for w in wind if w in df.columns]
    c = thermal[1]
    t = train.dropna(subset=feats + [c]).sample(n=3000, random_state=0).sort_index()
    return t, t[feats], t[c], pcol, c


def main() -> int:
    import lightgbm as lgb

    t, X, y, _pcol, _c = _sample_dataset()
    preds = []
    for _ in range(2):
        m = lgb.LGBMRegressor(**lgbm.LGBM_PARAMS)
        m.fit(X, y.to_numpy())
        preds.append(m.predict(X))

    identical = np.array_equal(preds[0], preds[1])
    max_diff = float(np.abs(preds[0] - preds[1]).max())
    print("two fits, same data, same params:")
    print(f"   predictions identical : {identical}")
    print(f"   max |difference|      : {max_diff:.10f}")
    print(f"   random_state in params: {lgbm.LGBM_PARAMS.get('random_state')}")
    print(f"   seed in params        : {lgbm.LGBM_PARAMS.get('seed')}")
    print()
    if not identical:
        print("   -> NOT reproducible. The committed CARE score cannot be re-derived.")
        return 1
    print("   -> reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
