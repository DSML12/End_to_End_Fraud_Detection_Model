"""Population drift via a discriminator: if a classifier can tell reference
from monitoring data (AUC above the flag margin), the joint input
distribution has shifted. Complements PSI, which is per-feature/marginal."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import FEATURE_COLUMNS

_PARAMS = {"objective": "binary", "learning_rate": 0.1,
           "num_leaves": 15, "verbose": -1}


def population_auc(reference: pd.DataFrame, monitoring: pd.DataFrame) -> float:
    """Train reference(0) vs monitoring(1) on a holdout; return discriminator AUC."""
    ref = reference[FEATURE_COLUMNS].assign(_y=0)
    mon = monitoring[FEATURE_COLUMNS].assign(_y=1)
    data = pd.concat([ref, mon], ignore_index=True).sample(frac=1, random_state=0)

    cut = int(len(data) * 0.8)
    tr, va = data.iloc[:cut], data.iloc[cut:]
    model = lgb.train(_PARAMS, lgb.Dataset(tr[FEATURE_COLUMNS], tr["_y"]),
                      num_boost_round=100)
    return float(roc_auc_score(va["_y"], model.predict(va[FEATURE_COLUMNS])))


def is_drifted(auc: float, threshold: float) -> bool:
    """Flag when the discriminator beats the margin (0.5 = pure noise floor)."""
    return auc > threshold
