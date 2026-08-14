"""Modeling only: fit LightGBM, calibrate with isotonic regression, and
report ranking/calibration metrics. No S3, no CLI — pure model work."""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
}


def train_model(X: pd.DataFrame, y: pd.Series) -> lgb.Booster:
    """Train the raw scorer. scale_pos_weight = neg/pos counters the
    severe class imbalance in the Sparkov data."""
    pos = int(y.sum())
    neg = int(len(y) - pos)
    params = {**_PARAMS, "scale_pos_weight": (neg / pos) if pos else 1.0}
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=300)

def calibrate(model: lgb.Booster, Xval: pd.DataFrame, yval: pd.Series) -> IsotonicRegression:
    """Map raw scores to well-calibrated probabilities using the val split."""
    raw = model.predict(Xval)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, yval)
    return iso


def evaluate(model: lgb.Booster, iso: IsotonicRegression,
             Xval: pd.DataFrame, yval: pd.Series) -> dict[str, float]:
    """ROC-AUC (ranking) and PR-AUC (imbalance-aware) on calibrated probs."""
    probs = iso.predict(model.predict(Xval))
    return {
        "roc_auc": float(roc_auc_score(yval, probs)),
        "pr_auc": float(average_precision_score(yval, probs)),
    }
