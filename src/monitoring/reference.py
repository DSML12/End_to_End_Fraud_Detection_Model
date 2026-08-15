"""Baseline artifacts frozen at train time. Monitoring compares against these
fixed values, never against live data — otherwise the yardstick drifts with
the thing being measured."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import FEATURE_COLUMNS
from src.monitoring.performance import calibration_drift


def build_reference(model, iso, Xval, yval, meta: pd.DataFrame,
                    dim: str = "category") -> dict:
    """Freeze: reference score deciles + per-segment reference PR-AUC and ECE.
    `meta` carries the segment column aligned to Xval."""
    probs = iso.predict(model.predict(Xval))
    ref_scores = np.sort(probs).tolist()

    seg_pr_auc, seg_ece = {}, {}
    y = yval.to_numpy()
    for seg, idx in meta.reset_index(drop=True).groupby(dim).indices.items():
        if y[idx].sum() == 0:  # PR-AUC undefined without positives
            continue
        seg_pr_auc[str(seg)] = float(average_precision_score(y[idx], probs[idx]))
        seg_ece[str(seg)] = calibration_drift(
            pd.DataFrame({"fraud_score": probs[idx], "is_fraud": y[idx]})
        )

    return {
        "dim": dim,
        "reference_scores": ref_scores,       # for PSI bin edges
        "segment_pr_auc": seg_pr_auc,         # for the control chart
        "segment_ece": seg_ece,               # calibration baseline
        "features": FEATURE_COLUMNS,
    }
