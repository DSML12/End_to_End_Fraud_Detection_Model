"""Label-gated checks: PR-AUC control chart and calibration drift. Both need
matured ground truth, so they run only in the labeled job."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Control-chart tolerance around the reference PR-AUC.
_BAND = 0.05


def pr_auc_chart(
    ref_pr_auc: float, periods: dict[str, pd.DataFrame],
    score: str = "fraud_score", label: str = "is_fraud",
) -> list[dict]:
    """Per-period PR-AUC vs a ±5% band around the reference. Out-of-band = flag."""
    lo, hi = ref_pr_auc * (1 - _BAND), ref_pr_auc * (1 + _BAND)
    rows = []
    for period, df in sorted(periods.items()):
        actual = average_precision_score(df[label], df[score])
        rows.append({"period": period, "pr_auc": float(actual),
                     "low": lo, "high": hi, "breach": actual < lo})
    return rows


def calibration_drift(
    df: pd.DataFrame, n_bins: int = 10,
    score: str = "fraud_score", label: str = "is_fraud",
) -> float:
    """Expected Calibration Error: mean |predicted - observed| across score bins.
    Rising ECE means probabilities no longer mean what they claim."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(df[score], edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            gap = abs(df[score][m].mean() - df[label][m].mean())
            ece += (m.mean()) * gap
    return float(ece)
