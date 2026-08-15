"""PSI/CSI primitive. PSI (score) and CSI (feature) are the same statistic on
different inputs — one implementation, called two ways."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6  # keeps empty bins from blowing up the log


def bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Decile edges from the reference; frozen and reused for every period."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, qs))
    edges[0], edges[-1] = -np.inf, np.inf  # catch out-of-range values
    return edges


def _distribution(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(x, bins=edges)[0].astype(float)
    return (counts + _EPS) / (counts.sum() + _EPS * len(counts))


def psi(reference: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
    """Population Stability Index between two samples on fixed bin edges."""
    r, a = _distribution(reference, edges), _distribution(actual, edges)
    return float(np.sum((a - r) * np.log(a / r)))


def band(value: float) -> str:
    """Standard fraud/credit interpretation bands."""
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "material"


def csi_by_feature(
    ref: pd.DataFrame, act: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    """Per-feature PSI — surfaces *which inputs* moved when score PSI is material."""
    out = {}
    for f in features:
        edges = bin_edges(ref[f].to_numpy())
        out[f] = psi(ref[f].to_numpy(), act[f].to_numpy(), edges)
    return out
