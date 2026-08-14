"""Split only: a leakage-safe, time-ordered train/val cut. Pure function —
no randomness, no I/O — so correctness is trivially testable."""
from __future__ import annotations

import pandas as pd


def split_index(n: int, val_frac: float = 0.2) -> int:
    """Row index where validation begins. Single source of the cut, so any
    frame aligned to X (e.g. meta) can be split on the same boundary."""
    return int(n * (1 - val_frac))


def chronological_split(
    X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """First (1 - val_frac) rows train, last val_frac validate.

    Assumes X/y are already in time order (feature pipeline guarantees this).
    """
    cut = split_index(len(X), val_frac)
    return (
        X.iloc[:cut], y.iloc[:cut],
        X.iloc[cut:], y.iloc[cut:],
    )
