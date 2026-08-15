"""Chronological split: sizes and time-order boundary."""
import pandas as pd

from src.train_pipeline.splitter import chronological_split


def test_split_sizes_and_order():
    X = pd.DataFrame({"amt": range(10)})
    y = pd.Series(range(10))
    Xtr, ytr, Xval, yval = chronological_split(X, y, val_frac=0.2)
    assert len(Xtr) == 8 and len(Xval) == 2
    # Val comes strictly after train (no shuffle).
    assert ytr.iloc[-1] < yval.iloc[0]
