"""Viability gate filters low-row and low-fraud segments."""
import pandas as pd

from src.monitoring.segments import viable_segments


def test_filters_by_rows_and_fraud():
    df = pd.DataFrame({
        "category": ["a"] * 100 + ["b"] * 100 + ["c"] * 5,
        "is_fraud": [1] * 30 + [0] * 70      # a: 30 fraud
                    + [1] * 2 + [0] * 98      # b: 2 fraud  (too few)
                    + [1] * 5,                # c: 5 rows   (too few)
    })
    keep = viable_segments(df, "category", min_rows=50, min_fraud=20)
    assert keep == ["a"]
