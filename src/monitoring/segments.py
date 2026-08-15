"""Segment viability gate — shared by every downstream check. A segment is
only trustworthy if it has enough rows AND enough fraud cases; noise on tiny
segments is the classic false-alarm source."""
from __future__ import annotations

import pandas as pd


def viable_segments(
    df: pd.DataFrame, dim: str, min_rows: int, min_fraud: int, label: str = "is_fraud"
) -> list[str]:
    """Return the values of `dim` that clear both thresholds."""
    g = df.groupby(dim)
    ok = g.agg(rows=(label, "size"), fraud=(label, "sum"))
    keep = ok[(ok["rows"] >= min_rows) & (ok["fraud"] >= min_fraud)]
    return keep.index.tolist()
