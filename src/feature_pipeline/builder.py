"""Transform only: vectorized replay of the feature logic over the sorted
frame. Windows use closed="left" so each row sees only that card's *prior*
events, and aggregates are pinned to the online definitions in src.features —
inclusive window edges, population std — so training and serving agree."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS

_W1H, _W24H, _W7D = "1h", "24h", "7D"


def _windows(values: pd.Series, index: pd.DatetimeIndex, win: str):
    """Yield each row's inclusive window as an array, current row last.

    Positional slicing (not label-based) so events sharing a timestamp are
    kept as history and only the current row is dropped by the caller.
    """
    start = index.searchsorted(index - pd.Timedelta(win), side="left")
    arr = values.to_numpy()
    return (arr[lo:hi + 1] for hi, lo in enumerate(start))


def _rolling(g: pd.DataFrame) -> pd.DataFrame:
    """Per-card windowed features, indexed by time, prior events only.

    Windows are closed="both" (inclusive left edge, matching online `>=`) with
    the current row's own contribution subtracted out. closed="left" cannot be
    used: it drops every row sharing the current timestamp, whereas online only
    the current transaction is excluded.
    """
    s = g.set_index("ts")
    amt = s["amt"]

    def w(win):
        return amt.rolling(win, closed="both")

    def prior_count(win):
        return w(win).count() - 1.0

    def prior_sum(win):
        return w(win).sum() - amt

    cnt_7d = prior_count(_W7D)
    mean_7d = (prior_sum(_W7D) / cnt_7d).where(cnt_7d > 0, 0.0)
    # Population std (ddof=0) over prior events only, two-pass to match
    # statistics.pstdev online; pandas .std() is the sample estimator.
    std_7d = pd.Series(
        [np.std(x[:-1]) if len(x) > 2 else 0.0 for x in _windows(amt, s.index, _W7D)],
        index=s.index, dtype=float,
    )

    # Distinct prior categories: unique codes in the window, current row dropped.
    codes = s["category"].astype("category").cat.codes
    distinct = pd.Series(
        [float(len(set(x[:-1]))) for x in _windows(codes, s.index, _W7D)],
        index=s.index, dtype=float,
    )

    out = pd.DataFrame({
        "cc_cnt_1h": prior_count(_W1H),
        "cc_amt_1h": prior_sum(_W1H),
        "cc_cnt_24h": prior_count(_W24H),
        "cc_amt_24h": prior_sum(_W24H),
        "cc_time_since_prev_s": (s.index.to_series() - s.index.to_series().shift())
                                .dt.total_seconds(),
        "cc_amt_mean_7d": mean_7d,
        "cc_amt_std_7d": std_7d,
        "cc_amt_zscore": ((amt - mean_7d) / std_7d).where(std_7d > 0, 0.0),
        "cc_distinct_cat_7d": distinct,
    }, index=s.index)
    return out.reset_index(drop=True)


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta) with the same columns/semantics as the online path."""
    df = df.sort_values(["cc_num", "ts"]).reset_index(drop=True)

    vel = df.groupby("cc_num", group_keys=False)[["ts", "amt", "category"]].apply(_rolling)

    # Static features are row-local and fully vectorized.
    hour = df["ts"].dt.hour
    dow = df["ts"].dt.dayofweek
    static = pd.DataFrame({
        "amt": df["amt"],
        "log_amt": np.log1p(df["amt"].clip(lower=0)),
        "hour": hour.astype(float),
        "dow": dow.astype(float),
        "is_weekend": (dow >= 5).astype(float),
        "is_night": ((hour < 6) | (hour >= 22)).astype(float),
    })

    X = pd.concat([static.reset_index(drop=True), vel.reset_index(drop=True)], axis=1)

    # Empty windows → 0 counts/sums; no prior txn → -1 sentinel; std/z → 0.
    X["cc_time_since_prev_s"] = X["cc_time_since_prev_s"].fillna(-1.0)
    X = X.fillna(0.0)[FEATURE_COLUMNS]

    y = df["is_fraud"].astype(int).reset_index(drop=True)
    
    meta = df[["category", "ts"]].reset_index(drop=True)  # segment + time, not features
    return X, y, meta
