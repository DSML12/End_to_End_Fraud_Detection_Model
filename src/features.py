"""Pure feature engineering — no I/O, no AWS. This is the parity contract:
the offline pipeline and the online path derive features identically."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from statistics import mean, pstdev

from src.config import FEATURE_COLUMNS

# A card's recent events, oldest→newest. Each: {"ts", "amt", "category"}.
Event = dict


def _time_features(amt: float, now: datetime) -> dict[str, float]:
    """Static features derived from amount and the server-assigned time."""
    hour = now.hour
    dow = now.weekday()  # Mon=0 .. Sun=6
    return {
        "amt": amt,
        "log_amt": math.log1p(max(amt, 0.0)),
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": float(dow >= 5),
        "is_night": float(hour < 6 or hour >= 22),
    }


def _velocity_features(
    amt: float, now: datetime, history: list[Event]
) -> dict[str, float]:
    """Streaming features from the card's prior events within the 7d window.

    Window edges are inclusive (>=) and std is population (ddof=0); the
    offline builder must match both exactly.
    """
    h1 = now - timedelta(hours=1)
    h24 = now - timedelta(hours=24)
    d7 = now - timedelta(days=7)

    amts_1h = [e["amt"] for e in history if e["ts"] >= h1]
    amts_24h = [e["amt"] for e in history if e["ts"] >= h24]
    amts_7d = [e["amt"] for e in history if e["ts"] >= d7]
    cats_7d = {e["category"] for e in history if e["ts"] >= d7}

    prev_ts = max((e["ts"] for e in history), default=None)
    time_since_prev = (now - prev_ts).total_seconds() if prev_ts else -1.0

    mean_7d = mean(amts_7d) if amts_7d else 0.0
    std_7d = pstdev(amts_7d) if len(amts_7d) > 1 else 0.0
    zscore = (amt - mean_7d) / std_7d if std_7d > 0 else 0.0

    return {
        "cc_cnt_1h": float(len(amts_1h)),
        "cc_amt_1h": float(sum(amts_1h)),
        "cc_cnt_24h": float(len(amts_24h)),
        "cc_amt_24h": float(sum(amts_24h)),
        "cc_time_since_prev_s": float(time_since_prev),
        "cc_amt_mean_7d": float(mean_7d),
        "cc_amt_std_7d": float(std_7d),
        "cc_amt_zscore": float(zscore),
        "cc_distinct_cat_7d": float(len(cats_7d)),
    }


def build_features(
    amt: float, category: str, now: datetime, history: list[Event]
) -> dict[str, float]:
    """Assemble the full feature dict for one transaction.

    `history` is the card's prior events (this txn excluded), so velocity
    counts reflect state *before* N and rise as the same card repeats.
    """
    feats = {**_time_features(amt, now), **_velocity_features(amt, now, history)}
    return {col: feats[col] for col in FEATURE_COLUMNS}  # enforce column order
