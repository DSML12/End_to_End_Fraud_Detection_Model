
"""Pure feature logic: velocity rises on repeats, time flags, zscore."""
from datetime import datetime, timedelta

from src.features import build_features

NOW = datetime(2025, 1, 4, 23, 0)  # Saturday, night


def _event(ts, amt=100.0, category="grocery_pos"):
    return {"ts": ts, "amt": amt, "category": category}


def test_time_flags():
    f = build_features(100.0, "grocery_pos", NOW, [])
    assert f["is_weekend"] == 1.0
    assert f["is_night"] == 1.0


def test_velocity_counts_rise_with_history():
    history = [_event(NOW - timedelta(minutes=m)) for m in (5, 10, 30)]
    f = build_features(100.0, "grocery_pos", NOW, history)
    assert f["cc_cnt_1h"] == 3.0
    assert f["cc_cnt_24h"] == 3.0


def test_no_history_sentinels():
    f = build_features(100.0, "grocery_pos", NOW, [])
    assert f["cc_cnt_1h"] == 0.0
    assert f["cc_time_since_prev_s"] == -1.0


def test_zscore_uses_prior_amounts():
    # Priors must vary so std > 0; a large current amt then scores positive.
    history = [
        _event(NOW - timedelta(hours=1), amt=10.0),
        _event(NOW - timedelta(hours=2), amt=20.0),
        _event(NOW - timedelta(hours=3), amt=30.0),
    ]
    f = build_features(1000.0, "grocery_pos", NOW, history)
    assert f["cc_amt_zscore"] > 0  # large amt vs modest, varied prior mean
