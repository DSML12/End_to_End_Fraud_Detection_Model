"""Train/serve parity: the vectorized builder must agree with the online
build_features replay on EVERY feature column, and must not leak the current
row into its own features."""
import random
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.feature_pipeline.builder import build_matrix
from src.features import build_features

T0 = datetime(2025, 1, 1, 12, 0)

# Spacings that land on the window edges the two implementations disagreed on:
# 0 = duplicate timestamp, 60/1440/10080 = exactly 1h/24h/7d.
_SPACINGS_MIN = [0, 0, 1, 17, 59, 60, 61, 240, 1439, 1440, 1441, 4000, 10080]


def _frame():
    rows = [
        {"cc_num": "c1", "amt": 10.0, "category": "a", "ts": T0, "is_fraud": 0},
        {"cc_num": "c1", "amt": 20.0, "category": "b",
         "ts": T0 + timedelta(minutes=10), "is_fraud": 0},
        {"cc_num": "c1", "amt": 30.0, "category": "a",
         "ts": T0 + timedelta(minutes=20), "is_fraud": 1},
    ]
    return pd.DataFrame(rows)


def _random_frame(seed: int, cards: int = 3, per_card: int = 40) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for c in range(cards):
        ts = T0
        for _ in range(per_card):
            ts += timedelta(minutes=rng.choice(_SPACINGS_MIN))
            rows.append({"cc_num": f"c{c}", "amt": round(rng.uniform(1, 900), 2),
                         "category": rng.choice(list("abcde")), "ts": ts,
                         "is_fraud": rng.randint(0, 1)})
    return pd.DataFrame(rows)


def _assert_parity(df: pd.DataFrame) -> None:
    """Replay every row through the online path and compare all columns."""
    X, _, _ = build_matrix(df)
    ordered = df.sort_values(["cc_num", "ts"], kind="stable").reset_index(drop=True)

    for i, row in enumerate(ordered.itertuples()):
        history = [{"ts": e.ts, "amt": e.amt, "category": e.category}
                   for e in ordered.iloc[:i].itertuples() if e.cc_num == row.cc_num]
        expected = build_features(row.amt, row.category, row.ts, history)
        actual = X.iloc[i]
        for col in FEATURE_COLUMNS:
            assert actual[col] == pytest.approx(expected[col], abs=1e-9), (
                f"row {i} ({row.cc_num}) column {col}: "
                f"offline={actual[col]!r} online={expected[col]!r}"
            )


def test_first_row_has_no_history():
    X, _, _ = build_matrix(_frame())
    assert X.iloc[0]["cc_cnt_1h"] == 0.0
    assert X.iloc[0]["cc_time_since_prev_s"] == -1.0


def test_matches_online_replay():
    _assert_parity(_frame())


@pytest.mark.parametrize("seed", [7, 13, 42, 99, 123, 2024])
def test_parity_on_randomized_histories(seed):
    """Multi-card histories over window boundaries — the case fixed assertions
    on a single card silently passed while three skews were live."""
    _assert_parity(_random_frame(seed))


def test_parity_on_duplicate_timestamps():
    """Same-instant events are distinct rows online (append_event uuid-suffixes
    the sort key), so the builder must keep them as history, not drop them."""
    rows = [{"cc_num": "c1", "amt": 10.0, "category": "a", "ts": T0, "is_fraud": 0},
            {"cc_num": "c1", "amt": 20.0, "category": "b", "ts": T0, "is_fraud": 0},
            {"cc_num": "c1", "amt": 30.0, "category": "c",
             "ts": T0 + timedelta(minutes=5), "is_fraud": 0}]
    _assert_parity(pd.DataFrame(rows))


@pytest.mark.parametrize("window", [timedelta(hours=1), timedelta(hours=24),
                                    timedelta(days=7)])
def test_parity_on_exact_window_edges(window):
    """An event landing exactly on a window boundary must be counted the same
    way by both paths (inclusive)."""
    rows = [{"cc_num": "c1", "amt": 10.0, "category": "a", "ts": T0, "is_fraud": 0},
            {"cc_num": "c1", "amt": 20.0, "category": "b", "ts": T0 + window,
             "is_fraud": 0}]
    _assert_parity(pd.DataFrame(rows))
