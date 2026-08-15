"""The single online scoring path, reused by both entrypoints (API, Lambda).

Per transaction: read prior state (sync) → build features → score → commit
state (sync, before N+1) → write prediction (async) → emit metric.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd

from src.config import FEATURE_COLUMNS, Settings, get_settings
from src.features import build_features
from src.metrics import emit
from src.storage import CardStateStore, ModelStore, PredictionSink


@dataclass
class Transaction:
    """One inbound transaction. Timestamp is server-assigned; txn_id is the
    stable join key to the later-arriving label."""
    cc_num: str
    amt: float
    category: str
    txn_id: str = ""  # assigned at ingest if not supplied

    def __post_init__(self):
        if not self.txn_id:
            self.txn_id = uuid.uuid4().hex


@dataclass
class Decision:
    """Scoring result returned to callers and surfaced in the UI."""
    fraud_score: float
    decision: str
    cc_cnt_1h: float
    cc_cnt_24h: float


class Scorer:
    """Holds the model + calibrator and maps a feature vector to a probability.

    Model artifacts load once per process (cold start) and are reused.
    """

    def __init__(self, settings: Settings | None = None, store: ModelStore | None = None):
        s = settings or get_settings()
        store = store or ModelStore(s)
        self._model = store.load_model()
        self._calibrator = store.load_calibrator()

    def score(self, features: dict[str, float]) -> float:
        """Calibrated fraud probability for one feature dict."""
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        raw = self._model.predict(X)[0]
        return float(self._calibrator.predict([raw])[0])


@lru_cache
def _scorer() -> Scorer:
    """Process-wide singleton so the model is loaded only once."""
    return Scorer()


def _classify(score: float, s: Settings) -> str:
    if score >= s.decline_threshold:
        return "DECLINE"
    if score >= s.review_threshold:
        return "STEP-UP"
    return "APPROVE"


def handle_transaction(
    txn: Transaction,
    *,
    state: CardStateStore | None = None,
    sink: PredictionSink | None = None,
    now: datetime | None = None,
    sync_sink: bool = False,
) -> Decision:
    """Score one transaction end-to-end.

    Ordering is deliberate: state is committed synchronously *before* we
    return, so the next transaction on this card sees updated velocity.
    """
    s = get_settings()
    state = state or CardStateStore(s)
    sink = sink or PredictionSink(s)
    now = now or datetime.now(timezone.utc)
    start = time.perf_counter()

    history = state.get_history(txn.cc_num, now)
    features = build_features(txn.amt, txn.category, now, history)
    score = _scorer().score(features)
    decision = _classify(score, s)

    # Synchronous: must persist before the N+1 transaction is scored.
    state.append_event(txn.cc_num, now, txn.amt, txn.category)

    # Async: prediction log is off the critical path.
    record = json.dumps({
        "txn_id": txn.txn_id,  # join key to the later-arriving label
        "cc_num": txn.cc_num, "amt": txn.amt, "category": txn.category,
        "ts": now.isoformat(), "fraud_score": score, "decision": decision,
        "features": features,
    }).encode()
    (sink.write if sync_sink else sink.write_async)(record, txn.txn_id)

    emit(score, (time.perf_counter() - start) * 1000, decision)

    return Decision(
        fraud_score=score,
        decision=decision,
        cc_cnt_1h=features["cc_cnt_1h"],
        cc_cnt_24h=features["cc_cnt_24h"],
    )
