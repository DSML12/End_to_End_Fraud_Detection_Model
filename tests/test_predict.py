"""End-to-end handle_transaction against moto: state commits before the
next txn, and the output schema is correct. Model is stubbed."""
from datetime import datetime, timezone

import boto3
import pytest
from moto import mock_aws

from src.config import get_settings
from src.inference_pipeline import predict as P
from src.inference_pipeline.predict import Transaction, handle_transaction

NOW = datetime(2025, 1, 4, 12, 0, tzinfo=timezone.utc)


class _StubScorer:
    """Deterministic: score scales with 1h velocity so repeats climb."""
    def score(self, features):
        return min(0.1 + 0.3 * features["cc_cnt_1h"], 0.99)


@pytest.fixture
def wired(monkeypatch):
    with mock_aws():
        s = get_settings()
        ddb = boto3.resource("dynamodb", region_name=s.region)
        ddb.create_table(
            TableName=s.ddb_table,
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"},
                       {"AttributeName": "sk", "KeyType": "RANGE"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"},
                                  {"AttributeName": "sk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        boto3.client("s3", region_name=s.region).create_bucket(
            Bucket=s.s3_bucket,
            CreateBucketConfiguration={"LocationConstraint": s.region},
        )
        from src.storage import CardStateStore, PredictionSink
        monkeypatch.setattr(P, "_scorer", lambda: _StubScorer())
        yield (CardStateStore(s, resource=ddb), PredictionSink(s))


def test_score_climbs_as_card_repeats(wired):
    state, sink = wired
    scores = []
    for i in range(3):
        d = handle_transaction(
            Transaction("000123456789", 100.0, "grocery_pos"),
            state=state, sink=sink,
            now=NOW.replace(minute=i),
        )
        scores.append(d.fraud_score)
    # Each call sees the prior committed events → strictly rising velocity.
    assert scores[0] < scores[1] < scores[2]


def test_response_schema(wired):
    state, sink = wired
    d = handle_transaction(
        Transaction("000123456789", 100.0, "grocery_pos"),
        state=state, sink=sink, now=NOW,
    )
    assert 0.0 <= d.fraud_score <= 1.0
    assert d.decision in {"APPROVE", "STEP-UP", "DECLINE"}
    assert d.cc_cnt_1h == 0.0  # first txn, no history
