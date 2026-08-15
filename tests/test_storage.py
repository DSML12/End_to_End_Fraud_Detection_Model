"""Sharded DynamoDB store against moto: consistency + pagination."""
from datetime import datetime, timedelta, timezone

import boto3
import pytest
from moto import mock_aws

from src.config import get_settings

NOW = datetime(2025, 1, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
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
        from src.storage import CardStateStore
        yield CardStateStore(s, resource=ddb)


def test_write_then_read_is_consistent(store):
    store.append_event("c1", NOW - timedelta(minutes=5), 100.0, "a")
    history = store.get_history("c1", NOW)
    assert len(history) == 1
    assert history[0]["amt"] == 100.0


def test_event_carries_ttl_beyond_the_read_window(store):
    """DynamoDB silently never reaps rows missing the TTL attribute, so pin
    both its name and that it outlives the feature window."""
    s = get_settings()
    ts = NOW - timedelta(minutes=5)
    store.append_event("c1", ts, 100.0, "a")

    item = store.table.scan()["Items"][0]
    expires_at = item["expires_at"]
    assert int(expires_at) == expires_at  # TTL requires an epoch-seconds Number
    assert int(expires_at) == int((ts + timedelta(days=s.state_ttl_days)).timestamp())
    assert s.state_ttl_days > s.state_history_days  # deletes never race reads


def test_history_windowed_and_paginated(store):
    for i in range(120):  # exceeds a single small page
        store.append_event("c1", NOW - timedelta(minutes=i + 1), float(i), "a")
    store.append_event("c1", NOW - timedelta(days=10), 999.0, "a")  # outside 7d
    history = store.get_history("c1", NOW)
    assert len(history) == 120  # old event excluded
