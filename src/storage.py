"""All AWS I/O behind small, injectable classes. Everything else in the
codebase stays boto3-free and unit-testable."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

from src.config import Settings, get_settings

log = logging.getLogger(__name__)


def _shard(cc_num: str, shard_count: int) -> int:
    """Deterministic shard so a card's events share a partition key across
    processes. Python's hash() is PYTHONHASHSEED-salted and unusable here."""
    return int(hashlib.sha256(cc_num.encode()).hexdigest()[:8], 16) % shard_count


class CardStateStore:
    """Sharded DynamoDB online feature store.

    PK = "<cc_num>#<shard>", SK = ISO timestamp. Writes are synchronous so a
    card's state is committed before its next transaction is scored.
    """

    def __init__(self, settings: Settings | None = None, resource=None):
        self.s = settings or get_settings()
        ddb = resource or boto3.resource("dynamodb", region_name=self.s.region)
        self.table = ddb.Table(self.s.ddb_table)

    def _pk(self, cc_num: str) -> str:
        return f"{cc_num}#{_shard(cc_num, self.s.shard_count)}"

    def get_history(self, cc_num: str, now: datetime) -> list[dict]:
        """Return the card's events within the 7d window, oldest→newest.

        Paginates so a high-velocity card is never truncated by a single page.
        """
        since = (now - timedelta(days=self.s.state_history_days)).isoformat()
        events: list[dict] = []
        kwargs = {
            "KeyConditionExpression": Key("pk").eq(self._pk(cc_num)) & Key("sk").gte(since),
            "ConsistentRead": True,  # must see the prior txn's write
        }
        while True:
            resp = self.table.query(**kwargs)
            events.extend(resp.get("Items", []))
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                break
            kwargs["ExclusiveStartKey"] = lek
        return [
            {"ts": datetime.fromisoformat(e["ts"]), "amt": float(e["amt"]),
             "category": e["category"]}
            for e in events
        ]

    def append_event(self, cc_num: str, ts: datetime, amt: float, category: str) -> None:
        """Synchronously record one event — the N+1 guarantee lives here."""
        iso = ts.isoformat()
        self.table.put_item(Item={
            "pk": self._pk(cc_num),
            "sk": f"{iso}#{uuid.uuid4().hex[:8]}",  # unique even on ts collisions
            "ts": iso,
            "amt": Decimal(str(amt)),  # native N type; str() avoids float artifacts
            "category": category,
            # DynamoDB TTL reclaims rows past the read window. Derived from the
            # event's own ts so replayed events expire on the right schedule.
            "expires_at": int((ts + timedelta(days=self.s.state_ttl_days)).timestamp()),
        })


class ModelStore:
    """Loads the model/calibrator and training data from S3."""

    def __init__(self, settings: Settings | None = None, client=None):
        self.s = settings or get_settings()
        self.s3 = client or boto3.client("s3", region_name=self.s.region)

    def _get_bytes(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.s.s3_bucket, Key=key)["Body"].read()

    def load_model(self):
        import lightgbm as lgb
        return lgb.Booster(model_str=self._get_bytes(self.s.model_key).decode())

    def load_calibrator(self):
        return pickle.loads(self._get_bytes(self.s.calibrator_key))

    def download_data(self, dest: str) -> str:
        self.s3.download_file(self.s.s3_bucket, self.s.data_key, dest)
        return dest

    def upload_model(self, model_str: str, calibrator) -> None:
        self.s3.put_object(Bucket=self.s.s3_bucket, Key=self.s.model_key,
                           Body=model_str.encode())
        self.s3.put_object(Bucket=self.s.s3_bucket, Key=self.s.calibrator_key,
                           Body=pickle.dumps(calibrator))

    def upload_bytes(self, key: str, body: bytes) -> None:
        self.s3.put_object(Bucket=self.s.s3_bucket, Key=key, Body=body)

    def load_json(self, key: str) -> dict:
        return json.loads(self._get_bytes(key))


class PredictionSink:
    """Writes each prediction to S3 off the request path.

    Bounded pool: prediction logging must never exhaust threads or mask a
    persistent S3 failure.
    """

    def __init__(self, settings: Settings | None = None, client=None, max_workers: int = 4):
        self.s = settings or get_settings()
        self.s3 = client or boto3.client("s3", region_name=self.s.region)
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="prediction-sink")

    def write_async(self, record: bytes, txn_id: str) -> None:
        self._pool.submit(self._put, record, txn_id)

    def write(self, record: bytes, txn_id: str) -> None:
        """Synchronous write. Lambda freezes the container on return, so the
        stream path must not rely on background threads."""
        self._put(record, txn_id)

    def _put(self, record: bytes, txn_id: str) -> None:
        key = f"{self.s.predictions_prefix}{datetime.now(timezone.utc):%Y/%m/%d}/{txn_id}.json"
        try:
            self.s3.put_object(Bucket=self.s.s3_bucket, Key=key, Body=record)
        except Exception:
            log.exception("prediction write failed key=%s", key)

