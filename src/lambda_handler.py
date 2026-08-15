"""Lambda entrypoint. Kinesis-triggered: decode each record and run it
through the same inference core the API uses. Parsing only — no scoring logic."""
from __future__ import annotations

import base64
import json

from src.inference_pipeline.predict import Transaction, handle_transaction


def _parse(record: dict) -> Transaction:
    """Decode one Kinesis record into a Transaction."""
    payload = json.loads(base64.b64decode(record["kinesis"]["data"]))
    return Transaction(
        cc_num=str(payload["cc_num"]),
        amt=float(payload["amt"]),
        category=str(payload["category"]),
    )


def handler(event: dict, context=None) -> dict:
    """Process a Kinesis batch. Records are handled sequentially so a card's
    state is committed before its next record is scored."""
    processed = 0
    for record in event.get("Records", []):
        # Synchronous sink: Lambda freezes the container the moment we return.
        handle_transaction(_parse(record), sync_sink=True)
        processed += 1
    return {"processed": processed}
