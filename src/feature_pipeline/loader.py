"""Ingest only: pull raw Sparkov transactions from S3 and put them in the
chronological, per-card order the builder replays. No feature logic here."""
from __future__ import annotations

import tempfile

import pandas as pd

from src.storage import ModelStore

# Minimal schema the builder depends on; everything else is ignored.
_REQUIRED = ["cc_num", "amt", "category", "trans_date_trans_time", "is_fraud"]


def load_raw(store: ModelStore | None = None) -> pd.DataFrame:
    """Download the training CSV and return a clean, time-sorted frame."""
    store = store or ModelStore()
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        store.download_data(tmp.name)
        df = pd.read_csv(tmp.name, usecols=_REQUIRED)

    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df["cc_num"] = df["cc_num"].astype(str)
    df["amt"] = df["amt"].astype(float)
    df["category"] = df["category"].astype(str)
    df = df.dropna(subset=["cc_num", "amt", "category", "ts"])

    # Global chronological order is the no-leakage precondition for replay.
    return df.sort_values("ts").reset_index(drop=True)
