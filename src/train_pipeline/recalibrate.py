"""Refit only the isotonic calibrator against fresh labeled data, keeping the
existing booster. Cheap remediation when ranking is fine but probabilities drift."""
from __future__ import annotations

import pandas as pd

from src.storage import ModelStore
from src.train_pipeline.trainer import calibrate


def main(labeled: str = "scored_labeled.parquet") -> None:
    store = ModelStore()
    model = store.load_model()
    df = pd.read_parquet(labeled)
    from src.config import FEATURE_COLUMNS
    iso = calibrate(model, df[FEATURE_COLUMNS], df["is_fraud"])
    store.upload_model(model.model_to_string(), iso)  # booster unchanged, calibrator fresh
    print("Recalibrated: isotonic refit, booster preserved.")
