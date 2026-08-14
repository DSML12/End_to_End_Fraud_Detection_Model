"""CLI entry: load matrix → split → train → calibrate → evaluate → upload
model.txt + calibrator.pkl to S3 models/."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.monitoring.reference import build_reference
from src.storage import ModelStore
from src.train_pipeline.splitter import chronological_split, split_index
from src.train_pipeline.trainer import calibrate, evaluate, train_model


def main(matrix: str = "matrix.parquet", val_frac: float = 0.2) -> dict[str, float]:
    df = pd.read_parquet(matrix)
    X = df.drop(columns=["is_fraud", "_category", "_ts"])
    y = df["is_fraud"]

    Xtr, ytr, Xval, yval = chronological_split(X, y, val_frac)
    # Same boundary keeps meta row-aligned with the validation split.
    cut = split_index(len(X), val_frac)
    meta_val = (df[["_category", "_ts"]].iloc[cut:]
                .rename(columns={"_category": "category"})
                .reset_index(drop=True))

    model = train_model(Xtr, ytr)
    iso = calibrate(model, Xval, yval)
    metrics = evaluate(model, iso, Xval, yval)
    print(f"Validation metrics: {metrics}")

    reference = build_reference(model, iso, Xval.reset_index(drop=True),
                                yval.reset_index(drop=True), meta_val)
    store = ModelStore()
    store.upload_model(model.model_to_string(), iso)
    store.upload_bytes("models/reference.json", json.dumps(reference).encode())
    print("Uploaded model + calibrator + reference.")
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="matrix.parquet")
    p.add_argument("--val-frac", type=float, default=0.2)
    a = p.parse_args()
    main(a.matrix, a.val_frac)
