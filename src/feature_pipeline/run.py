"""CLI entry: loader → builder → persist the matrix locally for training."""
from __future__ import annotations

import argparse

from src.feature_pipeline.builder import build_matrix
from src.feature_pipeline.loader import load_raw


def main(out: str = "matrix.parquet") -> str:
    X, y, meta = build_matrix(load_raw())
    # Prefix keeps meta columns from ever being mistaken for features.
    X.assign(is_fraud=y, _category=meta["category"], _ts=meta["ts"]).to_parquet(out, index=False)
    print(f"Wrote {len(X)} rows → {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="matrix.parquet")
    main(p.parse_args().out)
