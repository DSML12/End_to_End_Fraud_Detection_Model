"""Job B (runs when labels mature). Joins predictions/ to labels/ on txn_id,
then PR-AUC control chart + calibration drift per viable segment."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.monitoring.performance import calibration_drift, pr_auc_chart
from src.monitoring.segments import viable_segments
from src.storage import ModelStore

_FREQ = {"daily": "D", "weekly": "W", "monthly": "MS"}


def run(scored_labeled: pd.DataFrame, ref_meta: dict,
        *, dim: str, cadence: str, min_rows: int, min_fraud: int,
        ece_threshold: float) -> dict:
    period = scored_labeled["ts"].dt.to_period(_FREQ[cadence]).astype(str)
    report = {"segments": {}}
    for seg in viable_segments(scored_labeled, dim, min_rows, min_fraud):
        seg_df = scored_labeled[scored_labeled[dim] == seg]
        ref_pr = ref_meta["segment_pr_auc"].get(seg)
        if ref_pr is None:
            continue  # no frozen baseline for this segment
        periods = {p: c for p, c in seg_df.groupby(period)}
        chart = pr_auc_chart(ref_pr, periods)
        ece = calibration_drift(seg_df)
        report["segments"][seg] = {
            "pr_auc_chart": chart,
            "ece": ece,
            "ref_ece": ref_meta["segment_ece"].get(seg, 0.0),
            "ece_exceeded": ece > ece_threshold,
            "pr_auc_breach": any(r["breach"] for r in chart),
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", default="category")
    p.add_argument("--cadence", choices=_FREQ, default="monthly")
    p.add_argument("--min-rows", type=int, default=500)
    p.add_argument("--min-fraud", type=int, default=20)
    p.add_argument("--ece-threshold", type=float, default=0.05)
    a = p.parse_args()

    store = ModelStore()
    ref_meta = store.load_json("models/reference.json")
    # predictions/ joined to labels/ on txn_id, upstream of this job.
    scored_labeled = pd.read_parquet("scored_labeled.parquet")

    report = run(scored_labeled, ref_meta, dim=a.dim, cadence=a.cadence,
                 min_rows=a.min_rows, min_fraud=a.min_fraud,
                 ece_threshold=a.ece_threshold)
    store.upload_bytes("monitoring/labeled_report.json", json.dumps(report).encode())
    print(json.dumps(report, indent=2))
