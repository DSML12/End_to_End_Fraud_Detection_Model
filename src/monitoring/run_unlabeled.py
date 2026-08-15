"""Job A (cadence: daily/weekly/monthly). No labels required. Segments →
score PSI per period → CSI on material segments → population discriminator.
Writes a metrics report to S3 for the remediation job to consume."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.monitoring.drift import band, bin_edges, csi_by_feature, psi
from src.monitoring.population import is_drifted, population_auc
from src.monitoring.segments import viable_segments
from src.storage import ModelStore

_FREQ = {"daily": "D", "weekly": "W", "monthly": "MS"}


def run(reference: pd.DataFrame, monitoring: pd.DataFrame, ref_meta: dict,
        *, dim: str, cadence: str, min_rows: int, min_fraud: int,
        pop_auc_threshold: float) -> dict:
    edges = bin_edges(np.array(ref_meta["reference_scores"]))
    period = monitoring["ts"].dt.to_period(_FREQ[cadence]).astype(str)

    report = {"segments": {}, "population": {}}
    for seg in viable_segments(monitoring, dim, min_rows, min_fraud):
        seg_ref = reference[reference[dim] == seg]["fraud_score"].to_numpy()
        seg_mon = monitoring[monitoring[dim] == seg]
        by_period = {}
        for p, chunk in seg_mon.groupby(period):
            score_psi = psi(seg_ref, chunk["fraud_score"].to_numpy(), edges)
            entry = {"psi": score_psi, "band": band(score_psi)}
            # Drill into inputs only when the score distribution materially moved.
            if band(score_psi) == "material":
                ref_feats = reference[reference[dim] == seg]
                entry["csi"] = csi_by_feature(ref_feats, chunk, ref_meta["features"])
            by_period[p] = entry
        report["segments"][seg] = by_period

    auc = population_auc(reference, monitoring)
    report["population"] = {"auc": auc, "drifted": is_drifted(auc, pop_auc_threshold)}
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", default="category")
    p.add_argument("--cadence", choices=_FREQ, default="monthly")
    p.add_argument("--min-rows", type=int, default=500)
    p.add_argument("--min-fraud", type=int, default=20)
    p.add_argument("--pop-auc-threshold", type=float, default=0.55)
    a = p.parse_args()

    store = ModelStore()
    ref_meta = store.load_json("models/reference.json")
    reference = pd.read_parquet("reference_scored.parquet")   # baseline preds+features
    monitoring = pd.read_parquet("monitoring_scored.parquet")  # from predictions/

    report = run(reference, monitoring, ref_meta, dim=a.dim, cadence=a.cadence,
                 min_rows=a.min_rows, min_fraud=a.min_fraud,
                 pop_auc_threshold=a.pop_auc_threshold)
    store.upload_bytes("monitoring/unlabeled_report.json", json.dumps(report).encode())
    print(json.dumps(report, indent=2))
