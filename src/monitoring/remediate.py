"""Job C. Reads both reports, applies the pure policy, and triggers the
Phase 2 pipelines. Retrain/recalibrate execution reuses existing code —
no modeling logic duplicated here."""
from __future__ import annotations

import json

from botocore.exceptions import ClientError

from src.monitoring.policy import Action, decide
from src.storage import ModelStore


def _aggregate(unlabeled: dict, labeled: dict) -> dict:
    """Collapse per-segment reports into the booleans the policy consumes."""
    psi_material = any(
        e.get("band") == "material"
        for seg in unlabeled.get("segments", {}).values()
        for e in seg.values()
    )
    population_drift = unlabeled.get("population", {}).get("drifted", False)

    pr_auc_breach = calib_drift = ece_exceeded = None
    if labeled.get("segments"):
        segs = labeled["segments"].values()
        pr_auc_breach = any(s["pr_auc_breach"] for s in segs)
        calib_drift = any(s["ece"] > s["ref_ece"] for s in segs)
        ece_exceeded = any(s["ece_exceeded"] for s in segs)

    return {
        "psi_material": psi_material,
        "population_drift": population_drift,
        "pr_auc_breach": pr_auc_breach,
        "calibration_drift": calib_drift,
        "ece_threshold_exceeded": ece_exceeded,
    }


def main() -> None:
    store = ModelStore()
    unlabeled = store.load_json("monitoring/unlabeled_report.json")
    try:
        labeled = store.load_json("monitoring/labeled_report.json")
    except ClientError:
        labeled = {}  # report absent: labels not yet matured

    decision = decide(**_aggregate(unlabeled, labeled))
    print(f"Decision: {decision.action.value} — {decision.reason}")

    # Execution reuses the Phase 2 entrypoints; imported lazily so a NONE/ALERT
    # decision never pulls in training deps.
    if decision.action == Action.RETRAIN:
        from src.feature_pipeline.run import main as build
        from src.train_pipeline.run import main as train
        build()
        train()
    elif decision.action == Action.CALIBRATE_ONLY:
        from src.train_pipeline.recalibrate import main as recalibrate
        recalibrate()

    store.upload_bytes("monitoring/decision.json",
                       json.dumps({"action": decision.action.value,
                                   "reason": decision.reason}).encode())
