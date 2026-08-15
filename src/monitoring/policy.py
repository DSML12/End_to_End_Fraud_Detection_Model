"""Remediation policy — a PURE function from metrics to a decision, so the
'what should we do' logic is auditable and unit-testable without ever
triggering a retrain. Executing the decision lives elsewhere."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    NONE = "NONE"
    ALERT = "ALERT"                # unlabeled drift only — never auto-retrain on this
    CALIBRATE_ONLY = "CALIBRATE_ONLY"
    RETRAIN = "RETRAIN"


@dataclass
class RemediationDecision:
    action: Action
    reason: str


def decide(
    *,
    psi_material: bool,
    population_drift: bool,
    pr_auc_breach: bool | None,        # None = labels not yet available
    calibration_drift: bool | None,    # ECE worsened vs the frozen reference
    ece_threshold_exceeded: bool | None,
) -> RemediationDecision:
    """Decision table. Label-dependent signals gate the expensive actions:
    we never retrain on distribution shift alone."""
    # Confirmed ranking degradation → full retrain.
    if pr_auc_breach:
        return RemediationDecision(Action.RETRAIN, "PR-AUC breached tolerance band")

    # Ranking intact. Recalibrate only when the drift is both real (worse than
    # the reference) and material (over the absolute threshold).
    if calibration_drift and ece_threshold_exceeded:
        return RemediationDecision(Action.CALIBRATE_ONLY, "calibration drift, ranking intact")

    # Unlabeled signals fire but labels haven't confirmed harm → alert only.
    if psi_material or population_drift:
        return RemediationDecision(Action.ALERT, "input/score drift; awaiting label confirmation")

    return RemediationDecision(Action.NONE, "within tolerance")
