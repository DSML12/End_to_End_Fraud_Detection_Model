"""Decision truth table — pure, no AWS, no retrain triggered."""
from src.monitoring.policy import Action, decide


def test_pr_auc_breach_forces_retrain():
    d = decide(psi_material=True, population_drift=True, pr_auc_breach=True,
               calibration_drift=False, ece_threshold_exceeded=False)
    assert d.action == Action.RETRAIN


def test_calibration_only_when_ranking_intact():
    d = decide(psi_material=False, population_drift=False, pr_auc_breach=False,
               calibration_drift=True, ece_threshold_exceeded=True)
    assert d.action == Action.CALIBRATE_ONLY


def test_unlabeled_drift_alerts_not_retrains():
    d = decide(psi_material=True, population_drift=True, pr_auc_breach=None,
               calibration_drift=None, ece_threshold_exceeded=None)
    assert d.action == Action.ALERT


def test_clean_is_none():
    d = decide(psi_material=False, population_drift=False, pr_auc_breach=False,
               calibration_drift=None, ece_threshold_exceeded=None)
    assert d.action == Action.NONE
