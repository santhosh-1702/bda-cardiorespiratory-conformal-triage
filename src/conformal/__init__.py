"""
Conformal prediction package for distribution-free uncertainty quantification and clinical risk control.
"""

from src.conformal.calibrator import ConformalCalibrator
from src.conformal.cv_plus import CVPlusCalibrator, compute_cv_plus_quantiles
from src.conformal.artifact_simulator import inject_simulated_motion_artifacts, corrupt_single_ppg_segment
from src.conformal.safety_rules import evaluate_clinical_safety_rules

__all__ = [
    "ConformalCalibrator",
    "CVPlusCalibrator",
    "compute_cv_plus_quantiles",
    "inject_simulated_motion_artifacts",
    "corrupt_single_ppg_segment",
    "evaluate_clinical_safety_rules",
]
