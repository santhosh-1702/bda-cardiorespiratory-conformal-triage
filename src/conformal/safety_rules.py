"""
Phase 4: Clinical Safety Decision & Abstention Logic Module
===========================================================
Executes the risk-controlling safety rule:
If prediction interval width > 6 breaths/min, discard prediction and flag Safe Abstention.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error

from config.settings import ABSTENTION_THRESHOLD_BPM, ABSTENTION_FLAG_LABEL

def evaluate_clinical_safety_rules(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    threshold_bpm: float = ABSTENTION_THRESHOLD_BPM
) -> Dict[str, Any]:
    """
    Applies the clinical abstention rule and computes risk-control metrics.
    """
    widths = interval_upper - interval_lower
    covered = (y_true >= interval_lower) & (y_true <= interval_upper)
    empirical_coverage = float(np.mean(covered))

    # Safety Rule: Interval width > threshold -> Abstain
    abstain_mask = widths > threshold_bpm
    abstention_rate = float(np.mean(abstain_mask))
    retained_mask = ~abstain_mask

    overall_mae = float(mean_absolute_error(y_true, y_pred))

    if np.sum(retained_mask) > 0:
        retained_mae = float(mean_absolute_error(y_true[retained_mask], y_pred[retained_mask]))
        retained_count = int(np.sum(retained_mask))
    else:
        retained_mae = float("nan")
        retained_count = 0

    return {
        "empirical_coverage": empirical_coverage,
        "mean_width": float(np.mean(widths)),
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
        "abstention_threshold": threshold_bpm,
        "abstention_rate": abstention_rate,
        "abstained_count": int(np.sum(abstain_mask)),
        "retained_count": retained_count,
        "total_windows": len(y_true),
        "overall_mae": overall_mae,
        "retained_mae": retained_mae,
        "flag_label": ABSTENTION_FLAG_LABEL,
        "abstain_mask": abstain_mask,
        "widths": widths,
    }
