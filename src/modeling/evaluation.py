"""
Phase 3: Model Evaluation Module
======================================================
Evaluates regression predictions against ground-truth respiratory rate annotations.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MAE, RMSE, R-squared, and residual stats.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    residuals = y_true - y_pred

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
    }
