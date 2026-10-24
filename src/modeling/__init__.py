"""
Modeling package for CPU-optimized XGBoost regression and benchmark evaluation.
"""

from src.modeling.train import train_baseline_model, iterate_hyperparameters
from src.modeling.evaluation import evaluate_regression_model

from src.modeling.baselines import (
    evaluate_baseline_method,
    retrain_xgboost_validation_only,
    export_results_to_csv,
)

__all__ = [
    "train_baseline_model",
    "iterate_hyperparameters",
    "evaluate_regression_model",
    "evaluate_baseline_method",
    "retrain_xgboost_validation_only",
    "export_results_to_csv",
]
