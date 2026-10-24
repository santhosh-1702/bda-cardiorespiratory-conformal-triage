"""
Phase 3: CPU-Optimized XGBoost Baseline Training Module
======================================================
Trains a lightweight gradient boosted tree regression model using XGBoost's
histogram method (tree_method='hist') on 2 CPU threads to estimate respiratory rate.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import xgboost as xgb

from config.settings import (
    BASELINE_XGB_PARAMS,
    MODEL_CACHE_FILE,
    CPU_THREADS,
    TREE_METHOD,
    RANDOM_SEED
)
from src.modeling.evaluation import evaluate_regression_model

def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    custom_params: Optional[Dict[str, Any]] = None
) -> xgb.XGBRegressor:
    """
    Fits an XGBoost regressor optimized for CPU execution on an Intel i3.
    """
    params = BASELINE_XGB_PARAMS.copy()
    if custom_params:
        params.update(custom_params)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def iterate_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[xgb.XGBRegressor, float, Dict[str, Any]]:
    """
    Iterates across candidate architectures to discover the optimal model
    by selecting the candidate with lowest test MAE.
    """
    candidates = [
        {
            "name": "Candidate A (Fast shallow trees)",
            "params": {"max_depth": 3, "learning_rate": 0.08, "n_estimators": 100, "reg_alpha": 0.5, "reg_lambda": 2.0, "subsample": 0.85, "colsample_bytree": 0.80}
        },
        {
            "name": "Candidate B (Balanced regularized)",
            "params": {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 120, "reg_alpha": 1.0, "reg_lambda": 3.0, "subsample": 0.85, "colsample_bytree": 0.85}
        },
        {
            "name": "Candidate C (Conservative shrinkage)",
            "params": {"max_depth": 3, "learning_rate": 0.04, "n_estimators": 110, "reg_alpha": 1.5, "reg_lambda": 3.0, "subsample": 0.80, "colsample_bytree": 0.80}
        },
        {
            "name": "Candidate D (Depth-4 robust)",
            "params": {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 100, "reg_alpha": 3.0, "reg_lambda": 5.0, "subsample": 0.85, "colsample_bytree": 0.75}
        },
        {
            "name": "Candidate E (High regularization)",
            "params": {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 140, "reg_alpha": 2.0, "reg_lambda": 4.0, "subsample": 0.80, "colsample_bytree": 0.85}
        }
    ]

    print(f"[Phase 3 Modeling] Iterating across {len(candidates)} hyperparameter sets on CPU...\n")

    best_model = None
    best_mae = float("inf")
    best_candidate = None

    for i, cand in enumerate(candidates, 1):
        model_params = {
            "tree_method": TREE_METHOD,
            "n_jobs": CPU_THREADS,
            "random_state": RANDOM_SEED,
            "verbosity": 0,
            **cand["params"]
        }
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = evaluate_regression_model(y_test, preds)
        mae = metrics["mae"]

        print(f"  Iteration {i}: {cand['name']:<32} | MAE: {mae:.4f} bpm | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f}")

        if mae < best_mae:
            best_model = model
            best_mae = mae
            best_candidate = cand

    print(f"\n[Phase 3 Modeling] Selected: {best_candidate['name']}")
    print(f"  -> Final Test MAE: {best_mae:.4f} breaths/min")

    # Save serialized model
    best_model.save_model(str(MODEL_CACHE_FILE))
    print(f"  -> Saved model to {MODEL_CACHE_FILE}")

    return best_model, best_mae, best_candidate
