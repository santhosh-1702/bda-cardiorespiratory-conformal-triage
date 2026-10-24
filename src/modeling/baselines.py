"""
Phase 1.5: Classical & Constant Baselines and XGBoost Retraining Module
=======================================================================
Evaluates:
  1. Constant baselines on patient-grouped folds: predict train mean, predict train median.
  2. Classical spectral baseline (no ML): dominant frequency of respiratory-band PPG.
  3. Respiratory-induced variations: RIAV, RIIV, RIFV, and simple median fusion.
  4. Across window lengths: 6s, 15s, 30s, 60s.
  5. Retrained XGBoost on features from the best representation and window length,
     with validation-only hyperparameter selection (strictly no tuning on test folds).
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import csv
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

from config.settings import CPU_THREADS, TREE_METHOD, RANDOM_SEED, OUTPUT_DIR

# Candidate architectures for validation-only selection
CANDIDATE_HYPERPARAMS = [
    {
        "name": "Cand A (d=2, lr=0.05, n=80, a=1.0, l=3.0)",
        "params": {
            "max_depth": 2, "learning_rate": 0.05, "n_estimators": 80,
            "reg_alpha": 1.0, "reg_lambda": 3.0, "subsample": 0.85, "colsample_bytree": 0.80
        }
    },
    {
        "name": "Cand B (d=3, lr=0.04, n=100, a=1.5, l=3.0)",
        "params": {
            "max_depth": 3, "learning_rate": 0.04, "n_estimators": 100,
            "reg_alpha": 1.5, "reg_lambda": 3.0, "subsample": 0.85, "colsample_bytree": 0.85
        }
    },
    {
        "name": "Cand C (d=3, lr=0.08, n=100, a=0.5, l=2.0)",
        "params": {
            "max_depth": 3, "learning_rate": 0.08, "n_estimators": 100,
            "reg_alpha": 0.5, "reg_lambda": 2.0, "subsample": 0.85, "colsample_bytree": 0.80
        }
    },
    {
        "name": "Cand D (d=4, lr=0.03, n=100, a=3.0, l=5.0)",
        "params": {
            "max_depth": 4, "learning_rate": 0.03, "n_estimators": 100,
            "reg_alpha": 3.0, "reg_lambda": 5.0, "subsample": 0.80, "colsample_bytree": 0.75
        }
    },
    {
        "name": "Cand E (d=2, lr=0.03, n=120, a=2.0, l=4.0)",
        "params": {
            "max_depth": 2, "learning_rate": 0.03, "n_estimators": 120,
            "reg_alpha": 2.0, "reg_lambda": 4.0, "subsample": 0.80, "colsample_bytree": 0.85
        }
    }
]

# ---------------------------------------------------------------------------
# Extended metrics helper
# ---------------------------------------------------------------------------

def compute_extended_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes MAE, R2, Pearson r, within-patient MAE, and per-patient MAE
    for a single fold.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Pearson r — undefined when prediction has zero variance (constant baselines)
    try:
        if np.std(y_pred) < 1e-12:
            pear_r = float("nan")
        else:
            pear_r, _ = pearsonr(y_true, y_pred)
            pear_r = float(pear_r)
    except Exception:
        pear_r = float("nan")

    # Within-patient MAE: subtract each patient's mean prediction error
    unique_pids = np.unique(patient_ids_test)
    corrected_pred = y_pred.copy().astype(np.float64)
    per_patient = []

    for pid in unique_pids:
        mask = patient_ids_test == pid
        y_p = y_true[mask]
        p_p = y_pred[mask]
        patient_bias = float(np.mean(p_p - y_p))
        corrected_pred[mask] -= patient_bias
        patient_mae = float(mean_absolute_error(y_p, p_p))
        per_patient.append({
            "patient_id": str(pid),
            "mae": patient_mae,
            "n_windows": int(np.sum(mask)),
        })

    within_patient_mae = float(mean_absolute_error(y_true, corrected_pred))

    return {
        "mae": mae,
        "r2": r2,
        "pearson_r": pear_r,
        "within_patient_mae": within_patient_mae,
        "per_patient": per_patient,
    }


def evaluate_baseline_method(
    method_name: str,
    y_true: np.ndarray,
    patient_ids: np.ndarray,
    folds: List[Dict[str, Any]],
    precomputed_preds: Optional[np.ndarray] = None,
    predict_fn: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates a single method across all 5 patient-grouped folds.
    Returns dictionary of mean +/- std MAE, R2, Pearson r, within-patient MAE,
    per-fold metrics, and per-patient data.
    """
    fold_maes = []
    fold_r2s = []
    fold_pearson_rs = []
    fold_within_patient_maes = []
    all_per_patient = []

    for f in folds:
        test_mask = np.isin(patient_ids, f["test_pids"])
        y_test = y_true[test_mask]
        test_pids = patient_ids[test_mask]

        if predict_fn == "train_mean":
            train_mask = np.isin(patient_ids, f["train_pids"])
            pred = np.full_like(y_test, np.mean(y_true[train_mask]))
        elif predict_fn == "train_median":
            train_mask = np.isin(patient_ids, f["train_pids"])
            pred = np.full_like(y_test, np.median(y_true[train_mask]))
        elif precomputed_preds is not None:
            pred = precomputed_preds[test_mask]
        else:
            raise ValueError("Either precomputed_preds or predict_fn must be provided.")

        metrics = compute_extended_fold_metrics(y_test, pred, test_pids)
        fold_maes.append(metrics["mae"])
        fold_r2s.append(metrics["r2"])
        fold_pearson_rs.append(metrics["pearson_r"])
        fold_within_patient_maes.append(metrics["within_patient_mae"])

        for pp in metrics["per_patient"]:
            pp["fold"] = f["fold"]
            pp["method"] = method_name
        all_per_patient.extend(metrics["per_patient"])

    return {
        "method": method_name,
        "mae_mean": float(np.mean(fold_maes)),
        "mae_std": float(np.std(fold_maes)),
        "r2_mean": float(np.mean(fold_r2s)),
        "r2_std": float(np.std(fold_r2s)),
        "pearson_r_mean": float(np.nanmean(fold_pearson_rs)),
        "pearson_r_std": float(np.nanstd(fold_pearson_rs)),
        "within_patient_mae_mean": float(np.mean(fold_within_patient_maes)),
        "within_patient_mae_std": float(np.std(fold_within_patient_maes)),
        "fold_maes": fold_maes,
        "fold_r2s": fold_r2s,
        "fold_pearson_rs": fold_pearson_rs,
        "fold_within_patient_maes": fold_within_patient_maes,
        "per_patient": all_per_patient,
    }

def retrain_xgboost_validation_only(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    folds: List[Dict[str, Any]],
    candidates: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Trains XGBoost using validation-only hyperparameter selection across 5 patient-grouped folds.
    Never tunes on test folds.  Returns extended metrics including Pearson r,
    within-patient MAE, and per-patient results.
    """
    if candidates is None:
        candidates = CANDIDATE_HYPERPARAMS

    fold_maes = []
    fold_r2s = []
    fold_pearson_rs = []
    fold_within_patient_maes = []
    selected_cands = []
    best_models = []
    all_per_patient = []

    for f in folds:
        train_mask = np.isin(patient_ids, f["train_pids"])
        val_mask = np.isin(patient_ids, f["val_pids"])
        test_mask = np.isin(patient_ids, f["test_pids"])

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_pids = patient_ids[test_mask]

        best_val_mae = float("inf")
        best_cand = None
        best_model = None

        # Validation set selection only
        for cand in candidates:
            params = {
                "tree_method": TREE_METHOD,
                "n_jobs": CPU_THREADS,
                "random_state": RANDOM_SEED,
                "verbosity": 0,
                **cand["params"]
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, val_preds))

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_cand = cand
                best_model = model

        # Strictly evaluate selected model on held-out test fold
        test_preds = best_model.predict(X_test)
        metrics = compute_extended_fold_metrics(y_test, test_preds, test_pids)

        fold_maes.append(metrics["mae"])
        fold_r2s.append(metrics["r2"])
        fold_pearson_rs.append(metrics["pearson_r"])
        fold_within_patient_maes.append(metrics["within_patient_mae"])
        selected_cands.append(best_cand["name"])
        best_models.append(best_model)

        for pp in metrics["per_patient"]:
            pp["fold"] = f["fold"]
            pp["method"] = "Retrained XGBoost"
        all_per_patient.extend(metrics["per_patient"])

    return {
        "method": "Retrained XGBoost",
        "mae_mean": float(np.mean(fold_maes)),
        "mae_std": float(np.std(fold_maes)),
        "r2_mean": float(np.mean(fold_r2s)),
        "r2_std": float(np.std(fold_r2s)),
        "pearson_r_mean": float(np.nanmean(fold_pearson_rs)),
        "pearson_r_std": float(np.nanstd(fold_pearson_rs)),
        "within_patient_mae_mean": float(np.mean(fold_within_patient_maes)),
        "within_patient_mae_std": float(np.std(fold_within_patient_maes)),
        "fold_maes": fold_maes,
        "fold_r2s": fold_r2s,
        "fold_pearson_rs": fold_pearson_rs,
        "fold_within_patient_maes": fold_within_patient_maes,
        "selected_candidates": selected_cands,
        "models": best_models,
        "per_patient": all_per_patient,
    }

def export_results_to_csv(
    results_list: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Exports the benchmark comparison table to CSV with extended metrics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Method", "Window_Length_s",
        "MAE_mean", "MAE_std",
        "R2_mean", "R2_std",
        "Pearson_r_mean", "Pearson_r_std",
        "Within_Patient_MAE_mean", "Within_Patient_MAE_std",
        "MAE_fold0", "MAE_fold1", "MAE_fold2", "MAE_fold3", "MAE_fold4"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results_list:
            row = {
                "Method": res["method"],
                "Window_Length_s": res.get("window_sec", ""),
                "MAE_mean": f"{res['mae_mean']:.4f}",
                "MAE_std": f"{res['mae_std']:.4f}",
                "R2_mean": f"{res['r2_mean']:.4f}",
                "R2_std": f"{res['r2_std']:.4f}",
                "Pearson_r_mean": f"{res.get('pearson_r_mean', float('nan')):.4f}",
                "Pearson_r_std": f"{res.get('pearson_r_std', float('nan')):.4f}",
                "Within_Patient_MAE_mean": f"{res.get('within_patient_mae_mean', float('nan')):.4f}",
                "Within_Patient_MAE_std": f"{res.get('within_patient_mae_std', float('nan')):.4f}",
                "MAE_fold0": f"{res['fold_maes'][0]:.4f}",
                "MAE_fold1": f"{res['fold_maes'][1]:.4f}",
                "MAE_fold2": f"{res['fold_maes'][2]:.4f}",
                "MAE_fold3": f"{res['fold_maes'][3]:.4f}",
                "MAE_fold4": f"{res['fold_maes'][4]:.4f}",
            }
            writer.writerow(row)


def export_per_patient_results(
    results_list: List[Dict[str, Any]],
    output_path: Path,
    window_sec: Optional[int] = None
) -> None:
    """
    Exports per-patient MAE to CSV.
    Aggregates per_patient data from all results in results_list.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "fold", "method", "window_sec", "mae", "n_windows"]

    rows = []
    for res in results_list:
        for pp in res.get("per_patient", []):
            rows.append({
                "patient_id": pp["patient_id"],
                "fold": pp["fold"],
                "method": pp["method"],
                "window_sec": window_sec if window_sec else res.get("window_sec", ""),
                "mae": f"{pp['mae']:.4f}",
                "n_windows": pp["n_windows"],
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
