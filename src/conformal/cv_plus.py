"""
Phase 7: Jackknife+ / CV+ Conformal Prediction Module
=====================================================
Implements Jackknife+ / CV+ cross-conformal prediction (Barber, Candès, Ramdas & Tibshirani, 2021)
for continuous cardiorespiratory monitoring.

Instead of carving out a small dedicated calibration subset (8-9 patients), CV+ pools
out-of-fold residuals across all non-test patients (~42-44 patients) via inner GroupKFold
cross-fitting, testing whether a 5x larger calibration sample stabilizes fold-to-fold coverage spread.

Enforces strict patient-disjoint splits: no patient ever appears on both sides of any inner fold.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from config.settings import (
    CONFORMAL_CONFIDENCE_LEVEL,
    RANDOM_SEED,
    TREE_METHOD
)


def compute_cv_plus_quantiles(
    L: np.ndarray,
    U: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Computes exact finite-sample CV+ / Jackknife+ prediction intervals
    according to Barber et al. (2021) Section 3 / Definition of CV+.

    Parameters:
        L: (n_pool, n_test) array of lower candidate values:
           L[i, j] = \hat{\mu}_{-k(i)}(X_j) - R_i
        U: (n_pool, n_test) array of upper candidate values:
           U[i, j] = \hat{\mu}_{-k(i)}(X_j) + R_i
        alpha: significance level (e.g. 0.10 for 90% confidence)

    Returns:
        (lower_bounds, upper_bounds, interval_widths)
    """
    n_pool = L.shape[0]
    if n_pool == 0:
        n_test = L.shape[1]
        return np.zeros(n_test, dtype=np.float32), np.zeros(n_test, dtype=np.float32), np.zeros(n_test, dtype=np.float32)

    # Barber et al. (2021) finite-sample quantiles:
    # Lower bound: \lfloor \alpha (n + 1) \rfloor-th smallest value (1-indexed)
    idx_lower = max(0, min(n_pool - 1, int(np.floor(alpha * (n_pool + 1))) - 1))
    # Upper bound: \lceil (1 - \alpha) (n + 1) \rceil-th smallest value (1-indexed)
    idx_upper = max(0, min(n_pool - 1, int(np.ceil((1.0 - alpha) * (n_pool + 1))) - 1))

    # Fast partitioned quantile selection along non-test dimension (axis 0)
    lower_bounds = np.partition(L, idx_lower, axis=0)[idx_lower, :].astype(np.float32)
    upper_bounds = np.partition(U, idx_upper, axis=0)[idx_upper, :].astype(np.float32)
    widths = (upper_bounds - lower_bounds).astype(np.float32)

    return lower_bounds, upper_bounds, widths


class CVPlusCalibrator:
    """
    Manages inner K-fold cross-fitting and CV+ prediction intervals
    over pooled non-test patients with strict patient disjointness.
    """

    def __init__(
        self,
        n_inner_splits: int = 5,
        confidence_level: float = CONFORMAL_CONFIDENCE_LEVEL,
        tree_method: str = TREE_METHOD,
        estimator_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = 1,
        random_state: int = RANDOM_SEED
    ):
        self.n_inner_splits = n_inner_splits
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.random_state = random_state

        if estimator_params is None:
            self.estimator_params = {
                "max_depth": 3,
                "learning_rate": 0.06,
                "n_estimators": 100,
                "reg_alpha": 1.0,
                "reg_lambda": 3.0,
                "subsample": 0.85,
                "colsample_bytree": 0.80
            }
        else:
            self.estimator_params = dict(estimator_params)

        self.inner_models: List[xgb.XGBRegressor] = []
        self.oof_residuals: Optional[np.ndarray] = None
        self.assigned_inner_folds: Optional[np.ndarray] = None
        self.inner_val_pids_by_fold: Dict[int, List[str]] = {}
        self.inner_train_pids_by_fold: Dict[int, List[str]] = {}
        self.n_pool: int = 0
        self.is_calibrated: bool = False

    def fit_and_cross_calibrate(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        patient_ids_pool: np.ndarray
    ) -> None:
        """
        Executes inner K-fold cross-fitting over all non-test patients (train+val+cal).
        Guarantees that a patient never appears on both sides of any inner split.
        Computes one out-of-fold residual for every non-test window.
        """
        self.n_pool = len(y_pool)
        gkf = GroupKFold(n_splits=self.n_inner_splits)

        oof_preds = np.zeros(self.n_pool, dtype=np.float32)
        self.assigned_inner_folds = np.zeros(self.n_pool, dtype=int)
        self.inner_models = []
        self.inner_val_pids_by_fold = {}
        self.inner_train_pids_by_fold = {}

        splits = list(gkf.split(X_pool, y_pool, groups=patient_ids_pool))

        for k, (tr_idx, val_idx) in enumerate(splits):
            tr_pids = sorted(list(set(patient_ids_pool[tr_idx])))
            val_pids = sorted(list(set(patient_ids_pool[val_idx])))

            # Audit zero patient leakage across inner split
            overlap = set(tr_pids) & set(val_pids)
            assert len(overlap) == 0, f"Inner patient leakage detected in fold {k}: {overlap}"

            self.inner_train_pids_by_fold[k] = tr_pids
            self.inner_val_pids_by_fold[k] = val_pids
            self.assigned_inner_folds[val_idx] = k

            # Fit inner-fold model excluding fold k's patients
            params = {
                "tree_method": self.tree_method,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state + k,
                "verbosity": 0,
                **self.estimator_params
            }
            m = xgb.XGBRegressor(**params)
            m.fit(X_pool[tr_idx], y_pool[tr_idx])

            # Predict strictly out-of-fold for patients never seen during training
            oof_preds[val_idx] = m.predict(X_pool[val_idx])
            self.inner_models.append(m)

        # OOF residuals: R_i = |y_i - \hat{\mu}_{-k(i)}(X_i)|
        self.oof_residuals = np.abs(y_pool - oof_preds).astype(np.float32)
        self.is_calibrated = True

    def predict_intervals(
        self,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates ensemble point predictions and CV+ prediction intervals
        for test samples.

        Returns:
            (predictions, lower_bounds, upper_bounds, interval_widths)
        """
        if not self.is_calibrated:
            raise RuntimeError("CVPlusCalibrator must be cross-calibrated before predicting intervals.")

        n_test = len(X_test)
        # Predictions from each inner model at test points
        P_test = np.zeros((self.n_inner_splits, n_test), dtype=np.float32)
        for k, m in enumerate(self.inner_models):
            P_test[k] = m.predict(X_test)

        # Point prediction is the ensemble mean of inner cross-fitted models
        ens_preds = np.mean(P_test, axis=0).astype(np.float32)

        # Expand P_test according to each training point's inner fold
        # P_expanded[i, j] = \hat{\mu}_{-k(i)}(X_j)
        P_expanded = P_test[self.assigned_inner_folds]  # (n_pool, n_test)
        R_expanded = self.oof_residuals[:, np.newaxis]   # (n_pool, 1)

        L = P_expanded - R_expanded
        U = P_expanded + R_expanded

        lower_bounds, upper_bounds, widths = compute_cv_plus_quantiles(L, U, self.alpha)

        return ens_preds, lower_bounds, upper_bounds, widths
