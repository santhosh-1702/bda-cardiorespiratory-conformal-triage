"""
Phase 4: Conformal Calibrator Module
====================================
Implements split conformal prediction for continuous respiratory rate monitoring.
Supports:
  1. 'trivial_median': constant predictor (train median) with absolute-residual split conformal
  2. 'absolute_residual': standard split conformal intervals on base estimator
  3. 'residual_normalised': adaptive intervals scaled by secondary difficulty model
  4. 'mondrian_terciles': rank-based Mondrian grouped by predicted-RR terciles
  5. 'spectral_sharpness': model-free baseline widening intervals by within-window spectral peak sharpness

Supports patient-disjoint calibration sets (no data leakage).
Never describes empirical coverage as 'guaranteed'; reports as 'target coverage'.
"""

from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from config.settings import (
    CONFORMAL_CONFIDENCE_LEVEL,
    RANDOM_SEED,
    TREE_METHOD
)


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Computes exact finite-sample conformal quantile:
    level = ceil((n + 1) * (1 - alpha)) / n
    """
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


class ConformalCalibrator:
    """
    Manages conformal calibration and interval prediction for cardiorespiratory triage.
    Supports trivial baseline, absolute residual, difficulty-normalised, Mondrian terciles,
    and model-free spectral peak sharpness score functions.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        confidence_level: float = CONFORMAL_CONFIDENCE_LEVEL,
        score_type: str = "residual_normalised",
        rr_bins: Optional[List[float]] = None,
        train_median: Optional[float] = None,
        n_jobs: int = 1,
        random_state: int = RANDOM_SEED
    ):
        self.base_estimator = base_estimator
        self.confidence_level = confidence_level
        self.score_type = score_type
        self.rr_bins = rr_bins
        self.train_median = train_median
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.residual_estimator = None
        self.q_hat = None
        self.mondrian_q_hats = {}
        self.tercile_thresholds = None
        self.median_cal_dispersion = 1.0
        self.is_calibrated = False
        self.is_oof_residuals_trained = False

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        sharpness_cal: Optional[np.ndarray] = None,
        patient_ids_train: Optional[np.ndarray] = None,
        oof_residuals_train: Optional[np.ndarray] = None
    ) -> None:
        """
        Calibrates on an independent, patient-disjoint calibration set (X_cal, y_cal).
        Calibration data comes strictly from the fold's calibration patients.
        For difficulty-normalised conformal, the secondary difficulty estimator is trained
        strictly on grouped out-of-fold residuals (never in-sample residuals).
        """
        alpha = 1.0 - self.confidence_level

        if self.score_type == "trivial_median":
            if self.train_median is None:
                if y_train is not None and len(y_train) > 0:
                    self.train_median = float(np.median(y_train))
                else:
                    self.train_median = float(np.median(y_cal))
            y_cal_preds = np.full(len(y_cal), self.train_median, dtype=np.float32)
            raw_residuals = np.abs(y_cal - y_cal_preds)
            self.q_hat = compute_conformal_quantile(raw_residuals, alpha)

        elif self.score_type == "absolute_residual":
            if self.base_estimator is None:
                raise ValueError("base_estimator must be provided for 'absolute_residual'")
            y_cal_preds = self.base_estimator.predict(X_cal)
            raw_residuals = np.abs(y_cal - y_cal_preds)
            self.q_hat = compute_conformal_quantile(raw_residuals, alpha)

        elif self.score_type == "residual_normalised":
            if self.base_estimator is None:
                raise ValueError("base_estimator must be provided for 'residual_normalised'")
            y_cal_preds = self.base_estimator.predict(X_cal)
            raw_residuals = np.abs(y_cal - y_cal_preds)

            # Fit second model (difficulty estimator) on train patients: E[|y - y_hat| | x]
            # Strictly using grouped out-of-fold residuals, NEVER in-sample residuals.
            if oof_residuals_train is not None and X_train is not None:
                fit_X = X_train
                fit_r = oof_residuals_train
                self.is_oof_residuals_trained = True
            elif X_train is not None and y_train is not None:
                oof_residuals = np.zeros(len(y_train), dtype=np.float32)
                if patient_ids_train is not None and len(np.unique(patient_ids_train)) >= 2:
                    n_inner = min(3, len(np.unique(patient_ids_train)))
                    inner_cv = GroupKFold(n_splits=n_inner)
                    splits = inner_cv.split(X_train, y_train, groups=patient_ids_train)
                else:
                    n_inner = min(3, max(2, len(X_train) // 2))
                    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=self.random_state)
                    splits = inner_cv.split(X_train, y_train)

                for inner_tr_idx, inner_val_idx in splits:
                    inner_model = clone(self.base_estimator)
                    inner_model.fit(X_train[inner_tr_idx], y_train[inner_tr_idx])
                    inner_preds = inner_model.predict(X_train[inner_val_idx])
                    oof_residuals[inner_val_idx] = np.abs(y_train[inner_val_idx] - inner_preds)

                fit_X = X_train
                fit_r = oof_residuals
                self.is_oof_residuals_trained = True
            else:
                fit_X = X_cal
                fit_r = raw_residuals
                self.is_oof_residuals_trained = False

            self.residual_estimator = xgb.XGBRegressor(
                max_depth=3,
                n_estimators=50,
                learning_rate=0.08,
                tree_method=TREE_METHOD,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=0
            )
            self.residual_estimator.fit(fit_X, fit_r)

            # Predict calibration dispersion (bounded below by 0.10)
            sigma_cal = np.maximum(self.residual_estimator.predict(X_cal), 0.10)
            norm_scores = raw_residuals / sigma_cal
            self.q_hat = compute_conformal_quantile(norm_scores, alpha)

        elif self.score_type in ["mondrian_terciles", "mondrian"]:
            if self.base_estimator is None:
                raise ValueError("base_estimator must be provided for Mondrian methods")
            y_cal_preds = self.base_estimator.predict(X_cal)
            raw_residuals = np.abs(y_cal - y_cal_preds)
            self.q_hat = compute_conformal_quantile(raw_residuals, alpha)

            if self.score_type == "mondrian_terciles" or self.rr_bins is None:
                # Rank-based terciles computed from calibration predictions
                t1 = float(np.quantile(y_cal_preds, 1.0 / 3.0))
                t2 = float(np.quantile(y_cal_preds, 2.0 / 3.0))
                self.tercile_thresholds = (t1, t2)

                cal_bins = np.zeros(len(y_cal_preds), dtype=int)
                cal_bins[y_cal_preds > t1] = 1
                cal_bins[y_cal_preds > t2] = 2

                for b in [0, 1, 2]:
                    mask = cal_bins == b
                    if np.sum(mask) >= 10:
                        self.mondrian_q_hats[b] = compute_conformal_quantile(raw_residuals[mask], alpha)
                    else:
                        self.mondrian_q_hats[b] = self.q_hat
            else:
                # Clinically specified fixed RR bins (legacy support)
                cal_bin_ids = np.digitize(y_cal_preds, self.rr_bins)
                n_bins = len(self.rr_bins) + 1
                for b in range(n_bins):
                    b_mask = cal_bin_ids == b
                    if np.sum(b_mask) >= 10:
                        self.mondrian_q_hats[b] = compute_conformal_quantile(raw_residuals[b_mask], alpha)
                    else:
                        self.mondrian_q_hats[b] = self.q_hat

        elif self.score_type == "spectral_sharpness":
            if self.base_estimator is None:
                raise ValueError("base_estimator must be provided for 'spectral_sharpness'")
            y_cal_preds = self.base_estimator.predict(X_cal)
            raw_residuals = np.abs(y_cal - y_cal_preds)

            if sharpness_cal is None:
                # Fallback: assume sharpness is in the last column of X_cal
                sharpness_cal = X_cal[:, -1]

            # Dispersion inversely proportional to square-root of spectral peak sharpness
            disp_cal = 1.0 / np.sqrt(np.maximum(sharpness_cal, 1e-4))
            self.median_cal_dispersion = float(np.median(disp_cal))
            if self.median_cal_dispersion <= 1e-6:
                self.median_cal_dispersion = 1.0

            scaled_disp_cal = disp_cal / self.median_cal_dispersion
            sharp_scores = raw_residuals / np.maximum(scaled_disp_cal, 0.10)
            self.q_hat = compute_conformal_quantile(sharp_scores, alpha)

        else:
            raise ValueError(
                f"Unknown score_type '{self.score_type}'. "
                "Must be 'trivial_median', 'absolute_residual', 'residual_normalised', "
                "'mondrian_terciles', or 'spectral_sharpness'."
            )

        self.is_calibrated = True

    def fit_and_calibrate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        split_ratio: float = 0.20
    ) -> None:
        """
        Partition X_train into fit and calibration sets for standalone testing.
        """
        from sklearn.model_selection import train_test_split
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train, y_train, test_size=split_ratio, random_state=self.random_state
        )
        if self.base_estimator is not None and hasattr(self.base_estimator, "fit"):
            self.base_estimator.fit(X_fit, y_fit)
        self.calibrate(X_cal=X_cal, y_cal=y_cal, X_train=X_fit, y_train=y_fit)

    def predict_intervals(
        self,
        X: np.ndarray,
        sharpness: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates point predictions and conformal prediction intervals.
        Returns:
            (predictions, lower_bounds, upper_bounds, interval_widths)
        """
        if not self.is_calibrated:
            raise RuntimeError("Calibrator must be calibrated before generating intervals.")

        if self.score_type == "trivial_median":
            preds = np.full(len(X), self.train_median, dtype=np.float32)
            half_width = self.q_hat
            lower = preds - half_width
            upper = preds + half_width
            widths = np.full(len(X), 2.0 * half_width, dtype=np.float32)
            return preds, lower, upper, widths

        preds = self.base_estimator.predict(X)

        if self.score_type == "absolute_residual":
            half_width = self.q_hat
            lower = preds - half_width
            upper = preds + half_width
            widths = np.full(len(X), 2.0 * half_width, dtype=np.float32)

        elif self.score_type == "residual_normalised":
            sigma_test = np.maximum(self.residual_estimator.predict(X), 0.10)
            half_width = self.q_hat * sigma_test
            lower = preds - half_width
            upper = preds + half_width
            widths = 2.0 * half_width

        elif self.score_type == "mondrian_terciles":
            t1, t2 = self.tercile_thresholds
            test_bins = np.zeros(len(preds), dtype=int)
            test_bins[preds > t1] = 1
            test_bins[preds > t2] = 2
            half_width = np.array([self.mondrian_q_hats.get(b, self.q_hat) for b in test_bins], dtype=np.float32)
            lower = preds - half_width
            upper = preds + half_width
            widths = 2.0 * half_width

        elif self.score_type == "mondrian":
            test_bin_ids = np.digitize(preds, self.rr_bins)
            half_width = np.array([self.mondrian_q_hats.get(b, self.q_hat) for b in test_bin_ids], dtype=np.float32)
            lower = preds - half_width
            upper = preds + half_width
            widths = 2.0 * half_width

        elif self.score_type == "spectral_sharpness":
            if sharpness is None:
                sharpness = X[:, -1]
            disp_test = 1.0 / np.sqrt(np.maximum(sharpness, 1e-4))
            scaled_disp_test = np.maximum(disp_test / self.median_cal_dispersion, 0.10)
            half_width = self.q_hat * scaled_disp_test
            lower = preds - half_width
            upper = preds + half_width
            widths = 2.0 * half_width

        else:
            raise ValueError(f"Unknown score_type '{self.score_type}'.")

        return preds, lower, upper, widths

    @staticmethod
    def evaluate_patient_coverage(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        patient_ids: np.ndarray,
        target_coverage: float = CONFORMAL_CONFIDENCE_LEVEL
    ) -> Dict[str, Any]:
        """
        Evaluates empirical coverage globally and per-patient.
        Computes mean, std, min per-patient coverage, and the fraction of test patients
        whose empirical coverage falls below target_coverage - 0.05.
        """
        covered = (y_true >= lower) & (y_true <= upper)
        overall_cov = float(np.mean(covered))

        unique_pids = np.unique(patient_ids)
        patient_covs = []
        for pid in unique_pids:
            mask = patient_ids == pid
            if np.sum(mask) > 0:
                p_cov = float(np.mean(covered[mask]))
                patient_covs.append(p_cov)

        patient_covs = np.array(patient_covs)
        mean_p_cov = float(np.mean(patient_covs))
        std_p_cov = float(np.std(patient_covs))
        min_p_cov = float(np.min(patient_covs))
        undercovered_threshold = target_coverage - 0.05
        frac_undercovered = float(np.mean(patient_covs < undercovered_threshold))

        return {
            "overall_coverage": overall_cov,
            "patient_cov_mean": mean_p_cov,
            "patient_cov_std": std_p_cov,
            "patient_cov_min": min_p_cov,
            "target_coverage": target_coverage,
            "undercovered_threshold": undercovered_threshold,
            "fraction_patients_undercovered": frac_undercovered,
            "per_patient_coverages": {str(pid): float(c) for pid, c in zip(unique_pids, patient_covs)},
        }
