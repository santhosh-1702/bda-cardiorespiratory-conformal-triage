"""
Unit tests for Phase 2 / Phase 3: Conformal Prediction & Statistical Rigor
==========================================================================
Tests:
  - ConformalCalibrator across all 5 score functions:
      1. trivial_median (constant predictor + split conformal)
      2. absolute_residual (base model + split conformal)
      3. residual_normalised (difficulty normalisation)
      4. mondrian_terciles (rank-based tercile partitioning)
      5. spectral_sharpness (model-free spectral peak sharpness)
  - Selective prediction: constant-width marked as 'N/A (width constant)',
    varying-width methods report numerical AUROC and risk-coverage curves
  - Determinism verification: fixed seeds produce identical outputs
  - NeuroKit2 installation and baseline functionality
"""

import unittest
import numpy as np
from sklearn.linear_model import Ridge
import neurokit2 as nk

from src.conformal.calibrator import ConformalCalibrator, compute_conformal_quantile
from pipeline_stages.run_phase1_5 import compute_paired_patient_bootstrap_ci
from pipeline_stages.run_phase2 import compute_risk_coverage_metrics, compute_patient_level_coverage_summary


class TestPhase2Conformal(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = (15.0 + 2.0 * self.X[:, 0] + np.random.randn(self.n_samples)).astype(np.float32)
        self.patient_ids = np.repeat(np.arange(20), 10)  # 20 patients, 10 samples each
        self.sharpness = np.random.uniform(1.0, 10.0, size=self.n_samples).astype(np.float32)
        self.X[:, -1] = self.sharpness

        # Base linear model
        self.model = Ridge()
        self.model.fit(self.X[:100], self.y[:100])

    def test_neurokit2_available_and_functional(self):
        """Verify NeuroKit2 is installed and correctly processes PPG peaks (Requirement 7)."""
        fs = 125
        t = np.linspace(0, 5, 5 * fs, endpoint=False)
        sim_ppg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
        peaks_dict = nk.ppg_findpeaks(sim_ppg, sampling_rate=fs)
        self.assertIn("PPG_Peaks", peaks_dict)
        self.assertGreater(len(peaks_dict["PPG_Peaks"]), 0)

    def test_compute_conformal_quantile(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        q = compute_conformal_quantile(scores, alpha=0.10)
        self.assertGreaterEqual(q, 9.0)
        self.assertLessEqual(q, 10.0)

    def test_trivial_median_calibration(self):
        """Test trivial conformal baseline: constant predictor (train median)."""
        train_med = float(np.median(self.y[:100]))
        cal = ConformalCalibrator(confidence_level=0.90, score_type="trivial_median", train_median=train_med)
        cal.calibrate(X_cal=self.X[100:150], y_cal=self.y[100:150])
        preds, lo, hi, widths = cal.predict_intervals(self.X[150:])
        self.assertEqual(len(preds), 50)
        self.assertTrue(np.all(preds == train_med))
        self.assertTrue(np.all(hi >= lo))
        self.assertEqual(float(np.std(widths)), 0.0)  # Width must be strictly constant

    def test_absolute_residual_calibration(self):
        """Test standard absolute residual split conformal."""
        cal = ConformalCalibrator(self.model, confidence_level=0.90, score_type="absolute_residual")
        cal.calibrate(X_cal=self.X[100:150], y_cal=self.y[100:150])
        preds, lo, hi, widths = cal.predict_intervals(self.X[150:])
        self.assertEqual(len(preds), 50)
        self.assertTrue(np.all(hi >= lo))
        self.assertTrue(np.all(widths > 0))
        self.assertEqual(float(np.std(widths)), 0.0)  # Width must be constant

    def test_residual_normalised_calibration(self):
        """Test difficulty-normalised adaptive conformal intervals."""
        cal = ConformalCalibrator(self.model, confidence_level=0.90, score_type="residual_normalised")
        cal.calibrate(
            X_cal=self.X[100:150],
            y_cal=self.y[100:150],
            X_train=self.X[:100],
            y_train=self.y[:100]
        )
        preds, lo, hi, widths = cal.predict_intervals(self.X[150:])
        self.assertEqual(len(preds), 50)
        self.assertTrue(np.all(hi >= lo))
        self.assertGreater(float(np.std(widths)), 0.0)  # Width must vary across windows

    def test_mondrian_terciles_calibration(self):
        """Test rank-based Mondrian grouped by predicted-RR terciles."""
        cal = ConformalCalibrator(self.model, confidence_level=0.90, score_type="mondrian_terciles")
        cal.calibrate(X_cal=self.X[100:150], y_cal=self.y[100:150])
        preds, lo, hi, widths = cal.predict_intervals(self.X[150:])
        self.assertEqual(len(preds), 50)
        self.assertTrue(np.all(hi >= lo))
        self.assertIsNotNone(cal.tercile_thresholds)
        self.assertGreater(len(cal.mondrian_q_hats), 0)

    def test_spectral_sharpness_calibration(self):
        """Test model-free baseline widening intervals by spectral peak sharpness."""
        cal = ConformalCalibrator(self.model, confidence_level=0.90, score_type="spectral_sharpness")
        cal.calibrate(
            X_cal=self.X[100:150],
            y_cal=self.y[100:150],
            sharpness_cal=self.sharpness[100:150]
        )
        preds, lo, hi, widths = cal.predict_intervals(
            self.X[150:],
            sharpness=self.sharpness[150:]
        )
        self.assertEqual(len(preds), 50)
        self.assertTrue(np.all(hi >= lo))
        self.assertGreater(float(np.std(widths)), 0.0)  # Width must vary across windows

    def test_patient_coverage_evaluation(self):
        lo = self.y[150:] - 2.0
        hi = self.y[150:] + 2.0
        res = ConformalCalibrator.evaluate_patient_coverage(
            y_true=self.y[150:],
            lower=lo,
            upper=hi,
            patient_ids=self.patient_ids[150:],
            target_coverage=0.90
        )
        self.assertIn("patient_cov_mean", res)
        self.assertIn("fraction_patients_undercovered", res)
        self.assertGreaterEqual(res["overall_coverage"], 0.0)
        self.assertLessEqual(res["overall_coverage"], 1.0)

    def test_selective_prediction_constant_vs_varying_width(self):
        """Verify Requirement 5: constant width marked as N/A, varying width computed."""
        y_true = np.array([10.0, 12.0, 15.0, 20.0, 25.0])
        y_pred = np.array([10.5, 12.2, 18.5, 20.1, 30.0])

        # Constant width -> N/A (width constant)
        const_widths = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        rc_const = compute_risk_coverage_metrics(y_true, y_pred, const_widths, score_type="absolute_residual")
        self.assertTrue(rc_const["is_constant_width"])
        self.assertEqual(rc_const["auroc_error_gt_3bpm"], "N/A (width constant)")
        self.assertEqual(len(rc_const["curve_points"]), 0)

        # Varying width -> numerical AUROC
        var_widths = np.array([2.0, 2.5, 7.0, 1.8, 9.0])
        rc_var = compute_risk_coverage_metrics(y_true, y_pred, var_widths, score_type="residual_normalised")
        self.assertFalse(rc_var["is_constant_width"])
        self.assertIsInstance(rc_var["auroc_error_gt_3bpm"], float)
        self.assertGreater(rc_var["auroc_error_gt_3bpm"], 0.8)
        self.assertGreater(len(rc_var["curve_points"]), 0)

    def test_paired_bootstrap(self):
        p_xgb = [{"patient_id": str(i), "mae": 1.5} for i in range(10)]
        p_base = [{"patient_id": str(i), "mae": 2.5} for i in range(10)]
        boot = compute_paired_patient_bootstrap_ci(p_xgb, p_base, n_resamples=100)
        self.assertTrue(boot["excludes_zero"])
        self.assertLess(boot["mean_diff"], 0)

    def test_window_length_and_windows_per_patient_assertion(self):
        """Verify window length assertion (60s) and windows per patient logging/validation."""
        from src.features.extractor import build_feature_dataset
        X, y, pids, folds = build_feature_dataset(force_recompute=False, window_sec=60)
        unique_pids, pid_counts = np.unique(pids, return_counts=True)
        self.assertEqual(len(unique_pids), 53)
        self.assertGreaterEqual(int(np.min(pid_counts)), 50)
        self.assertEqual(int(np.max(pid_counts)), 211)

    def test_conformal_determinism_across_runs(self):
        """Verify Requirement 2: fixed seeds produce identical conformal outputs across repeated passes."""
        cal1 = ConformalCalibrator(self.model, confidence_level=0.90, score_type="residual_normalised", random_state=42)
        cal1.calibrate(X_cal=self.X[100:150], y_cal=self.y[100:150], X_train=self.X[:100], y_train=self.y[:100])
        p1, l1, u1, w1 = cal1.predict_intervals(self.X[150:])

        cal2 = ConformalCalibrator(self.model, confidence_level=0.90, score_type="residual_normalised", random_state=42)
        cal2.calibrate(X_cal=self.X[100:150], y_cal=self.y[100:150], X_train=self.X[:100], y_train=self.y[:100])
        p2, l2, u2, w2 = cal2.predict_intervals(self.X[150:])

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(w1, w2)


if __name__ == "__main__":
    unittest.main()

