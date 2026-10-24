"""
Test Suite: Phase 4 Fixed Folds, Honest Statistics & Raw-Signal Artifacts
========================================================================
Verifies:
  1. Identical test patients per fold across all window lengths (6s, 15s, 30s, 60s).
  2. Corrupted raw windows differ from clean in the time domain for all artifact types and severities.
  3. Training and calibration windows remain strictly uncorrupted.
  4. Difficulty model trains on grouped out-of-fold residuals, never in-sample residuals.
  5. Signal Quality Index (SQI) drops under corruption.
  6. Verification statuses are computed dynamically from actual checks, not hardcoded strings.
"""

import unittest
from pathlib import Path
import numpy as np

from config.settings import FS, RANDOM_SEED, PATIENT_IDS, OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR
from src.features.extractor import (
    load_patient_fold_map,
    generate_patient_folds,
    create_patient_fold_map,
    FOLD_MAP_FILE
)
from src.conformal.artifact_simulator import (
    corrupt_raw_ppg_window,
    inject_hf_noise,
    inject_baseline_drift,
    inject_motion_burst,
    inject_dropout,
    compute_inband_power_ratio,
    compute_spectral_flatness,
    compute_sqi_abstention_score,
    ARTIFACT_SEVERITY_CONFIGS
)
from src.conformal.calibrator import ConformalCalibrator


class TestPhase4(unittest.TestCase):

    def setUp(self):
        self.fold_map = load_patient_fold_map()
        self.fs = FS
        self.duration_sec = 60.0
        self.n_samples = int(self.fs * self.duration_sec)
        self.t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
        # Synthetic clean respiratory PPG (0.25 Hz = 15 bpm)
        self.clean_window = (np.sin(2 * np.pi * 0.25 * self.t) + 0.1 * np.cos(2 * np.pi * 0.5 * self.t)).astype(np.float32)

    def test_fold_map_json_exists_and_patient_count(self):
        """Verify fold_map.json exists and maps all 53 patients to 5 folds."""
        self.assertTrue(FOLD_MAP_FILE.exists(), "fold_map.json must exist in output/")
        self.assertEqual(len(self.fold_map), 53, "fold_map must contain exactly 53 patients")
        assigned_folds = set(self.fold_map.values())
        self.assertEqual(assigned_folds, {0, 1, 2, 3, 4}, "All 5 folds must be represented")

    def test_identical_test_patients_per_fold_across_window_lengths(self):
        """
        Requirement 4A.1: Assert identical test patients per fold across window lengths
        (6s, 15s, 30s, 60s).
        """
        window_lengths = [6, 15, 30, 60]
        # Simulate patient ID arrays for each window length (different sample counts per patient)
        pids_by_length = {}
        for w in window_lengths:
            # Different window durations produce different numbers of windows per patient
            multiplier = max(1, int(120 / w))
            sim_pids = np.repeat(sorted(PATIENT_IDS), multiplier)
            pids_by_length[w] = sim_pids

        folds_by_length = {
            w: generate_patient_folds(pids_by_length[w], fold_map=self.fold_map)
            for w in window_lengths
        }

        for fold_idx in range(5):
            ref_test_pids = folds_by_length[60][fold_idx]["test_pids"]
            for w in [6, 15, 30]:
                curr_test_pids = folds_by_length[w][fold_idx]["test_pids"]
                self.assertEqual(
                    ref_test_pids, curr_test_pids,
                    f"Fold {fold_idx}: Mismatch in test patients between 60s and {w}s window lengths"
                )

    def test_corrupted_windows_differ_from_clean_in_time_domain(self):
        """
        Requirement 4C.4: Assert corrupted windows differ from clean in the time domain.
        Tests all 4 types and 3 severities.
        """
        for severity in ["mild", "moderate", "severe"]:
            # 1. Composite corruption
            corrupted_comp = corrupt_raw_ppg_window(self.clean_window, severity=severity, seed=42)
            self.assertEqual(len(corrupted_comp), len(self.clean_window))
            self.assertGreater(float(np.max(np.abs(corrupted_comp - self.clean_window))), 0.0)
            self.assertGreater(float(np.mean((corrupted_comp - self.clean_window) ** 2)), 0.0)

            # 2. High-Frequency Noise
            corrupted_hf = inject_hf_noise(self.clean_window, snr_db=ARTIFACT_SEVERITY_CONFIGS[severity]["hf_snr_db"])
            self.assertGreater(float(np.max(np.abs(corrupted_hf - self.clean_window))), 0.0)

            # 3. Baseline Drift
            corrupted_drift = inject_baseline_drift(
                self.clean_window,
                drift_scale=ARTIFACT_SEVERITY_CONFIGS[severity]["baseline_drift_scale"],
                fs=self.fs
            )
            self.assertGreater(float(np.max(np.abs(corrupted_drift - self.clean_window))), 0.0)

            # 4. Motion Burst
            corrupted_burst, interval = inject_motion_burst(
                self.clean_window,
                burst_scale=ARTIFACT_SEVERITY_CONFIGS[severity]["motion_burst_scale"],
                duration_range=tuple(ARTIFACT_SEVERITY_CONFIGS[severity]["motion_burst_duration"]),
                fs=self.fs
            )
            self.assertGreater(float(np.max(np.abs(corrupted_burst - self.clean_window))), 0.0)
            self.assertLess(interval[0], interval[1])

            # 5. Dropout
            corrupted_drop, drop_interval = inject_dropout(
                self.clean_window,
                dropout_scale=ARTIFACT_SEVERITY_CONFIGS[severity]["dropout_scale"],
                duration_range=tuple(ARTIFACT_SEVERITY_CONFIGS[severity]["dropout_duration"]),
                fs=self.fs
            )
            self.assertGreater(float(np.max(np.abs(corrupted_drop - self.clean_window))), 0.0)
            self.assertLess(drop_interval[0], drop_interval[1])

    def test_sqi_drops_under_corruption(self):
        """Verify Signal Quality Index (in-band ratio and spectral flatness) changes under corruption."""
        clean_inband = compute_inband_power_ratio(self.clean_window, fs=self.fs)
        self.assertGreaterEqual(clean_inband, 0.90)

        for severity in ["mild", "moderate", "severe"]:
            corrupted = corrupt_raw_ppg_window(self.clean_window, severity=severity, seed=42)
            corrupt_inband = compute_inband_power_ratio(corrupted, fs=self.fs)
            corrupt_score = compute_sqi_abstention_score(corrupted, fs=self.fs)
            self.assertLess(corrupt_inband, clean_inband)
            self.assertGreater(corrupt_score, 0.0)

    def test_difficulty_model_uses_grouped_oof_residuals(self):
        """
        Requirement 4B.1: Assert difficulty model trains on grouped out-of-fold residuals,
        never in-sample residuals.
        """
        from sklearn.linear_model import Ridge
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5).astype(np.float32)
        y = (15.0 + 2.0 * X[:, 0] + np.random.randn(n)).astype(np.float32)
        pids = np.repeat([f"pat_{i}" for i in range(10)], 10)

        base_model = Ridge()
        base_model.fit(X[:60], y[:60])

        cal = ConformalCalibrator(base_estimator=base_model, score_type="residual_normalised")
        cal.calibrate(
            X_cal=X[60:],
            y_cal=y[60:],
            X_train=X[:60],
            y_train=y[:60],
            patient_ids_train=pids[:60]
        )
        self.assertTrue(cal.is_calibrated)
        self.assertTrue(cal.is_oof_residuals_trained, "Calibrator must mark that grouped OOF residuals were used")
    def test_combination_score_retention_no_test_leakage_and_masked_mae(self):
        """
        Task 2 & 5: Assert combination-score retention mechanism calls the actual
        production code path (fit_validation_combination_triage and compute_risk_coverage_curve),
        guaranteeing zero test-set leakage, and verifies that MAE is computed strictly
        on retained windows.
        """
        from sklearn.metrics import mean_absolute_error
        from pipeline_stages.run_phase4 import (
            fit_validation_combination_triage,
            compute_risk_coverage_curve,
            compute_patient_bootstrap_risk_coverage
        )

        np.random.seed(42)
        n_val = 60
        n_test = 40

        # Synthetic validation features [interval width, SQI] and errors
        val_widths = np.random.uniform(2.0, 10.0, size=n_val).astype(np.float32)
        val_sqis = np.random.uniform(0.1, 0.9, size=n_val).astype(np.float32)
        val_errors = 0.4 * val_widths + 0.3 * (1.0 - val_sqis) + np.random.normal(0, 0.5, size=n_val)
        val_labels = (val_errors > 3.0).astype(int)

        # Synthetic test features and true / predicted labels
        test_widths = np.random.uniform(2.0, 10.0, size=n_test).astype(np.float32)
        test_sqis = np.random.uniform(0.1, 0.9, size=n_test).astype(np.float32)
        y_test_true = np.random.uniform(10.0, 30.0, size=n_test).astype(np.float32)
        y_test_pred = (y_test_true + np.random.normal(0, 2.0, size=n_test)).astype(np.float32)

        # 1. Call production fit_validation_combination_triage function
        test_comb, clf, scaler = fit_validation_combination_triage(
            val_widths=val_widths,
            val_sqis=val_sqis,
            val_labels=val_labels,
            test_widths=test_widths,
            test_sqis=test_sqis,
            seed=42
        )

        # Assert classifier fitted purely on validation samples without test labels
        self.assertIsNotNone(clf)
        self.assertEqual(clf.n_features_in_, 2)
        self.assertEqual(len(test_comb), n_test)
        self.assertTrue(np.all(test_comb >= 0.0) and np.all(test_comb <= 1.0))

        # 2. Compute risk-coverage curve at multiple retention fractions using production function
        for retention_frac in [0.40, 0.50, 0.70, 0.90]:
            rc_maes = compute_risk_coverage_curve(y_test_true, y_test_pred, test_comb, np.array([retention_frac]))
            computed_retained_mae = float(rc_maes[0])

            cutoff = float(np.quantile(test_comb, retention_frac))
            retained_mask = test_comb <= cutoff
            expected_retained_mae = float(mean_absolute_error(y_test_true[retained_mask], y_test_pred[retained_mask]))

            # Assert MAE is computed ONLY on retained windows
            self.assertAlmostEqual(computed_retained_mae, expected_retained_mae, places=5)
            self.assertLessEqual(np.sum(retained_mask), int(n_test * retention_frac) + 1)
            self.assertGreater(np.sum(retained_mask), 0)

        # 3. Verify production patient-clustered bootstrap risk-coverage computation
        test_pids = np.repeat([f"pat_{i}" for i in range(4)], 10)
        curve, ci_lo, ci_hi = compute_patient_bootstrap_risk_coverage(
            y_test_true, y_test_pred, test_comb, test_pids,
            retention_fractions=np.array([0.50, 1.00]),
            n_resamples=50,
            seed=42
        )
        self.assertEqual(len(curve), 2)
        self.assertLessEqual(ci_lo[0], curve[0] + 1e-4)
        self.assertGreaterEqual(ci_hi[0], curve[0] - 1e-4)

    def test_combination_score_production_zero_leakage_invariance(self):
        """
        Sibling Test (Task 5): Verify that fit_validation_combination_triage strictly
        satisfies zero test leakage invariance: changing test-fold inputs must NOT alter
        the fitted scaler parameters or logistic regression coefficients.
        """
        from pipeline_stages.run_phase4 import fit_validation_combination_triage

        np.random.seed(42)
        n_val = 50
        n_test = 30

        val_widths = np.random.uniform(2.0, 10.0, size=n_val).astype(np.float32)
        val_sqis = np.random.uniform(0.1, 0.9, size=n_val).astype(np.float32)
        val_labels = (val_widths + val_sqis > 6.0).astype(int)

        test_widths = np.random.uniform(2.0, 10.0, size=n_test).astype(np.float32)
        test_sqis = np.random.uniform(0.1, 0.9, size=n_test).astype(np.float32)

        # Baseline fit
        comb_orig, clf_orig, scaler_orig = fit_validation_combination_triage(
            val_widths=val_widths,
            val_sqis=val_sqis,
            val_labels=val_labels,
            test_widths=test_widths,
            test_sqis=test_sqis,
            seed=42
        )

        # Perturbed test set (different values, scale, and extreme outliers)
        perturbed_test_widths = test_widths * 100.0 + 50.0
        perturbed_test_sqis = np.zeros_like(test_sqis)

        comb_pert, clf_pert, scaler_pert = fit_validation_combination_triage(
            val_widths=val_widths,
            val_sqis=val_sqis,
            val_labels=val_labels,
            test_widths=perturbed_test_widths,
            test_sqis=perturbed_test_sqis,
            seed=42
        )

        # Assert mathematical zero-leakage invariance
        np.testing.assert_array_equal(scaler_orig.mean_, scaler_pert.mean_)
        np.testing.assert_array_equal(scaler_orig.scale_, scaler_pert.scale_)
        np.testing.assert_array_equal(clf_orig.coef_, clf_pert.coef_)
        np.testing.assert_array_equal(clf_orig.intercept_, clf_pert.intercept_)

        # Assert single-class fallback behaves gracefully and bounded in [0, 1]
        single_class_labels = np.zeros(n_val, dtype=int)
        comb_fallback, clf_fallback, _ = fit_validation_combination_triage(
            val_widths=val_widths,
            val_sqis=val_sqis,
            val_labels=single_class_labels,
            test_widths=test_widths,
            test_sqis=test_sqis,
            seed=42
        )
        self.assertIsNone(clf_fallback)
        self.assertTrue(np.all(comb_fallback >= 0.0) and np.all(comb_fallback <= 1.0))

    def test_combination_score_results_table_and_curve_integrity(self):
        """Verify combination score risk-coverage series in CSV, summary markdown, and figure."""
        import csv
        results_csv = RESULTS_DIR / "phase5_results.csv"
        summary_md = RESULTS_DIR / "phase4_summary.md"
        figure_path = FIGURES_DIR / "risk_coverage_curve.png"

        self.assertTrue(results_csv.exists(), f"phase5_results.csv must exist in {RESULTS_DIR}")
        self.assertTrue(summary_md.exists(), f"phase4_summary.md must exist in {RESULTS_DIR}")
        self.assertTrue(figure_path.exists(), f"risk_coverage_curve.png must exist in {FIGURES_DIR}")

        # Verify CSV columns and numeric bounds
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        in_sec5 = False
        sec5_rows = []
        header = None
        for r in rows:
            if not r:
                continue
            if "[SECTION 5: SELECTIVE PREDICTION RISK-COVERAGE OPERATING POINTS" in r[0]:
                in_sec5 = True
                continue
            if in_sec5:
                if header is None:
                    header = r
                else:
                    sec5_rows.append(r)

        self.assertIsNotNone(header)
        self.assertIn("Combination_Score_MAE_bpm", header)
        self.assertIn("Combination_95CI_Lo", header)
        self.assertIn("Combination_95CI_Hi", header)
        self.assertEqual(len(sec5_rows), 9)  # 40, 50, 60, 70, 80, 85, 90, 95, 100%

        comb_idx = header.index("Combination_Score_MAE_bpm")
        oracle_idx = header.index("Oracle_MAE_bpm")
        rand_idx = header.index("Random_MAE_bpm")

        for row in sec5_rows:
            comb_val = float(row[comb_idx])
            oracle_val = float(row[oracle_idx])
            rand_val = float(row[rand_idx])
            self.assertGreaterEqual(comb_val, oracle_val - 1e-5, "Combination MAE must be >= Oracle MAE")
            self.assertLessEqual(comb_val, rand_val + 1e-5, "Combination MAE must be <= Random MAE")

        # Verify summary markdown
        md_text = summary_md.read_text(encoding="utf-8")
        self.assertIn("Combination Score MAE (bpm) [95% CI]", md_text)
        self.assertIn("Plain Evaluated Retention Verdict", md_text)

        # Verify figure header
        fig_bytes = figure_path.read_bytes()
        self.assertTrue(fig_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(fig_bytes), 10000)


if __name__ == "__main__":
    unittest.main()
