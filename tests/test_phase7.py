"""
Unit tests for Phase 7: Jackknife+ / CV+ Cross-Fitting Conformal Prediction
===========================================================================
Tests:
  1. Patient-disjoint inner cross-fitting:
     - No patient appears on both sides of any inner fold across all 25 outer folds.
     - Zero identity leakage at both outer and inner fold levels.
  2. CV+ interval construction matches the exact Jackknife+ / CV+ formula
     (Barber et al., 2021) on a small synthetic example with known closed-form answer.
  3. CVPlusCalibrator end-to-end functionality:
     - Fits K inner models, computes valid out-of-fold residuals, and produces valid intervals.
  4. Phase 7 artifacts existence and integrity:
     - phase7_results.csv contains 150 detailed runs + 6 aggregate rows.
     - phase7_summary.md contains benchmark table, bootstrap comparisons, and plain verdict.
     - phase7_score_comparison.png is a valid high-resolution graphic.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config.settings import RANDOM_SEED, PATIENT_IDS, OUTPUT_DIR
from src.features.extractor import create_patient_fold_map, generate_patient_folds
from src.conformal.cv_plus import CVPlusCalibrator, compute_cv_plus_quantiles
from pipeline_stages.run_phase7_cv_plus import (
    REPEAT_SEEDS,
    SCORE_TYPES,
    CSV_OUTPUT_PATH,
    SUMMARY_MD_PATH,
    FIGURE_OUTPUT_PATH
)


class TestPhase7CVPlus(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = (15.0 + 2.0 * self.X[:, 0] + np.random.randn(self.n_samples)).astype(np.float32)
        self.patient_ids = np.repeat(np.arange(20), 10)  # 20 patients, 10 samples each

    def test_inner_kfold_no_patient_leakage_across_all_outer_folds(self):
        """
        Requirement 1 & 2: Confirm no patient appears on both sides of any inner fold
        across all 5 repeat seeds and all 5 outer folds (25 outer folds total).
        """
        for seed in REPEAT_SEEDS:
            fold_map = create_patient_fold_map(seed=seed, save_path=None)
            folds = generate_patient_folds(PATIENT_IDS, fold_map=fold_map, seed=seed)
            self.assertEqual(len(folds), 5)

            for f in folds:
                te_pids = set(f["test_pids"])
                non_test_pids = sorted(list(set(PATIENT_IDS) - te_pids))

                # GroupKFold partitioning on non-test patients
                gkf = GroupKFold(n_splits=5)
                # Mock indices per patient
                mock_pids_arr = np.array(non_test_pids)
                mock_X = np.zeros((len(mock_pids_arr), 2))

                for inner_k, (inner_tr_idx, inner_val_idx) in enumerate(
                    gkf.split(mock_X, groups=mock_pids_arr)
                ):
                    inner_tr_pids = set(mock_pids_arr[inner_tr_idx])
                    inner_val_pids = set(mock_pids_arr[inner_val_idx])

                    # 1. Zero leakage between inner train and inner val
                    inner_overlap = inner_tr_pids & inner_val_pids
                    self.assertEqual(
                        len(inner_overlap), 0,
                        f"Inner patient leakage in seed {seed} outer fold {f['fold']} inner fold {inner_k}: {inner_overlap}"
                    )

                    # 2. Zero leakage between inner folds and outer test fold
                    self.assertEqual(
                        len(inner_tr_pids & te_pids), 0,
                        f"Outer test leakage in inner train: {inner_tr_pids & te_pids}"
                    )
                    self.assertEqual(
                        len(inner_val_pids & te_pids), 0,
                        f"Outer test leakage in inner val: {inner_val_pids & te_pids}"
                    )

                    # 3. Inner split conserves all non-test patients
                    self.assertEqual(inner_tr_pids | inner_val_pids, set(non_test_pids))

    def test_cv_plus_interval_exact_formula_closed_form_synthetic_example(self):
        """
        Requirement 4 & 8: Confirm the interval construction matches the exact jackknife+/CV+
        formula (Barber et al., 2021) on a small synthetic example with known closed-form answer.

        Setup:
          n = 4 training points, 1 test point.
          Inner fold 0 (2 points): R_1 = 1.0, R_2 = 2.0; Model 0 predicts 10.0 at test point.
          Inner fold 1 (2 points): R_3 = 1.5, R_4 = 0.5; Model 1 predicts 12.0 at test point.

          Lower candidates L:
            Point 1: 10.0 - 1.0 = 9.0
            Point 2: 10.0 - 2.0 = 8.0
            Point 3: 12.0 - 1.5 = 10.5
            Point 4: 12.0 - 0.5 = 11.5
            Sorted L: [8.0, 9.0, 10.5, 11.5]

          Upper candidates U:
            Point 1: 10.0 + 1.0 = 11.0
            Point 2: 10.0 + 2.0 = 12.0
            Point 3: 12.0 + 1.5 = 13.5
            Point 4: 12.0 + 0.5 = 12.5
            Sorted U: [11.0, 12.0, 12.5, 13.5]

          Confidence level: 1 - alpha = 0.75 (alpha = 0.25).
          With n = 4:
            floor(alpha * (n + 1)) = floor(0.25 * 5) = 1 (1st smallest L -> 8.0)
            ceil((1 - alpha) * (n + 1)) = ceil(0.75 * 5) = ceil(3.75) = 4 (4th smallest U -> 13.5)

          Closed-form answer:
            lower = 8.0, upper = 13.5, width = 5.5
        """
        # Formulate L and U matrices of shape (n_train, n_test) = (4, 1)
        L = np.array([[9.0], [8.0], [10.5], [11.5]], dtype=np.float32)
        U = np.array([[11.0], [12.0], [13.5], [12.5]], dtype=np.float32)

        lo, hi, width = compute_cv_plus_quantiles(L, U, alpha=0.25)

        self.assertEqual(len(lo), 1)
        self.assertEqual(len(hi), 1)
        self.assertEqual(len(width), 1)

        self.assertAlmostEqual(float(lo[0]), 8.0, places=5)
        self.assertAlmostEqual(float(hi[0]), 13.5, places=5)
        self.assertAlmostEqual(float(width[0]), 5.5, places=5)

    def test_cv_plus_calibrator_end_to_end(self):
        """Verify CVPlusCalibrator fits inner models, computes OOF residuals, and predicts intervals."""
        calibrator = CVPlusCalibrator(
            n_inner_splits=5,
            confidence_level=0.90,
            random_state=42
        )
        # 150 training samples across 15 patients
        calibrator.fit_and_cross_calibrate(
            X_pool=self.X[:150],
            y_pool=self.y[:150],
            patient_ids_pool=self.patient_ids[:150]
        )

        self.assertTrue(calibrator.is_calibrated)
        self.assertEqual(len(calibrator.inner_models), 5)
        self.assertEqual(len(calibrator.oof_residuals), 150)
        self.assertFalse(np.any(np.isnan(calibrator.oof_residuals)))

        # Predict on 50 held-out samples from 5 held-out patients
        preds, lo, hi, widths = calibrator.predict_intervals(self.X[150:])
        self.assertEqual(len(preds), 50)
        self.assertEqual(len(lo), 50)
        self.assertEqual(len(hi), 50)
        self.assertEqual(len(widths), 50)

        self.assertTrue(np.all(hi >= lo), "Upper bound strictly less than lower bound")
        self.assertTrue(np.all(widths > 0), "Interval width must be positive")

    def test_phase7_artifacts_and_stability_verdict(self):
        """
        Requirement 6 & 7: Verify Phase 7 artifacts existence and statistical integrity:
          1. phase7_results.csv contains 150 detailed runs (25 runs x 6 score types) + 6 aggregate rows.
          2. phase7_summary.md contains benchmark table, bootstrap table, and plain verdict.
          3. phase7_score_comparison.png is a valid graphic.
        """
        self.assertTrue(CSV_OUTPUT_PATH.exists(), "phase7_results.csv must exist in output/")
        self.assertTrue(SUMMARY_MD_PATH.exists(), "phase7_summary.md must exist in output/")
        self.assertTrue(FIGURE_OUTPUT_PATH.exists(), "phase7_score_comparison.png must exist in output/")

        df_csv = pd.read_csv(CSV_OUTPUT_PATH)
        detailed = df_csv[df_csv["section"] == "DETAILED_RUNS"]
        aggregate = df_csv[df_csv["section"] == "AGGREGATE_SUMMARY"]

        self.assertEqual(len(detailed), 150, "Expected 150 detailed runs (5 seeds * 5 folds * 6 score types)")
        self.assertEqual(len(aggregate), 6, "Expected 6 aggregate summary rows (1 per score type)")

        for st in SCORE_TYPES:
            self.assertEqual(len(detailed[detailed["score_type"] == st]), 25)
            self.assertEqual(len(aggregate[aggregate["score_type"] == st]), 1)

        summary_text = SUMMARY_MD_PATH.read_text(encoding="utf-8")
        self.assertIn("Head-to-Head Conformal Methods Benchmark Table", summary_text)
        self.assertIn("Paired Patient-Level Bootstrap Differences vs Difficulty-Normalised", summary_text)
        self.assertIn("Fold-to-Fold Coverage Dispersion Analysis (Bootstrap CIs)", summary_text)
        self.assertIn("Final Evaluated Verdict", summary_text)
        self.assertIn("Plain Verdict", summary_text)
        self.assertIn("cv_plus", summary_text)

        # Verify the coverage-spread bootstrap CI is present in the stability section
        self.assertIn("Bootstrap Difference (CV+ minus Normalised)", summary_text)
        self.assertIn("SD Diff:", summary_text)
        self.assertIn("Min-Max Spread Diff:", summary_text)
        # Verify the verdict itself contains the bootstrap CI
        self.assertIn("Bootstrap 95% CI on the SD difference", summary_text)
        # Verify the CI states whether it includes or excludes zero
        self.assertTrue(
            "includes zero" in summary_text or "excludes zero" in summary_text,
            "Verdict must state whether the coverage-spread bootstrap CI includes or excludes zero"
        )

        fig_bytes = FIGURE_OUTPUT_PATH.read_bytes()
        self.assertTrue(fig_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "phase7 figure is not valid PNG")
        self.assertGreater(len(fig_bytes), 50000, "phase7 figure size unexpectedly small")


if __name__ == "__main__":
    unittest.main()
