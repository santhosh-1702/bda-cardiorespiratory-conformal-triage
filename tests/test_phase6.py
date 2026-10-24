"""
Unit tests for Phase 6: Multi-Seed Conformal Score Functions Comparison
=======================================================================
Tests:
  1. Patient-disjoint splits preserved across all 5 repeat seeds:
     {train, val, cal, test} pairwise disjoint for all 25 folds (0 patient leakage).
  2. Conformal calibrator determinism across all 5 score functions.
  3. Paired patient bootstrap CI computation and mathematical sanity.
  4. Phase 6 artifacts existence and integrity (CSV, Summary MD, Figure).
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config.settings import RANDOM_SEED, PATIENT_IDS, OUTPUT_DIR
from src.features.extractor import create_patient_fold_map, generate_patient_folds
from src.conformal.calibrator import ConformalCalibrator
from pipeline_stages.run_phase6_score_comparison import (
    REPEAT_SEEDS,
    SCORE_TYPES,
    CSV_OUTPUT_PATH,
    SUMMARY_MD_PATH,
    FIGURE_OUTPUT_PATH
)


class TestPhase6ScoreComparison(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = (15.0 + 2.0 * self.X[:, 0] + np.random.randn(self.n_samples)).astype(np.float32)
        self.patient_ids = np.repeat(np.arange(20), 10)  # 20 patients, 10 samples each
        self.sharpness = np.random.uniform(1.0, 10.0, size=self.n_samples).astype(np.float32)
        self.X[:, -1] = self.sharpness

        self.model = Ridge()
        self.model.fit(self.X[:100], self.y[:100])

    def test_patient_disjoint_splits_across_all_repeat_seeds(self):
        """
        Verify that all 5 repeat seeds generate folds that are strictly
        pairwise disjoint across train, val, cal, and test patient subsets.
        Zero identity leakage.
        """
        for seed in REPEAT_SEEDS:
            fold_map = create_patient_fold_map(seed=seed, save_path=None)
            folds = generate_patient_folds(PATIENT_IDS, fold_map=fold_map, seed=seed)
            self.assertEqual(len(folds), 5)

            for f in folds:
                tr = set(f["train_pids"])
                va = set(f["val_pids"])
                ca = set(f["cal_pids"])
                te = set(f["test_pids"])

                self.assertEqual(len(tr & va), 0, f"Train & Val leakage in seed {seed} fold {f['fold']}")
                self.assertEqual(len(tr & ca), 0, f"Train & Cal leakage in seed {seed} fold {f['fold']}")
                self.assertEqual(len(tr & te), 0, f"Train & Test leakage in seed {seed} fold {f['fold']}")
                self.assertEqual(len(va & ca), 0, f"Val & Cal leakage in seed {seed} fold {f['fold']}")
                self.assertEqual(len(va & te), 0, f"Val & Test leakage in seed {seed} fold {f['fold']}")
                self.assertEqual(len(ca & te), 0, f"Cal & Test leakage in seed {seed} fold {f['fold']}")

                # Total unique patients in this fold must equal cohort size
                all_pts = tr | va | ca | te
                self.assertEqual(len(all_pts), len(set(PATIENT_IDS)))

    def test_conformal_calibrator_all_five_score_types(self):
        """
        Verify that all 5 score functions calibrate and predict without error,
        producing valid intervals where upper >= lower and positive widths.
        """
        for st in SCORE_TYPES:
            train_med = float(np.median(self.y[:100]))
            cal = ConformalCalibrator(
                base_estimator=self.model if st != "trivial_median" else None,
                confidence_level=0.90,
                score_type=st,
                train_median=train_med if st == "trivial_median" else None,
                n_jobs=1,
                random_state=42
            )
            cal.calibrate(
                X_cal=self.X[100:150],
                y_cal=self.y[100:150],
                X_train=self.X[:100],
                y_train=self.y[:100],
                sharpness_cal=self.sharpness[100:150] if st == "spectral_sharpness" else None
            )

            preds, lo, hi, widths = cal.predict_intervals(
                self.X[150:],
                sharpness=self.sharpness[150:] if st == "spectral_sharpness" else None
            )

            self.assertEqual(len(preds), 50)
            self.assertTrue(np.all(hi >= lo), f"hi < lo detected for score_type {st}")
            self.assertTrue(np.all(widths > 0), f"non-positive widths for score_type {st}")
            self.assertGreater(cal.q_hat, 0.0)

    def test_paired_bootstrap_identical_predictors_and_divergent_predictors(self):
        """
        Verify bootstrap CI logic:
        1. When two methods have identical point predictions, mean difference is 0 and CI is [0, 0].
        2. When one method is strictly worse, difference is non-zero and CI excludes 0.
        """
        from pipeline_stages.run_phase4 import compute_patient_bootstrap_ci_paired

        y_true = np.array([12.0, 15.0, 18.0, 20.0, 22.0, 25.0])
        pids = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])

        # Identical predictions
        y_pred1 = np.array([12.2, 14.8, 18.1, 19.9, 21.8, 25.2])
        y_pred2 = np.copy(y_pred1)

        boot_same = compute_patient_bootstrap_ci_paired(y_true, y_pred1, y_pred2, pids, n_resamples=100)
        self.assertAlmostEqual(boot_same["mean_diff"], 0.0, places=5)
        self.assertAlmostEqual(boot_same["ci_lower"], 0.0, places=5)
        self.assertAlmostEqual(boot_same["ci_upper"], 0.0, places=5)
        self.assertFalse(boot_same["excludes_zero"])

        # Significantly worse predictor
        y_pred_bad = y_pred1 + 10.0
        boot_diff = compute_patient_bootstrap_ci_paired(y_true, y_pred_bad, y_pred1, pids, n_resamples=100)
        self.assertGreater(boot_diff["mean_diff"], 5.0)
        self.assertTrue(boot_diff["excludes_zero"])

    def test_phase6_artifacts_and_stability_verdict(self):
        """
        Verify Phase 6 artifacts existence and statistical integrity:
        1. phase6_results.csv contains 125 detailed runs + 5 aggregate rows.
        2. phase6_summary.md contains benchmark table, bootstrap table, and verdict.
        3. phase6_score_comparison.png is a valid high-resolution graphic.
        """
        self.assertTrue(CSV_OUTPUT_PATH.exists(), "phase6_results.csv must exist in output/")
        self.assertTrue(SUMMARY_MD_PATH.exists(), "phase6_summary.md must exist in output/")
        self.assertTrue(FIGURE_OUTPUT_PATH.exists(), "phase6_score_comparison.png must exist in output/")

        df_csv = pd.read_csv(CSV_OUTPUT_PATH)
        detailed = df_csv[df_csv["section"] == "DETAILED_RUNS"]
        aggregate = df_csv[df_csv["section"] == "AGGREGATE_SUMMARY"]

        self.assertEqual(len(detailed), 125, "Expected 125 detailed runs (5 seeds * 5 folds * 5 score types)")
        self.assertEqual(len(aggregate), 5, "Expected 5 aggregate summary rows (1 per score type)")

        for st in SCORE_TYPES:
            self.assertEqual(len(detailed[detailed["score_type"] == st]), 25)
            self.assertEqual(len(aggregate[aggregate["score_type"] == st]), 1)

        summary_text = SUMMARY_MD_PATH.read_text(encoding="utf-8")
        self.assertIn("Head-to-Head Conformal Methods Benchmark Table", summary_text)
        self.assertIn("Paired Patient-Level Bootstrap Differences vs Difficulty-Normalised", summary_text)
        self.assertIn("Fold-to-Fold Quantile & Coverage Stability Analysis", summary_text)
        self.assertIn("Final Evaluated Verdict", summary_text)
        self.assertIn("small-sample calibration noise", summary_text)

        fig_bytes = FIGURE_OUTPUT_PATH.read_bytes()
        self.assertTrue(fig_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "phase6 figure is not valid PNG")
        self.assertGreater(len(fig_bytes), 50000, "phase6 figure size unexpectedly small")


if __name__ == "__main__":
    unittest.main()
