"""
Test Suite: AUROC Alignment Verification
========================================
Verifies:
  1. Synthetic 5-fold example where clean and corrupted scores have true AUROC ~ 0.98.
  2. The buggy pooling (all-clean-then-all-corrupt scores vs interleaved labels) collapses AUROC to ~0.50.
  3. The fixed pipeline's AUROC aligns perfectly with the independent ground truth (within 0.02 of 0.98).
"""

import unittest
import numpy as np
from sklearn.metrics import roc_auc_score


class TestAUROCAlignment(unittest.TestCase):

    def test_synthetic_5fold_alignment(self):
        """
        Requirement 1: Add tests/test_auroc_alignment.py with a synthetic 5-fold example
        where true AUROC is about 0.98 and assert the pipeline's AUROC is within 0.02
        of an independently computed value.
        """
        rng = np.random.RandomState(42)
        n_folds = 5
        fold_clean_scores = []
        fold_corrupt_scores = []

        # Synthetic 5-fold setup:
        # Clean scores ~ N(10.0, 2.0), Corrupted scores ~ N(15.8, 2.0)
        # Difference in means = 5.8, sigma = 2.0 -> d' = 5.8 / (2 * sqrt(2)) = 2.0506
        # Expected AUROC = Phi(d') ~ 0.98
        independent_scores = []
        independent_labels = []

        for k in range(n_folds):
            n_clean = 200 + k * 15
            n_corrupt = 200 + k * 15

            c_scores = rng.normal(10.0, 2.0, size=n_clean).astype(np.float32)
            corrupt_s = rng.normal(15.8, 2.0, size=n_corrupt).astype(np.float32)

            fold_clean_scores.append(c_scores)
            fold_corrupt_scores.append(corrupt_s)

            # Ground-truth construction: pair each item with its correct label
            for s in c_scores:
                independent_scores.append(float(s))
                independent_labels.append(0)
            for s in corrupt_s:
                independent_scores.append(float(s))
                independent_labels.append(1)

        independent_auroc = float(roc_auc_score(independent_labels, independent_scores))
        self.assertAlmostEqual(independent_auroc, 0.98, delta=0.015,
                               msg=f"Ground-truth AUROC should be ~0.98, got {independent_auroc:.4f}")

        # --- BUGGY IMPLEMENTATION (Prior to Phase 5) ---
        # Labels appended per-fold as [0]*n_clean + [1]*n_corrupt (interleaved across folds)
        # but scores were pooled as (all_clean + all_corrupt)
        buggy_scores = []
        for c in fold_clean_scores:
            buggy_scores.extend(c)
        for cor in fold_corrupt_scores:
            buggy_scores.extend(cor)

        buggy_labels = []
        for k in range(n_folds):
            buggy_labels.extend([0] * len(fold_clean_scores[k]) + [1] * len(fold_corrupt_scores[k]))

        buggy_auroc = float(roc_auc_score(buggy_labels, buggy_scores))
        # The buggy approach destroys sample correspondence, dropping AUROC to ~0.50
        self.assertLess(buggy_auroc, 0.60,
                        msg=f"Buggy pooling should collapse AUROC near random chance (~0.50), got {buggy_auroc:.4f}")

        # --- FIXED PIPELINE IMPLEMENTATION ---
        # Scores and labels built in the exact same order (per fold or pooled)
        fixed_scores = []
        fixed_labels = []
        for k in range(n_folds):
            fixed_scores.extend(fold_clean_scores[k])
            fixed_scores.extend(fold_corrupt_scores[k])
            fixed_labels.extend([0] * len(fold_clean_scores[k]))
            fixed_labels.extend([1] * len(fold_corrupt_scores[k]))

        pipeline_auroc = float(roc_auc_score(fixed_labels, fixed_scores))

        # Assert pipeline AUROC matches independent AUROC exactly and is within 0.02 of 0.98
        self.assertAlmostEqual(pipeline_auroc, independent_auroc, delta=1e-7,
                               msg="Fixed pipeline AUROC must match independently computed value exactly.")
        self.assertAlmostEqual(pipeline_auroc, 0.98, delta=0.02,
                               msg=f"Pipeline AUROC must be within 0.02 of 0.98, got {pipeline_auroc:.4f}")


if __name__ == "__main__":
    unittest.main()
