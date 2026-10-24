"""
Test Suite: Zero-Leakage Patient-Grouped Cross-Validation
==========================================================
Verifies that:
1. Train, Validation, Calibration, and Test patient sets are strictly pairwise disjoint for every fold.
2. Window indices for Train, Validation, Calibration, and Test are strictly pairwise disjoint for every fold.
3. Every window's patient_id strictly belongs to the partition assigned to that window.
4. The outer test sets across all 5 folds form a partition of the entire patient cohort (53 patients).
5. The cached features file (features_split.npz) contains patient_ids and conforms to all zero-leakage constraints.
"""

import unittest
from pathlib import Path
import numpy as np

from config.settings import (
    FEATURES_CACHE_FILE,
    N_FOLDS,
    TOTAL_PATIENTS,
    RANDOM_SEED
)
from src.features.extractor import (
    build_feature_dataset,
    generate_patient_folds,
    load_feature_dataset,
    get_patient_folds,
)

class TestNoLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure feature dataset with patient tracking is built/loaded
        cls.X, cls.y, cls.patient_ids, cls.folds = build_feature_dataset(force_recompute=False)
        cls.unique_patients = set(np.unique(cls.patient_ids))

    def test_cohort_completeness(self):
        """Assert all 53 patients are represented in the dataset."""
        self.assertEqual(len(self.unique_patients), TOTAL_PATIENTS,
                         f"Expected {TOTAL_PATIENTS} patients, found {len(self.unique_patients)}")
        self.assertEqual(len(self.folds), N_FOLDS,
                         f"Expected {N_FOLDS} folds, found {len(self.folds)}")

    def test_pairwise_disjoint_patient_sets_per_fold(self):
        """
        Assert that for EVERY fold:
        train_pids, val_pids, cal_pids, and test_pids are pairwise disjoint.
        """
        for fold_info in self.folds:
            fold = fold_info["fold"]
            train_pids = set(fold_info["train_pids"])
            val_pids = set(fold_info["val_pids"])
            cal_pids = set(fold_info["cal_pids"])
            test_pids = set(fold_info["test_pids"])

            pairs = [
                ("Train", train_pids, "Val", val_pids),
                ("Train", train_pids, "Cal", cal_pids),
                ("Train", train_pids, "Test", test_pids),
                ("Val", val_pids, "Cal", cal_pids),
                ("Val", val_pids, "Test", test_pids),
                ("Cal", cal_pids, "Test", test_pids),
            ]

            for name_a, set_a, name_b, set_b in pairs:
                overlap = set_a & set_b
                self.assertEqual(
                    len(overlap), 0,
                    f"Fold {fold} LEAKAGE DETECTED: {name_a} and {name_b} share patients: {overlap}"
                )

            # Union of all 4 sets must equal the complete patient cohort
            all_fold_pts = train_pids | val_pids | cal_pids | test_pids
            self.assertEqual(
                all_fold_pts, self.unique_patients,
                f"Fold {fold}: Union of patient sets does not match total cohort."
            )

    def test_pairwise_disjoint_window_indices_per_fold(self):
        """
        Assert that for EVERY fold:
        train_idx, val_idx, cal_idx, and test_idx are pairwise disjoint.
        """
        n_total_windows = len(self.patient_ids)

        for fold_info in self.folds:
            fold = fold_info["fold"]
            train_idx = set(fold_info["train_idx"])
            val_idx = set(fold_info["val_idx"])
            cal_idx = set(fold_info["cal_idx"])
            test_idx = set(fold_info["test_idx"])

            idx_pairs = [
                ("train_idx", train_idx, "val_idx", val_idx),
                ("train_idx", train_idx, "cal_idx", cal_idx),
                ("train_idx", train_idx, "test_idx", test_idx),
                ("val_idx", val_idx, "cal_idx", cal_idx),
                ("val_idx", val_idx, "test_idx", test_idx),
                ("cal_idx", cal_idx, "test_idx", test_idx),
            ]

            for name_a, set_a, name_b, set_b in idx_pairs:
                overlap = set_a & set_b
                self.assertEqual(
                    len(overlap), 0,
                    f"Fold {fold} WINDOW LEAKAGE DETECTED: {name_a} and {name_b} overlap on {len(overlap)} windows"
                )

            # Check that indices partition the full window set
            union_idx = train_idx | val_idx | cal_idx | test_idx
            self.assertEqual(
                len(union_idx), n_total_windows,
                f"Fold {fold}: Total window union {len(union_idx)} does not match {n_total_windows}"
            )

    def test_window_patient_mapping_integrity(self):
        """
        Assert that every window's patient_id matches the patient set of its partition.
        """
        for fold_info in self.folds:
            fold = fold_info["fold"]
            for part_name, idx_key, pid_key in [
                ("Train", "train_idx", "train_pids"),
                ("Val", "val_idx", "val_pids"),
                ("Cal", "cal_idx", "cal_pids"),
                ("Test", "test_idx", "test_pids"),
            ]:
                assigned_pids = set(fold_info[pid_key])
                window_pids = set(self.patient_ids[fold_info[idx_key]])
                unexpected = window_pids - assigned_pids
                self.assertEqual(
                    len(unexpected), 0,
                    f"Fold {fold} {part_name}: Windows contain patients outside assigned set: {unexpected}"
                )

    def test_outer_test_folds_partition_cohort(self):
        """
        Assert that the 5 test folds partition the entire patient cohort (GroupKFold property).
        Each patient must appear in exactly one test fold.
        """
        seen_test_patients = []
        for fold_info in self.folds:
            seen_test_patients.extend(fold_info["test_pids"])

        self.assertEqual(len(seen_test_patients), TOTAL_PATIENTS)
        self.assertEqual(len(set(seen_test_patients)), TOTAL_PATIENTS)
        self.assertEqual(set(seen_test_patients), self.unique_patients)

    def test_cached_file_contains_patient_ids_and_folds(self):
        """
        Assert that features_split.npz contains patient_ids, X, y, and per-fold indices.
        """
        self.assertTrue(FEATURES_CACHE_FILE.exists(), f"{FEATURES_CACHE_FILE} does not exist")
        cached = np.load(FEATURES_CACHE_FILE, allow_pickle=True)

        self.assertIn("patient_ids", cached)
        self.assertIn("X", cached)
        self.assertIn("y", cached)
        self.assertEqual(len(cached["patient_ids"]), len(cached["X"]))
        self.assertEqual(len(cached["y"]), len(cached["X"]))

        for k in range(N_FOLDS):
            self.assertIn(f"fold_{k}_train_idx", cached)
            self.assertIn(f"fold_{k}_val_idx", cached)
            self.assertIn(f"fold_{k}_cal_idx", cached)
            self.assertIn(f"fold_{k}_test_idx", cached)
            self.assertIn(f"fold_{k}_train_pids", cached)
            self.assertIn(f"fold_{k}_val_pids", cached)
            self.assertIn(f"fold_{k}_cal_pids", cached)
            self.assertIn(f"fold_{k}_test_pids", cached)

            tr_p = set(cached[f"fold_{k}_train_pids"])
            te_p = set(cached[f"fold_{k}_test_pids"])
            val_p = set(cached[f"fold_{k}_val_pids"])
            cal_p = set(cached[f"fold_{k}_cal_pids"])

            self.assertEqual(len(tr_p & te_p), 0)
            self.assertEqual(len(val_p & te_p), 0)
            self.assertEqual(len(cal_p & te_p), 0)
            self.assertEqual(len(tr_p & val_p), 0)
            self.assertEqual(len(tr_p & cal_p), 0)
            self.assertEqual(len(val_p & cal_p), 0)

    def test_reproducibility(self):
        """
        Assert that generate_patient_folds with fixed seed produces identical splits across runs.
        """
        folds_a = generate_patient_folds(self.patient_ids, seed=RANDOM_SEED)
        folds_b = generate_patient_folds(self.patient_ids, seed=RANDOM_SEED)

        for fa, fb in zip(folds_a, folds_b):
            self.assertEqual(fa["train_pids"], fb["train_pids"])
            self.assertEqual(fa["val_pids"], fb["val_pids"])
            self.assertEqual(fa["cal_pids"], fb["cal_pids"])
            self.assertEqual(fa["test_pids"], fb["test_pids"])
            np.testing.assert_array_equal(fa["train_idx"], fb["train_idx"])
            np.testing.assert_array_equal(fa["val_idx"], fb["val_idx"])
            np.testing.assert_array_equal(fa["cal_idx"], fb["cal_idx"])
            np.testing.assert_array_equal(fa["test_idx"], fb["test_idx"])

if __name__ == "__main__":
    unittest.main()
