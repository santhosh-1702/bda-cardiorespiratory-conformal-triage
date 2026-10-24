"""
Unit Test Suite: CapnoBase External Validation Dataset Import
============================================================
Verifies:
  1. Synthetic biosignal resampling (300 Hz -> 125 Hz) and Butterworth filtering
     produce sane, numerically stable, zero-NaN output preserving respiratory frequency.
  2. Single CapnoBase record parsing loads signal, parameters, metadata, and 1 Hz ground truth.
  3. Feature extraction produces standard 28-dimensional multi-domain feature vector with 0 NaNs.
  4. Cached capnobase_features.npz dataset integrity:
     - Exactly 42 external patient cases.
     - 8,764 total 60s windows (min=157, mean=208.7, max=211 windows/case).
     - Feature dimensionality exactly 28.
     - Target respiratory rates bounded strictly within physiological bounds [6.0, 45.0] bpm.
     - Zero NaNs across all feature columns and targets.
  5. Zero overlap between CapnoBase cases and BIDMC ICU training cohort.
  6. Documentation and formal citation statement exist in output/paper/.
"""

import unittest
from pathlib import Path
import numpy as np

from config.settings import (
    FS,
    PATIENT_IDS,
    PROJECT_ROOT,
    CAPNOBASE_CACHE_FILE,
    CAPNOBASE_TOTAL_CASES,
)
from src.external_validation.capnobase_loader import (
    locate_capnobase_dir,
    get_capnobase_case_ids,
    resample_signal_125hz,
    load_capnobase_record,
    load_capnobase_dataset,
    parse_capnobase_subject_metadata,
    get_capnobase_ventilation_stratification,
)
from src.features.extractor import extract_window_features_from_segments
from src.preprocessing.filter import (
    ButterworthRespiratoryFilter,
    ButterworthWidebandFilter,
)


class TestCapnoBaseImport(unittest.TestCase):

    def setUp(self):
        self.target_fs = FS  # 125.0 Hz
        self.cache_file = CAPNOBASE_CACHE_FILE

    def test_resampling_and_filtering_on_synthetic_signal(self):
        """
        Verify that resample_signal_125hz and Butterworth filters produce sane,
        zero-NaN, numerically stable outputs on synthetic multi-frequency signal.
        """
        orig_fs = 300.0
        duration_sec = 60.0
        n_orig = int(orig_fs * duration_sec)
        t_orig = np.linspace(0, duration_sec, n_orig, endpoint=False)

        # Synthetic signal: 0.25 Hz respiration (15 bpm) + 1.25 Hz cardiac carrier (75 bpm)
        resp_component = 0.5 * np.sin(2 * np.pi * 0.25 * t_orig)
        cardiac_component = 1.0 * np.sin(2 * np.pi * 1.25 * t_orig)
        synthetic_raw = (resp_component + cardiac_component).astype(np.float32)

        # Resample from 300 Hz to 125 Hz
        resampled = resample_signal_125hz(synthetic_raw, orig_fs=orig_fs, target_fs=self.target_fs)
        expected_len = int(round(n_orig * self.target_fs / orig_fs))
        self.assertEqual(len(resampled), expected_len)
        self.assertFalse(np.any(np.isnan(resampled)))

        # Apply respiratory filter (0.10–0.60 Hz)
        resp_filter = ButterworthRespiratoryFilter(fs=self.target_fs)
        filtered_resp = resp_filter.apply(resampled)
        self.assertEqual(len(filtered_resp), expected_len)
        self.assertFalse(np.any(np.isnan(filtered_resp)))

        # Verify cardiac pulse is attenuated and 0.25 Hz respiration is preserved
        # FFT peak should be close to 0.25 Hz
        fft_mags = np.abs(np.fft.rfft(filtered_resp))
        freqs = np.fft.rfftfreq(len(filtered_resp), d=1.0 / self.target_fs)
        peak_freq = freqs[np.argmax(fft_mags)]
        self.assertAlmostEqual(peak_freq, 0.25, delta=0.03, msg="Respiratory filter must preserve 0.25 Hz modulation")

    def test_capnobase_directory_and_case_count(self):
        """Verify CapnoBase dataset directory is found and contains exactly 42 cases."""
        data_dir = locate_capnobase_dir()
        self.assertTrue(data_dir.exists(), f"CapnoBase directory not found at {data_dir}")

        case_ids = get_capnobase_case_ids()
        self.assertEqual(
            len(case_ids),
            CAPNOBASE_TOTAL_CASES,
            f"Expected {CAPNOBASE_TOTAL_CASES} CapnoBase cases, found {len(case_ids)}"
        )

    def test_single_record_parsing_and_structure(self):
        """Test loading and standardization of a single representative recording."""
        rec = load_capnobase_record("0009")

        # Verify contract keys matching BIDMC streaming format
        expected_keys = [
            "record_id", "raw_ppg", "filtered_ppg", "filtered_wide",
            "gt_rr_1hz", "fs", "duration_sec", "meta"
        ]
        for k in expected_keys:
            self.assertIn(k, rec, f"Missing key '{k}' in record dict")

        self.assertEqual(rec["record_id"], "capno_0009")
        self.assertEqual(rec["fs"], 125.0)
        self.assertAlmostEqual(rec["duration_sec"], 480.0, delta=0.5)

        # 480s at 125 Hz is 60,000 samples (plus/minus 1 due to endpoint in resampling)
        self.assertAlmostEqual(len(rec["raw_ppg"]), 60000, delta=2)
        self.assertAlmostEqual(len(rec["filtered_ppg"]), 60000, delta=2)
        self.assertAlmostEqual(len(rec["filtered_wide"]), 60000, delta=2)

        # Ground truth RR array at 1 Hz
        gt_rr = rec["gt_rr_1hz"]
        self.assertEqual(len(gt_rr), 480)
        valid_rr = gt_rr[np.isfinite(gt_rr)]
        self.assertGreater(len(valid_rr), 400, "Expected at least 400 valid 1 Hz RR annotations")
        self.assertTrue(np.all((valid_rr >= 6.0) & (valid_rr <= 45.0)))

    def test_unmodified_feature_extraction_single_window(self):
        """
        Verify that extract_window_features_from_segments() executes unmodified
        on a CapnoBase window segment and returns exactly 28 finite features.
        """
        rec = load_capnobase_record("0009")
        w_len = int(60 * self.target_fs)
        win_resp = rec["filtered_ppg"][:w_len]
        win_wide = rec["filtered_wide"][:w_len]

        feats = extract_window_features_from_segments(win_resp, win_wide, fs=self.target_fs)
        self.assertEqual(feats.shape, (28,), "Feature vector must be exactly 28-dimensional")
        self.assertEqual(feats.dtype, np.float32)
        self.assertFalse(np.any(np.isnan(feats)), "Feature vector contains unexpected NaNs")
        self.assertFalse(np.any(np.isinf(feats)), "Feature vector contains unexpected Infs")

    def test_cached_capnobase_features_npz(self):
        """
        Verify the cached CapnoBase features file exists and satisfies all
        validation dataset invariants.
        """
        self.assertTrue(
            self.cache_file.exists(),
            f"Cached CapnoBase file missing at {self.cache_file}"
        )

        X, y, pids = load_capnobase_dataset(cache_path=self.cache_file)

        # Feature matrix shape
        self.assertEqual(X.shape[1], 28, "Expected 28 features")
        self.assertGreaterEqual(X.shape[0], 8000, "Expected >= 8000 total 60s windows across 42 cases")
        self.assertEqual(len(y), len(X))
        self.assertEqual(len(pids), len(X))

        # Check zero NaNs across all rows and columns
        self.assertEqual(np.isnan(X).sum(), 0, "Feature matrix X contains NaNs")
        self.assertEqual(np.isnan(y).sum(), 0, "Target vector y contains NaNs")

        # Physiological bounds on ground truth
        self.assertGreaterEqual(float(y.min()), 6.0, "Target RR below physiological minimum (6 bpm)")
        self.assertLessEqual(float(y.max()), 45.0, "Target RR above physiological maximum (45 bpm)")

        # Unique patient count
        unique_pids, counts = np.unique(pids, return_counts=True)
        self.assertEqual(len(unique_pids), 42, "Expected exactly 42 unique CapnoBase cases")
        self.assertEqual(int(np.max(counts)), 211, "Max candidate windows per 480s recording must be 211")
        self.assertGreaterEqual(int(np.min(counts)), 150, "Min valid windows per recording should be >= 150")

    def test_zero_patient_overlap_with_bidmc(self):
        """
        Verify that CapnoBase patient IDs have 0% overlap with BIDMC cohort IDs,
        confirming strict external validity.
        """
        _, _, capno_pids = load_capnobase_dataset(cache_path=self.cache_file)
        capno_set = set(np.unique(capno_pids))
        bidmc_set = set(PATIENT_IDS)

        overlap = capno_set & bidmc_set
        self.assertEqual(len(overlap), 0, f"External validation leakage detected: {overlap}")

    def test_documentation_file_exists_and_contains_citation(self):
        """Verify the documentation and formal citation exist in output/paper/."""
        doc_path = PROJECT_ROOT / "output" / "paper" / "CAPNOBASE_EXTERNAL_VALIDATION.md"
        self.assertTrue(doc_path.exists(), f"Documentation file missing at {doc_path}")

        doc_text = doc_path.read_text(encoding="utf-8")
        self.assertIn("Karlen", doc_text)
        self.assertIn("Multiparameter Respiratory Rate Estimation", doc_text)
        self.assertIn("10.1109/TBME.2013.2246160", doc_text)
        self.assertIn("CapnoBase", doc_text)

    def test_capnobase_ventilation_metadata_and_stratification(self):
        """Verify ventilation mode parsing and exact case counts (23 spontaneous, 19 controlled)."""
        meta = parse_capnobase_subject_metadata()
        spont_cases = [c for c, m in meta.items() if not c.startswith('capno_') and m.get('case_ventilation') == 'spontaneous']
        vent_cases = [c for c, m in meta.items() if not c.startswith('capno_') and m.get('case_ventilation') == 'controlled']
        self.assertEqual(len(spont_cases), 23, "Expected 23 spontaneous breathing cases total.")
        self.assertEqual(len(vent_cases), 19, "Expected 19 mechanically ventilated cases total.")

        vent_strat = get_capnobase_ventilation_stratification()
        self.assertEqual(len(vent_strat["adult_spontaneous"]), 5, "Expected 5 adult spontaneous cases.")
        self.assertEqual(len(vent_strat["adult_ventilated"]), 8, "Expected 8 adult mechanically ventilated cases.")
        self.assertEqual(len(vent_strat["pediatric_spontaneous"]), 18, "Expected 18 pediatric spontaneous cases.")
        self.assertEqual(len(vent_strat["pediatric_ventilated"]), 11, "Expected 11 pediatric mechanically ventilated cases.")


if __name__ == "__main__":
    unittest.main()
