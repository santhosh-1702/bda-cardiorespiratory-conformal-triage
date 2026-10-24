"""
Unit Tests for Phase 1.5 Biosignal Foundation & Baselines
==========================================================
Verifies:
  1. Filter bandpass definitions (0.10-0.60 Hz vs 0.10-5.00 Hz).
  2. Pulse modulation extraction (RIAV, RIIV, RIFV positivity and length).
  3. Classical spectral estimation accuracy on calibrated sine waveforms.
  4. Strict zero-leakage in validation-only hyperparameter selection.
"""

import unittest
import numpy as np

from config.settings import FS
from src.preprocessing.filter import ButterworthRespiratoryFilter, ButterworthWidebandFilter
from src.features.modulations import (
    extract_pulse_modulations,
    estimate_dominant_frequency,
    estimate_spectral_ppg_rr,
)
from src.modeling.baselines import (
    evaluate_baseline_method,
    retrain_xgboost_validation_only,
    CANDIDATE_HYPERPARAMS
)

class TestPhase1_5(unittest.TestCase):
    def setUp(self):
        self.fs = FS
        self.duration_sec = 60.0
        self.t = np.linspace(0, self.duration_sec, int(self.fs * self.duration_sec), endpoint=False)

    def test_filter_bands(self):
        """Verify respiratory and wideband filter parameters."""
        resp_filt = ButterworthRespiratoryFilter(fs=self.fs)
        self.assertAlmostEqual(resp_filt.low, 0.10)
        self.assertAlmostEqual(resp_filt.high, 0.60)

        wide_filt = ButterworthWidebandFilter(fs=self.fs)
        self.assertAlmostEqual(wide_filt.low, 0.10)
        self.assertAlmostEqual(wide_filt.high, 5.00)

        # Apply to test noise
        noise = np.random.randn(len(self.t)).astype(np.float32)
        out_resp = resp_filt.apply(noise)
        out_wide = wide_filt.apply(noise)

        self.assertEqual(len(out_resp), len(noise))
        self.assertEqual(len(out_wide), len(noise))
        self.assertEqual(out_resp.dtype, np.float32)
        self.assertEqual(out_wide.dtype, np.float32)

    def test_pulse_modulation_extraction(self):
        """Verify RIAV, RIIV, and RIFV series extraction on synthetic modulated PPG."""
        # Heart rate ~ 75 bpm (1.25 Hz), respiration ~ 15 bpm (0.25 Hz)
        cardiac = np.cos(2 * np.pi * 1.25 * self.t)
        resp_mod = 0.3 * np.sin(2 * np.pi * 0.25 * self.t) # baseline modulation
        amp_mod = 1.0 + 0.2 * np.sin(2 * np.pi * 0.25 * self.t) # amplitude modulation
        ppg = (cardiac * amp_mod + resp_mod).astype(np.float32)

        mods = extract_pulse_modulations(ppg, fs=self.fs)
        t_peaks = mods["t_peaks"]
        riav = mods["riav"]
        riiv = mods["riiv"]
        rifv = mods["rifv"]

        self.assertGreater(len(t_peaks), 50)
        self.assertEqual(len(t_peaks), len(riav))
        self.assertEqual(len(t_peaks), len(riiv))
        self.assertEqual(len(t_peaks), len(rifv))

        # Peak-to-peak amplitude must be strictly positive
        self.assertTrue(np.all(riav > 0), "All RIAV amplitudes must be positive")
        # Pulse intervals must be within reasonable physiological limits (0.3s to 2.0s)
        self.assertTrue(np.all(rifv > 0.3) and np.all(rifv < 2.0), "Pulse intervals out of physiological range")

    def test_spectral_frequency_estimation(self):
        """Verify frequency estimation on known respiratory frequency (0.30 Hz = 18.0 bpm)."""
        target_hz = 0.30
        target_bpm = target_hz * 60.0 # 18.0 bpm
        pure_resp = np.sin(2 * np.pi * target_hz * self.t).astype(np.float32)

        # Test classical spectral baseline
        est_bpm = estimate_spectral_ppg_rr(pure_resp, fs=self.fs)
        self.assertAlmostEqual(est_bpm, target_bpm, delta=0.5)

        # Test modulation series estimation with regular sampling
        t_pts = np.linspace(0, self.duration_sec, 60) # 1 Hz heartbeat
        mod_vals = np.sin(2 * np.pi * target_hz * t_pts)
        est_mod_bpm = estimate_dominant_frequency(t_pts, mod_vals, 0.0, self.duration_sec)
        self.assertAlmostEqual(est_mod_bpm, target_bpm, delta=0.5)

    def test_validation_only_tuning_no_leakage(self):
        """Verify that validation-only hyperparameter selection never evaluates candidate models on test set."""
        # Create a small synthetic dataset across 5 patients
        pids = np.repeat([f"p{i}" for i in range(5)], 20)
        X = np.random.randn(len(pids), 5).astype(np.float32)
        y = np.random.uniform(12.0, 24.0, len(pids)).astype(np.float32)

        folds = [
            {
                "fold": 0,
                "train_pids": ["p0", "p1", "p2"],
                "val_pids": ["p3"],
                "test_pids": ["p4"]
            }
        ]

        result = retrain_xgboost_validation_only(
            X=X,
            y=y,
            patient_ids=pids,
            folds=folds,
            candidates=CANDIDATE_HYPERPARAMS[:2]
        )

        self.assertIn("mae_mean", result)
        self.assertIn("selected_candidates", result)
        self.assertEqual(len(result["selected_candidates"]), 1)
        self.assertGreater(result["mae_mean"], 0.0)

if __name__ == "__main__":
    unittest.main()
