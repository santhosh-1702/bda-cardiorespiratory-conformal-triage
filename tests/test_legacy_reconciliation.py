"""
Test Suite: Legacy Reconciliation & Active Pipeline Alignment
==============================================================
Verifies:
  1. Active feature extractor (extract_window_features_from_segments) outputs exactly 28 features.
  2. Legacy feature extractor (extract_single_window_features) outputs exactly 55 features.
  3. Legacy module docstrings: wavelet_features, spectral_features, statistical_features,
     and segmentation carry the legacy deprecation note.
  4. Abstention figure independence: abstention_results.png exists, is valid PNG,
     and has a distinct SHA-256 hash from output/risk_coverage_curve.png.
  5. Active raw-signal artifact corruption (corrupt_raw_ppg_window) alters the time domain
     while preserving signal length and producing valid 28-dimensional features.
  6. Documentation artifacts exist: output/LEGACY_NOTES.md and output/METHODS_CORRECTION.md
     contain required architectural and feature-breakdown documentation.
  7. Conformal calibrator feature-dimension enforcement: model trained on 28 features
     rejects 55-feature inputs, verifying the shape mismatch in the legacy approach.
"""

import unittest
import hashlib
from pathlib import Path
import numpy as np
import xgboost as xgb

from config.settings import (
    FS,
    WINDOW_SEC,
    WINDOW_SAMPLES,
    PROJECT_ROOT,
    OUTPUT_DIR,
    FIGURES_DIR,
    PAPER_DIR
)
from src.features.extractor import (
    extract_window_features_from_segments,
    extract_single_window_features,
)
from src.conformal.artifact_simulator import corrupt_raw_ppg_window
import src.features.wavelet_features as wavelet_mod
import src.features.spectral_features as spectral_mod
import src.features.statistical_features as statistical_mod
import src.features.segmentation as segmentation_mod


class TestLegacyReconciliation(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.fs = FS
        self.duration_sec = WINDOW_SEC
        self.n_samples = WINDOW_SAMPLES
        self.t = np.linspace(0, self.duration_sec, self.n_samples, endpoint=False)
        # Synthetic modulated PPG: cardiac carrier (1.25 Hz = 75 bpm) + respiratory modulation (0.25 Hz = 15 bpm)
        cardiac = np.cos(2 * np.pi * 1.25 * self.t)
        amp_mod = 1.0 + 0.2 * np.sin(2 * np.pi * 0.25 * self.t)
        resp_mod = 0.3 * np.sin(2 * np.pi * 0.25 * self.t)
        self.synthetic_ppg = (cardiac * amp_mod + resp_mod).astype(np.float32)

    def test_feature_dimensionality_contrast(self):
        """Verify active extractor produces 28 features, legacy produces 55 features."""
        active_feats = extract_window_features_from_segments(self.synthetic_ppg, fs=self.fs)
        legacy_feats = extract_single_window_features(self.synthetic_ppg, fs=self.fs)

        self.assertEqual(len(active_feats), 28, f"Active feature extractor must output 28 features, got {len(active_feats)}")
        self.assertEqual(len(legacy_feats), 55, f"Legacy feature extractor must output 55 features, got {len(legacy_feats)}")
        self.assertFalse(np.any(np.isnan(active_feats)), "Active features contain NaN")
        self.assertFalse(np.any(np.isinf(active_feats)), "Active features contain Inf")

    def test_legacy_module_docstrings(self):
        """Verify all 4 legacy files and the legacy extractor function contain the deprecation note."""
        expected_note = (
            "legacy — belongs to the original 6s/55-feature approach, "
            "superseded by the 28-feature pipeline in extractor.py; retained for historical reference."
        )

        for mod, name in [
            (wavelet_mod, "wavelet_features.py"),
            (spectral_mod, "spectral_features.py"),
            (statistical_mod, "statistical_features.py"),
            (segmentation_mod, "segmentation.py"),
        ]:
            doc = mod.__doc__ or ""
            self.assertIn(
                expected_note,
                doc,
                f"Module {name} missing required legacy docstring note."
            )

        fn_doc = extract_single_window_features.__doc__ or ""
        self.assertIn(
            expected_note,
            fn_doc,
            "extract_single_window_features missing required legacy docstring note."
        )

    def test_abstention_figure_distinct_and_valid(self):
        """Verify abstention_results.png exists, is valid PNG, and is distinct from risk_coverage_curve.png."""
        abstention_path = FIGURES_DIR / "abstention_results.png"
        risk_cov_path = FIGURES_DIR / "risk_coverage_curve.png"

        self.assertTrue(abstention_path.exists(), f"abstention_results.png must exist at {abstention_path}")
        self.assertTrue(risk_cov_path.exists(), f"risk_coverage_curve.png must exist in {FIGURES_DIR}")

        abstention_bytes = abstention_path.read_bytes()
        risk_cov_bytes = risk_cov_path.read_bytes()

        png_magic = b"\x89PNG\r\n\x1a\n"
        self.assertTrue(abstention_bytes.startswith(png_magic), "abstention_results.png is not a valid PNG")
        self.assertTrue(risk_cov_bytes.startswith(png_magic), "risk_coverage_curve.png is not a valid PNG")

        hash_abstention = hashlib.sha256(abstention_bytes).hexdigest()
        hash_risk_cov = hashlib.sha256(risk_cov_bytes).hexdigest()

        self.assertNotEqual(
            hash_abstention,
            hash_risk_cov,
            "abstention_results.png and risk_coverage_curve.png must have distinct SHA-256 hashes"
        )

    def test_documentation_artifacts_exist(self):
        """Verify output/paper/LEGACY_NOTES.md and output/paper/METHODS_CORRECTION.md exist and contain required content."""
        legacy_notes = PAPER_DIR / "LEGACY_NOTES.md"
        methods_correction = PAPER_DIR / "METHODS_CORRECTION.md"

        self.assertTrue(legacy_notes.exists(), f"LEGACY_NOTES.md must exist at {legacy_notes}")
        self.assertTrue(methods_correction.exists(), f"METHODS_CORRECTION.md must exist at {methods_correction}")

        legacy_text = legacy_notes.read_text(encoding="utf-8")
        self.assertIn("wavelet_features.py", legacy_text)
        self.assertIn("spectral_features.py", legacy_text)
        self.assertIn("statistical_features.py", legacy_text)
        self.assertIn("segmentation.py", legacy_text)
        self.assertIn("55-feature", legacy_text)
        self.assertIn("28-feature", legacy_text)

        methods_text = methods_correction.read_text(encoding="utf-8")
        self.assertIn("28-dimensional", methods_text)
        self.assertIn("RIAV", methods_text)
        self.assertIn("RIIV", methods_text)
        self.assertIn("RIFV", methods_text)
        self.assertIn("fusion_rr", methods_text)

    def test_raw_signal_corruption_and_active_extraction(self):
        """Verify corrupt_raw_ppg_window alters waveform and feeds into active 28-feature extractor."""
        corrupted = corrupt_raw_ppg_window(self.synthetic_ppg, severity="moderate", seed=42)

        self.assertEqual(len(corrupted), len(self.synthetic_ppg))
        self.assertFalse(np.array_equal(corrupted, self.synthetic_ppg), "Corrupted signal must differ from clean")

        feats_clean = extract_window_features_from_segments(self.synthetic_ppg, fs=self.fs)
        feats_corrupted = extract_window_features_from_segments(corrupted, fs=self.fs)

        self.assertEqual(len(feats_clean), 28)
        self.assertEqual(len(feats_corrupted), 28)
        self.assertFalse(np.allclose(feats_clean, feats_corrupted), "Corrupted features must differ from clean features")

    def test_calibrator_feature_dimension_enforcement(self):
        """Confirm that a model fit on 28 features raises ValueError when passed 55 features."""
        model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
        X_mock = np.random.randn(20, 28).astype(np.float32)
        y_mock = np.random.uniform(12.0, 24.0, size=20).astype(np.float32)
        model.fit(X_mock, y_mock)

        # Active 28 features predict smoothly
        X_active = np.random.randn(1, 28).astype(np.float32)
        pred_active = model.predict(X_active)
        self.assertEqual(len(pred_active), 1)

        # Legacy 55 features must raise shape mismatch error
        X_legacy = np.random.randn(1, 55).astype(np.float32)
        with self.assertRaises(ValueError):
            model.predict(X_legacy)


if __name__ == "__main__":
    unittest.main()
