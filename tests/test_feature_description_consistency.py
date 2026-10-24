"""
Test Suite: Feature Description Consistency & Documentation Integrity
======================================================================
Verifies:
  1. Both output/PAPER_RESULTS_REFERENCE.md and output/METHODS_CORRECTION.md exist.
  2. Both documents define exactly 28 total feature dimensions.
  3. Both documents name the identical four feature groups:
       - PPG Spectral Descriptors (4 features)
       - Pulse-Modulation Spectral Descriptors (12 features)
       - Cardiorespiratory / Pulse-Modulation Statistics (8 features)
       - Waveform Morphology & Spectral Sharpness Descriptors (4 features)
  4. Both documents reference all three pulse-modulation series (RIAV, RIIV, RIFV)
     and all 12 individual morphology and statistics features:
     fusion_rr, riav_mean, riav_std, riiv_mean, riiv_std, rifv_mean, rifv_std,
     mean_hr, sig_std, sig_ptp, sig_rms, sharpness.
  5. Neither document describes Daubechies/db4 wavelets, Hjorth parameters, or
     obsolete RIAM/RIFM/BW proxies in its active feature space.
  6. Verified directly against extract_window_features_from_segments() in
     src/features/extractor.py, ensuring the active pipeline outputs exactly 28 features.
"""

import unittest
import re
from pathlib import Path
import numpy as np

from config.settings import PAPER_DIR, OUTPUT_DIR, FS, WINDOW_SEC
from src.features.extractor import extract_window_features_from_segments


class TestFeatureDescriptionConsistency(unittest.TestCase):

    def setUp(self):
        self.ref_path = PAPER_DIR / "PAPER_RESULTS_REFERENCE.md"
        self.correction_path = PAPER_DIR / "METHODS_CORRECTION.md"

        self.assertTrue(self.ref_path.exists(), f"Missing {self.ref_path}")
        self.assertTrue(self.correction_path.exists(), f"Missing {self.correction_path}")

        self.ref_text = self.ref_path.read_text(encoding="utf-8")
        self.correction_text = self.correction_path.read_text(encoding="utf-8")

        # Extract active feature sections
        ref_match = re.search(
            r"-\s+\*\*Active Feature Space \(28 Dimensions\)\*\*:(.*?)(?=\n-\s+\*\*Regression Architecture|\n##|\Z)",
            self.ref_text,
            re.DOTALL
        )
        self.assertIsNotNone(ref_match, "Failed to isolate Active Feature Space section in PAPER_RESULTS_REFERENCE.md")
        self.ref_feature_section = ref_match.group(1)

        corr_match = re.search(
            r"## Detailed Feature Decomposition \(28 Dimensions Total\)(.*?)(?=\n##|\Z)",
            self.correction_text,
            re.DOTALL
        )
        self.assertIsNotNone(corr_match, "Failed to isolate Detailed Feature Decomposition section in METHODS_CORRECTION.md")
        self.correction_feature_section = corr_match.group(1)

    def test_total_dimensionality_matches_28(self):
        """Assert both documents explicitly assert 28 dimensions/features total."""
        self.assertIn("28 Dimensions", self.ref_text)
        self.assertIn("28 features", self.ref_text)

        self.assertIn("28 Dimensions Total", self.correction_text)
        self.assertIn("28 features", self.correction_text)

        # Code verification: extract_window_features_from_segments returns 28 features
        synthetic_win = np.sin(2 * np.pi * 0.25 * np.linspace(0, WINDOW_SEC, int(WINDOW_SEC * FS))).astype(np.float32)
        feats = extract_window_features_from_segments(synthetic_win, fs=FS)
        self.assertEqual(len(feats), 28, f"Active feature extractor must output 28 features, got {len(feats)}")

    def test_four_feature_groups_present_in_both(self):
        """Assert both documents name the identical four feature groups."""
        four_groups = [
            "PPG Spectral Descriptors",
            "Pulse-Modulation Spectral Descriptors",
            "Cardiorespiratory / Pulse-Modulation Statistics",
            "Waveform Morphology & Spectral Sharpness Descriptors"
        ]

        for group in four_groups:
            self.assertIn(
                group,
                self.ref_feature_section,
                f"Feature group '{group}' missing from PAPER_RESULTS_REFERENCE.md"
            )
            self.assertIn(
                group,
                self.correction_feature_section,
                f"Feature group '{group}' missing from METHODS_CORRECTION.md"
            )

    def test_all_individual_feature_names_present_in_both(self):
        """Assert all 12 individual morphology/statistics features and modulations are named in both."""
        required_features = [
            "fusion_rr",
            "riav_mean", "riav_std",
            "riiv_mean", "riiv_std",
            "rifv_mean", "rifv_std",
            "mean_hr",
            "sig_std", "sig_ptp", "sig_rms",
            "sharpness",
        ]
        for feat in required_features:
            self.assertIn(
                feat,
                self.ref_feature_section,
                f"Feature '{feat}' missing from PAPER_RESULTS_REFERENCE.md"
            )
            self.assertIn(
                feat,
                self.correction_feature_section,
                f"Feature '{feat}' missing from METHODS_CORRECTION.md"
            )

        # Pulse modulation series
        for mod in ["RIAV", "RIIV", "RIFV"]:
            self.assertIn(mod, self.ref_feature_section)
            self.assertIn(mod, self.correction_feature_section)

    def test_no_obsolete_features_in_active_descriptions(self):
        """Assert no wavelets, Hjorth, or RIAM/RIFM/BW proxies exist in active feature descriptions."""
        obsolete_terms = [
            "Daubechies",
            "db4",
            "Wavelet",
            "Hjorth",
            "RIAM",
            "RIFM",
            "baseline wander (BW)",
        ]
        for term in obsolete_terms:
            self.assertNotIn(
                term,
                self.ref_feature_section,
                f"Obsolete term '{term}' found in PAPER_RESULTS_REFERENCE.md active feature section!"
            )
            self.assertNotIn(
                term,
                self.correction_feature_section,
                f"Obsolete term '{term}' found in METHODS_CORRECTION.md active feature section!"
            )


if __name__ == "__main__":
    unittest.main()
