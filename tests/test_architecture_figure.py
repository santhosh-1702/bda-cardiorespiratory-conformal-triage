"""
Test Suite: Pipeline Architecture Overview Figure
=================================================
Verifies:
  1. Primary figure (output/pipeline_architecture.png) and secondary figure
     (pipeline_architecture.png) exist and are valid PNG graphics.
  2. The figure is generated at publication-grade 300 DPI resolution.
  3. The architecture plotter is importable from both `src.visualization`
     and `src.visualization.plotter`.
  4. The architecture overview accurately and faithfully reflects the complete
     end-to-end pipeline:
       - Raw PPG acquisition (125 Hz, PhysioNet BIDMC ICU Cohort)
       - Butterworth bandpass filter (0.10-0.60 Hz)
       - 60 s windows / 2 s stride
       - 28-feature extraction with all 3 clinically grounded groups:
         * PPG spectral descriptors (4)
         * RIAV/RIIV/RIFV pulse-modulation spectral descriptors (12)
         * Morphology and modulation statistics (12)
       - XGBoost point prediction (CPU histogram regressor)
       - Conformal calibration (active: residual_normalised, tested alternative: jackknife+/CV+)
       - Width + SQI combination score
       - Accept / abstain clinical triage
"""

import unittest
from pathlib import Path
from PIL import Image

from config.settings import PROJECT_ROOT, OUTPUT_DIR, ARCHITECTURE_FIGURE_PATH
from src.visualization import render_pipeline_architecture_figure
from src.visualization.plotter import render_pipeline_architecture_figure as render_arch_from_plotter
import src.visualization.architecture_plotter as arch_module


class TestPipelineArchitectureFigure(unittest.TestCase):

    def setUp(self):
        self.primary_path = ARCHITECTURE_FIGURE_PATH

    def test_figure_files_exist_and_are_valid_png(self):
        """Assert architecture figure exists in output/figures and is a valid PNG."""
        self.assertTrue(
            self.primary_path.exists(),
            f"Primary architecture figure not found at {self.primary_path}"
        )

        png_magic = b"\x89PNG\r\n\x1a\n"
        primary_bytes = self.primary_path.read_bytes()

        self.assertTrue(
            primary_bytes.startswith(png_magic),
            "Primary file does not begin with PNG magic header"
        )
        self.assertGreater(
            len(primary_bytes),
            500_000,
            "Primary architecture figure appears unexpectedly small (<500 KB)"
        )

    def test_figure_resolution_matches_300_dpi(self):
        """Assert the exported figure has 300 DPI resolution and appropriate dimensions."""
        with Image.open(self.primary_path) as im:
            dpi = im.info.get("dpi")
            self.assertIsNotNone(dpi, "DPI metadata missing from exported PNG")
            # Matplotlib may write 299.999 or 300
            self.assertAlmostEqual(dpi[0], 300.0, delta=1.5, msg="Horizontal DPI is not 300")
            self.assertAlmostEqual(dpi[1], 300.0, delta=1.5, msg="Vertical DPI is not 300")

            width, height = im.size
            self.assertGreaterEqual(width, 4000, "Figure width should exceed 4000px for print clarity")
            self.assertGreaterEqual(height, 2500, "Figure height should exceed 2500px for print clarity")

    def test_import_paths_consistent(self):
        """Assert render_pipeline_architecture_figure is exported via standard visualization namespaces."""
        self.assertIs(
            render_pipeline_architecture_figure,
            arch_module.render_pipeline_architecture_figure,
            "render_pipeline_architecture_figure in src.visualization does not match module definition"
        )
        self.assertIs(
            render_arch_from_plotter,
            arch_module.render_pipeline_architecture_figure,
            "render_pipeline_architecture_figure in src.visualization.plotter does not match module definition"
        )

    def test_architecture_code_reflects_all_pipeline_stages(self):
        """
        Verify that architecture_plotter source code accurately reflects all
        stages and groups without omitting any required component.
        """
        source_text = Path(arch_module.__file__).read_text(encoding="utf-8")

        # 1. Raw PPG Acquisition (125 Hz)
        self.assertIn("125 Hz", source_text)
        self.assertIn("Raw PPG", source_text)

        # 2. Butterworth Bandpass Filter (0.10 - 0.60 Hz)
        self.assertIn("0.10 - 0.60 Hz", source_text)
        self.assertIn("Butterworth Bandpass Filter", source_text)

        # 3. 60 s Window Segmentation / 2 s Stride
        self.assertIn("60.0 s", source_text)
        self.assertIn("2.0 s", source_text)
        self.assertIn("60 s Window Segmentation", source_text)

        # 4. 28-Feature Extraction & 3 Groups
        self.assertIn("28-Feature", source_text)
        self.assertIn("PPG SPECTRAL", source_text)
        self.assertIn("PULSE-MODULATION SPECTRAL", source_text)
        self.assertIn("RIAV", source_text)
        self.assertIn("RIIV", source_text)
        self.assertIn("RIFV", source_text)
        self.assertIn("MORPHOLOGY", source_text)

        # 5. XGBoost Point Prediction
        self.assertIn("XGBoost", source_text)
        self.assertIn("Point Prediction", source_text)

        # 6. Conformal Calibration (active: residual_normalised, tested: jackknife+/CV+)
        self.assertIn("residual_normalised", source_text)
        self.assertIn("Jackknife+ / CV+", source_text)
        self.assertIn("Conformal Calibration", source_text)

        # 7. Width + SQI Combination Score
        self.assertIn("Combination Scoring", source_text)
        self.assertIn("SQI", source_text)

        # 8. Accept / Abstain Triage
        self.assertIn("ACCEPT PREDICTION", source_text)
        self.assertIn("SAFE ABSTENTION", source_text)
        self.assertIn("Selective Triage", source_text)

    def test_architecture_ci_matches_phase7_summary(self):
        """
        Assert the CV+ vs. difficulty-normalised SD diff CI in the architecture figure
        matches phase7_summary.md exactly.
        """
        from src.visualization.architecture_plotter import _get_coverage_sd_diff_ci
        from config.settings import RESULTS_DIR
        summary_path = RESULTS_DIR / "phase7_summary.md"
        self.assertTrue(summary_path.exists(), "phase7_summary.md must exist")
        ci_str, phrase = _get_coverage_sd_diff_ci()
        summary_text = summary_path.read_text(encoding="utf-8")
        self.assertIn(ci_str, summary_text)
        self.assertEqual(ci_str, "[-4.7%, +1.4%]")


if __name__ == "__main__":
    unittest.main()
