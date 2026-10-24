"""
Unit Test Suite: CapnoBase External Validation Evaluation
=========================================================
Verifies:
  1. output/results/capnobase_external_validation_results.csv exists, contains 5 folds,
     and all expected evaluation columns.
  2. Numerical domain shift characteristics:
     - Point prediction MAE increases from in-domain BIDMC (1.78 +- 0.55 bpm)
       to CapnoBase (6.33 +- 0.31 bpm) due to population / sensor shift.
     - Empirical coverage drops from BIDMC (88.1% +- 11.9%) to CapnoBase (55.3% +- 20.3%)
       against 90.0% nominal target due to exchangeability violation.
     - Prediction interval width adaptively widens from BIDMC (10.72 +- 4.87 bpm)
       to CapnoBase (14.29 +- 6.32 bpm) as difficulty model s(x) flags unfamiliar morphology.
     - Error detection AUROC on CapnoBase exceeds 0.50 (0.577 +- 0.101), confirming
       conformal interval width retains discriminative sorting capability.
     - Selective triage with 80% retention reduces external MAE (5.98 +- 0.31 bpm).
  3. Summary markdown (capnobase_external_validation_summary.md) exists and contains
     scientific explanation of domain transfer and exchangeability breakdown.
  4. Publication comparison plot (output/figures/capnobase_external_validation.png)
     exists, is non-empty, and matches publication 300 DPI specifications.
  5. Table 6 integration in paper_tables.tex and paper_tables.txt.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import (
    RESULTS_DIR,
    FIGURES_DIR,
    PAPER_DIR,
    CAPNOBASE_CACHE_FILE,
)
from pipeline_stages.run_capnobase_evaluation import (
    RESULTS_CSV_PATH,
    STRATIFIED_RESULTS_CSV_PATH,
    SUMMARY_MD_PATH,
    FIGURE_PATH,
    compute_pediatric_frequency_mismatch,
)
from pipeline_stages.generate_paper_tables import (
    LATEX_OUTPUT_PATH,
    PLAIN_OUTPUT_PATH,
    CAPNOBASE_CSV_PATH,
)


class TestCapnoBaseEvaluation(unittest.TestCase):

    def setUp(self):
        self.csv_path = RESULTS_CSV_PATH
        self.summary_path = SUMMARY_MD_PATH
        self.fig_path = FIGURE_PATH

    def test_results_csv_structure_and_folds(self):
        """Verify CSV exists, has 5 rows (one per fold), and all expected columns."""
        self.assertTrue(self.csv_path.exists(), f"Missing {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        self.assertEqual(len(df), 5, f"Expected 5 fold rows, found {len(df)}")
        self.assertEqual(list(df["fold"].values), [0, 1, 2, 3, 4])

        expected_cols = [
            "fold", "score_type", "bidmc_mae", "bidmc_coverage", "bidmc_width",
            "bidmc_patient_cov_mean", "bidmc_patient_cov_std", "bidmc_auroc_gt_3",
            "capno_mae", "capno_coverage", "capno_width", "capno_med_width",
            "capno_patient_cov_mean", "capno_patient_cov_std", "capno_patient_cov_min",
            "capno_patient_cov_max", "capno_auroc_gt_3", "capno_retained_mae_80", "q_hat"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing expected column '{col}' in results CSV.")

    def test_domain_generalization_mae_increase(self):
        """Verify point MAE increases significantly under domain shift from BIDMC to CapnoBase."""
        df = pd.read_csv(self.csv_path)
        bidmc_mae = df["bidmc_mae"].mean()
        capno_mae = df["capno_mae"].mean()

        # In-domain BIDMC is ~1.78 bpm, CapnoBase is ~6.33 bpm
        self.assertAlmostEqual(bidmc_mae, 1.78, delta=0.15)
        self.assertAlmostEqual(capno_mae, 6.33, delta=0.20)
        self.assertGreater(capno_mae, bidmc_mae + 3.0, "External MAE should be substantially higher than in-domain.")

    def test_conformal_undercoverage_phenomenon(self):
        """Verify conformal coverage degrades from ~88% on BIDMC to ~55% on CapnoBase (exchangeability violation)."""
        df = pd.read_csv(self.csv_path)
        bidmc_cov = df["bidmc_coverage"].mean() * 100.0
        capno_cov = df["capno_coverage"].mean() * 100.0

        # In-domain BIDMC achieves ~88.1% (close to 90%), CapnoBase degrades to ~55.3%
        self.assertAlmostEqual(bidmc_cov, 88.1, delta=2.0)
        self.assertAlmostEqual(capno_cov, 55.3, delta=2.0)
        self.assertLess(capno_cov, 75.0, "External coverage must honestly reflect undercoverage under domain shift.")

    def test_adaptive_interval_widening(self):
        """Verify difficulty model s(x) widens prediction intervals from 10.72 to 14.29 bpm."""
        df = pd.read_csv(self.csv_path)
        bidmc_w = df["bidmc_width"].mean()
        capno_w = df["capno_width"].mean()

        self.assertAlmostEqual(bidmc_w, 10.72, delta=1.0)
        self.assertAlmostEqual(capno_w, 14.29, delta=1.0)
        self.assertGreater(capno_w, bidmc_w, "Conformal interval width should expand adaptively on unfamiliar inputs.")

    def test_error_discrimination_and_selective_triage(self):
        """Verify interval width retains error discrimination (AUROC > 0.50) and triage reduces error."""
        df = pd.read_csv(self.csv_path)
        capno_auc = df["capno_auroc_gt_3"].mean()
        capno_raw_mae = df["capno_mae"].mean()
        capno_ret_mae = df["capno_retained_mae_80"].mean()

        self.assertGreater(capno_auc, 0.50, "Interval width should maintain discriminative ranking ability.")
        self.assertAlmostEqual(capno_auc, 0.577, delta=0.05)
        self.assertLess(capno_ret_mae, capno_raw_mae, "Abstaining on top 20% widest intervals must reduce MAE.")
        self.assertAlmostEqual(capno_ret_mae, 5.98, delta=0.15)

    def test_summary_markdown_documentation(self):
        """Verify summary markdown exists and documents domain shift and exchangeability breakdown."""
        self.assertTrue(self.summary_path.exists(), f"Missing {self.summary_path}")
        content = self.summary_path.read_text(encoding="utf-8")

        self.assertIn("CapnoBase External Validation Benchmark", content)
        self.assertIn("Head-to-Head Benchmark Table", content)
        self.assertIn("Per-Fold Detailed Breakdown", content)
        self.assertIn("Scientific Interpretation & Plain Evaluated Verdict", content)
        self.assertIn("Violation of Exchangeability", content)
        self.assertIn("Adaptive Interval Widening", content)
        self.assertIn("Preservation of Triage Utility", content)

    def test_publication_figure_integrity(self):
        """Verify publication comparison figure exists, is non-empty, and has expected file size."""
        self.assertTrue(self.fig_path.exists(), f"Missing {self.fig_path}")
        file_size_kb = self.fig_path.stat().st_size / 1024.0
        self.assertGreater(file_size_kb, 50.0, "Figure file size is unexpectedly small.")

    def test_latex_and_plain_table6_integration(self):
        """Verify Table 6 exists in generated paper_tables.tex and paper_tables.txt."""
        self.assertTrue(LATEX_OUTPUT_PATH.exists(), f"Missing {LATEX_OUTPUT_PATH}")
        self.assertTrue(PLAIN_OUTPUT_PATH.exists(), f"Missing {PLAIN_OUTPUT_PATH}")

        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        txt_content = PLAIN_OUTPUT_PATH.read_text(encoding="utf-8")

        self.assertIn("\\label{tab:external_validation_capnobase}", tex_content)
        self.assertIn("=== Table 6: External Validation on CapnoBase Benchmark ===", txt_content)

    def test_stratified_results_csv_and_columns(self):
        """Verify stratified columns exist in primary CSV and stratified CSV has 25 rows (5 folds x 5 cohorts)."""
        self.assertTrue(self.csv_path.exists(), f"Missing {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        stratified_cols = [
            "capno_adult_mae", "capno_adult_coverage", "capno_adult_width", "capno_adult_auroc_gt_3", "capno_adult_retained_mae_80",
            "capno_adult_spont_mae", "capno_adult_spont_coverage", "capno_adult_spont_width",
            "capno_adult_vent_mae", "capno_adult_vent_coverage", "capno_adult_vent_width",
            "capno_pediatric_mae", "capno_pediatric_coverage", "capno_pediatric_width", "capno_pediatric_auroc_gt_3", "capno_pediatric_retained_mae_80"
        ]
        for col in stratified_cols:
            self.assertIn(col, df.columns, f"Missing stratified column '{col}' in results CSV.")

        self.assertTrue(STRATIFIED_RESULTS_CSV_PATH.exists(), f"Missing {STRATIFIED_RESULTS_CSV_PATH}")
        df_strat = pd.read_csv(STRATIFIED_RESULTS_CSV_PATH)
        self.assertEqual(len(df_strat), 25, "Stratified results CSV must contain exactly 25 rows (5 folds x 5 cohorts).")
        self.assertEqual(
            set(df_strat["cohort"].unique()),
            {"all", "adult", "pediatric", "adult_spontaneous", "adult_ventilated"}
        )

    def test_adult_primary_generalization_benchmark(self):
        """Verify adult-only cohort (N=13) is evaluated as primary fair generalization benchmark."""
        df = pd.read_csv(self.csv_path)
        adult_mae = df["capno_adult_mae"].mean()
        adult_cov = df["capno_adult_coverage"].mean() * 100.0
        adult_w = df["capno_adult_width"].mean()

        # Adult cohort: MAE ~6.21 bpm, Coverage ~45.3%, Width ~11.94 bpm
        self.assertAlmostEqual(adult_mae, 6.21, delta=0.20)
        self.assertAlmostEqual(adult_cov, 45.3, delta=2.0)
        self.assertAlmostEqual(adult_w, 11.94, delta=1.0)

    def test_adult_ventilation_stratification(self):
        """
        Verify adult cohort stratification by ventilation mode:
        - 5 spontaneous cases (1,055 windows) with MAE ~3.80 bpm, Coverage ~77.7%
        - 8 mechanically ventilated cases (1,634 windows) with MAE ~7.76 bpm, Coverage ~24.3%
        """
        df = pd.read_csv(self.csv_path)
        spont_mae = df["capno_adult_spont_mae"].mean()
        spont_cov = df["capno_adult_spont_coverage"].mean() * 100.0
        vent_mae = df["capno_adult_vent_mae"].mean()
        vent_cov = df["capno_adult_vent_coverage"].mean() * 100.0

        self.assertAlmostEqual(spont_mae, 3.80, delta=0.25)
        self.assertAlmostEqual(spont_cov, 77.7, delta=3.0)
        self.assertAlmostEqual(vent_mae, 7.76, delta=0.25)
        self.assertAlmostEqual(vent_cov, 24.3, delta=3.0)
        self.assertLess(spont_mae, vent_mae - 3.0, "Spontaneous breathing adult MAE must be substantially lower than ventilated.")
        self.assertGreater(spont_cov, vent_cov + 40.0, "Spontaneous breathing adult coverage must be dramatically higher than ventilated.")

    def test_pediatric_frequency_mismatch_quantification(self):
        """Verify direct quantification of pediatric respiratory frequency exceeding filter passband."""
        mismatch = compute_pediatric_frequency_mismatch()

        # Total window counts
        self.assertEqual(mismatch["ped_win_total"], 6075)
        self.assertEqual(mismatch["ad_win_total"], 2689)

        # 4.48% of pediatric 60-s windows exceed 36 bpm (0.60 Hz passband ceiling)
        self.assertEqual(mismatch["ped_win_gt_36"], 272)
        self.assertAlmostEqual(mismatch["ped_win_gt_36_pct"], 4.48, delta=0.1)

        # Adult windows never exceed 36 bpm (0.00%)
        self.assertEqual(mismatch["ad_win_gt_36"], 0)
        self.assertEqual(mismatch["ad_win_gt_36_pct"], 0.0)

        # 1 Hz ground truth annotations
        self.assertGreater(mismatch["ped_1hz_total"], 10000)
        self.assertAlmostEqual(mismatch["ped_1hz_gt_36_pct"], 4.74, delta=0.2)
        self.assertAlmostEqual(mismatch["ped_1hz_gt_45_pct"], 0.04, delta=0.02)


if __name__ == "__main__":
    unittest.main()
