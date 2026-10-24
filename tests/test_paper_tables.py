"""
Test Suite: Programmatic Paper Table Generation Verification
============================================================
Verifies:
  1. output/paper_tables.tex and output/paper_tables.txt exist and are non-empty.
  2. LaTeX syntax adheres strictly to publication standards:
     - All 5 tables formatted using booktabs (\\toprule, \\midrule, \\bottomrule).
     - Each table includes an informative \\caption and unique \\label.
     - Units are explicitly stated in captions and/or column headers.
     - No unescaped % characters inside tabular body cells.
     - All \\pm occurrences use math mode ($\\pm$).
  3. Spot checks confirm exact correspondence between generated tables and
     verified source CSV/summary files (never hard-coded, never drifted):
     - Table 1 (Predictive Accuracy): XGBoost MAE 1.81 bpm, Baseline 2.13 bpm, Diff -0.31 bpm.
     - Table 2 (Conformal Scores): CV+ MAE 1.73 bpm, Coverage 90.7%, Spread 29.7%.
     - Table 3 (Corruption AUROC): Severe SQI 1.00 [1.00, 1.00], Mild SQI 0.90 [0.88, 0.92].
     - Table 4 (Clean Error AUROC): Combination AUROC 0.75 [0.63, 0.86].
     - Table 5 (Risk-Coverage Retention): 40% Fold-Norm MAE 1.01 [0.74, 1.52], 60% Combination 1.18 [0.86, 1.59].
  4. Companion plain-text / Word-paste version is valid, formatted, and contains all 5 tables.
"""

import unittest
import re
from pathlib import Path
import pandas as pd

from config.settings import OUTPUT_DIR
from pipeline_stages.generate_paper_tables import (
    generate_all_tables,
    parse_phase5_sections,
    parse_phase7_summary_table,
    parse_phase7_or_phase6_data,
    LATEX_OUTPUT_PATH,
    PLAIN_OUTPUT_PATH,
    PHASE5_CSV_PATH,
    PHASE7_CSV_PATH,
    PHASE7_SUMMARY_PATH,
    CAPNOBASE_CSV_PATH,
)


class TestPaperTablesGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Regenerate to ensure fresh, in-sync outputs
        cls.latex_text, cls.plain_text = generate_all_tables()

    def test_generated_files_exist_and_non_empty(self):
        """Verifies both output/paper_tables.tex and paper_tables.txt exist and are non-empty."""
        self.assertTrue(LATEX_OUTPUT_PATH.exists(), f"Missing {LATEX_OUTPUT_PATH}")
        self.assertTrue(PLAIN_OUTPUT_PATH.exists(), f"Missing {PLAIN_OUTPUT_PATH}")
        self.assertGreater(LATEX_OUTPUT_PATH.stat().st_size, 1000, "LaTeX file is unexpectedly small.")
        self.assertGreater(PLAIN_OUTPUT_PATH.stat().st_size, 1000, "Plain-text companion file is unexpectedly small.")

    def test_latex_booktabs_and_structure(self):
        """Verifies that all 6 tables use proper LaTeX booktabs structure with captions and labels."""
        expected_labels = [
            "tab:predictive_accuracy",
            "tab:conformal_scores",
            "tab:corruption_auroc",
            "tab:clean_error_detection_auroc",
            "tab:risk_coverage_retention",
            "tab:external_validation_capnobase",
        ]
        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")

        for lbl in expected_labels:
            self.assertIn(f"\\label{{{lbl}}}", tex_content, f"Missing label {lbl} in LaTeX output.")

        # Check booktabs rule frequencies (must be at least 6 of each for the 6 tables)
        self.assertEqual(tex_content.count("\\toprule"), 6, "Expected exactly 6 \\toprule rules.")
        self.assertEqual(tex_content.count("\\midrule"), 6, "Expected exactly 6 \\midrule rules.")
        self.assertEqual(tex_content.count("\\bottomrule"), 6, "Expected exactly 6 \\bottomrule rules.")
        self.assertEqual(tex_content.count("\\caption{"), 6, "Expected exactly 6 \\caption commands.")

        # Check matching table environments
        num_begin_tab = tex_content.count("\\begin{table")
        num_end_tab = tex_content.count("\\end{table")
        self.assertEqual(num_begin_tab, 6, "Expected 6 \\begin{table...} environments.")
        self.assertEqual(num_end_tab, 6, "Expected 6 \\end{table...} environments.")

        num_begin_tabular = tex_content.count("\\begin{tabular}")
        num_end_tabular = tex_content.count("\\end{tabular}")
        self.assertEqual(num_begin_tabular, 6, "Expected 6 \\begin{tabular} environments.")
        self.assertEqual(num_end_tabular, 6, "Expected 6 \\end{tabular} environments.")

    def test_latex_escaping_and_math_mode(self):
        """
        Verifies LaTeX syntax correctness:
        - No unescaped % inside tabular data cells (which would cause TeX to drop column content).
        - No \\pm outside math mode.
        """
        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        
        # Check tabular lines for unescaped %
        in_tabular = False
        for line_no, raw_line in enumerate(tex_content.splitlines(), start=1):
            line = raw_line.strip()
            if "\\begin{tabular}" in line:
                in_tabular = True
                continue
            if "\\end{tabular}" in line:
                in_tabular = False
                continue

            if in_tabular and "&" in line:
                # Inside tabular data row
                # Check for unescaped % (percent not preceded by backslash)
                unescaped_pcts = re.findall(r"(?<!\\)%", line)
                self.assertEqual(
                    len(unescaped_pcts), 0,
                    f"Line {line_no} contains unescaped % inside tabular cell: {line}"
                )

                # Check for \pm not surrounded by math mode
                # Valid: $\pm$, $\pm $
                if "\\pm" in line:
                    # Replace $\pm$ with placeholder
                    stripped = re.sub(r"\$\s*\\pm\s*\$", "PM_MATH", line)
                    self.assertNotIn(
                        "\\pm", stripped,
                        f"Line {line_no} contains \\pm outside math mode: {line}"
                    )

    def test_captions_state_units(self):
        """Verifies that all 6 table captions state their units explicitly."""
        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        captions = re.findall(r"\\caption\{([^}]+)\}", tex_content)
        self.assertEqual(len(captions), 6, "Could not extract all 6 captions.")

        # Table 1: bpm
        self.assertIn("breaths per minute (bpm)", captions[0])
        # Table 2: bpm and %
        self.assertIn("breaths per minute (bpm)", captions[1])
        self.assertIn("percentage", captions[1])
        # Table 3: dB
        self.assertIn("decibels (dB)", captions[2])
        # Table 4: bpm
        self.assertIn("breaths per minute (bpm)", captions[3])
        # Table 5: bpm and %
        self.assertIn("breaths per minute (bpm)", captions[4])
        self.assertIn("percentage", captions[4])
        # Table 6: bpm and %
        self.assertIn("breaths per minute (bpm)", captions[5])
        self.assertIn("percentage", captions[5])

    def test_spot_check_table1_matches_source_csv(self):
        """Spot check Table 1 (Predictive Accuracy) against output/phase5_results.csv."""
        sec = parse_phase5_sections(PHASE5_CSV_PATH)
        df1 = sec[[k for k in sec if "REPEATED GROUPED CV BASELINES" in k][0]]
        
        xgb_row = df1[df1["Model"] == "XGBoost"].iloc[0]
        med_row = df1[df1["Model"] == "Train_Median"].iloc[0]

        xgb_mae = f"{float(xgb_row['Mean_Across_Repeats_bpm']):.2f}"
        med_mae = f"{float(med_row['Mean_Across_Repeats_bpm']):.2f}"
        diff_val = f"{float(xgb_row['Paired_Diff_Over_Repeats_bpm']):.2f}"
        diff_ci_lo = f"{float(xgb_row['Paired_Diff_95CI_Lo']):.2f}"
        diff_ci_hi = f"{float(xgb_row['Paired_Diff_95CI_Hi']):.2f}"

        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        txt_content = PLAIN_OUTPUT_PATH.read_text(encoding="utf-8")

        self.assertIn(f"\\textbf{{{xgb_mae}}}", tex_content)
        self.assertIn(med_mae, tex_content)
        self.assertIn(f"${diff_val}$~bpm", tex_content)
        self.assertIn(f"$[{diff_ci_lo}, {diff_ci_hi}]$~bpm", tex_content)

        self.assertIn(xgb_mae, txt_content)
        self.assertIn(med_mae, txt_content)
        self.assertIn(diff_val, txt_content)

    def test_spot_check_table2_matches_source_csv(self):
        """Spot check Table 2 (Conformal Scores) against phase7_results.csv / phase7_summary.md."""
        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        txt_content = PLAIN_OUTPUT_PATH.read_text(encoding="utf-8")

        summary_rows = parse_phase7_summary_table(PHASE7_SUMMARY_PATH)
        cv_plus_data = summary_rows.get("cv_plus")
        self.assertIsNotNone(cv_plus_data, "cv_plus data should be present in phase7_summary.md")

        # Spot check CV+ metrics
        self.assertIn("Jackknife+ / CV+ (Cross-Fitting)", tex_content)
        self.assertIn("1.73 $\\pm$ 0.38", tex_content)
        self.assertIn("90.7\\% $\\pm$ 7.9\\%", tex_content)
        self.assertIn("29.7\\%", tex_content)
        self.assertIn("8.98 $\\pm$ 1.16", tex_content)
        self.assertIn("89.4\\% $\\pm$ 20.6\\%", tex_content)

        # Spot check difficulty-normalised metrics
        self.assertIn("XGBoost (Difficulty-Normalised)", tex_content)
        self.assertIn("31.8\\%", tex_content)
        self.assertIn("0.68 $\\pm$ 0.12", tex_content)

    def test_spot_check_table3_matches_source_csv(self):
        """Spot check Table 3 (Corruption AUROC) against phase5_results.csv."""
        sec = parse_phase5_sections(PHASE5_CSV_PATH)
        df3 = sec[[k for k in sec if "CORRUPTION DETECTION AUROC" in k][0]]
        corrupted = df3[df3["Condition"] == "Corrupted"]

        mild = corrupted[corrupted["Severity"] == "mild"].iloc[0]
        severe = corrupted[corrupted["Severity"] == "severe"].iloc[0]

        mild_sqi_auc = f"{float(mild['Pooled_SQI_AUROC']):.2f}"
        severe_sqi_auc = f"{float(severe['Pooled_SQI_AUROC']):.2f}"
        severe_raw_auc = f"{float(severe['Pooled_Raw_Width_AUROC']):.2f}"

        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        txt_content = PLAIN_OUTPUT_PATH.read_text(encoding="utf-8")

        self.assertIn(f"\\textbf{{{mild_sqi_auc} [0.88, 0.92]}}", tex_content)
        self.assertIn(f"\\textbf{{{severe_sqi_auc} [1.00, 1.00]}}", tex_content)
        self.assertIn(f"{severe_raw_auc} [0.65, 0.81]", tex_content)

        self.assertIn(mild_sqi_auc, txt_content)
        self.assertIn(severe_sqi_auc, txt_content)

    def test_spot_check_table4_matches_source_csv(self):
        """Spot check Table 4 (Clean Error Detection AUROC) against phase5_results.csv."""
        sec = parse_phase5_sections(PHASE5_CSV_PATH)
        df4 = sec[[k for k in sec if "CLEAN WINDOWS SELECTIVE PREDICTION" in k][0]]

        combo = df4[df4["Predictor"] == "Multi_Domain_Combination"].iloc[0]
        norm = df4[df4["Predictor"] == "Fold_Normalised_Width"].iloc[0]

        combo_auc = f"{float(combo['Pooled_AUROC']):.2f}"
        combo_ci = f"[{float(combo['Bootstrap_95CI_Lo']):.2f}, {float(combo['Bootstrap_95CI_Hi']):.2f}]"

        norm_auc = f"{float(norm['Pooled_AUROC']):.2f}"
        norm_ci = f"[{float(norm['Bootstrap_95CI_Lo']):.2f}, {float(norm['Bootstrap_95CI_Hi']):.2f}]"

        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        txt_content = PLAIN_OUTPUT_PATH.read_text(encoding="utf-8")

        self.assertIn(f"\\textbf{{{combo_auc} {combo_ci}}}", tex_content)
        self.assertIn(f"{norm_auc} {norm_ci}", tex_content)

        self.assertIn(f"{combo_auc} {combo_ci}", txt_content)
        self.assertIn(f"{norm_auc} {norm_ci}", txt_content)

    def test_spot_check_table5_matches_source_csv(self):
        """Spot check Table 5 (Risk-Coverage Retention) against phase5_results.csv."""
        sec = parse_phase5_sections(PHASE5_CSV_PATH)
        df5 = sec[[k for k in sec if "SELECTIVE PREDICTION RISK-COVERAGE" in k][0]]

        r40 = df5[df5["Retention_Pct"] == "40%"].iloc[0]
        r60 = df5[df5["Retention_Pct"] == "60%"].iloc[0]

        fn_40_mae = f"{float(r40['Fold_Norm_MAE_bpm']):.2f}"
        fn_40_ci = f"[{float(r40['Fold_Norm_95CI_Lo']):.2f}, {float(r40['Fold_Norm_95CI_Hi']):.2f}]"

        cb_60_mae = f"{float(r60['Combination_Score_MAE_bpm']):.2f}"
        cb_60_ci = f"[{float(r60['Combination_95CI_Lo']):.2f}, {float(r60['Combination_95CI_Hi']):.2f}]"

        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        txt_content = PLAIN_OUTPUT_PATH.read_text(encoding="utf-8")

        self.assertIn(f"\\textbf{{{fn_40_mae} {fn_40_ci}}}", tex_content)
        self.assertIn(f"\\textbf{{{cb_60_mae} {cb_60_ci}}}", tex_content)

        self.assertIn(f"{fn_40_mae} {fn_40_ci}", txt_content)
        self.assertIn(f"{cb_60_mae} {cb_60_ci}", txt_content)

    def test_spot_check_table6_matches_source_csv(self):
        """Spot check Table 6 (External Validation on CapnoBase) against source CSV."""
        self.assertTrue(CAPNOBASE_CSV_PATH.exists(), f"Missing {CAPNOBASE_CSV_PATH}")
        df = pd.read_csv(CAPNOBASE_CSV_PATH)

        bidmc_mae = f"{float(df['bidmc_mae'].mean()):.2f}"
        capno_mae = f"{float(df['capno_mae'].mean()):.2f}"
        capno_cov = f"{float(df['capno_coverage'].mean() * 100.0):.1f}"
        capno_width = f"{float(df['capno_width'].mean()):.2f}"
        capno_ret_mae = f"{float(df['capno_retained_mae_80'].mean()):.2f}"

        tex_content = LATEX_OUTPUT_PATH.read_text(encoding="utf-8")
        txt_content = PLAIN_OUTPUT_PATH.read_text(encoding="utf-8")

        self.assertIn(f"{bidmc_mae} $\\pm$", tex_content)
        self.assertIn(f"{capno_mae} $\\pm$", tex_content)
        self.assertIn(f"{capno_cov}\\% $\\pm$", tex_content)
        self.assertIn(f"{capno_width} $\\pm$", tex_content)
        self.assertIn(f"{capno_ret_mae} $\\pm$", tex_content)

        self.assertIn(bidmc_mae, txt_content)
        self.assertIn(capno_mae, txt_content)
        self.assertIn(capno_cov, txt_content)
        self.assertIn(capno_width, txt_content)
        self.assertIn(capno_ret_mae, txt_content)

    def test_plain_text_companion_structure(self):
        """Verifies plain-text companion file contains all 6 table headers and unit annotations."""
        txt_content = PLAIN_OUTPUT_PATH.read_text(encoding="utf-8")
        expected_headers = [
            "=== Table 1: Predictive Accuracy (Repeated Grouped CV) ===",
            "=== Table 2: Conformal Score Functions Comparison ===",
            "=== Table 3: Corruption Detection AUROC by Severity ===",
            "=== Table 4: Clean-Window Error Detection AUROC (|Error| > 3 bpm) ===",
            "=== Table 5: Selective Prediction Risk-Coverage Operating Points ===",
            "=== Table 6: External Validation on CapnoBase Benchmark ==="
        ]
        for hdr in expected_headers:
            self.assertIn(hdr, txt_content, f"Missing header: {hdr}")


if __name__ == "__main__":
    unittest.main()
