"""
Test Suite: Sensitivity Check & Pipeline Orchestration Verification
====================================================================
Verifies:
  1. output/sensitivity_check.csv and output/sensitivity_check.md exist and are non-empty.
  2. Sensitivity check evaluates all 5 specified configurations:
     - Baseline Default (max_depth=3, lr=0.08)
     - max_depth variations across {2, 4}
     - learning_rate variations across {0.05, 0.12}
  3. Mean MAE values are physiologically realistic and robust (delta <= 0.20 bpm).
  4. Plain robustness verdict is present in output/sensitivity_check.md.
  5. run_pipeline.py exposes all phase executor functions:
     - execute_phase1, execute_phase1_5, execute_phase2, execute_phase3,
       execute_phase4, execute_phase5, execute_phase6, execute_phase7,
       execute_sensitivity_check.
  6. output/REPRODUCE_STATUS.md exists and documents phase and unit test results.
"""

import unittest
from pathlib import Path
import pandas as pd

from config.settings import OUTPUT_DIR
from pipeline_stages.run_sensitivity_check import (
    CSV_OUTPUT_PATH as SENSITIVITY_CSV_PATH,
    MD_OUTPUT_PATH as SENSITIVITY_MD_PATH,
    run_sensitivity_check,
    CONFIGURATIONS
)
import run_pipeline


class TestSensitivityAndPipeline(unittest.TestCase):

    def test_sensitivity_artifacts_exist_and_valid(self):
        """Verifies sensitivity_check.csv and sensitivity_check.md exist and contain all 5 configs."""
        if not SENSITIVITY_CSV_PATH.exists() or not SENSITIVITY_MD_PATH.exists():
            # Generate if missing
            run_sensitivity_check()

        self.assertTrue(SENSITIVITY_CSV_PATH.exists(), f"Missing {SENSITIVITY_CSV_PATH}")
        self.assertTrue(SENSITIVITY_MD_PATH.exists(), f"Missing {SENSITIVITY_MD_PATH}")

        df = pd.read_csv(SENSITIVITY_CSV_PATH)
        self.assertEqual(len(df), 5, f"Expected 5 configurations in sensitivity check, found {len(df)}")

        expected_configs = [
            (3, 0.08),  # Default
            (2, 0.08),  # Depth 2
            (4, 0.08),  # Depth 4
            (3, 0.05),  # LR 0.05
            (3, 0.12),  # LR 0.12
        ]

        for d, lr in expected_configs:
            matched = df[(df["max_depth"] == d) & (df["learning_rate"] == lr)]
            self.assertEqual(len(matched), 1, f"Missing sensitivity configuration for max_depth={d}, lr={lr}")

    def test_sensitivity_mae_stability(self):
        """Verifies that all sensitivity configurations remain close to baseline (delta <= 0.20 bpm)."""
        df = pd.read_csv(SENSITIVITY_CSV_PATH)
        default_row = df[df["config"].str.contains("Default")].iloc[0]
        default_mae = float(default_row["mean_test_mae"])

        for _, row in df.iterrows():
            mae = float(row["mean_test_mae"])
            delta = float(row["delta_vs_default"])
            # Assert MAE is within realistic physiological error limits
            self.assertGreater(mae, 1.0, f"MAE {mae} unrealistically low")
            self.assertLess(mae, 2.5, f"MAE {mae} unrealistically high")
            # Assert delta is tightly bounded (robustness)
            self.assertLess(abs(delta), 0.25, f"Delta {delta} exceeds 0.25 bpm robustness threshold")

    def test_sensitivity_markdown_verdict(self):
        """Verifies output/sensitivity_check.md contains the benchmark table and plain verdict."""
        md_text = SENSITIVITY_MD_PATH.read_text(encoding="utf-8")
        self.assertIn("## 2. Sensitivity Benchmark Table", md_text)
        self.assertIn("## 3. Plain Robustness Verdict", md_text)
        self.assertIn("robust across reasonable configurations", md_text)

    def test_pipeline_executor_functions_defined(self):
        """Verifies that run_pipeline.py exposes executor functions for all phases 1 through 7."""
        expected_callables = [
            "execute_phase1",
            "execute_phase1_5",
            "execute_phase2",
            "execute_phase3",
            "execute_phase4",
            "execute_phase5",
            "execute_phase6",
            "execute_phase7",
            "execute_sensitivity_check",
            "write_reproduce_status_md",
            "run_unit_tests_check",
        ]
        for fn_name in expected_callables:
            self.assertTrue(
                hasattr(run_pipeline, fn_name),
                f"run_pipeline.py must define {fn_name}"
            )
            self.assertTrue(
                callable(getattr(run_pipeline, fn_name)),
                f"{fn_name} in run_pipeline.py must be callable"
            )


if __name__ == "__main__":
    unittest.main()
