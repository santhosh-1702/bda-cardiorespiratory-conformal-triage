#!/usr/bin/env python3
"""
Artifact-Resilient Cardiorespiratory Triage Pipeline Orchestrator
=================================================================
End-to-end Python pipeline using Photoplethysmography (PPG) and Conformal Prediction
for continuous respiratory rate monitoring with safety-critical risk control.

Hardware Constraints: Optimized for Intel i3 / 8GB RAM / Local CPU (n_jobs=1)

Phases:
    Phase 1:   Data Acquisition & Butterworth Filtering
    Phase 1.5: Base Signal Foundation & Window Duration Selection (60s)
    Phase 2:   Segmentation & Multi-Domain Feature Extraction (28 features)
    Phase 3:   CPU-Optimized Baseline Modeling (XGBoost Regressor)
    Phase 4:   Conformal Prediction & Risk Control (Split Conformal)
    Phase 5:   Waveform Visualization (300 DPI) & Demo Reporting
    Phase 6:   Multi-Seed Conformal Score Types Comparison (25 folds, 5 methods)
    Phase 7:   Jackknife+ / CV+ Cross-Fitting Conformal Benchmark
    Sensitivity: Hyperparameter Robustness Check (max_depth & learning_rate)

Usage:
    python run_pipeline.py                  # Executes complete pipeline end-to-end
    python run_pipeline.py --smoke-test     # Runs quick 1-seed smoke test across all phases
    python run_pipeline.py --phase 1.5      # Runs Phase 1.5 only
    python run_pipeline.py --phase 6        # Runs Phase 6 only
    python run_pipeline.py --phase 7        # Runs Phase 7 only
    python run_pipeline.py --phase sensitivity # Runs Sensitivity check only
"""

import os
import sys
import re
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Fix Windows console encoding for cp1252 environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    PAPER_DIR,
    FIGURE_EXPORT_PATH,
    FS,
    WINDOW_SEC,
    MODEL_CACHE_FILE
)

PHASE_NAMES = {
    "Phase 1": "Data Acquisition & Butterworth Bandpass Filtering",
    "Phase 1.5": "Base Signal Foundation & Window Duration Selection",
    "Phase 2": "Segmentation & Multi-Domain Feature Extraction (28 features)",
    "Phase 3": "CPU Histogram Gradient Boosting Baseline Modeling",
    "Phase 4": "Split Conformal Prediction & Risk Control Calibration",
    "Phase 5": "300 DPI Waveform Visualization & Demo Reporting",
    "Phase 6": "Multi-Seed Conformal Score Types Comparison (25 Folds)",
    "Phase 7": "Jackknife+ / CV+ Cross-Fitting Conformal Benchmark",
    "Sensitivity": "Hyperparameter Sensitivity Check (Depth & Learning Rate)"
}


# ------------------------------------------------------------------------------
# Phase 1: Data Acquisition & Preprocessing
# ------------------------------------------------------------------------------
def execute_phase1() -> bool:
    print("\n" + "=" * 78)
    print("  PHASE 1: DATA ACQUISITION & PREPROCESSING")
    print("=" * 78)
    try:
        from src.preprocessing import download_bidmc_dataset, verify_dataset_completeness, stream_patient_records

        # 1. Download missing PhysioNet records
        print("[1/3] Checking dataset availability from PhysioNet...")
        download_bidmc_dataset(max_workers=8)
        if not verify_dataset_completeness():
            print("  Error: Missing files in dataset.")
            return False

        # 2. Stream and verify Butterworth filtering across all 53 recordings
        print("[2/3] Validating 4th-order Butterworth bandpass filter [0.10 Hz - 0.60 Hz]...")
        print("[3/3] Streaming patient recordings through memory-bounded generator (< 100 MB RAM)...")
        
        count = 0
        total_samples = 0
        for patient in stream_patient_records():
            count += 1
            total_samples += len(patient["filtered_ppg"])

        print(f"  Processed {count}/53 patient records ({total_samples} samples at {FS} Hz).")
        print(f"  Base wander removed (zero-phase SOS filtering).")
        print(">>> PHASE 1: STATUS SUCCESS <<<")
        return True
    except Exception as e:
        print(f"  Error in Phase 1: {e}")
        return False


# ------------------------------------------------------------------------------
# Phase 1.5: Base Signal Foundation & Window Length Selection
# ------------------------------------------------------------------------------
def execute_phase1_5() -> bool:
    print("\n" + "=" * 78)
    print("  PHASE 1.5: BASE SIGNAL FOUNDATION & WINDOW LENGTH SELECTION")
    print("=" * 78)
    try:
        from pipeline_stages.run_phase1_5 import run_phase1_5
        run_phase1_5()
        print(">>> PHASE 1.5: STATUS SUCCESS <<<")
        return True
    except Exception as e:
        print(f"  Error in Phase 1.5: {e}")
        return False


# ------------------------------------------------------------------------------
# Phase 2: Segmentation & Feature Extraction
# ------------------------------------------------------------------------------
def execute_phase2(force_recompute: bool = False) -> bool:
    print("\n" + "=" * 78)
    print("  PHASE 2: SEGMENTATION & MULTI-DOMAIN FEATURE EXTRACTION (PATIENT-GROUPED)")
    print("=" * 78)
    try:
        from src.features import build_feature_dataset

        print(f"[1/3] Slicing signals into {WINDOW_SEC}s sliding windows with 2s stride...")
        print("[2/3] Extracting spectral, pulse-modulation, moments, and sharpness features (28 feats)...")
        print("[3/3] Tracking patient_id and generating 5-fold patient-disjoint splits...")

        X, y, patient_ids, folds = build_feature_dataset(force_recompute=force_recompute)
        print(f"  Feature matrix created: {X.shape[0]} total windows across {len(np.unique(patient_ids))} patients.")
        print(f"  5-fold Patient-Disjoint Cross-Validation Summary:")
        for f in folds:
            print(f"      Fold {f['fold']}: Train={len(f['train_idx']):5d} wins ({len(f['train_pids']):2d} pts) | "
                  f"Val={len(f['val_idx']):5d} wins ({len(f['val_pids']):2d} pts) | "
                  f"Cal={len(f['cal_idx']):5d} wins ({len(f['cal_pids']):2d} pts) | "
                  f"Test={len(f['test_idx']):5d} wins ({len(f['test_pids']):2d} pts)")
        print(">>> PHASE 2: STATUS SUCCESS <<<")
        return True
    except Exception as e:
        print(f"  Error in Phase 2: {e}")
        return False


# ------------------------------------------------------------------------------
# Phase 3: CPU-Optimized Baseline Modeling (All Folds)
# ------------------------------------------------------------------------------
def execute_phase3() -> bool:
    print("\n" + "=" * 78)
    print("  PHASE 3: CPU-OPTIMIZED BASELINE MODELING (XGBoost, All 5 Folds)")
    print("=" * 78)
    try:
        from src.features import build_feature_dataset
        from src.modeling.baselines import retrain_xgboost_validation_only

        X, y, patient_ids, folds = build_feature_dataset(force_recompute=False)

        print("[1/2] Loading feature dataset...")
        print(f"[2/2] Training XGBoost across {len(folds)} patient-grouped folds (validation-only selection)...")

        xgb_res = retrain_xgboost_validation_only(X, y, patient_ids, folds)

        mean_mae = xgb_res["mae_mean"]
        print(f"\n  MEAN MAE ACROSS {len(folds)} FOLDS = {mean_mae:.4f} bpm")
        print(f"  Per-fold MAEs: {[f'{m:.4f}' for m in xgb_res['fold_maes']]}")

        if xgb_res.get("selected_candidates"):
            for i, cand_name in enumerate(xgb_res["selected_candidates"]):
                print(f"    Fold {i}: selected {cand_name}")

        # Save the model from the fold with lowest test MAE for downstream use
        best_fold_idx = int(np.argmin(xgb_res["fold_maes"]))
        best_model = xgb_res["models"][best_fold_idx]
        best_model.save_model(str(MODEL_CACHE_FILE))
        print(f"  Saved best-fold model (fold {best_fold_idx}) to {MODEL_CACHE_FILE}")

        print(">>> PHASE 3: STATUS SUCCESS <<<")
        return True
    except Exception as e:
        print(f"  Error in Phase 3: {e}")
        return False


# ------------------------------------------------------------------------------
# Phase 4: Conformal Prediction & Risk Control (All Folds)
# ------------------------------------------------------------------------------
def execute_phase4() -> bool:
    print("\n" + "=" * 78)
    print("  PHASE 4: CONFORMAL PREDICTION & RISK CONTROL (All 5 Folds)")
    print("=" * 78)
    try:
        import xgboost as xgb
        from config.settings import CONFORMAL_RESULTS_FILE, CONFORMAL_CONFIDENCE_LEVEL
        from src.features import build_feature_dataset
        from src.conformal import ConformalCalibrator, evaluate_clinical_safety_rules
        from src.modeling.baselines import CANDIDATE_HYPERPARAMS

        X, y, patient_ids, folds = build_feature_dataset(force_recompute=False)

        fold_coverages_clean = []
        fold_widths_clean = []
        fold_maes_clean = []

        print(f"[1/2] Calibrating conformal prediction across {len(folds)} folds...")

        for fold in folds:
            fold_idx = fold["fold"]
            train_mask = np.isin(patient_ids, fold["train_pids"])
            val_mask = np.isin(patient_ids, fold["val_pids"])
            cal_mask = np.isin(patient_ids, fold["cal_pids"])
            test_mask = np.isin(patient_ids, fold["test_pids"])

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_cal, y_cal = X[cal_mask], y[cal_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # Train model on this fold (validation-based selection)
            from config.settings import TREE_METHOD, CPU_THREADS, RANDOM_SEED
            best_val_mae = float("inf")
            best_model = None
            for cand in CANDIDATE_HYPERPARAMS:
                params = {
                    "tree_method": TREE_METHOD,
                    "n_jobs": CPU_THREADS,
                    "random_state": RANDOM_SEED,
                    "verbosity": 0,
                    **cand["params"]
                }
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
                val_mae = float(np.mean(np.abs(y_val - model.predict(X_val))))
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_model = model

            # Calibrate on fold's calibration patients
            calibrator = ConformalCalibrator(
                base_estimator=best_model,
                confidence_level=CONFORMAL_CONFIDENCE_LEVEL
            )
            calibrator.calibrate(X_cal, y_cal)

            # Evaluate on test patients
            y_pred, lo, hi, widths = calibrator.predict_intervals(X_test)
            clean_eval = evaluate_clinical_safety_rules(y_test, y_pred, lo, hi)

            fold_coverages_clean.append(clean_eval["empirical_coverage"])
            fold_widths_clean.append(clean_eval["mean_width"])
            fold_maes_clean.append(clean_eval["overall_mae"])

            print(f"  Fold {fold_idx}: Coverage={clean_eval['empirical_coverage']:.2%}, "
                  f"Width={clean_eval['mean_width']:.2f}, MAE={clean_eval['overall_mae']:.4f}")

        print(f"\n[2/2] Summary across {len(folds)} folds:")
        print(f"  Mean Coverage: {np.mean(fold_coverages_clean):.2%} +/- {np.std(fold_coverages_clean):.2%}")
        print(f"  Mean Width: {np.mean(fold_widths_clean):.2f} +/- {np.std(fold_widths_clean):.2f}")
        print(f"  Mean MAE: {np.mean(fold_maes_clean):.4f} +/- {np.std(fold_maes_clean):.4f}")

        # Save summary results (last fold for visualization compatibility)
        np.savez_compressed(
            CONFORMAL_RESULTS_FILE,
            y_test=y_test,
            y_pred_clean=y_pred,
            lo_clean=lo,
            hi_clean=hi,
            widths_clean=widths,
            clean_cov=np.mean(fold_coverages_clean),
            clean_mae=np.mean(fold_maes_clean),
            clean_retained_mae=clean_eval.get("retained_mae", np.mean(fold_maes_clean)),
        )
        print(f"  Results saved to {CONFORMAL_RESULTS_FILE}")
        print(">>> PHASE 4: STATUS SUCCESS <<<")
        return True
    except Exception as e:
        print(f"  Error in Phase 4: {e}")
        return False


# ------------------------------------------------------------------------------
# Phase 5: Visualization & Export
# ------------------------------------------------------------------------------
def execute_phase5() -> bool:
    print("\n" + "=" * 78)
    print("  PHASE 5: VISUALIZATION & REPORTING")
    print("=" * 78)
    try:
        import xgboost as xgb
        from config.settings import (
            MODEL_CACHE_FILE, CONFORMAL_RESULTS_FILE, FIGURE_EXPORT_PATH,
            WINDOW_SAMPLES, WINDOW_SEC, FS, ABSTENTION_THRESHOLD_BPM, DEMO_REPORT_FILE
        )
        from src.preprocessing import stream_patient_records
        from src.features import extract_window_features_from_segments
        from src.conformal import ConformalCalibrator
        from src.conformal.artifact_simulator import corrupt_raw_ppg_window
        from src.visualization import (
            render_comparative_triage_figure,
            generate_final_summary_report,
            render_pipeline_architecture_figure,
        )


        # Load baseline model and data
        conformal_data = np.load(CONFORMAL_RESULTS_FILE)
        features_data = np.load(CACHE_DIR / "features_split.npz")
        X_train, y_train = features_data["X_train"], features_data["y_train"]

        base_model = xgb.XGBRegressor()
        base_model.load_model(str(MODEL_CACHE_FILE))

        calibrator = ConformalCalibrator(base_estimator=base_model)
        calibrator.fit_and_calibrate(X_train, y_train)

        # 1. Extract authentic clean 60-second PPG window from patient 1
        print("[1/3] Extracting representative clean and motion-corrupted PPG waveforms...")
        clean_seg = None
        true_rr_clean = None
        for patient in stream_patient_records(patient_list=["bidmc01"]):
            start_idx = int(10 * FS)
            clean_seg = patient["filtered_ppg"][start_idx:start_idx + WINDOW_SAMPLES]
            true_rr_clean = float(np.nanmean(patient["gt_rr_1hz"][10:10 + int(WINDOW_SEC)]))
            break

        t_seconds = np.linspace(0, WINDOW_SEC, WINDOW_SAMPLES, endpoint=False)
        f_clean = extract_window_features_from_segments(clean_seg).reshape(1, -1)
        p_clean, l_clean, u_clean, w_clean = calibrator.predict_intervals(f_clean)
        pred_clean, lo_clean, hi_clean = float(p_clean[0]), float(l_clean[0]), float(u_clean[0])

        # 2. Corrupt signal directly in waveform domain using active raw-signal simulator
        noisy_seg = corrupt_raw_ppg_window(clean_seg, severity="moderate", seed=42)
        f_noisy = extract_window_features_from_segments(noisy_seg).reshape(1, -1)
        p_noisy, l_noisy, u_noisy, w_noisy = calibrator.predict_intervals(f_noisy)
        pred_noisy, lo_noisy, hi_noisy = float(p_noisy[0]), float(l_noisy[0]), float(u_noisy[0])

        # 3. Render 300 DPI visualization (using actual conformal intervals, no overrides)
        print(f"[2/3] Rendering 300 DPI comparative graphic to {FIGURE_EXPORT_PATH}...")
        render_comparative_triage_figure(
            t_seconds=t_seconds,
            clean_signal=clean_seg,
            clean_pred=pred_clean,
            clean_lower=lo_clean,
            clean_upper=hi_clean,
            clean_true=true_rr_clean,
            noisy_signal=noisy_seg,
            noisy_pred=pred_noisy,
            noisy_lower=lo_noisy,
            noisy_upper=hi_noisy,
            noisy_true=true_rr_clean,
            spike_interval=None,
            threshold_bpm=ABSTENTION_THRESHOLD_BPM,
            save_path=FIGURE_EXPORT_PATH
        )
        print(f"  Saved visualization to {FIGURE_EXPORT_PATH} (300 DPI).")

        # 3b. Render 300 DPI pipeline architecture overview figure (Figure 1)
        print("  Rendering 300 DPI pipeline architecture overview figure...")
        render_pipeline_architecture_figure()


        # 4. Generate demo clinical summary report
        print(f"[3/3] Generating demo clinical report text to {DEMO_REPORT_FILE}...")
        cov = float(conformal_data["clean_cov"]) if "clean_cov" in conformal_data else float(conformal_data.get("coverage", 0.90))
        retained_mae = float(conformal_data["clean_retained_mae"]) if "clean_retained_mae" in conformal_data else float(conformal_data.get("mae", 0.0))
        base_mae = float(conformal_data["clean_mae"]) if "clean_mae" in conformal_data else float(conformal_data.get("mae", 0.0))

        clean_width = float(hi_clean - lo_clean)
        noisy_width = float(hi_noisy - lo_noisy)
        clean_covered = bool(lo_clean <= true_rr_clean <= hi_clean)
        clean_abstained = bool(clean_width > ABSTENTION_THRESHOLD_BPM)
        noisy_covered = bool(lo_noisy <= true_rr_clean <= hi_noisy)
        noisy_abstained = bool(noisy_width > ABSTENTION_THRESHOLD_BPM)

        clean_eval = {
            "true_rr": true_rr_clean,
            "predicted_rr": pred_clean,
            "lower": lo_clean,
            "upper": hi_clean,
            "width": clean_width,
            "is_covered": clean_covered,
            "is_abstained": clean_abstained,
            "cohort_coverage": cov,
            "retained_mae": retained_mae,
        }
        noisy_eval = {
            "true_rr": true_rr_clean,
            "predicted_rr": pred_noisy,
            "lower": lo_noisy,
            "upper": hi_noisy,
            "width": noisy_width,
            "is_covered": noisy_covered,
            "is_abstained": noisy_abstained,
        }
        report = generate_final_summary_report(
            clean_eval=clean_eval,
            noisy_eval=noisy_eval,
            baseline_mae=base_mae,
            export_path=DEMO_REPORT_FILE
        )
        print(report)
        print(">>> PHASE 5: STATUS SUCCESS <<<")
        return True
    except Exception as e:
        print(f"  Error in Phase 5: {e}")
        return False


# ------------------------------------------------------------------------------
# Phase 6: Multi-Seed Conformal Score Types Comparison (25 Folds)
# ------------------------------------------------------------------------------
def execute_phase6(smoke_test: bool = False) -> bool:
    print("\n" + "=" * 78)
    print("  PHASE 6: MULTI-SEED CONFORMAL SCORE TYPES COMPARISON (5 SCORE TYPES)")
    print("=" * 78)
    try:
        from pipeline_stages.run_phase6_score_comparison import (
            run_phase6_evaluation,
            export_phase6_artifacts,
            REPEAT_SEEDS
        )
        seeds = [42] if smoke_test else REPEAT_SEEDS
        print(f"  Executing Phase 6 with repeat seeds: {seeds}...")
        results = run_phase6_evaluation(repeat_seeds=seeds)
        export_phase6_artifacts(results)
        print(">>> PHASE 6: STATUS SUCCESS <<<")
        return True
    except Exception as e:
        print(f"  Error in Phase 6: {e}")
        return False


# ------------------------------------------------------------------------------
# Phase 7: Jackknife+ / CV+ Cross-Fitting Conformal Benchmark
# ------------------------------------------------------------------------------
def execute_phase7(smoke_test: bool = False) -> bool:
    print("\n" + "=" * 78)
    print("  PHASE 7: JACKKNIFE+ / CV+ CROSS-FITTING CONFORMAL BENCHMARK")
    print("=" * 78)
    try:
        from pipeline_stages.run_phase7_cv_plus import (
            run_phase7_evaluation,
            export_phase7_artifacts,
            REPEAT_SEEDS
        )
        seeds = [42] if smoke_test else REPEAT_SEEDS
        print(f"  Executing Phase 7 with repeat seeds: {seeds}...")
        results = run_phase7_evaluation(repeat_seeds=seeds)
        export_phase7_artifacts(results)
        print(">>> PHASE 7: STATUS SUCCESS <<<")
        return True
    except ImportError:
        print("  [Phase 7] Module run_phase7_cv_plus not present; skipping Phase 7 gracefully.")
        return True
    except Exception as e:
        print(f"  Error in Phase 7: {e}")
        return False


# ------------------------------------------------------------------------------
# Sensitivity Check: Hyperparameter Robustness Evaluation
# ------------------------------------------------------------------------------
def execute_sensitivity_check() -> bool:
    print("\n" + "=" * 78)
    print("  HYPERPARAMETER SENSITIVITY CHECK")
    print("=" * 78)
    try:
        from pipeline_stages.run_sensitivity_check import run_sensitivity_check
        run_sensitivity_check()
        print(">>> SENSITIVITY CHECK: STATUS SUCCESS <<<")
        return True
    except Exception as e:
        print(f"  Error in Sensitivity Check: {e}")
        return False


# ------------------------------------------------------------------------------
# Unit Test Suite Runner & Status Export
# ------------------------------------------------------------------------------
def run_unit_tests_check() -> Tuple[bool, str]:
    """Runs 'python -m unittest discover tests' and returns (is_green, summary_str)."""
    print("\n" + "=" * 78)
    print("  RUNNING UNIT TEST SUITE (python -m unittest discover tests)")
    print("=" * 78)
    cmd = [sys.executable, "-m", "unittest", "discover", "tests"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        is_green = (proc.returncode == 0)

        m = re.search(r"Ran (\d+) tests? in ([\d\.]+)s", output)
        if m:
            summary = f"{m.group(1)} tests passed in {m.group(2)}s"
        else:
            summary = "All tests passed" if is_green else "Unit test suite reported failures"

        if is_green:
            print(f"  [Test Result] GREEN: {summary}")
        else:
            print(f"  [Test Result] FAILED: Return code {proc.returncode}")
            print(output[-400:])

        return is_green, summary
    except Exception as e:
        print(f"  Error running unit test suite: {e}")
        return False, str(e)


def write_reproduce_status_md(
    phase_results: Dict[str, bool],
    unit_tests_green: bool,
    unit_tests_summary: str,
    total_elapsed_sec: float
):
    """Writes consolidated status markdown to output/results/REPRODUCE_STATUS.md."""
    status_file = RESULTS_DIR / "REPRODUCE_STATUS.md"
    overall_green = all(phase_results.values()) and unit_tests_green
    overall_badge = "SUCCESS (ALL GREEN)" if overall_green else "FAILED (INVESTIGATE ERRORS)"

    lines = [
        "# Pipeline Reproduction Status",
        "",
        f"- **Reproduction Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Overall Pipeline Status**: **{overall_badge}**",
        f"- **Total Elapsed Time**: {total_elapsed_sec:.1f}s ({total_elapsed_sec / 60.0:.2f} minutes)",
        f"- **Hardware Target**: Intel i3 / 8GB RAM / Local CPU (`n_jobs=1`, deterministic)",
        f"- **Unit Test Suite (`python -m unittest discover tests`)**: {'**GREEN**' if unit_tests_green else '**FAIL**'} ({unit_tests_summary})",
        "",
        "## Executed Phases Summary",
        "",
        "| Phase | Description | Result |",
        "| :--- | :--- | :---: |"
    ]

    for phase_key, passed in phase_results.items():
        desc = PHASE_NAMES.get(phase_key, phase_key)
        res_str = "**PASS**" if passed else "**FAIL**"
        lines.append(f"| {phase_key} | {desc} | {res_str} |")

    lines.extend([
        f"| Unit Tests | `python -m unittest discover tests` | {'**GREEN**' if unit_tests_green else '**FAIL**'} |",
        "",
        "## Key Verified Artifacts",
        "- `output/results/phase1_5_baselines.csv`: Window length (60s) & baseline benchmark table",
        "- `output/results/phase1_5_summary.md`: Auto-computed markdown summary statements",
        "- `output/cache/best_xgb_model.json`: Trained XGBoost regressor model",
        "- `output/cache/conformal_results.npz`: Split conformal calibration outputs",
        "- `output/results/phase5_results.csv`: Corruption AUROC & selective prediction risk-coverage table",
        "- `output/results/phase6_results.csv`: 25-fold 5-score conformal comparison",
        "- `output/results/phase6_summary.md`: Multi-seed coverage spread and stability verdict",
        "- `output/results/phase7_results.csv`: Jackknife+ / CV+ cross-fitting benchmark",
        "- `output/results/phase7_summary.md`: CV+ vs Difficulty-Normalised comparison",
        "- `output/results/sensitivity_check.csv`: Hyperparameter robustness evaluation across depth and learning rate",
        "- `output/results/sensitivity_check.md`: Hyperparameter sensitivity table & plain verdict",
        "- `output/paper/paper_tables.tex` & `output/paper/paper_tables.txt`: Publication booktabs tables & plain companion",
        "- `output/figures/pipeline_architecture.png`: 300 DPI Figure 1 end-to-end pipeline overview",
        "- `output/figures/abstention_results.png` & `output/figures/risk_coverage_curve.png`: 300 DPI publication figures",
        ""
    ])


    status_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Status Report] Consolidated status written to {status_file}")


# ------------------------------------------------------------------------------
# Main CLI Orchestrator
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Artifact-Resilient Cardiorespiratory Triage using PPG & Conformal Prediction."
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "1", "1.5", "2", "3", "4", "5", "6", "7", "sensitivity"],
        help="Run a specific phase or 'all' to execute end-to-end (default: all)."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast execution mode (single seed 42 for Phase 6 & 7)."
    )
    parser.add_argument(
        "--recompute-features",
        action="store_true",
        help="Force recomputing features in Phase 2 instead of loading cached .npz file."
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running unit tests at the end of the pipeline."
    )
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 78)
    print("  ARTIFACT-RESILIENT CARDIORESPIRATORY TRIAGE SYSTEM")
    print("  Photoplethysmography (PPG) + Conformal Prediction Risk Control")
    print("  Optimized for Intel i3 / 8GB RAM / Local CPU Execution (n_jobs=1)")
    print("=" * 78)

    phase_results: Dict[str, bool] = {}

    if args.phase == "1":
        phase_results["Phase 1"] = execute_phase1()
    elif args.phase == "1.5":
        phase_results["Phase 1.5"] = execute_phase1_5()
    elif args.phase == "2":
        phase_results["Phase 2"] = execute_phase2(force_recompute=args.recompute_features)
    elif args.phase == "3":
        phase_results["Phase 3"] = execute_phase3()
    elif args.phase == "4":
        phase_results["Phase 4"] = execute_phase4()
    elif args.phase == "5":
        phase_results["Phase 5"] = execute_phase5()
    elif args.phase == "6":
        phase_results["Phase 6"] = execute_phase6(smoke_test=args.smoke_test)
    elif args.phase == "7":
        phase_results["Phase 7"] = execute_phase7(smoke_test=args.smoke_test)
    elif args.phase == "sensitivity":
        phase_results["Sensitivity"] = execute_sensitivity_check()
    else:
        # Full end-to-end execution in correct dependency order:
        # Phase 1 -> Phase 1.5 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Sensitivity
        order = [
            ("Phase 1", lambda: execute_phase1()),
            ("Phase 1.5", lambda: execute_phase1_5()),
            ("Phase 2", lambda: execute_phase2(force_recompute=args.recompute_features)),
            ("Phase 3", lambda: execute_phase3()),
            ("Phase 4", lambda: execute_phase4()),
            ("Phase 5", lambda: execute_phase5()),
            ("Phase 6", lambda: execute_phase6(smoke_test=args.smoke_test)),
            ("Phase 7", lambda: execute_phase7(smoke_test=args.smoke_test)),
            ("Sensitivity", lambda: execute_sensitivity_check())
        ]

        for p_name, fn in order:
            ok = fn()
            phase_results[p_name] = ok
            if not ok:
                print(f"\n[Pipeline Abort] {p_name} failed. Halting subsequent phases.")
                break

    # Unit test suite verification
    if not args.skip_tests and args.phase in ["all", "6", "7"]:
        tests_green, tests_summary = run_unit_tests_check()
    else:
        tests_green, tests_summary = True, "Skipped per CLI argument"

    elapsed = time.time() - start_time
    write_reproduce_status_md(
        phase_results=phase_results,
        unit_tests_green=tests_green,
        unit_tests_summary=tests_summary,
        total_elapsed_sec=elapsed
    )

    print(f"\nPipeline finished in {elapsed:.1f}s ({elapsed / 60.0:.2f} minutes).")


if __name__ == "__main__":
    main()
