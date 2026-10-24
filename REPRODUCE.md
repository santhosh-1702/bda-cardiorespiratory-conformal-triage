# End-to-End Pipeline Reproduction Guide
**Artifact-Resilient Cardiorespiratory Triage Using Photoplethysmography (PPG) and Conformal Prediction**
*PhysioNet BIDMC Cohort (53 ICU Patients) | 5-Fold Patient-Disjoint Cross-Validation*

---

## 1. System Requirements & Hardware Target

This pipeline is strictly engineered and verified for lightweight local execution:
- **Target Hardware**: Intel Core i3 (dual-core / quad-thread), 8 GB RAM, no GPU required (strictly CPU-only).
- **Execution Constraints**: Deterministic execution (`n_jobs=1`), fixed random seeds, memory-bounded batch streaming (< 100 MB RAM during signal preprocessing).
- **Software Requirements**: Python 3.10+ (tested on Python 3.10 through 3.14 on Windows and Linux).

### Environment Setup
```bash
# Clone the repository and enter directory
git clone <repository_url>
cd "BDA PROJECT"

# Install dependencies
pip install -r requirements.txt
```

---

## 2. One-Command Full Pipeline Reproduction

To execute all phases sequentially from raw data preprocessing through cross-conformal benchmarking and automated verification:

```bash
python run_pipeline.py
```

### Expected Runtime on Target Hardware (Intel i3 / 8GB RAM)
| Mode | Command | Expected Duration | Description |
| :--- | :--- | :---: | :--- |
| **Standard Full Run** | `python run_pipeline.py` | ~2–3 minutes | Runs Phases 1–7 (using cached 28-feature matrix), sensitivity check, and unit test suite. |
| **Quick Smoke Test** | `python run_pipeline.py --smoke-test` | ~60–90 seconds | Runs all phases end-to-end with single-seed evaluation (Seed 42) for Phases 6 & 7. |
| **Cold Start (From Scratch)** | `python run_pipeline.py --recompute-features` | ~6–8 minutes | Re-streams all 53 raw BIDMC records, re-extracts all 28 features, and runs all phases. |

---

## 3. Pipeline Phases & Dependency Order

The pipeline executes in strict dependency order, ensuring that each phase receives verified inputs from preceding stages:

1. **Phase 1: Data Acquisition & Preprocessing** (`python run_pipeline.py --phase 1`)
   - Verifies BIDMC PhysioNet dataset (53 patients, 125 Hz).
   - Validates zero-phase 4th-order Butterworth bandpass filter [0.10–0.60 Hz].
2. **Phase 1.5: Base Signal Foundation & Window Selection** (`python run_pipeline.py --phase 1.5` or `python pipeline_stages/run_phase1_5.py`)
   - Evaluates window lengths (6s, 15s, 30s, 60s) using validation folds only.
   - Selects 60-second window duration across all folds (`output/results/phase1_5_baselines.csv`).
3. **Phase 2: Patient-Grouped Feature Engineering** (`python run_pipeline.py --phase 2` or `python pipeline_stages/run_phase2.py`)
   - Slices signals into 60s windows with 2s stride (11,026 clean windows total).
   - Extracts 28 multi-domain features (PPG spectral descriptors, RIAV/RIIV/RIFV pulse modulations, moments, and spectral peak sharpness).
   - Partitions into 5 strictly patient-disjoint folds (`output/cache/fold_map.json`).
4. **Phase 3: CPU Baseline Modeling** (`python run_pipeline.py --phase 3`)
   - Trains histogram XGBoost regressor on CPU with validation-only tuning.
   - Saves best model to `output/cache/best_xgb_model.json`.
5. **Phase 4: Conformal Prediction & Risk Control** (`python run_pipeline.py --phase 4` or `python pipeline_stages/run_phase4.py`)
   - Calibrates split conformal prediction at 90.0% nominal confidence.
   - Applies clinical abstention threshold (6.0 bpm width).
   - Exports `output/cache/conformal_results.npz`.
6. **Phase 5: Visualization & Demo Reporting** (`python run_pipeline.py --phase 5`)
   - Renders 300 DPI publication graphic (`output/figures/abstention_results.png`).
   - Renders 300 DPI Figure 1 pipeline architecture diagram (`output/figures/pipeline_architecture.png`).
   - Generates representative single-window clinical demo report (`output/results/demo_window_report.txt`).
7. **Phase 6: Multi-Seed Conformal Score Types Comparison** (`python run_pipeline.py --phase 6` or `python pipeline_stages/run_phase6_score_comparison.py`)
   - Evaluates 5 conformal score types across 5 repeat seeds (25 folds total).
   - Exports full 125-run breakdown to `output/results/phase6_results.csv` and `output/results/phase6_summary.md`.
8. **Phase 7: Jackknife+ / CV+ Cross-Fitting Benchmark** (`python run_pipeline.py --phase 7` or `python pipeline_stages/run_phase7_cv_plus.py`)
   - Pools out-of-fold residuals across ~42–44 non-test patients per fold via inner GroupKFold cross-fitting.
   - Exports results to `output/results/phase7_results.csv` and `output/results/phase7_summary.md`.
9. **Sensitivity Check** (`python run_pipeline.py --phase sensitivity` or `python pipeline_stages/run_sensitivity_check.py`)
   - Retrains and tests tree depth across {2, 3, 4} and learning rate across {0.05, 0.08, 0.12} against default.
   - Exports `output/results/sensitivity_check.csv` and `output/results/sensitivity_check.md`.
10. **Paper Tables Generation** (`python pipeline_stages/generate_paper_tables.py`)
    - Programmatically formats publication booktabs LaTeX tables (`output/paper/paper_tables.tex`) and companion plain-text tables (`output/paper/paper_tables.txt`).

---

## 4. Automated Verification & Testing

Run the automated test suite before and after any modification:

```bash
python -m unittest discover tests
```

Expected result: **All 72 unit tests pass in ~2 seconds (100% green)**.

Key verification tests include:
- `tests/test_no_leakage.py`: Asserts zero patient overlap across train, val, cal, and test sets.
- `tests/test_phase2.py`: Asserts bit-identical determinism and cohort window assertions.
- `tests/test_auroc_alignment.py`: Asserts bug-free AUROC alignment against independent ground truth.
- `tests/test_paper_tables.py`: Asserts exact spot-check correspondence between generated LaTeX tables and source CSV files.
- `tests/test_architecture_figure.py`: Asserts publication-grade 300 DPI resolution and faithful stage representation.

---

## 5. Success Confirmation Artifacts Checklist

Upon completion of `python run_pipeline.py`, verify the following primary artifacts in `output/`:

| Artifact | File Path | Expected Contents / Verification Criteria |
| :--- | :--- | :--- |
| **Reproduction Status** | `output/results/REPRODUCE_STATUS.md` | Consolidated report confirming all phases and unit tests passed. |
| **Window Selection** | `output/results/phase1_5_baselines.csv` | 60s window selected across all 5 folds (MAE: 1.81 bpm). |
| **Model Checkpoint** | `output/cache/best_xgb_model.json` | Valid serialized XGBoost regressor model. |
| **Conformal Calibration** | `output/cache/conformal_results.npz` | Clean coverage ~90%, test predictions, prediction interval bounds. |
| **Selective Risk-Coverage** | `output/results/phase5_results.csv` | AUROC by corruption severity and retention operating points (40%–100%). |
| **Score Comparison** | `output/results/phase6_results.csv` | 25-fold results across all 5 split-conformal methods. |
| **CV+ Cross-Fitting** | `output/results/phase7_results.csv` | Jackknife+ / CV+ benchmark across 25 folds (coverage: 90.7% ± 7.9%). |
| **Sensitivity Check** | `output/results/sensitivity_check.csv` | MAE variation across depth {2,3,4} and learning rate {0.05,0.08,0.12}. |
| **Sensitivity Summary** | `output/results/sensitivity_check.md` | Plain robustness verdict confirming MAE moves by ≤ 0.10 bpm. |
| **Manuscript Tables (LaTeX)** | `output/paper/paper_tables.tex` | Booktabs LaTeX tables for all 5 manuscript sections. |
| **Manuscript Tables (Plain)** | `output/paper/paper_tables.txt` | Monospace companion tables for Word/Google Docs copy-pasting. |
| **Publication Figures** | `output/figures/abstention_results.png`<br>`output/figures/pipeline_architecture.png`<br>`output/figures/risk_coverage_curve.png` | 300 DPI figures showing adaptive abstention, pipeline architecture, and risk-coverage curves. |
