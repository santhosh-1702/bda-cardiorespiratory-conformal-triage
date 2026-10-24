# Pipeline Reproduction Status

- **Reproduction Date**: 2026-10-17 07:02:22 UTC
- **Overall Pipeline Status**: **SUCCESS (ALL GREEN)**
- **Total Elapsed Time**: 587.8s (9.80 minutes)
- **Hardware Target**: Intel i3 / 8GB RAM / Local CPU (`n_jobs=1`, deterministic)
- **Unit Test Suite (`python -m unittest discover tests`)**: **GREEN** (72 tests passed in 4.748s)

## Executed Phases Summary

| Phase | Description | Result |
| :--- | :--- | :---: |
| Phase 1 | Data Acquisition & Butterworth Bandpass Filtering | **PASS** |
| Phase 1.5 | Base Signal Foundation & Window Duration Selection | **PASS** |
| Phase 2 | Segmentation & Multi-Domain Feature Extraction (28 features) | **PASS** |
| Phase 3 | CPU Histogram Gradient Boosting Baseline Modeling | **PASS** |
| Phase 4 | Split Conformal Prediction & Risk Control Calibration | **PASS** |
| Phase 5 | 300 DPI Waveform Visualization & Demo Reporting | **PASS** |
| Phase 6 | Multi-Seed Conformal Score Types Comparison (25 Folds) | **PASS** |
| Phase 7 | Jackknife+ / CV+ Cross-Fitting Conformal Benchmark | **PASS** |
| Sensitivity | Hyperparameter Sensitivity Check (Depth & Learning Rate) | **PASS** |
| Unit Tests | `python -m unittest discover tests` | **GREEN** |

## Key Verified Artifacts
- `output/results/phase1_5_baselines.csv`: Window length (60s) & baseline benchmark table
- `output/results/phase1_5_summary.md`: Auto-computed markdown summary statements
- `output/cache/best_xgb_model.json`: Trained XGBoost regressor model
- `output/cache/conformal_results.npz`: Split conformal calibration outputs
- `output/results/phase5_results.csv`: Corruption AUROC & selective prediction risk-coverage table
- `output/results/phase6_results.csv`: 25-fold 5-score conformal comparison
- `output/results/phase6_summary.md`: Multi-seed coverage spread and stability verdict
- `output/results/phase7_results.csv`: Jackknife+ / CV+ cross-fitting benchmark
- `output/results/phase7_summary.md`: CV+ vs Difficulty-Normalised comparison
- `output/results/sensitivity_check.csv`: Hyperparameter robustness evaluation across depth and learning rate
- `output/results/sensitivity_check.md`: Hyperparameter sensitivity table & plain verdict
- `output/paper/paper_tables.tex` & `output/paper/paper_tables.txt`: Publication booktabs tables & plain companion
- `output/figures/pipeline_architecture.png`: 300 DPI Figure 1 end-to-end pipeline overview
- `output/figures/abstention_results.png` & `output/figures/risk_coverage_curve.png`: 300 DPI publication figures
