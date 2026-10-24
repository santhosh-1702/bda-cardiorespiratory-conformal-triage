# Phase 3: Corrected Conformal Prediction & Selective Triage Benchmark

## Conformal Score Functions Comparison
- **Target Coverage**: 90.0% across all methods (reported as target coverage, never guaranteed).
- **Window Duration**: Validation-chosen 60 s window per fold (zero test-fold leakage).
- **Threshold for Undercoverage**: Target minus 5 percentage points (85.0%).

### 1. Score Function Benchmark Table

| Method | Test MAE (bpm) | Mean Fold Coverage | Mean Interval Width (bpm) | Patient Cov (Mean ± Std) | Patient Cov Min | Patients < 85% Cov (%) | Error > 3 bpm AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Trivial Baseline (Train Median) | 2.1215 ± 0.3264 | 85.17% ± 17.43% | 10.51 | 83.48% ± 31.93% | 0.00% | 24.5% (13/53) | N/A (width constant) |
| XGBoost (Absolute Residual) | 1.7801 ± 0.4943 | 86.80% ± 10.79% | 9.86 | 85.25% ± 28.31% | 0.00% | 28.3% (15/53) | N/A (width constant) |
| XGBoost (Difficulty-Normalised) | 1.7801 ± 0.4943 | 88.10% ± 10.63% | 10.72 | 86.64% ± 27.98% | 0.00% | 18.9% (10/53) | 0.5911 |
| XGBoost (Mondrian Terciles) | 1.7801 ± 0.4943 | 85.44% ± 10.99% | 9.25 | 83.89% ± 28.54% | 0.00% | 32.1% (17/53) | 0.4020 |
| XGBoost (Model-Free Spectral Sharpness) | 1.7801 ± 0.4943 | 85.89% ± 11.62% | 10.13 | 84.37% ± 29.58% | 0.00% | 26.4% (14/53) | 0.4350 |

### 2. Selective Triage Risk-Coverage Operating Points (Difficulty-Normalised)

| Retention (%) | Retained MAE (bpm) | Clean Data Abstention (%) | Interval Width Threshold (bpm) |
| :---: | :---: | :---: | :---: |
| 40% | 1.4110 | 60% | 7.40 |
| 50% | 1.5849 | 50% | 9.02 |
| 60% | 1.6950 | 40% | 10.98 |
| 70% | 1.6816 | 30% | 12.75 |
| 80% | 1.7917 | 20% | 15.17 |
| 85% | 1.8139 | 15% | 17.10 |
| 90% | 1.8176 | 10% | 19.95 |
| 95% | 1.7889 | 5% | 24.66 |
| 100% | 1.7991 | 0% | 54.20 |

### 3. Auto-Computed Conformal Statements
- **Trivial Baseline (Train Median)**: Test MAE = 2.1215 bpm. Empirical coverage is 85.17% (target: 90.0%). Mean width is 10.51 bpm. 13 of 53 patients (24.5%) fall below 85.0% coverage. Error > 3 bpm AUROC: N/A (width constant).
- **XGBoost (Absolute Residual)**: Test MAE = 1.7801 bpm. Empirical coverage is 86.80% (target: 90.0%). Mean width is 9.86 bpm. 15 of 53 patients (28.3%) fall below 85.0% coverage. Error > 3 bpm AUROC: N/A (width constant).
- **XGBoost (Difficulty-Normalised)**: Test MAE = 1.7801 bpm. Empirical coverage is 88.10% (target: 90.0%). Mean width is 10.72 bpm. 10 of 53 patients (18.9%) fall below 85.0% coverage. Error > 3 bpm AUROC: 0.5911.
- **XGBoost (Mondrian Terciles)**: Test MAE = 1.7801 bpm. Empirical coverage is 85.44% (target: 90.0%). Mean width is 9.25 bpm. 17 of 53 patients (32.1%) fall below 85.0% coverage. Error > 3 bpm AUROC: 0.4020.
- **XGBoost (Model-Free Spectral Sharpness)**: Test MAE = 1.7801 bpm. Empirical coverage is 85.89% (target: 90.0%). Mean width is 10.13 bpm. 14 of 53 patients (26.4%) fall below 85.0% coverage. Error > 3 bpm AUROC: 0.4350.

---
*Verification Status*: Evaluated strictly across 5 patient-disjoint folds on 53 BIDMC patients. Zero test-fold leakage. Calibration on patient-disjoint calibration sets.