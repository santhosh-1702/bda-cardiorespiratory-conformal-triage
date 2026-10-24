# Phase 6: Multi-Seed Grouped CV & Conformal Score Function Stability Benchmark

## 1. Study Design & Rigor Parameters
- **Cohort**: PhysioNet BIDMC (53 ICU patients), 60-second windows (active 28-feature pipeline).
- **Repetition Scheme**: 5 independent fold-map seeds (`[42, 101, 202, 303, 404]`) × 5 patient-disjoint folds = 25 evaluations per score type (125 total model calibrations).
- **Nominal Target Coverage**: 90.0% (split conformal calibration, never guaranteed per patient).
- **Patient Leakage Audit**: Strictly pairwise disjoint across `train`, `validation`, `calibration`, and `test` patient subsets (0 overlap across all 25 runs).
- **Hardware Execution**: CPU-only Intel i3, 8GB RAM, fixed seeds, deterministic single-thread (`n_jobs=1`).

## 2. Head-to-Head Conformal Methods Benchmark Table

| Score Function | Point Test MAE (bpm) | Mean Coverage (%) | Coverage Spread [Min, Max] | q-hat (Mean ± Std) | q-hat Spread [Min, Max] | Mean Width (bpm) | AUROC (|Error| > 3 bpm) | Patient Cov (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline (Train Median)** | 2.1279 ± 0.4089 | 86.55% ± 10.37% | **32.7%** [67.3%, 100.0%] | 4.796 ± 1.418 | 4.633 [2.667, 7.300] | 9.59 ± 2.84 | 0.5000 ± 0.0000 | 85.3% ± 26.0% |
| **XGBoost (Absolute Residual)** | 1.8123 ± 0.4348 | 87.54% ± 9.81% | **34.5%** [65.5%, 100.0%] | 4.271 ± 1.429 | 4.914 [2.204, 7.118] | 8.54 ± 2.86 | 0.5000 ± 0.0000 | 86.3% ± 22.6% |
| **XGBoost (Difficulty-Normalised)** | 1.8123 ± 0.4348 | 87.38% ± 9.68% | **31.8%** [68.2%, 100.0%] | 2.362 ± 1.048 | 4.044 [1.138, 5.182] | 9.30 ± 4.87 | 0.6838 ± 0.1222 | 86.3% ± 20.5% |
| **XGBoost (Mondrian Terciles)** | 1.8123 ± 0.4348 | 84.50% ± 10.28% | **38.0%** [62.0%, 100.0%] | 4.271 ± 1.429 | 4.914 [2.204, 7.118] | 7.71 ± 2.56 | 0.5192 ± 0.1207 | 83.3% ± 23.7% |
| **XGBoost (Spectral Sharpness)** | 1.8123 ± 0.4348 | 87.42% ± 10.49% | **35.5%** [64.5%, 100.0%] | 4.162 ± 1.469 | 5.206 [2.020, 7.226] | 8.64 ± 3.16 | 0.5995 ± 0.0785 | 86.2% ± 22.8% |

## 3. Paired Patient-Level Bootstrap Differences vs Difficulty-Normalised
*(1,000 patient-level resamples; positive difference indicates higher error than residual_normalised)*

| Alternative Method | Mean Point MAE Diff (bpm) | 95% Bootstrap CI | Excludes Zero? | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Trivial Baseline (Train Median)** | +0.3065 | [+0.1103, +0.5053] | True | XGBoost significantly outperforms median baseline. |
| **XGBoost (Absolute Residual)** | +0.0000 | [+0.0000, +0.0000] | False | Identical point predictor (XGBoost); difference lies solely in interval width & calibration. |
| **XGBoost (Mondrian Terciles)** | +0.0000 | [+0.0000, +0.0000] | False | Identical point predictor (XGBoost); difference lies solely in interval width & calibration. |
| **XGBoost (Spectral Sharpness)** | +0.0000 | [+0.0000, +0.0000] | False | Identical point predictor (XGBoost); difference lies solely in interval width & calibration. |

## 4. Fold-to-Fold Quantile & Coverage Stability Analysis
- **Residual-Normalised Swings**: Over the 25 grouped folds, `residual_normalised` exhibits empirical test coverage ranging from **68.2% to 100.0%** (spread: 31.8%), with conformal quantile $q$-hat ranging from **1.14 to 5.18**.
- **Alternative Methods Comparison**: The alternative score types show comparable spread dynamics:
  - *Trivial Baseline (Train Median)*: Coverage spread 32.7% [67.3%, 100.0%]; q-hat spread 4.633 [2.667, 7.300].
  - *XGBoost (Absolute Residual)*: Coverage spread 34.5% [65.5%, 100.0%]; q-hat spread 4.914 [2.204, 7.118].
  - *XGBoost (Mondrian Terciles)*: Coverage spread 38.0% [62.0%, 100.0%]; q-hat spread 4.914 [2.204, 7.118].
  - *XGBoost (Spectral Sharpness)*: Coverage spread 35.5% [64.5%, 100.0%]; q-hat spread 5.206 [2.020, 7.226].

## 5. Final Evaluated Verdict
> **Plain Verdict**: Across all 25 grouped cross-validation evaluations, none of the five conformal score functions materially stabilizes the fold-to-fold coverage spread (residual_normalised spread: 31.8%, vs Trivial Baseline (Train Median) spread: 32.7%), confirming that fold-to-fold quantile swings are primarily driven by small-sample calibration noise (8–9 calibration patients per fold) rather than scoring-function inadequacy.

---
*Verification Status*: Evaluated strictly across 5 repeated fold seeds and 5 disjoint folds (n_runs=125). Patient-disjoint splits guaranteed.
