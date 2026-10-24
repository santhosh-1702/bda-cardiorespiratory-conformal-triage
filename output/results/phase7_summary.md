# Phase 7: Jackknife+ / CV+ Cross-Fitting Conformal Prediction Benchmark

## 1. Study Design & Rigor Parameters
- **Cohort**: PhysioNet BIDMC (53 ICU patients), 60-second windows (active 28-feature pipeline).
- **Repetition Scheme**: 5 independent fold-map seeds (`[42, 101, 202, 303, 404]`) × 5 patient-disjoint folds = 25 evaluations per score type (150 total model calibrations).
- **Nominal Target Coverage**: 90.0% (split conformal and cross-conformal, never guaranteed per patient).
- **Cross-Fitting Scheme (CV+)**: 5-fold inner GroupKFold over all non-test patients (train+val+cal combined, ~42-44 patients per outer fold). Every non-test window receives an out-of-fold residual from an inner model that never saw its patient.
- **Patient Leakage Audit**: Strictly pairwise disjoint across `train`, `val`, `cal`, and `test` patient subsets at the outer level, and strictly disjoint across inner train and inner val folds at the cross-fitting level (0 patient overlap across all splits).
- **Hardware Execution**: CPU-only Intel i3, 8GB RAM, fixed seeds, deterministic single-thread (`n_jobs=1`).

## 2. Head-to-Head Conformal Methods Benchmark Table

| Score Function | Point Test MAE (bpm) | Mean Coverage (%) | Coverage Spread [Min, Max] | Mean Width (bpm) | AUROC (|Error| > 3 bpm) | Patient Cov (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Trivial Baseline (Train Median) | 2.1279 ± 0.4089 | 86.55% ± 10.37% | **32.7%** [67.3%, 100.0%] | 9.59 ± 2.84 | 0.5000 ± 0.0000 | 85.3% ± 26.0% |
| XGBoost (Absolute Residual) | 1.8123 ± 0.4348 | 87.54% ± 9.81% | **34.5%** [65.5%, 100.0%] | 8.54 ± 2.86 | 0.5000 ± 0.0000 | 86.3% ± 22.6% |
| **XGBoost (Difficulty-Normalised)** | 1.8123 ± 0.4348 | 87.38% ± 9.68% | **31.8%** [68.2%, 100.0%] | 9.30 ± 4.87 | 0.6838 ± 0.1222 | 86.3% ± 20.5% |
| XGBoost (Mondrian Terciles) | 1.8123 ± 0.4348 | 84.50% ± 10.28% | **38.0%** [62.0%, 100.0%] | 7.71 ± 2.56 | 0.5192 ± 0.1207 | 83.3% ± 23.7% |
| XGBoost (Spectral Sharpness) | 1.8123 ± 0.4348 | 87.42% ± 10.49% | **35.5%** [64.5%, 100.0%] | 8.64 ± 3.16 | 0.5995 ± 0.0785 | 86.2% ± 22.8% |
| **Jackknife+ / CV+ (Ensemble Cross-Fitting)** | 1.7276 ± 0.3816 | 90.69% ± 7.94% | **29.7%** [70.2%, 100.0%] | 8.98 ± 1.16 | 0.5733 ± 0.1586 | 89.4% ± 20.6% |

## 3. Paired Patient-Level Bootstrap Differences vs Difficulty-Normalised
*(1,000 patient-level resamples; positive MAE difference indicates higher error; positive coverage difference indicates higher coverage)*

| Method | Mean Point MAE Diff (bpm) | 95% Bootstrap CI (MAE) | Mean Coverage Diff (%) | 95% Bootstrap CI (Coverage) | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Trivial Baseline (Train Median)** | +0.3065 | [+0.1103, +0.5053] | -0.86% | [-5.20%, +3.00%] | Trivial baseline has significantly higher point error. |
| **XGBoost (Absolute Residual)** | +0.0000 | [+0.0000, +0.0000] | +0.06% | [-2.32%, +2.24%] | Identical base XGBoost point predictions; intervals differ. |
| **XGBoost (Mondrian Terciles)** | +0.0000 | [+0.0000, +0.0000] | -2.88% | [-5.37%, -0.55%] | Identical base XGBoost point predictions; intervals differ. |
| **XGBoost (Spectral Sharpness)** | +0.0000 | [+0.0000, +0.0000] | -0.03% | [-2.43%, +1.97%] | Identical base XGBoost point predictions; intervals differ. |
| **Jackknife+ / CV+ (Ensemble Cross-Fitting)** | -0.0851 | [-0.1523, -0.0052] | +3.24% | [+0.90%, +5.61%] | CV+ ensemble point predictor (-0.0851 bpm); coverage diff +3.24%. |

## 4. Fold-to-Fold Coverage Dispersion Analysis (Bootstrap CIs)
- **Difficulty-Normalised**: Over 25 outer folds, `residual_normalised` exhibits empirical coverage ranging from **68.2% to 100.0%** (spread: **31.8%**, 95% CI [26.7%, 31.8%]; SD: **9.5%**, 95% CI [6.9%, 11.2%]).
- **Jackknife+ / CV+**: Over the same 25 outer folds, `cv_plus` exhibits empirical coverage ranging from **70.2% to 100.0%** (spread: **29.7%**, 95% CI [19.7%, 29.7%]; SD: **7.8%**, 95% CI [5.6%, 9.5%]).
- **Bootstrap Difference (CV+ minus Normalised)**:
  - Min-Max Spread Diff: -2.1pp, 95% CI [-11.6%, +3.1%], includes zero.
  - SD Diff: -1.7pp, 95% CI [-4.7%, +1.4%], includes zero.
- **Comparison with Other Split Conformal Baselines**:
  - Trivial Baseline (Train Median): 32.7%
  - XGBoost (Absolute Residual): 34.5%
  - XGBoost (Mondrian Terciles): 38.0%
  - XGBoost (Spectral Sharpness): 35.5%

## 5. Final Evaluated Verdict
> **Plain Verdict**: CV+/jackknife+ cross-fitting does NOT meaningfully narrow the fold-to-fold coverage spread (CV+ coverage spread: 29.7% [70.2%, 100.0%], vs difficulty-normalised spread: 31.8% [68.2%, 100.0%]). Bootstrap 95% CI on the SD difference (CV+ minus Normalised): [-4.7%, +1.4%] (includes zero). This confirms that coverage volatility is NOT merely an artifact of carving out a small (8-9 patient) calibration subset, but reflects genuine between-patient physiological heterogeneity in the ICU cohort that persists even when pooling cross-fitted residuals across ~42-44 non-test patients (~5x more patients).

---
*Verification Status*: Evaluated strictly across 5 repeated fold seeds and 5 disjoint folds (n_runs=150). Zero patient leakage at outer and inner cross-fitting levels. Coverage-spread CIs computed via 1,000 fold-level bootstrap resamples.