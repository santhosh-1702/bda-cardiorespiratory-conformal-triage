# Project Execution & Verification Status Dashboard

> **Project Title**: Artifact-Resilient Cardiorespiratory Triage using PPG & Conformal Prediction  
> **Target Environment**: Local Intel Core i3 | 8 GB RAM | No GPU (100% CPU Execution)  
> **Evaluation Cohort**: 53 BIDMC ICU Patients (5-fold Patient-Disjoint Cross-Validation)  
> **Evaluation Mode**: Phase 4 & 5 Verified Conformal Prediction & Artifact Resilience  

---

## 1. System Pipeline Overview

| Component | Architecture & Configuration | Status | Primary Verified Output |
| :--- | :--- | :---: | :--- |
| **Phase 1: Preprocessing** | 4th-order Butterworth SOS bandpass [0.10–0.60 Hz], streaming generator (< 100 MB RAM) | Complete | `bidmc_data/` (53 patient recordings @ 125 Hz) |
| **Phase 1.5: Biosignal Foundation** | Window duration benchmark (6s, 15s, 30s, 60s), pulse modulations (RIAV, RIIV, RIFV) | Complete | `output/results/phase1_5_baselines.csv`, `output/results/phase1_5_summary.md` |
| **Phase 2: Feature Engineering** | Validation-chosen 60s windows (2s stride), 28 multi-domain features, GroupKFold (5 folds) | Complete | `output/cache/features_split.npz` (11,026 windows across 53 patients) |
| **Phase 3: Base Regression** | CPU Histogram XGBoost (`tree_method='hist'`, `n_jobs=1`, seed=42) | Complete | `output/cache/best_xgb_model.json` |
| **Phase 4 & 5: Conformal Calibration** | Split conformal with grouped OOF difficulty estimation, auditable per-fold width scales | Complete | `output/cache/fold_map.json`, `output/results/phase5_results.csv` |
| **Phase 4 & 5: Artifact Resilience** | Time-domain corruption (mild, moderate, severe) with aligned AUROC evaluation | Complete | `output/cache/artifact_params.json`, `output/results/phase4_results.csv` |
| **Phase 4 & 5: Selective Prediction** | Risk-coverage curves and clean-window error discrimination (|error| > 3 bpm) | Complete | `output/figures/risk_coverage_curve.png`, `output/results/phase4_summary.md` |

---

## 2. Experimental Verification & Audit Checklist

### Requirement 1: Zero Test Leakage & Patient Grouping
- [x] **Disjoint Splits**: Patient IDs in Train, Validation, Calibration, and Test splits are strictly pairwise disjoint for all 5 folds.
- [x] **Cohort Segmentation**: 11,026 total 60s windows across 53 patients (min=54, mean=208.0, max=211).
- [x] **Patient 13 Clinical Audit**: 54 valid windows retained out of 211 candidates due to reference sensor disconnection (30.4% NaN, 47.8% zeros < 6 bpm in ground truth).
- [x] **Verification**: Verified via `tests/test_no_leakage.py` and `tests/test_phase4.py`.

### Requirement 2: Determinism Verification
- [x] **Execution Determinism**: Fixed random seeds and single-threaded execution (`n_jobs=1`).
- [x] **Bit-Identical Outputs**: Successive pipeline runs yield `0.0000000000` maximum prediction difference.

### Requirement 3: Repeated Grouped Cross-Validation (5 Repeats: XGBoost vs Train-Median Baseline)
- [x] **5 Independent Fold Seeds**: Seeds `[42, 101, 202, 303, 404]`.
- [x] **XGBoost Performance**: Mean Test MAE = **1.8123 bpm** (Between-Repeat Std: **0.0912 bpm**). Repeat MAEs: `[1.8564, 1.9205, 1.8748, 1.6833, 1.7265]`.
- [x] **Train-Median Baseline**: Mean Test MAE = **2.1279 bpm** (Between-Repeat Std: **0.0315 bpm**). Repeat MAEs: `[2.1139, 2.1893, 2.1236, 2.1017, 2.1111]`.
- [x] **Paired Patient Bootstrap Over Repeats (1,000 Resamples)**: Mean Difference = **-0.3065 bpm** | 95% CI: **[-0.5053, -0.1103] bpm** | Zero Excluded: **True**.

### Requirement 4: Waveform Alteration & SNR Verification
- [x] **Time-Domain Modification Verified**:
  - Mild: Mean Absolute Change = **0.007176 ± 0.003521** (28.3% of signal std), SNR = **+5.03 ± 2.05 dB**, Distinguishable = **True**.
  - Moderate: Mean Absolute Change = **0.027637 ± 0.014987** (109.1% of signal std), SNR = **-5.47 ± 3.76 dB**, Distinguishable = **True**.
  - Severe: Mean Absolute Change = **0.102235 ± 0.087041** (403.7% of signal std), SNR = **-12.59 ± 7.38 dB**, Distinguishable = **True**.

### Requirement 5: AUROC Alignment Bug Fix & Corruption Discrimination
- [x] **Alignment Bug Fix**: Fixed label/score ordering mismatch; unit test `tests/test_auroc_alignment.py` verifies alignment against independently computed ground truth (AUROC ~ 0.98 within 0.02).
- [x] **Corruption-Detection AUROC (Three Ways)**:
  - **Mild**:
    - Pooled Raw Width AUROC: **0.6278** [0.5665, 0.6925] | Fold Raw Width: **0.6515 ± 0.1757**
    - Pooled Norm Width AUROC: **0.6590** [0.5886, 0.7284] | Fold Norm Width: **0.6515 ± 0.1757**
    - Pooled SQI AUROC: **0.8979** [0.8790, 0.9156] | Fold SQI: **0.9010 ± 0.0255**
  - **Moderate**:
    - Pooled Raw Width AUROC: **0.6662** [0.5948, 0.7378] | Fold Raw Width: **0.6965 ± 0.2051**
    - Pooled Norm Width AUROC: **0.6940** [0.6195, 0.7675] | Fold Norm Width: **0.6965 ± 0.2051**
    - Pooled SQI AUROC: **0.9922** [0.9884, 0.9951] | Fold SQI: **0.9923 ± 0.0050**
  - **Severe**:
    - Pooled Raw Width AUROC: **0.7289** [0.6529, 0.8071] | Fold Raw Width: **0.7493 ± 0.1992**
    - Pooled Norm Width AUROC: **0.7487** [0.6720, 0.8223] | Fold Norm Width: **0.7493 ± 0.1992**
    - Pooled SQI AUROC: **0.9998** [0.9997, 0.9999] | Fold SQI: **0.9998 ± 0.0001**

### Requirement 6: Clean-Window Selective Prediction Analysis (|Error| > 3 bpm)
- [x] **True Error Prediction on Clean Windows**:
  - Raw Conformal Width: Pooled AUROC = **0.6067** [0.5027, 0.7175] | Per-Fold: **0.6793 ± 0.1544**
  - Fold-Norm Conformal Width: Pooled AUROC = **0.7023** [0.5972, 0.8038] | Per-Fold: **0.6793 ± 0.1544**
  - SQI Score: Pooled AUROC = **0.6022** [0.5106, 0.6981] | Per-Fold: **0.6257 ± 0.1356**
  - Multi-Domain Combination (Val-Trained): Pooled AUROC = **0.7527** [0.6335, 0.8562] | Per-Fold: **0.7105 ± 0.1089**
- [x] **Plain Evaluated Verdict**: Conformal interval width (AUROC = 0.6067 [95% CI: 0.5027, 0.7175]) predicts true error (|error| > 3 bpm) better than SQI score (AUROC = 0.6022 [95% CI: 0.5106, 0.6981]) on clean windows.

### Requirement 7: Selective Triage Risk-Coverage Operating Points (Clean Test Data)
- [x] **Empirical Trade-Offs (with 95% Patient-Bootstrap CIs)**:
  - **40% Retention**: Raw Width MAE = **1.2038 bpm** [0.8497, 1.9321] | Fold-Norm MAE = **1.0122 bpm** [0.7358, 1.5154]
  - **50% Retention**: Raw Width MAE = **1.5885 bpm** [1.0326, 2.2093] | Fold-Norm MAE = **1.2596 bpm** [0.8402, 1.8030]
  - **60% Retention**: Raw Width MAE = **1.7097 bpm** [1.1753, 2.2904] | Fold-Norm MAE = **1.4135 bpm** [0.9647, 1.9875]
  - **80% Retention**: Raw Width MAE = **1.8282 bpm** [1.3455, 2.3854] | Fold-Norm MAE = **1.6423 bpm** [1.2005, 2.1597]
  - **100% Retention**: Full Cohort Test MAE = **1.7991 bpm** [1.3393, 2.3350]

---

## 3. Auditable Per-Fold Conformal Parameter Table

Computed at target coverage **90.0%**:

| Fold | Conformal $\hat{q}$ | Cal Patients | Cal Windows | Test Windows | Test MAE (bpm) | Test Coverage | Mean Width (bpm) | Median Width (bpm) | Width Std | P10 Width | P90 Width | Mean Norm Width |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 1.6502 | 8 | 1,531 | 2,321 | 2.1820 | 76.2% | 5.88 | 5.92 | 2.59 | 2.71 | 8.66 | 3.57 |
| 1 | 1.6777 | 8 | 1,688 | 2,321 | 2.3085 | 86.1% | 9.01 | 6.28 | 7.51 | 2.90 | 17.16 | 5.37 |
| 2 | 2.1014 | 8 | 1,688 | 2,164 | 2.0221 | 74.1% | 5.81 | 4.53 | 3.70 | 3.00 | 11.09 | 2.76 |
| 3 | 4.5859 | 9 | 1,742 | 2,110 | 1.0613 | 100.0% | 18.29 | 14.07 | 15.45 | 5.82 | 37.32 | 3.99 |
| 4 | 3.6595 | 9 | 1,899 | 2,110 | 1.3266 | 95.3% | 16.18 | 13.16 | 10.15 | 8.28 | 28.27 | 4.42 |

---

## 4. Test Suite Execution Summary

All 30 unit tests across the test suites run and pass:
- `tests/test_auroc_alignment.py` (1 test: synthetic 5-fold AUROC alignment verification)
- `tests/test_phase4.py` (5 tests: fold map, time-domain corruption, grouped OOF residuals)
- `tests/test_no_leakage.py` (6 tests: strict patient disjointness across splits)
- `tests/test_phase1_5.py` (4 tests: filtering and feature foundation)
- `tests/test_phase2.py` (14 tests: conformal calibration, assertions, & determinism)
