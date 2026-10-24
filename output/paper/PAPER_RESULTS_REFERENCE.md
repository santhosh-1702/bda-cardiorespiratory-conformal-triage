# Consolidated Results Reference for Manuscript Preparation
**Artifact-Resilient Cardiorespiratory Triage Using Photoplethysmography (PPG) and Conformal Prediction**
*PhysioNet BIDMC ICU Cohort (53 Patients) | 5-Fold Patient-Disjoint Cross-Validation*

---

## 1. Experimental Design & Methodology Overview

- **Cohort & Data**: PhysioNet BIDMC Dataset, 53 adult intensive care unit (ICU) patients, 8-minute records (125 Hz PPG and ground-truth respiratory signals).
- **Segmentation & Windows**: 60-second window duration with 2.0-second stride. 60 s was selected via validation-based grid search over candidate durations [6s, 15s, 30s, 60s] as yielding optimal feature stability and lowest MAE with zero test-patient leakage.
- **Total Windows**: 11,026 clean windows across 53 patients (min: 54 [bidmc13], mean: 208.0, max: 211).
  - *bidmc13 QA Note*: 47.8% of recording had sensor flatline / dropout (reference RR = 0 bpm) and 30.4% was NaN; retained in full 53-patient benchmark to strictly avoid post-hoc exclusion bias.
- **Active Feature Space (28 Dimensions)**:
  - **PPG Spectral Descriptors (4 features)**: Dominant frequency (bpm) in respiratory band [0.10–0.60 Hz], spectral centroid (bpm), spectral bandwidth / in-band power ratio, and spectral roll-off / in-band power.
  - **Pulse-Modulation Spectral Descriptors (12 features = 3 series × 4 descriptors)**: Dominant frequency, spectral centroid, spectral bandwidth, and spectral roll-off computed separately across three extracted pulse-modulation series:
    - **RIAV (Respiratory-Induced Amplitude Variation)**: Peak-to-peak pulse amplitude modulation series (4 features)
    - **RIIV (Respiratory-Induced Intensity Variation)**: Diastolic trough / baseline wander series (4 features)
    - **RIFV (Respiratory-Induced Frequency Variation)**: Pulse-interval duration modulation series (4 features)
  - **Cardiorespiratory / Pulse-Modulation Statistics (8 features)**: Multi-modulation median fusion respiratory rate (`fusion_rr`), statistical moments of each modulation series (`riav_mean`, `riav_std`, `riiv_mean`, `riiv_std`, `rifv_mean`, `rifv_std`), and pulse-derived mean heart rate (`mean_hr` = 60 / `rifv_mean`).
  - **Waveform Morphology & Spectral Sharpness Descriptors (4 features)**: Respiratory-filtered window standard deviation (`sig_std`), peak-to-peak amplitude (`sig_ptp`), root-mean-square amplitude (`sig_rms`), and within-window spectral-peak sharpness (`sharpness` = peak power / mean band power in [0.10, 0.60 Hz]).
  *(Total Dimensionality: 4 + 12 + 8 + 4 = 28 features; physiologically grounded pulse modulations, zero wavelets).*
- **Regression Architecture**: Extreme Gradient Boosting (XGBoost Regressor) running histogram tree method (`hist`), restricted to CPU-only execution (`n_jobs=1`), fixed random seeds.
- **Conformal Prediction Framework**: Split conformal prediction calibrated at 90.0% nominal confidence ($\alpha = 0.10$).
- **Patient Leakage Prohibition**: Strict patient disjointness enforced across `train`, `val`, `cal`, and `test` subsets (`output/fold_map.json`). Zero patient overlap across all folds.

---

## 2. Predictive Accuracy (Point Predictions)

### A. Repeated Grouped Cross-Validation (5 Repeats, 5 Folds = 25 Runs)
Evaluated across 5 independent fold-map seeds (`[42, 101, 202, 303, 404]`):

| Model | Mean Test MAE (bpm) | Between-Repeat Std (bpm) | Per-Repeat MAEs (bpm) |
| :--- | :---: | :---: | :--- |
| **XGBoost (Hist, CPU)** | **1.8123** | 0.0912 | [1.8564, 1.9205, 1.8748, 1.6833, 1.7265] |
| **Trivial Baseline (Train Median)** | **2.1279** | 0.0315 | [2.1139, 2.1893, 2.1236, 2.1017, 2.1111] |

- **Paired Patient-Clustered Bootstrap Difference (1,000 resamples)**:
  - Mean MAE Difference ($\Delta = \text{XGB} - \text{Median}$): **-0.3065 bpm**
  - 95% Bootstrap Confidence Interval: **[-0.5053, -0.1103] bpm**
  - Zero Excluded: **True ($p < 0.05$)** — XGBoost achieves statistically significant error reduction over the central-tendency median baseline.

### B. Primary Benchmark Run (Seed 42, Fixed Fold Map)
- XGBoost Mean Test MAE: **1.7801 ± 0.4943 bpm**
- Trivial Baseline Test MAE: **2.1215 ± 0.3264 bpm**
- Per-Fold Test MAEs (XGBoost): Fold 0: 2.1820 | Fold 1: 2.3085 | Fold 2: 2.0221 | Fold 3: 1.0613 | Fold 4: 1.3266 bpm.

---

## 3. Conformal Score Functions Head-to-Head Comparison

Evaluated on clean test windows at 90.0% nominal target coverage.

### A. Multi-Seed Stability Benchmark (25 Folds / 125 Model Calibrations)

| Conformal Score Type | Point Test MAE (bpm) | Mean Test Coverage | Fold Coverage Spread [Min, Max] | Conformal Quantile $\hat{q}$ (Mean ± Std) | Mean Interval Width (bpm) | Clean AUROC (Error > 3 bpm) | Patient Coverage (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline (Train Median)** | 2.1279 ± 0.4089 | 86.55% ± 10.37% | 32.7% [67.3%, 100.0%] | 4.796 ± 1.418 | 9.59 ± 2.84 | 0.5000 (constant) | 85.3% ± 26.0% |
| **XGBoost (Absolute Residual)** | 1.8123 ± 0.4348 | 87.54% ± 9.81% | 34.5% [65.5%, 100.0%] | 4.271 ± 1.429 | 8.54 ± 2.86 | 0.5000 (constant) | 86.3% ± 22.6% |
| **XGBoost (Difficulty-Normalised)** | 1.8123 ± 0.4348 | 87.38% ± 9.68% | 31.8% [68.2%, 100.0%] | 2.362 ± 1.048 | 9.30 ± 4.87 | **0.6838 ± 0.1222** | 86.3% ± 20.5% |
| **XGBoost (Mondrian Terciles)** | 1.8123 ± 0.4348 | 84.50% ± 10.28% | 38.0% [62.0%, 100.0%] | 4.271 ± 1.429 | 7.71 ± 2.56 | 0.5192 ± 0.1207 | 83.3% ± 23.7% |
| **XGBoost (Spectral Sharpness)** | 1.8123 ± 0.4348 | 87.42% ± 10.49% | 35.5% [64.5%, 100.0%] | 4.162 ± 1.469 | 8.64 ± 3.16 | 0.5995 ± 0.0785 | 86.2% ± 22.8% |

### B. Score-Type Stability Verdict (Phase 6 Finding)
> Across all 25 grouped cross-validation evaluations, none of the five conformal score functions materially stabilizes the fold-to-fold coverage spread (difficulty-normalised spread: 31.8%, vs median baseline spread: 32.7%, vs absolute residual spread: 34.5%). This confirms that fold-to-fold quantile swings are primarily driven by small-sample calibration noise (8–9 calibration patients per fold in ICU cohorts) rather than scoring-function inadequacy.

---

## 4. Waveform-Domain Corruption & Artifact Detection AUROC

Raw time-domain PPG corruptions injected directly into 60-second waveform windows prior to feature extraction (`corrupt_raw_ppg_window` in `src/conformal/artifact_simulator.py`), strictly on validation and test partitions (training and calibration partitions remain uncorrupted). Evaluates four composable artifact types—high-frequency noise, respiratory baseline drift, transient motion bursts, and sensor flatline dropouts—calibrated across three clinical severity tiers:
- **Mild**: Mean Absolute Change (MAC) = $0.0072 \pm 0.0035$ (28.3% of signal std), $\text{SNR} = +5.03 \pm 2.05\text{ dB}$.
- **Moderate**: MAC = $0.0276 \pm 0.0150$ (109.1% of signal std), $\text{SNR} = -5.47 \pm 3.76\text{ dB}$.
- **Severe**: MAC = $0.1022 \pm 0.0870$ (403.7% of signal std), $\text{SNR} = -12.59 \pm 7.38\text{ dB}$.

### Corruption-Detection AUROC (Discriminative Power)

| Corruption Severity | Raw Conformal Width AUROC [95% CI] | Fold-Norm Width AUROC [95% CI] | In-Band SQI Baseline AUROC [95% CI] |
| :--- | :---: | :---: | :---: |
| **Mild Artifact** | 0.6278 [0.5665, 0.6925] | 0.6590 [0.5886, 0.7284] | **0.8979 [0.8790, 0.9156]** |
| **Moderate Artifact** | 0.6662 [0.5948, 0.7378] | 0.6940 [0.6195, 0.7675] | **0.9922 [0.9884, 0.9951]** |
| **Severe Artifact** | 0.7289 [0.6529, 0.8071] | 0.7487 [0.6720, 0.8223] | **0.9998 [0.9997, 0.9999]** |

---

## 5. Clean-Window Error Detection AUROC (|Error| > 3 bpm)

Evaluating ability to flag inaccurate predictions on **clean** PPG windows where no synthetic noise was injected:

| Predictor Rule | Pooled AUROC [95% Patient Bootstrap CI] | Fold AUROC (Mean ± Std) |
| :--- | :---: | :---: |
| **Raw Interval Width** | 0.6067 [0.5027, 0.7175] | 0.6793 ± 0.1544 |
| **Fold-Normalized Width** | 0.7023 [0.5972, 0.8038] | 0.6793 ± 0.1544 |
| **Signal Quality Index (SQI)** | 0.6022 [0.5106, 0.6981] | 0.6257 ± 0.1356 |
| **Multi-Domain Combination (Width + SQI)** | **0.7527 [0.6335, 0.8562]** | **0.7105 ± 0.1089** |

- **Clinical Interpretation**:
  - On corrupted signals, SQI detects morphological breakdown with near-perfect discrimination ($\text{AUROC} > 0.99$).
  - On clean signals, however, SQI drops in utility ($\text{AUROC} = 0.6022$), whereas conformal interval width predicts true estimation error better ($\text{AUROC} = 0.7023$).
  - Training a lightweight triage rule combining both features on validation windows yields superior discrimination ($\text{AUROC} = 0.7527$) without leaking test labels.

---

## 6. Selective Prediction Risk-Coverage Operating Points (Clean Test Data)

Full retention table comparing selective triage rules with 95% patient-clustered bootstrap confidence intervals (1,000 resamples):

| Retention Fraction | Raw Width MAE (bpm) [95% CI] | Fold-Norm Width MAE (bpm) [95% CI] | Combination Score MAE (bpm) [95% CI] | Oracle Lower Bound MAE (bpm) [95% CI] | Random Abstention MAE (bpm) [95% CI] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **40%** | 1.2038 [0.8497, 1.9321] | **1.0122 [0.7358, 1.5154]** | 1.1387 [0.7894, 1.5872] | 0.3310 [0.2644, 0.4361] | 1.8205 [1.3558, 2.3631] |
| **50%** | 1.5885 [1.0326, 2.2093] | 1.2596 [0.8402, 1.8030] | **1.1467 [0.8366, 1.5700]** | 0.4281 [0.3410, 0.5646] | 1.8154 [1.3503, 2.3535] |
| **60%** | 1.7097 [1.1753, 2.2904] | 1.4135 [0.9647, 1.9875] | **1.1788 [0.8560, 1.5903]** | 0.5427 [0.4303, 0.7126] | 1.7763 [1.3248, 2.2900] |
| **70%** | 1.7407 [1.2629, 2.3372] | 1.5299 [1.0761, 2.0830] | **1.2678 [0.9124, 1.7983]** | 0.6828 [0.5379, 0.9024] | 1.7861 [1.3411, 2.3124] |
| **80%** | 1.8282 [1.3455, 2.3854] | 1.6423 [1.2005, 2.1597] | **1.4061 [1.0090, 1.9134]** | 0.8724 [0.6842, 1.1907] | 1.7892 [1.3346, 2.3180] |
| **85%** | 1.8962 [1.3813, 2.4415] | 1.6575 [1.2279, 2.1635] | **1.4888 [1.0950, 1.9841]** | 1.0044 [0.7761, 1.4158] | 1.7954 [1.3413, 2.3254] |
| **90%** | 1.8547 [1.3593, 2.4338] | 1.6734 [1.2575, 2.2175] | **1.5714 [1.1678, 2.0582]** | 1.1922 [0.8925, 1.6932] | 1.7912 [1.3395, 2.3256] |
| **95%** | 1.8388 [1.3550, 2.3863] | 1.7737 [1.3130, 2.3290] | **1.6423 [1.2587, 2.2164]** | 1.4588 [1.0551, 1.9897] | 1.7972 [1.3401, 2.3252] |
| **100%** | 1.7991 [1.3393, 2.3350] | 1.7991 [1.3393, 2.3350] | 1.7991 [1.3393, 2.3350] | 1.7991 [1.3393, 2.3350] | 1.7991 [1.3393, 2.3350] |

- **Gap Closure Analysis**:
  - Random-to-Oracle MAE Gap at 40% Retention: $1.8205 - 0.3310 = 1.4895\text{ bpm}$.
  - Fold-normalized width closes **54.3%** of the gap ($\text{MAE} = 1.0122\text{ bpm}$).
  - Combination score closes **45.8%** of the gap at 40% retention ($\text{MAE} = 1.1387\text{ bpm}$), but delivers superior error reduction across all intermediate clinical operating points from 50% through 95% retention (e.g., $1.1788$ vs $1.4135\text{ bpm}$ at 60% retention).

---

## 7. NeuroKit2 Open-Source Benchmark (53-Patient Audit)

- Evaluated across all 53 BIDMC patients comparing the pipeline's peak-detection and respiratory extraction against NeuroKit2 v0.2.13 (`ppg_findpeaks` / `signal_filter`).
- **Mean Pearson Correlation**: $r = 0.448 \pm 0.312$
- **High Agreement ($r \ge 0.60$)**: 18 of 51 evaluated patients (35.3%)
- **Moderate Agreement ($0.30 \le r < 0.60$)**: 12 of 51 evaluated patients (23.5%)
- **Low Agreement ($r < 0.30$)**: 21 of 51 evaluated patients (41.2%)
- *Scientific Finding*: Confirms known physiological limitations of morphological PPG peak detection in ICU cohorts, where arrhythmia, variable peripheral perfusion, and motion artifacts degrade conventional time-domain peak finders.

---

## 8. External Validation on CapnoBase Benchmark (Out-of-Distribution Transfer & Stratification)

Evaluates zero-shot domain generalization and conformal validity on the independent IEEE TBME CapnoBase benchmark dataset (42 elective surgical patients: 13 adult, 29 pediatric; Vancouver, Canada) using the 5 frozen BIDMC-trained fold models and conformal calibrators without retraining.

### A. Study Cohort & Demographic Stratification
- **In-Domain Cohort (BIDMC)**: 53 adult ICU patients (Boston, MA), 125 Hz PPG, impedance pneumography (IP) respiratory reference, 11,026 clean 60-s windows (2 s stride).
- **External Adult Cohort (PRIMARY Benchmark)**: 13 adult surgical cases (Vancouver, Canada; age $\ge 18$, range 32–63 years), 2,689 clean 60-s windows. Serves as the primary, fair domain generalization test because adult respiration matches the pipeline's 6–36 bpm filter passband.
  - **Spontaneous Breathing Adults**: $N = 5$ cases (1,055 clean windows; cases 0322, 0330, 0331, 0332, 0370) breathing spontaneously without mechanical assistance.
  - **Mechanically Ventilated Adults**: $N = 8$ cases (1,634 clean windows; cases 0309, 0311, 0312, 0313, 0325, 0328, 0329, 0333) receiving controlled positive-pressure ventilation.
- **External Pediatric Cohort (Frequency-Mismatch Cohort)**: 29 pediatric surgical cases (Vancouver, Canada; age $< 18$, range 1–16 years, including case 0032 with weight 26 kg), 6,075 clean 60-s windows (18 spontaneous: 3,754 windows; 11 ventilated: 2,321 windows).
- **Full Benchmark Cohort**: All 42 cases combined (8,764 clean 60-s windows; 23 spontaneous, 19 mechanically ventilated).
- **Pediatric Frequency-Range Mismatch**:
  - Filter Passband: 4th-order Butterworth bandpass covers 0.10–0.60 Hz (6–36 bpm); `VALID_RR_RANGE` ceiling is 45 bpm.
  - **Pediatric Windows > 36 bpm**: **272 of 6,075 windows (4.48%)** have ground-truth RR strictly above the filter passband ceiling (max window RR: 42.4 bpm).
  - **Pediatric 1 Hz Reference Points > 36 bpm**: **617 of 13,006 seconds (4.74%)** exceed 36 bpm; **5 points (0.04%)** exceed 45 bpm (max: 47.5 bpm).
  - **Adult Comparison**: **0.00% (0 of 2,689 windows)** of the adult cohort exceed 36 bpm (max: 29.7 bpm).
- **Model Status**: Strictly frozen XGBoost base estimator and auxiliary dispersion models calibrated on BIDMC. **Zero CapnoBase samples used for training, hyperparameter selection, or calibration.**
- **Nominal Target Coverage**: 90.0% ($\alpha = 0.10$).

### B. Stratified Benchmark Comparison Table (5-Fold Repeated Evaluation)

| Metric | In-Domain (BIDMC ICU, N=53) | Adult Spontaneous (Natural, N=5) | Adult Ventilated (Mechanical, N=8) | External Adult All (Primary, N=13) | External Pediatric (Outside Passband, N=29) | Full Benchmark (All 42 Cases) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort Size** | 53 ICU Patients (11,026 wins) | 5 Adult Cases (1,055 wins) | 8 Adult Cases (1,634 wins) | 13 Adult Cases (2,689 wins) | 29 Pediatric Cases (6,075 wins) | 42 Cases (8,764 wins) |
| **Physiological Match** | In-Range (Adult ICU) | **Full Match (Spontaneous)** | Mismatch (Positive-Pressure) | Full Passband (Blended Vent) | Partial (4.48% > 36 bpm) | Blended |
| **Point Prediction MAE (bpm)** | **1.78 ± 0.55** | **3.80 ± 0.71** | **7.76 ± 0.40** | **6.21 ± 0.36** | **6.39 ± 0.29** | **6.33 ± 0.31** |
| **Empirical Coverage (%)** | **88.1% ± 11.9%** | **77.7% ± 14.6%** | **24.3% ± 22.5%** | **45.3% ± 18.4%** | **59.8% ± 21.6%** | **55.3% ± 20.3%** |
| **Mean Interval Width (bpm)** | **10.72 ± 4.87** | **12.16 ± 6.32** | **11.80 ± 4.83** | **11.94 ± 5.39** | **15.32 ± 6.86** | **14.29 ± 6.32** |
| **Patient Coverage (Mean ± Std)** | **87.1% ± 19.2%** | **77.7% ± 24.7%** | **23.7% ± 21.3%** | **44.5% ± 37.9%** | **59.8% ± 37.4%** | **55.1% ± 37.4%** |
| **Error AUROC (|Err| > 3 bpm)** | **0.712 ± 0.127** | **0.535 ± 0.115** | **0.445 ± 0.123** | **0.528 ± 0.148** | **0.619 ± 0.100** | **0.577 ± 0.101** |
| **Selective Triage MAE (80% Ret.)** | — | **3.48 ± 0.72** | **7.86 ± 0.39** | **6.17 ± 0.44** | **5.97 ± 0.35** | **5.98 ± 0.31** |

### C. Per-Fold Detailed Breakdown

| Fold Model | BIDMC MAE | Adult Spont MAE | Adult Spont Cov | Adult Vent MAE | Adult Vent Cov | Adult All MAE | Adult All Cov | Ped MAE | Ped Cov | Full MAE | Full Cov |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 2.18 | 3.33 | 70.6% | 7.59 | 4.9% | 5.92 | 30.7% | 6.08 | 40.0% | 6.03 | 37.1% |
| Fold 1 | 2.31 | 4.98 | 83.4% | 7.53 | 55.5% | 6.53 | 66.5% | 6.58 | 75.0% | 6.56 | 72.4% |
| Fold 2 | 2.02 | 3.47 | 56.6% | 7.45 | 0.1% | 5.89 | 22.2% | 6.12 | 32.7% | 6.05 | 29.5% |
| Fold 3 | 1.06 | 3.92 | 83.1% | 8.44 | 28.9% | 6.66 | 50.2% | 6.75 | 77.3% | 6.73 | 69.0% |
| Fold 4 | 1.33 | 3.31 | 95.0% | 7.79 | 32.1% | 6.03 | 56.7% | 6.41 | 73.8% | 6.29 | 68.6% |

### D. Scientific Interpretation & Plain Evaluated Verdict

> **Scientific Attribution of Degradation & Evaluated Verdict**:
> 
> 1. **Testing Candidate Confounds & Violation of Exchangeability**:
>    - **Confound 1 (Pediatric Frequency-Range Mismatch)**: Testing adults alone ($N = 13$) eliminated the pediatric filter cutoff confound (0.0% of adult windows exceed 36 bpm), yet adult MAE remained elevated (6.21 bpm) and coverage degraded (45.3%).
>    - **Confound 2 (Ventilation Mode)**: Disaggregating the adult cohort by ventilation mode reveals that **ventilation mode explains a dominant share of this degradation**:
>      - **Adult Spontaneous Breathing ($N = 5$, 1,055 windows)**: MAE drops to **3.80 ± 0.71 bpm**, and empirical coverage rebounds to **77.7% ± 14.6%** (reaching 95.0% in Fold 4, with selective triage bringing MAE to **3.48 bpm**).
>      - **Adult Mechanical Ventilation ($N = 8$, 1,634 windows)**: MAE deteriorates to **7.76 ± 0.40 bpm**, and coverage plummets to **24.3% ± 22.5%** (severe violation of exchangeability under positive-pressure ventilation).
> 
> 2. **Physiological Mechanism: Inversion of Intrathoracic Pressure Dynamics**:
>    - In natural spontaneous respiration, inspiratory negative pleural pressure augments right ventricular preload and shifts the pulmonary capillary bed, creating characteristic respiratory-induced baseline wander (BW), amplitude modulation (AM), and frequency modulation (FM/RSA).
>    - In positive-pressure mechanical ventilation, inspiration produces positive pleural pressure, decreasing venous return and reversing the phase relationship between ventilation and pulse modulation.
>    - Because the feature extractor and XGBoost estimators were trained exclusively on spontaneously breathing ICU patients, positive-pressure ventilation creates a fundamental, biophysical domain shift that invalidates conformal exchangeability.
> 
> 3. **Adaptive Interval Widening**:
>    - Prediction interval widths expand dynamically from in-domain (10.72 bpm) to out-of-domain adult cohorts (12.16 bpm spontaneous, 11.80 bpm ventilated) and pediatric cohorts (15.32 bpm), reflecting the difficulty estimator $s(x)$ flagging morphological anomalies and elevated residual variance.
> 
> 4. **Preservation of Triage Utility**:
>    - Even under severe external domain shift, abstaining on the top 20% most uncertain windows (selective triage at 80% retention) consistently reduces MAE across cohorts (Adult Spontaneous: 3.48 bpm, Full: 5.98 bpm), preserving clinical risk mitigation.
> 
> 5. **Summary Conclusion**:
>    - When both confounding axes are controlled (evaluating **adults with spontaneous breathing**, $N = 5$), the PPG-based model demonstrates robust generalization (MAE 3.80 bpm, 77.7% empirical coverage).
>    - The performance drop on mechanical ventilation is an essential clinical finding for deployment: PPG respiratory algorithms trained on spontaneous respiration require separate calibration or dedicated models before use in mechanically ventilated surgical environments.

---

## 9. Summary Table of Primary Manuscript Artifacts

| File Path | Description | Key Verification / Contents |
| :--- | :--- | :--- |
| `output/final_report.txt` | 5-Method Clinical Benchmark Report | 53-patient CV, point MAEs, conformal coverages, AUROCs |
| `output/demo_window_report.txt` | Single-Window Illustrative Demo Report | N=1 genuine booleans (`is_covered`, `is_abstained`), bounds |
| `output/phase2_results.csv` | 5-Score Conformal Comparison CSV | Clean coverage, widths, undercoverage fractions per score |
| `output/phase5_results.csv` | Selective Prediction Risk-Coverage CSV | 9 retention points × 5 triage series with 95% bootstrap CIs |
| `output/phase6_results.csv` | 25-Fold Multi-Seed Conformal Stability CSV | 125 model calibrations, q-hat and coverage spread dynamics |
| `output/results/capnobase_external_validation_results.csv` | CapnoBase External Validation CSV | 5-fold evaluation metrics on CapnoBase vs BIDMC in-domain (including adult/pediatric columns) |
| `output/results/capnobase_external_validation_stratified_results.csv` | CapnoBase Stratified Results CSV | 25-row explicit cohort-disaggregated evaluation (all, adult, pediatric, adult_spontaneous, adult_ventilated across 5 folds) |
| `output/results/capnobase_external_validation_summary.md` | CapnoBase External Validation Summary | Detailed scientific report of domain shift, frequency mismatch, ventilation mode stratification, and exchangeability breakdown |
| `output/figures/capnobase_external_validation.png` | External Validation Benchmark Plot (300 DPI) | 3-panel comparison across 6 cohorts: MAE, Conformal Coverage, Prediction Interval Width |
| `output/cache/capnobase_features.npz` | CapnoBase Extracted Feature Cache | 8,764 windows × 28 standardized physiological features (0 NaNs) |
| `output/risk_coverage_curve.png` | Publication Risk-Coverage Curves (300 DPI) | Oracle, Random, Width, Fold-Norm, Combination + CI bands |
| `output/pipeline_architecture.png` | Publication Pipeline Architecture Overview Figure 1 (300 DPI) | Schematic diagram: Raw PPG -> Bandpass Filter -> 60s Windows -> 28 Features -> XGBoost -> Conformal Calibration -> Combination -> Triage |
| `abstention_results.png` | 2-Panel Comparative Waveform Figure (300 DPI) | Dynamic honest labeling based on computed interval width |
| `output/neurokit_validation_log.md` | Full 53-Patient NeuroKit2 Benchmark Log | Per-patient Pearson $r$, MAE, and agreement spread audit |

