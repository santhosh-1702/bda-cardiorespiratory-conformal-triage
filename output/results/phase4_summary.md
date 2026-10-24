# Phase 4 & 5: Experimental Results & Statistical Audit Summary

## 1. System Verification Status
- Status: Bit-Identical Determinism Verified | Zero Test Leakage Verified
- Maximum Absolute Determinism Difference: 0.0000000000
- Cohort Size: 53 Patients | Folds: 5 Patient-Disjoint Folds

## 2. Repeated Grouped Cross-Validation (5 Repeats: XGBoost vs Train-Median Baseline)
- XGBoost Mean Test MAE Across Repeats: 1.8123 bpm (Between-Repeat Std: 0.0912 bpm)
- XGBoost Repeat MAEs: [1.8564, 1.9205, 1.8748, 1.6833, 1.7265]
- Train-Median Mean Test MAE Across Repeats: 2.1279 bpm (Between-Repeat Std: 0.0315 bpm)
- Train-Median Repeat MAEs: [2.1139, 2.1893, 2.1236, 2.1017, 2.1111]
- Paired Patient Bootstrap Over Repeats (1000 resamples): Mean Diff = -0.3065 bpm | 95% CI: [-0.5053, -0.1103] bpm | Zero Excluded: True

## 3. Patient With 54 Windows Clinical Audit (bidmc13)
- Total Seconds in Numerics: 481
- NaN Seconds: 146 (30.4%)
- Below 6 bpm (Zero/Dropout) Seconds: 230 (47.8%)
- Valid Physiological Seconds: 105 (21.8%)
- Candidate 60s Windows: 211
- Sub-physiological Discarded Windows: 157
- Valid Retained Windows: 54
- Audit Note: Exclude in clinical acquisition QA due to prolonged sensor flatline / dropout (47.8% of recording had reference RR=0 bpm and 30.4% was NaN; reference sensor was disconnected). Retained in 53-patient benchmark to strictly avoid post-hoc exclusion bias.

## 4. Waveform Alteration & SNR Verification per Severity

| Severity | Mean Absolute Change (MAC) | Relative to Signal Std (%) | Achieved SNR (dB) | Distinguishable from Clean |
| :--- | :---: | :---: | :---: | :---: |
| Mild | 0.007176 ± 0.003521 | 28.3% | 5.03 ± 2.05 dB | True |
| Moderate | 0.027637 ± 0.014987 | 109.1% | -5.47 ± 3.76 dB | True |
| Severe | 0.102235 ± 0.087041 | 403.7% | -12.59 ± 7.38 dB | True |

## 5. Per-Fold Conformal Quantile (q_hat) & Width Scale Audit Table

| Fold | q_hat | Cal Pts | Cal Wins | Test Wins | Test MAE (bpm) | Test Cov | Mean Width (bpm) | Median Width (bpm) | Width Std | P10 Width | P90 Width | Mean Norm Width |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 1.6502 | 8 | 1531 | 2321 | 2.1820 | 76.2% | 5.88 | 5.92 | 2.59 | 2.71 | 8.66 | 3.57 |
| 1 | 1.6777 | 8 | 1688 | 2321 | 2.3085 | 86.1% | 9.01 | 6.28 | 7.51 | 2.90 | 17.16 | 5.37 |
| 2 | 2.1014 | 8 | 1688 | 2164 | 2.0221 | 74.1% | 5.81 | 4.53 | 3.70 | 3.00 | 11.09 | 2.76 |
| 3 | 4.5859 | 9 | 1742 | 2110 | 1.0613 | 100.0% | 18.29 | 14.07 | 15.45 | 5.82 | 37.32 | 3.99 |
| 4 | 3.6595 | 9 | 1899 | 2110 | 1.3266 | 95.3% | 16.18 | 13.16 | 10.15 | 8.28 | 28.27 | 4.42 |

## 6. Raw-Signal Artifact Experiment: Corruption-Detection AUROC (Three Ways)

| Severity | Condition | Test MAE (bpm) | Coverage | Mean Width (bpm) | Abstention (%) | Pooled Raw Width AUROC [95% CI] | Fold Raw Width AUROC (Mean ± Std) | Pooled Norm Width AUROC [95% CI] | Fold Norm Width AUROC (Mean ± Std) | Pooled SQI AUROC [95% CI] | Fold SQI AUROC (Mean ± Std) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mild | Clean | 1.7801 | 86.3% | 11.03 | 15.8% | N/A | N/A | N/A | N/A | N/A | N/A |
| Mild | Corrupted | 1.9933 | 90.1% | 12.93 | 27.6% | 0.6278 [0.5665, 0.6925] | 0.6515 ± 0.1757 | 0.6590 [0.5886, 0.7284] | 0.6515 ± 0.1757 | 0.8979 [0.8790, 0.9156] | 0.9010 ± 0.0255 |
| Moderate | Clean | 1.7801 | 86.3% | 11.03 | 15.8% | N/A | N/A | N/A | N/A | N/A | N/A |
| Moderate | Corrupted | 2.3369 | 88.7% | 12.99 | 36.6% | 0.6662 [0.5948, 0.7378] | 0.6965 ± 0.2051 | 0.6940 [0.6195, 0.7675] | 0.6965 ± 0.2051 | 0.9922 [0.9884, 0.9951] | 0.9923 ± 0.0050 |
| Severe | Clean | 1.7801 | 86.3% | 11.03 | 15.8% | N/A | N/A | N/A | N/A | N/A | N/A |
| Severe | Corrupted | 2.3900 | 91.2% | 14.37 | 47.0% | 0.7289 [0.6529, 0.8071] | 0.7493 ± 0.1992 | 0.7487 [0.6720, 0.8223] | 0.7493 ± 0.1992 | 0.9998 [0.9997, 0.9999] | 0.9998 ± 0.0001 |

## 7. Clean-Window Selective Prediction Analysis (|Error| > 3 bpm)
- Plain Evaluated Verdict: Conformal interval width (AUROC = 0.6067 [95% CI: 0.5027, 0.7175]) predicts true error (|error| > 3 bpm) better than SQI score (AUROC = 0.6022 [95% CI: 0.5106, 0.6981]).

| Predictor | Pooled AUROC [95% Patient Bootstrap CI] | Fold AUROC (Mean ± Std) |
| :--- | :---: | :---: |
| (a) Conformal Interval Width (Raw) | 0.6067 [0.5027, 0.7175] | 0.6793 ± 0.1544 |
| (a) Conformal Interval Width (Fold-Norm) | 0.7023 [0.5972, 0.8038] | 0.6793 ± 0.1544 |
| (b) Signal Quality Index (SQI) Score | 0.6022 [0.5106, 0.6981] | 0.6257 ± 0.1356 |
| (c) Multi-Domain Combination (Val-Trained) | 0.7527 [0.6335, 0.8562] | 0.7105 ± 0.1089 |

## 8. Selective Prediction Risk-Coverage Operating Points (Clean Test Data)
- Plain Evaluated Retention Verdict: At 40% retention, fold-normalized width alone closes 54.3% of the random-to-oracle gap (MAE = 1.0122 bpm), whereas combination score retention closes 45.8% (MAE = 1.1387 bpm [0.7894, 1.5872]); however, combination retention achieves superior error reduction across 50%–95% retention fractions (e.g., MAE = 1.1788 vs 1.4135 bpm at 60% retention).

| Retention (%) | Raw Width MAE (bpm) [95% CI] | Fold-Norm MAE (bpm) [95% CI] | Combination Score MAE (bpm) [95% CI] | Oracle MAE (bpm) [95% CI] | Random MAE (bpm) [95% CI] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 40% | 1.2038 [0.8497, 1.9321] | 1.0122 [0.7358, 1.5154] | 1.1387 [0.7894, 1.5872] | 0.3310 [0.2644, 0.4361] | 1.8205 [1.3558, 2.3631] |
| 50% | 1.5885 [1.0326, 2.2093] | 1.2596 [0.8402, 1.8030] | 1.1467 [0.8366, 1.5700] | 0.4281 [0.3410, 0.5646] | 1.8154 [1.3503, 2.3535] |
| 60% | 1.7097 [1.1753, 2.2904] | 1.4135 [0.9647, 1.9875] | 1.1788 [0.8560, 1.5903] | 0.5427 [0.4303, 0.7126] | 1.7763 [1.3248, 2.2900] |
| 70% | 1.7407 [1.2629, 2.3372] | 1.5299 [1.0761, 2.0830] | 1.2678 [0.9124, 1.7983] | 0.6828 [0.5379, 0.9024] | 1.7861 [1.3411, 2.3124] |
| 80% | 1.8282 [1.3455, 2.3854] | 1.6423 [1.2005, 2.1597] | 1.4061 [1.0090, 1.9134] | 0.8724 [0.6842, 1.1907] | 1.7892 [1.3346, 2.3180] |
| 85% | 1.8962 [1.3813, 2.4415] | 1.6575 [1.2279, 2.1635] | 1.4888 [1.0950, 1.9841] | 1.0044 [0.7761, 1.4158] | 1.7954 [1.3413, 2.3254] |
| 90% | 1.8547 [1.3593, 2.4338] | 1.6734 [1.2575, 2.2175] | 1.5714 [1.1678, 2.0582] | 1.1922 [0.8925, 1.6932] | 1.7912 [1.3395, 2.3256] |
| 95% | 1.8388 [1.3550, 2.3863] | 1.7737 [1.3130, 2.3290] | 1.6423 [1.2587, 2.2164] | 1.4588 [1.0551, 1.9897] | 1.7972 [1.3401, 2.3252] |
| 100% | 1.7991 [1.3393, 2.3350] | 1.7991 [1.3393, 2.3350] | 1.7991 [1.3393, 2.3350] | 1.7991 [1.3393, 2.3350] | 1.7991 [1.3393, 2.3350] |