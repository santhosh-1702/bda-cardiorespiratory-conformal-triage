# Phase 1.5: Biosignal Foundation & Baseline Evaluation

## Auto-Computed Summary Statements
- **Fold Wins**: XGBoost beats train median in 4 of 5 folds (60s window MAE: 1.8084 vs 2.1215 bpm).
- **Validation-Chosen Windows**: Selected window lengths across folds: [60, 60, 60, 60, 60] s (Test MAE: 1.8084 ± 0.4235 bpm).
- **Paired Patient Bootstrap (1000 resamples)**: 95% CI for [MAE(XGBoost) − MAE(Train Median)] = [-0.5302, -0.0253] bpm. Status: excludes 0 (statistically significant).

## Full Benchmark Results Table

| Method | Window (s) | MAE (mean ± std bpm) | R² (mean ± std) | Pearson r (mean ± std) | Within-Patient MAE (bpm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Train Mean | 6 | 2.2750 ± 0.3575 | -0.0500 ± 0.0872 | N/A | 0.9233 |
| Train Median | 6 | 2.3004 ± 0.3549 | -0.0454 ± 0.0820 | N/A | 0.9233 |
| Classical Spectral | 6 | 3.8639 ± 0.4568 | -2.4052 ± 0.8774 | 0.2850 | 3.5647 |
| RIAV | 6 | 5.6295 ± 0.5276 | -5.5583 ± 1.3798 | 0.1329 | 5.2762 |
| RIIV | 6 | 4.8994 ± 0.5174 | -4.2133 ± 0.9145 | 0.2261 | 4.5131 |
| RIFV | 6 | 7.4295 ± 0.5347 | -9.5732 ± 1.6296 | 0.1004 | 6.9786 |
| Fusion (Median) | 6 | 4.4468 ± 0.4041 | -3.3302 ± 0.7864 | 0.2309 | 4.1189 |
| Retrained XGBoost | 6 | 2.2511 ± 0.2960 | -0.0655 ± 0.1412 | 0.2134 | 1.1204 |
| Train Mean | 15 | 2.2526 ± 0.3572 | -0.0521 ± 0.0910 | N/A | 0.8735 |
| Train Median | 15 | 2.2586 ± 0.3465 | -0.0369 ± 0.0713 | N/A | 0.8735 |
| Classical Spectral | 15 | 2.9955 ± 0.6068 | -1.7741 ± 0.9358 | 0.3389 | 2.7901 |
| RIAV | 15 | 5.5252 ± 0.9019 | -5.8376 ± 1.5607 | 0.0822 | 4.6798 |
| RIIV | 15 | 4.2920 ± 0.7851 | -3.7222 ± 1.1098 | 0.2064 | 3.7546 |
| RIFV | 15 | 7.4913 ± 0.4533 | -9.7611 ± 2.5406 | 0.1036 | 6.2672 |
| Fusion (Median) | 15 | 4.4853 ± 0.5666 | -4.0900 ± 1.2520 | 0.1864 | 3.8366 |
| Retrained XGBoost | 15 | 2.1338 ± 0.4033 | -0.0463 ± 0.2261 | 0.2824 | 1.0743 |
| Train Mean | 30 | 2.2128 ± 0.3539 | -0.0554 ± 0.0971 | N/A | 0.7851 |
| Train Median | 30 | 2.2065 ± 0.3423 | -0.0365 ± 0.0678 | N/A | 0.7851 |
| Classical Spectral | 30 | 2.6050 ± 0.6616 | -1.4482 ± 1.0401 | 0.3943 | 2.3125 |
| RIAV | 30 | 4.5746 ± 0.9366 | -4.6980 ± 1.6811 | 0.0944 | 3.7806 |
| RIIV | 30 | 3.4310 ± 0.8145 | -2.7573 ± 1.2558 | 0.2759 | 2.9542 |
| RIFV | 30 | 6.4586 ± 0.5339 | -8.2037 ± 2.1499 | 0.1104 | 5.3465 |
| Fusion (Median) | 30 | 3.5553 ± 0.5981 | -2.9878 ± 1.2720 | 0.2390 | 3.0594 |
| Retrained XGBoost | 30 | 1.9532 ± 0.4623 | 0.0301 ± 0.2873 | 0.3781 | 0.9838 |
| Train Mean | 60 | 2.1355 ± 0.3422 | -0.0598 ± 0.1057 | N/A | 0.6335 |
| Train Median | 60 | 2.1215 ± 0.3264 | -0.0366 ± 0.0660 | N/A | 0.6335 |
| Classical Spectral | 60 | 2.5368 ± 0.6661 | -1.5845 ± 1.2848 | 0.4200 | 2.1996 |
| RIAV | 60 | 3.9508 ± 1.0010 | -4.1891 ± 1.9385 | 0.1136 | 3.0692 |
| RIIV | 60 | 2.9461 ± 0.9496 | -2.4756 ± 1.7401 | 0.3383 | 2.3305 |
| RIFV | 60 | 5.8187 ± 0.5277 | -7.4888 ± 2.0788 | 0.1311 | 4.5925 |
| Fusion (Median) | 60 | 3.0825 ± 0.6498 | -2.7119 ± 1.5936 | 0.2768 | 2.5204 |
| Retrained XGBoost | 60 | 1.8084 ± 0.4235 | 0.0914 ± 0.2872 | 0.4139 | 0.8274 |
| XGBoost (Val-Chosen Window) | val-selected | 1.8084 ± 0.4235 | 0.0914 ± 0.2872 | 0.4139 | 0.8274 |

## Fold Assignment Comparison (Requirement 4A.3)
- **Old vs New Fold Map**: Old Phase 1.5 folds were generated via GroupKFold on window arrays, balancing window sample counts across folds. Because window counts differed per patient across durations (e.g. bidmc13 had 54 windows vs ~211), test patient assignments varied across window lengths. Phase 4 enforces ONE unified, seeded patient->fold map (output/fold_map.json) generated purely from the 53 patient IDs. The resulting redistribution of heterogeneous patients across test folds accounts for the observed difference in 60s baseline metrics.

---
*Verification Status*: Evaluated across 53 BIDMC patients via unified patient->fold map. Test folds held out strictly for final evaluation.