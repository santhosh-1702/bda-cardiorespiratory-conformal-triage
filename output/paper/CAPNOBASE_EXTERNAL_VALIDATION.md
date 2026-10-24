# CapnoBase External Validation Dataset & Provenance Statement

This document provides formal, publication-ready Data Availability, Provenance, Bibliographic Citation, and Stratified Benchmark statements for the external validation benchmark dataset used in:

> **Artifact-Resilient Cardiorespiratory Triage using Photoplethysmography and Split Conformal Prediction**  
> *External Out-of-Distribution Validation Benchmark • CapnoBase IEEE TBME Respiratory Rate Dataset (42 Cases)*

---

## 1. External Validation Study Design & Clinical Rationale

To evaluate the out-of-distribution generalization and clinical portability of the cardiorespiratory triage pipeline trained on the PhysioNet BIDMC ICU cohort (53 patients), the pipeline incorporates the **CapnoBase IEEE TBME Respiratory Rate Benchmark** as a strictly frozen, zero-shot external test set:

1. **Independent Patient Population**:
   - BIDMC: Adult intensive care unit (ICU) patients at Beth Israel Deaconess Medical Center, Boston, MA.
   - CapnoBase: Pediatric and adult surgical patients undergoing general anesthesia at BC Children's Hospital, Vancouver, Canada (29 pediatric, 13 adult cases).
2. **Independent Respiratory Reference Modality**:
   - BIDMC: Synchronous impedance pneumography (IP) chest-impedance respiratory waveforms.
   - CapnoBase: Capnography (mainstream/sidestream expired end-tidal CO2 waveform) with expert human-annotated breaths and pulse peaks.
3. **Independent Sensor Hardware & Native Sampling Rate**:
   - BIDMC: 125 Hz pulse oximeter waveforms.
   - CapnoBase: 300 Hz S/5 anesthesia monitor fingertip transmission pulse oximeter waveforms.
4. **Strictly External / Frozen Evaluation**:
   - In accordance with the CapnoBase benchmark protocol and conformal prediction principles, zero CapnoBase recordings or windows were included in model hyperparameter tuning, feature selection, or split conformal calibration. The entire dataset serves strictly as an out-of-distribution evaluation set.

---

## 2. Demographic Stratification & Physiological Frequency-Range Mismatch

### Cutoff Definition & Cohort Partition
In accordance with CapnoBase benchmark documentation (Karlen et al., 2013 [1]), cases are partitioned into distinct age cohorts and ventilation modes:
- **Adult Cohort (Primary Fair Generalization Benchmark)**: Age $\ge 18$ years ($N = 13$ cases, 2,689 clean 60-s windows, mean age 47.0 years, range 32–63 years).
  - **Adult Spontaneous Breathing**: $N = 5$ cases (1,055 clean windows; cases 0322, 0330, 0331, 0332, 0370).
  - **Adult Mechanical Ventilation**: $N = 8$ cases (1,634 clean windows; cases 0309, 0311, 0312, 0313, 0325, 0328, 0329, 0333).
- **Pediatric Cohort (Quantified Frequency Mismatch)**: Age $< 18$ years ($N = 29$ cases, 6,075 clean 60-s windows, mean age 7.5 years, range 1–16 years, including case 0032 with weight 26 kg; 18 spontaneous, 11 ventilated).
- **Full Benchmark Cohort**: All 42 cases combined ($N = 42$ cases, 8,764 clean 60-s windows; 23 spontaneous, 19 ventilated).

### Quantified Frequency-Range Mismatch
- **Adult-Tuned Filter Passband**: The pipeline's 4th-order Butterworth filter passband is **0.10–0.60 Hz (6–36 bpm)** and `VALID_RR_RANGE` is **(6.0, 45.0) bpm**, configured specifically for adult ICU physiology.
- **Pediatric Respiratory Range**: Pediatric resting breathing rates naturally reach 24–40+ bpm:
  - **Pediatric Windows > 36 bpm**: **272 of 6,075 windows (4.48%)** have ground-truth respiratory rates strictly above the filter passband ceiling (maximum observed window RR: 42.4 bpm).
  - **Pediatric 1 Hz Reference Points > 36 bpm**: **617 of 13,006 seconds (4.74%)** exceed 36 bpm.
  - **Pediatric 1 Hz Reference Points > 45 bpm**: **5 of 13,006 seconds (0.04%)** exceed the pipeline's 45 bpm ceiling (maximum instantaneous RR: 47.5 bpm).
  - **Adult Cohort Comparison**: In contrast, **0.00% (0 of 2,689 windows)** of the CapnoBase adult cohort exceed 36 bpm (maximum observed adult window RR: 29.7 bpm).

> **Methodological Consequence**: The adult-only comparison eliminates frequency-range mismatch. For pediatric cases, frequency components above 36 bpm are physically filtered out by the adult-tuned bandpass filter, representing a hardware/filter passband mismatch.

---

## 3. Impact of Ventilation Mode: Resolving the Adult Generalization Gap

Disaggregating the 13 adult cases by ventilation mode identifies the underlying biophysical driver of the domain transfer gap:
1. **Adult Spontaneous Respiration ($N = 5$ cases, 1,055 windows)**:
   - Evaluated against frozen BIDMC models, spontaneous breathing adults achieve **MAE = 3.80 ± 0.71 bpm** and empirical coverage of **77.7% ± 14.6%** (reaching 95.0% in Fold 4).
   - Selective triage at 80% retention reduces MAE to **3.48 ± 0.72 bpm**.
2. **Adult Mechanical Ventilation ($N = 8$ cases, 1,634 windows)**:
   - On mechanically ventilated adults, point error rises to **MAE = 7.76 ± 0.40 bpm**, and conformal coverage collapses to **24.3% ± 22.5%**.
3. **Biophysical Mechanism**: Positive-pressure ventilation reverses the thoracic pressure cycle (positive pleural pressure on inspiration rather than negative pleural pressure), suppressing venous return and altering the amplitude, baseline, and frequency modulation components in fingertip PPG that the models learned from spontaneously breathing ICU patients.
4. **Conclusion**: **Ventilation mode accounts for a dominant share of the adult performance degradation.** Natural spontaneous respiration generalizes well across clinical institutions, whereas positive-pressure ventilation introduces an exchangeability-violating physiological shift requiring dedicated calibration.

---

## 4. Stratified Benchmark Results Summary (5-Fold Repeated Evaluation)

| Metric | In-Domain (BIDMC ICU, N=53) | Adult Spont (Natural, N=5) | Adult Vent (Mech, N=8) | External Adult All (Primary, N=13) | External Pediatric (Outside Passband, N=29) | Full Benchmark (All 42 Cases) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort Size** | 53 ICU Patients (11,026 wins) | 5 Adult Cases (1,055 wins) | 8 Adult Cases (1,634 wins) | 13 Adult Cases (2,689 wins) | 29 Pediatric Cases (6,075 wins) | 42 Cases (8,764 wins) |
| **Physiological Match** | In-Range (Adult ICU) | **Full Match (Spontaneous)** | Mismatch (Positive-Pressure) | Full Passband (Blended Vent) | Partial (4.48% > 36 bpm) | Blended |
| **Point Prediction MAE (bpm)** | **1.78 ± 0.55** | **3.80 ± 0.71** | **7.76 ± 0.40** | **6.21 ± 0.36** | **6.39 ± 0.29** | **6.33 ± 0.31** |
| **Empirical Coverage (%)** | **88.1% ± 11.9%** | **77.7% ± 14.6%** | **24.3% ± 22.5%** | **45.3% ± 18.4%** | **59.8% ± 21.6%** | **55.3% ± 20.3%** |
| **Mean Interval Width (bpm)** | **10.72 ± 4.87** | **12.16 ± 6.32** | **11.80 ± 4.83** | **11.94 ± 5.39** | **15.32 ± 6.86** | **14.29 ± 6.32** |
| **Patient Coverage (Mean ± Std)** | **87.1% ± 19.2%** | **77.7% ± 24.7%** | **23.7% ± 21.3%** | **44.5% ± 37.9%** | **59.8% ± 37.4%** | **55.1% ± 38.5%** |
| **Error AUROC (|Err| > 3 bpm)** | **0.712 ± 0.127** | **0.535 ± 0.115** | **0.445 ± 0.123** | **0.528 ± 0.148** | **0.619 ± 0.100** | **0.577 ± 0.101** |
| **Selective Triage MAE (80% Ret.)** | — | **3.48 ± 0.72** | **7.86 ± 0.39** | **6.17 ± 0.44** | **5.97 ± 0.35** | **5.98 ± 0.31** |

---

## 5. Manuscript Data Availability Statement (Verbatim for Direct Adaptation)

### External Validation Data Availability
The external validation data supporting this study are openly available in the CapnoBase research resource as the **CapnoBase IEEE TBME Respiratory Rate Benchmark Dataset** (Karlen et al., 2013 [1]) hosted on Borealis, the Canadian Dataverse Repository (https://doi.org/10.5683/SP2/NLB8IT). The benchmark comprises 42 8-minute recordings (300 Hz continuous PPG, synchronous expired CO2 waveform, and ECG) collected from 42 elective surgery patients under anesthesia. Each recording includes expert-adjudicated breath onset annotations and capnometry-derived reference respiratory rates. Access is provided under the terms of the CapnoBase research initiative (http://www.capnobase.org / https://doi.org/10.5683/SP2/NLB8IT). Data files are parsed, resampled to 125 Hz using polyphase FIR anti-aliasing filtering, and processed through the exact frozen feature extraction pipeline without retraining or parameter adjustment.

### Non-Redistribution Notice & Data Acquisition Instructions
Under CapnoBase's Terms of Use, copying or redistributing the raw recordings without authorization is strictly prohibited. Consequently, the raw CapnoBase dataset is **not bundled or redistributed with this repository**. Individual researchers must acquire the dataset directly from Borealis Dataverse under their own account:
- **Direct Repository**: Borealis Dataverse (https://doi.org/10.5683/SP2/NLB8IT) or CapnoBase project repository (http://www.capnobase.org).

### Local Placement & Environment Configuration
When downloading the raw CapnoBase files (CSV format or raw archives), place the data in one of the following locations recognized automatically by the loader:
1. **Adjacent External Directory (Recommended)**: Place the files at `../capnobase_data` (a sibling directory alongside the project repository root).
2. **Environment Variable**: Set `CAPNOBASE_DIR` to the folder containing the downloaded case files (`export CAPNOBASE_DIR=/path/to/capnobase_data` or `set CAPNOBASE_DIR=D:\path\to\capnobase_data`).
3. **Local Gitignored Directory**: Place inside the repository at `data/capnobase/` (this path is protected by `.gitignore` to prevent committing raw data).

*Offline Execution Note*: The repository includes pre-extracted and verified feature matrices (`output/cache/capnobase_features.npz`). The complete 5-fold out-of-distribution evaluation and conformal coverage analysis can be fully executed and verified (`run_pipeline.py --stage capnobase` or `python -m unittest discover tests`) without downloading the raw 140MB signal files.

---

## 6. Formal Bibliographic Citation

### [1] CapnoBase IEEE TBME Benchmark Primary Publication
```bibtex
@article{karlen2013multiparameter,
  author    = {Karlen, Walter and Raman, Sanjeev and Ansermino, J. Mark and Dumont, Guy A.},
  title     = {Multiparameter Respiratory Rate Estimation from the Photoplethysmogram},
  journal   = {IEEE Transactions on Biomedical Engineering},
  volume    = {60},
  number    = {7},
  pages     = {1946--1953},
  year      = {2013},
  publisher = {IEEE},
  doi       = {10.1109/TBME.2013.2246160}
}
```

**Plain Text Citation:**  
Karlen, W., Raman, S., Ansermino, J.M., & Dumont, G.A. (2013). "Multiparameter respiratory rate estimation from the photoplethysmogram." *IEEE Transactions on Biomedical Engineering*, 60(7), pp. 1946–1953. DOI: [10.1109/TBME.2013.2246160](https://doi.org/10.1109/TBME.2013.2246160).

---

## 7. Dataset Preprocessing & Standardization Summary

| Parameter | PhysioNet BIDMC (In-Domain) | CapnoBase Benchmark (External) | Standardization Applied |
| :--- | :--- | :--- | :--- |
| **Cohort Size** | 53 ICU Patients | 42 Surgical Cases (13 Adult, 29 Pediatric) | Patient-level disjoint sets |
| **Recording Length** | 8.0 minutes (480 s) | 8.0 minutes (480 s) | Identical duration |
| **Native Sampling Rate** | 125.0 Hz | 300.0 Hz | Resampled to 125 Hz via `resample_poly(up=5, down=12)` |
| **Respiratory Bandpass** | 0.10–0.60 Hz (4th-order Butterworth SOS) | 0.10–0.60 Hz (4th-order Butterworth SOS) | Exact same filter instance |
| **Wideband Filter** | 0.10–5.00 Hz (4th-order Butterworth SOS) | 0.10–5.00 Hz (4th-order Butterworth SOS) | Exact same filter instance |
| **Feature Extraction** | 28 features (spectral, modulation, stats) | 28 features (spectral, modulation, stats) | Unmodified `extract_window_features_from_segments()` |
| **Windowing Scheme** | 60 s window / 2 s stride | 60 s window / 2 s stride | Identical 211 candidate steps / recording |
| **Ground-Truth Granularity** | 1 Hz numerics array (`gt_rr_1hz`) | 1 Hz interpolated capnography reference | Identical 1 Hz temporal resolution |
| **Role in Pipeline** | Train (50%), Val (20%), Cal (15%), Test (15%) | 100% External Holdout Test | Completely frozen; zero tuning |

---

## 8. License & Governance Summary

- **Software Pipeline Code**: Licensed under the [MIT License](LICENSE).
- **CapnoBase Benchmark Data**: Created and distributed by the CapnoBase research group and hosted on Borealis Dataverse under the CapnoBase Terms of Use for non-commercial academic research. Copying or redistribution is prohibited without authorization; the raw dataset is **not redistributed** with this repository and remains governed by its original copyright holders. Researchers must obtain access directly from Borealis Dataverse (https://doi.org/10.5683/SP2/NLB8IT).
