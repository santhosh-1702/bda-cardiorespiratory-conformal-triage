# Data & Code Availability Statement

This document provides formal, publication-ready Data and Code Availability statements for the manuscript and repository:

> **Artifact-Resilient Cardiorespiratory Triage using Photoplethysmography and Split Conformal Prediction**  
> *PhysioNet BIDMC ICU Cohort (53 Patients) • Patient-Disjoint 5-Fold Cross-Validation*

---

## 1. Manuscript Availability Statement (Verbatim for Direct Adaptation)

### In-Domain Training & Evaluation Data (PhysioNet BIDMC)
The physiological biosignal data supporting the primary model development, cross-validation, and in-domain testing of this study are openly available in the PhysioNet repository as the **BIDMC Respiratory Rate Dataset (v1.0.0)** at https://doi.org/10.13026/C2201R (Pimentel et al., 2016 [1]). The dataset comprises continuous fingertip photoplethysmogram (PPG) and synchronous impedance pneumography respiratory reference waveforms recorded from 53 adult intensive care unit (ICU) patients at Beth Israel Deaconess Medical Center, Boston, MA. Data collection and distribution were managed by the PhysioNet research resource (Goldberger et al., 2000 [2]) and are governed under the Open Data Commons Attribution License (ODC-By v1.0). No clinical data was acquired outside of this publicly accessible cohort.

### External Validation Benchmark Data (CapnoBase IEEE TBME)
The out-of-distribution external validation data supporting this study are openly available from Borealis, the Canadian Dataverse Repository, as the **CapnoBase IEEE TBME Respiratory Rate Benchmark Dataset** at https://doi.org/10.5683/SP2/NLB8IT (Karlen et al., 2013 [3]). The benchmark comprises 42 8-minute recordings (300 Hz continuous fingertip PPG, synchronous expired CO2 capnography waveform, and ECG) collected from 42 surgical patients (13 adult, 29 pediatric) undergoing general anesthesia at BC Children's Hospital, Vancouver, Canada.

**Non-Redistribution Notice & Data Acquisition:**
In accordance with CapnoBase's Terms of Use, copying or redistributing the raw dataset without authorization is strictly prohibited. Consequently, the raw CapnoBase dataset is **not bundled or redistributed with this repository**. Each user or researcher wishing to parse raw signals must acquire the dataset directly from Borealis Dataverse under their own account.

**Local Directory Configuration:**
Once downloaded, place the dataset files in any of the following supported locations (auto-detected by `src/external_validation/capnobase_loader.py` in priority order):
1. **External directory (Recommended)**: Place at `../capnobase_data` (a sibling directory adjacent to the repository root).
2. **Environment variable**: Set the `CAPNOBASE_DIR` environment variable to the path containing the dataset (e.g., `export CAPNOBASE_DIR=/path/to/capnobase_data` or `set CAPNOBASE_DIR=D:\path\to\capnobase_data`).
3. **Local gitignored directory**: Place within the repository at `data/capnobase/` (this path is explicitly excluded in `.gitignore` to safeguard against accidental commits).

*Note: The repository ships with deterministic cached feature matrices (`output/cache/capnobase_features.npz`), allowing full reproduction of out-of-distribution conformal benchmark evaluations without requiring the 140MB raw signal download.*

### Code Availability
All source code, feature engineering routines, cross-validation splits, model training scripts, conformal prediction calibration algorithms, and visualization tools are open source under the MIT License and publicly accessible at https://github.com/santhosh-1702/bda-cardiorespiratory-conformal-triage. The repository includes:
1. Complete reproduction orchestrators (`run_pipeline.py`) capable of executing all pipeline phases on CPU-only hardware (Intel Core i3, 8GB RAM, no GPU requirement).
2. Exactly pinned dependency lockfiles (`requirements.txt`).
3. Machine-readable citation metadata (`CITATION.cff`).
4. An automated deterministic test suite (`python -m unittest discover tests`) verifying zero patient leakage across folds and numerical reproducibility.

---

## 2. Formal Bibliographic Citations

### [1] BIDMC Respiratory Rate Dataset Primary Publication
```bibtex
@article{pimentel2016towards,
  author    = {Pimentel, Marco A. F. and Johnson, Alistair E. W. and Charlton, Peter H. and Birrenkott, David and Watkinson, Peter J. and Tarassenko, Lionel and Clifton, David A.},
  title     = {Towards a Robust Estimation of Respiratory Rate from Pulse Oximeters},
  journal   = {IEEE Transactions on Biomedical Engineering},
  volume    = {64},
  number    = {8},
  pages     = {1914--1923},
  year      = {2016},
  publisher = {IEEE},
  doi       = {10.1109/TBME.2016.2613124}
}
```
**Plain Text Citation:**  
Pimentel, M.A.F., Johnson, A.E.W., Charlton, P.H., Birrenkott, D., Watkinson, P.J., Tarassenko, L., & Clifton, D.A. (2016). "Towards a Robust Estimation of Respiratory Rate from Pulse Oximeters." *IEEE Transactions on Biomedical Engineering*, 64(8), pp. 1914–1923. DOI: [10.1109/TBME.2016.2613124](https://doi.org/10.1109/TBME.2016.2613124).

---

### [2] PhysioNet Core Resource Publication
```bibtex
@article{goldberger2000physiobank,
  author    = {Goldberger, Ary L. and Amaral, Luis A. N. and Glass, Leon and Hausdorff, Jeffrey M. and Ivanov, Plamen Ch. and Mark, Roger G. and Mietus, Joseph E. and Moody, George B. and Peng, Chung-Kang and Stanley, H. Eugene},
  title     = {PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals},
  journal   = {Circulation},
  volume    = {101},
  number    = {23},
  pages     = {e215--e220},
  year      = {2000},
  publisher = {American Heart Association},
  doi       = {10.1161/01.CIR.101.23.e215}
}
```
**Plain Text Citation:**  
Goldberger, A.L., Amaral, L.A.N., Glass, L., Hausdorff, J.M., Ivanov, P.Ch., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.-K., & Stanley, H.E. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals." *Circulation*, 101(23), pp. e215–e220. DOI: [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215).

---

### [3] CapnoBase External Validation Dataset Primary Publication
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

## 3. License & Governance Summary

- **Software Code**: Licensed under the [MIT License](LICENSE).
- **BIDMC Clinical Data**: Governed by PhysioNet under the Open Data Commons Attribution License (ODC-By v1.0). The dataset is downloaded directly from PhysioNet servers and is **not relicensed** by this repository.
- **CapnoBase Benchmark Data**: Governed by the CapnoBase Research Group and Borealis Dataverse under CapnoBase Terms of Use for non-commercial academic research. Copying or redistribution is prohibited without authorization; raw data files are **not redistributed** within this repository and must be acquired directly from [Borealis Dataverse](https://doi.org/10.5683/SP2/NLB8IT).
