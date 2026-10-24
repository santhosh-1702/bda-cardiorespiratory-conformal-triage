# Phase 2A.5: NeuroKit2 Validation Reference Log

## Quantitative Comparison
Benchmarking custom signal processing against `neurokit2` (v0.2.13) on 53 BIDMC recordings.

### 1. Results Table

| Patient | Custom Peaks | NK2 Peaks | Matched Peaks (%) | Median |Δt| (ms) | Mean HR (Custom) | Mean HR (NK2) | Spectral RR Pearson r | Spec Inter-Method MAE (bpm) | Custom vs GT MAE | NK2 vs GT MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bidmc01 | 717 | 717 | 99.3% | 0.0 ms | 89.7 bpm | 89.7 bpm | 0.591 | 1.68 bpm | 1.82 bpm | 1.84 bpm |
| bidmc02 | 732 | 726 | 99.2% | 0.0 ms | 91.5 bpm | 90.9 bpm | 0.042 | 1.14 bpm | 1.24 bpm | 0.33 bpm |
| bidmc03 | 611 | 608 | 99.5% | 0.0 ms | 76.4 bpm | 76.1 bpm | 0.705 | 1.83 bpm | 2.60 bpm | 2.66 bpm |
| bidmc04 | 715 | 701 | 97.5% | 0.0 ms | 89.3 bpm | 87.7 bpm | 0.659 | 1.17 bpm | 1.51 bpm | 1.71 bpm |
| bidmc05 | 767 | 762 | 99.0% | 0.0 ms | 95.9 bpm | 95.3 bpm | 0.250 | 2.33 bpm | 3.31 bpm | 3.93 bpm |
| bidmc06 | 655 | 653 | 99.7% | 0.0 ms | 81.8 bpm | 81.7 bpm | nan | 0.40 bpm | 0.51 bpm | 0.13 bpm |
| bidmc07 | 722 | 720 | 99.7% | 0.0 ms | 90.2 bpm | 90.2 bpm | -0.024 | 0.38 bpm | 0.43 bpm | 0.11 bpm |
| bidmc08 | 798 | 798 | 100.0% | 0.0 ms | 99.8 bpm | 99.8 bpm | 0.401 | 3.83 bpm | 3.04 bpm | 4.11 bpm |
| bidmc09 | 614 | 614 | 100.0% | 0.0 ms | 76.8 bpm | 76.8 bpm | 0.006 | 0.68 bpm | 0.66 bpm | 0.02 bpm |
| bidmc10 | 660 | 636 | 96.2% | 0.0 ms | 82.5 bpm | 79.5 bpm | 0.644 | 1.81 bpm | 2.38 bpm | 2.55 bpm |
| bidmc11 | 743 | 718 | 96.6% | 0.0 ms | 93.0 bpm | 89.9 bpm | 0.267 | 1.23 bpm | 1.72 bpm | 1.48 bpm |
| bidmc12 | 744 | 735 | 98.8% | 0.0 ms | 93.1 bpm | 92.1 bpm | 0.745 | 0.61 bpm | 0.45 bpm | 0.62 bpm |
| bidmc13 | 536 | 534 | 99.6% | 0.0 ms | 67.1 bpm | 66.9 bpm | 0.826 | 1.80 bpm | 7.94 bpm | 6.82 bpm |
| bidmc14 | 752 | 691 | 91.5% | 0.0 ms | 94.0 bpm | 86.5 bpm | 0.430 | 0.74 bpm | 6.82 bpm | 6.78 bpm |
| bidmc15 | 858 | 855 | 99.7% | 0.0 ms | 107.3 bpm | 107.0 bpm | 0.430 | 0.77 bpm | 1.53 bpm | 1.34 bpm |
| bidmc16 | 876 | 872 | 99.5% | 0.0 ms | 109.5 bpm | 109.2 bpm | 0.578 | 1.19 bpm | 1.28 bpm | 1.32 bpm |
| bidmc17 | 650 | 644 | 99.1% | 0.0 ms | 81.3 bpm | 80.5 bpm | 0.057 | 4.51 bpm | 4.48 bpm | 2.29 bpm |
| bidmc18 | 721 | 720 | 99.9% | 0.0 ms | 90.1 bpm | 90.1 bpm | nan | 0.56 bpm | 0.69 bpm | 0.13 bpm |
| bidmc19 | 593 | 577 | 97.0% | 0.0 ms | 74.2 bpm | 72.2 bpm | 0.696 | 1.49 bpm | 3.45 bpm | 3.44 bpm |
| bidmc20 | 715 | 688 | 96.2% | 0.0 ms | 89.3 bpm | 86.1 bpm | 0.309 | 0.50 bpm | 0.67 bpm | 0.26 bpm |
| bidmc21 | 988 | 964 | 97.6% | 0.0 ms | 123.5 bpm | 120.5 bpm | 0.205 | 4.40 bpm | 5.28 bpm | 6.17 bpm |
| bidmc22 | 664 | 651 | 97.9% | 0.0 ms | 83.0 bpm | 81.5 bpm | 0.324 | 1.56 bpm | 1.28 bpm | 1.21 bpm |
| bidmc23 | 696 | 503 | 72.0% | 0.0 ms | 87.1 bpm | 62.9 bpm | 0.591 | 2.32 bpm | 4.33 bpm | 4.38 bpm |
| bidmc24 | 761 | 742 | 97.5% | 0.0 ms | 95.2 bpm | 93.0 bpm | 0.431 | 0.58 bpm | 0.63 bpm | 0.29 bpm |
| bidmc25 | 976 | 949 | 97.2% | 0.0 ms | 122.1 bpm | 118.8 bpm | 0.165 | 1.12 bpm | 1.18 bpm | 0.62 bpm |
| bidmc26 | 657 | 564 | 85.7% | 0.0 ms | 82.1 bpm | 70.5 bpm | 0.382 | 3.10 bpm | 3.62 bpm | 2.84 bpm |
| bidmc27 | 702 | 696 | 98.9% | 0.0 ms | 87.7 bpm | 87.1 bpm | 0.214 | 0.35 bpm | 0.76 bpm | 0.68 bpm |
| bidmc28 | 665 | 617 | 92.5% | 0.0 ms | 83.2 bpm | 77.2 bpm | -0.060 | 1.37 bpm | 1.34 bpm | 0.75 bpm |
| bidmc29 | 726 | 701 | 96.6% | 0.0 ms | 90.7 bpm | 87.6 bpm | 0.434 | 2.03 bpm | 2.34 bpm | 2.41 bpm |
| bidmc30 | 701 | 696 | 99.3% | 0.0 ms | 87.7 bpm | 87.1 bpm | 0.341 | 4.36 bpm | 6.25 bpm | 6.97 bpm |
| bidmc31 | 671 | 662 | 98.7% | 0.0 ms | 83.8 bpm | 82.8 bpm | 0.802 | 0.87 bpm | 1.09 bpm | 0.78 bpm |
| bidmc32 | 648 | 633 | 97.7% | 0.0 ms | 81.0 bpm | 79.2 bpm | 0.776 | 1.45 bpm | 2.57 bpm | 2.10 bpm |
| bidmc33 | 661 | 653 | 98.8% | 0.0 ms | 82.7 bpm | 81.8 bpm | 0.332 | 3.57 bpm | 6.52 bpm | 6.14 bpm |
| bidmc34 | 652 | 646 | 99.1% | 0.0 ms | 81.6 bpm | 80.8 bpm | 0.179 | 2.31 bpm | 2.22 bpm | 2.06 bpm |
| bidmc35 | 805 | 764 | 94.5% | 0.0 ms | 100.6 bpm | 95.6 bpm | 0.130 | 2.40 bpm | 3.06 bpm | 2.63 bpm |
| bidmc36 | 584 | 582 | 99.7% | 0.0 ms | 73.0 bpm | 72.9 bpm | 0.451 | 2.16 bpm | 8.10 bpm | 9.72 bpm |
| bidmc37 | 720 | 715 | 99.2% | 0.0 ms | 90.1 bpm | 89.4 bpm | 0.469 | 2.03 bpm | 2.46 bpm | 1.53 bpm |
| bidmc38 | 668 | 640 | 95.5% | 0.0 ms | 83.5 bpm | 80.1 bpm | 0.287 | 3.80 bpm | 4.12 bpm | 2.15 bpm |
| bidmc39 | 604 | 583 | 96.5% | 0.0 ms | 75.5 bpm | 73.0 bpm | -0.011 | 1.38 bpm | 0.79 bpm | 0.89 bpm |
| bidmc40 | 817 | 714 | 84.6% | 0.0 ms | 102.1 bpm | 89.3 bpm | -0.002 | 3.12 bpm | 3.36 bpm | 2.10 bpm |
| bidmc41 | 684 | 607 | 88.6% | 0.0 ms | 85.5 bpm | 75.9 bpm | 0.691 | 1.85 bpm | 3.06 bpm | 2.49 bpm |
| bidmc42 | 669 | 669 | 100.0% | 0.0 ms | 83.7 bpm | 83.7 bpm | 0.469 | 0.69 bpm | 0.71 bpm | 0.30 bpm |
| bidmc43 | 715 | 711 | 99.4% | 0.0 ms | 89.4 bpm | 89.0 bpm | 0.277 | 2.20 bpm | 6.51 bpm | 7.72 bpm |
| bidmc44 | 693 | 692 | 98.6% | 0.0 ms | 86.6 bpm | 86.6 bpm | -0.081 | 1.23 bpm | 1.07 bpm | 0.21 bpm |
| bidmc45 | 486 | 482 | 99.2% | 0.0 ms | 60.8 bpm | 60.3 bpm | 0.481 | 1.96 bpm | 2.44 bpm | 1.62 bpm |
| bidmc46 | 853 | 845 | 99.1% | 0.0 ms | 106.7 bpm | 105.7 bpm | -0.137 | 4.65 bpm | 6.28 bpm | 6.18 bpm |
| bidmc47 | 1287 | 660 | 51.3% | 0.0 ms | 160.9 bpm | 82.6 bpm | 0.449 | 1.66 bpm | 1.68 bpm | 2.15 bpm |
| bidmc48 | 803 | 790 | 98.3% | 0.0 ms | 100.5 bpm | 98.9 bpm | 0.529 | 2.58 bpm | 2.49 bpm | 2.82 bpm |
| bidmc49 | 855 | 663 | 77.2% | 0.0 ms | 107.0 bpm | 83.0 bpm | 0.247 | 1.57 bpm | 2.11 bpm | 1.16 bpm |
| bidmc50 | 611 | 603 | 98.5% | 0.0 ms | 76.4 bpm | 75.4 bpm | 0.582 | 0.89 bpm | 0.82 bpm | 0.51 bpm |
| bidmc51 | 720 | 720 | 100.0% | 0.0 ms | 90.0 bpm | 90.0 bpm | 0.237 | 0.90 bpm | 0.79 bpm | 0.38 bpm |
| bidmc52 | 921 | 913 | 99.1% | 0.0 ms | 115.1 bpm | 114.2 bpm | 0.247 | 0.77 bpm | 0.69 bpm | 0.60 bpm |
| bidmc53 | 734 | 710 | 96.7% | 0.0 ms | 91.7 bpm | 88.9 bpm | 0.347 | 2.99 bpm | 3.41 bpm | 1.73 bpm |

### 2. Auto-Computed Comparison Metrics
- Cohort size: 53 recordings evaluated
- Matched peak percentage: min=51.3%, mean=95.7%, max=100.0%
- Median timing offset |Δt|: min=0.0 ms, mean=0.0 ms, max=0.0 ms
- Mean HR difference: min=0.00 bpm, mean=4.34 bpm, max=78.31 bpm across evaluated subjects
- Spectral RR inter-method MAE: min=0.35 bpm, mean=1.83 bpm, max=4.65 bpm
- Spectral RR Pearson r: min=-0.137, mean=0.361, max=0.826
- Low-agreement cohort spread: 21/51 evaluated patients fall below Pearson r < 0.3 threshold (honest spread reporting).

---
*Verification Status*: Full-cohort validation run (53 patients) executed without modifying Phase 1/1.5 results.