# Legacy Pipeline Notes: 6-Second / 55-Feature Architecture

## Context & Architectural Pivot
During early pipeline development, the project explored 6-second window segmentation with a 55-dimensional feature set consisting of Discrete Wavelet Transform (DWT db4, level 4), fast Fourier transform spectral metrics, and time-domain statistical moments. However, systematic evaluation demonstrated that random window-level cross-validation induced patient identity leakage across partitions. 

The pipeline subsequently pivoted to:
1. **60-second window segmentation** (strictly patient-grouped 5-fold cross-validation with disjoint train, validation, calibration, and test partitions).
2. **28-dimensional (28-feature) physiologically grounded pulse-modulation feature set** (`extract_window_features_from_segments()` in `src/features/extractor.py`).

## Retained Legacy Files & Functions
To ensure backwards compatibility, reproducibility of initial explorations, and historical documentation, the legacy 55-feature components are retained with explicit docstring annotations:

1. **`src/features/wavelet_features.py`**:
   - `extract_dwt_features()`: Extracts 37 Daubechies-4 subband energy, statistical, and Shannon entropy metrics over 6s windows.
2. **`src/features/spectral_features.py`**:
   - `extract_spectral_features()`: Extracts 9 FFT spectral moments, peak powers, and spectral centroid over the 0.10–0.60 Hz band.
3. **`src/features/statistical_features.py`**:
   - `extract_statistical_features()`: Extracts 9 time-domain statistical descriptors (variance, std, peak-to-peak, zero crossings, etc.).
4. **`src/features/segmentation.py`**:
   - `segment_signal_into_windows()`: Sliding window segmentation initially drafted for 6s slices. Retained for reference; active window segmentation is handled via patient streaming and window-level feature generation.
5. **`extract_single_window_features()` in `src/features/extractor.py`**:
   - Combines DWT, spectral, and statistical feature extractors (37 + 9 + 9 = 55 features). Superseded by `extract_window_features_from_segments()`.
6. **`corrupt_single_ppg_segment()` in `src/conformal/artifact_simulator.py`**:
   - Synthesizes 6s waveform corruption. Superseded by `corrupt_raw_ppg_window()` supporting modular high-frequency noise, baseline wander, motion bursts, and sensor dropouts across mild, moderate, and severe clinical tiers.

## Active Pipeline References
- Active feature extractor: `extract_window_features_from_segments()` in `src/features/extractor.py` (28 features).
- Active artifact simulator: `corrupt_raw_ppg_window()` in `src/conformal/artifact_simulator.py`.
- Active orchestration: `run_phase2.py`, `run_phase4.py`, and `run_pipeline.py`.
