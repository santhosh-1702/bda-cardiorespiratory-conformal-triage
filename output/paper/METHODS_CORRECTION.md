# Corrected Methods Draft: 28-Dimensional Pulse-Modulation Feature Pipeline

## Abstract / Methods Replacement Text

> **Proposed Replacement Sentences (for Abstract and Methodology Sections):**
> 
> "From each 60-second photoplethysmogram (PPG) window, we extracted a 28-dimensional physiologically grounded feature representation comprising 4 PPG spectral descriptors (dominant frequency, spectral centroid, spectral bandwidth, spectral roll-off), 12 spectral descriptors across three respiratory-induced pulse modulations (4 descriptors each for the RIAV, RIIV, and RIFV modulation series), and 12 cardiorespiratory morphology features. These morphology features include a multi-modulation median fusion respiratory rate, the mean and standard deviation of each modulation series (6 features), pulse-derived mean heart rate, signal std/peak-to-peak/RMS descriptors, and within-window spectral-peak sharpness over the 0.10–0.60 Hz physiological band (28 features total). This formulation captures respiration-induced hemodynamic modulations while remaining robust to patient identity leakage under strict patient-disjoint cross-validation."

---

## Detailed Feature Decomposition (28 Dimensions Total)

1. **PPG Spectral Descriptors (4 features):**
   - Dominant frequency (bpm) in respiratory band [0.10–0.60 Hz]
   - Spectral centroid (bpm)
   - Spectral bandwidth / in-band power ratio
   - Spectral roll-off / in-band power
2. **Pulse-Modulation Spectral Descriptors (12 features = 3 series × 4 descriptors):**
   - **RIAV (Respiratory-Induced Amplitude Variation)**: Dominant freq, centroid, bandwidth, roll-off (4)
   - **RIIV (Respiratory-Induced Intensity Variation)**: Dominant freq, centroid, bandwidth, roll-off (4)
   - **RIFV (Respiratory-Induced Frequency Variation)**: Dominant freq, centroid, bandwidth, roll-off (4)
3. **Cardiorespiratory / Pulse-Modulation Statistics (8 features):**
   - `fusion_rr`: Multi-modulation median frequency estimate (RIAV, RIIV, RIFV)
   - `riav_mean`, `riav_std`: Statistical moments of the RIAV envelope
   - `riiv_mean`, `riiv_std`: Statistical moments of baseline intensity wander
   - `rifv_mean`, `rifv_std`: Statistical moments of pulse-interval durations
   - `mean_hr`: Mean heart rate derived from reciprocal median pulse interval (60 / RIFV_mean)
4. **Waveform Morphology & Spectral Sharpness Descriptors (4 features):**
   - `sig_std`: Respiratory-filtered window standard deviation
   - `sig_ptp`: Window peak-to-peak amplitude
   - `sig_rms`: Window root-mean-square amplitude
   - `sharpness`: Spectral peak power divided by mean band power in [0.10, 0.60 Hz]

*(Note: Groups 3 and 4 together constitute the 12 cardiorespiratory & waveform morphology features described above).*

**Total Dimensionality:** 4 + 12 + 8 + 4 = **28 features**.
