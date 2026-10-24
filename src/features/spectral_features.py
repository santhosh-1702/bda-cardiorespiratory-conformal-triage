"""
Phase 2: Frequency-Domain / Spectral Feature Extraction Module
=============================================================
Computes discrete Fourier transform (FFT) features over the respiratory
frequency spectrum (0.1 Hz – 0.6 Hz = 6 – 36 breaths/min).

NOTE: legacy — belongs to the original 6s/55-feature approach, superseded by the 28-feature pipeline in extractor.py; retained for historical reference.
"""

from typing import List
import numpy as np

from config.settings import FS, FILTER_LOW_HZ, FILTER_HIGH_HZ

def extract_spectral_features(window_segment: np.ndarray, fs: float = FS) -> List[float]:
    """
    Computes spectral energy, peak frequencies, and spectral moments.

    Returns:
        List of 9 float values representing frequency-domain metrics.
    """
    n = len(window_segment)
    fft_vals = np.abs(np.fft.rfft(window_segment))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power_spectrum = fft_vals ** 2
    total_power = float(np.sum(power_spectrum)) + 1e-12

    # Respiratory band mask [0.10 Hz – 0.60 Hz]
    resp_mask = (freqs >= FILTER_LOW_HZ) & (freqs <= FILTER_HIGH_HZ)
    resp_power = float(np.sum(power_spectrum[resp_mask]))
    resp_ratio = float(resp_power / total_power)

    # Peak frequency within respiratory band
    if np.any(resp_mask) and np.max(fft_vals[resp_mask]) > 0:
        masked_fft = fft_vals.copy()
        masked_fft[~resp_mask] = 0
        peak_idx = int(np.argmax(masked_fft))
        peak_freq_hz = float(freqs[peak_idx])
        peak_freq_bpm = peak_freq_hz * 60.0   # Breaths per minute
        peak_amplitude = float(fft_vals[peak_idx])
    else:
        peak_freq_hz = 0.25
        peak_freq_bpm = 15.0
        peak_amplitude = 0.0

    # Spectral centroid (center of gravity of spectrum)
    centroid = float(np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-12))

    # Spectral bandwidth (spread around centroid)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_vals) / (np.sum(fft_vals) + 1e-12)))

    # Spectral roll-off (frequency below which 85% of power resides)
    cumulative_power = np.cumsum(power_spectrum)
    roll_idx = int(np.searchsorted(cumulative_power, 0.85 * total_power))
    roll_freq = float(freqs[min(roll_idx, len(freqs) - 1)])

    return [
        total_power,
        resp_power,
        resp_ratio,
        peak_freq_hz,
        peak_freq_bpm,
        peak_amplitude,
        centroid,
        bandwidth,
        roll_freq
    ]
