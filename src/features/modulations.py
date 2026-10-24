"""
Phase 1.5: Respiratory-Induced Variations & Modulation Extraction Module
========================================================================
Extracts respiratory-induced variations from wide-band filtered PPG (0.10 - 5.00 Hz):
  - RIAV: Respiratory-Induced Amplitude Variation (peak-to-peak amplitude series)
  - RIIV: Respiratory-Induced Intensity Variation (baseline wander series)
  - RIFV: Respiratory-Induced Frequency Variation (pulse-interval series)
Provides classical spectral RR estimation and multi-domain modulation feature extraction.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import scipy.signal as signal

from config.settings import FS, FILTER_LOW_HZ, FILTER_HIGH_HZ

def extract_pulse_modulations(
    filt_wide_ppg: np.ndarray,
    fs: float = FS
) -> Dict[str, np.ndarray]:
    """
    Detects systolic peaks and diastolic troughs from 0.1 - 5.0 Hz filtered PPG,
    and extracts RIAV, RIIV, and RIFV series alongside their timestamps.

    Returns:
        dict containing:
            - 't_peaks': timestamps of peaks (in seconds)
            - 'riav': peak-to-peak amplitude series
            - 'riiv': baseline variation series (trough amplitudes)
            - 'rifv': pulse-interval series (in seconds)
            - 'peaks': peak sample indices
            - 'troughs': trough sample indices
    """
    if len(filt_wide_ppg) == 0:
        return {
            "t_peaks": np.array([]), "riav": np.array([]),
            "riiv": np.array([]), "rifv": np.array([]),
            "peaks": np.array([]), "troughs": np.array([])
        }

    # Cardiac pulse peaks detection (minimum distance ~ 0.32s = max ~185 bpm)
    prom = max(1e-4, float(np.std(filt_wide_ppg) * 0.15))
    peaks, _ = signal.find_peaks(filt_wide_ppg, distance=int(fs * 0.32), prominence=prom)

    if len(peaks) < 3:
        return {
            "t_peaks": np.array([]), "riav": np.array([]),
            "riiv": np.array([]), "rifv": np.array([]),
            "peaks": peaks, "troughs": np.array([])
        }

    # Diastolic troughs between consecutive peaks
    troughs = np.array([
        peaks[i] + int(np.argmin(filt_wide_ppg[peaks[i]:peaks[i + 1]]))
        for i in range(len(peaks) - 1)
    ])

    t_peaks = peaks[:-1] / fs
    riav = (filt_wide_ppg[peaks[:-1]] - filt_wide_ppg[troughs]).astype(np.float32)
    riiv = filt_wide_ppg[troughs].astype(np.float32)
    rifv = ((peaks[1:] - peaks[:-1]) / fs).astype(np.float32)

    return {
        "t_peaks": t_peaks,
        "riav": riav,
        "riiv": riiv,
        "rifv": rifv,
        "peaks": peaks,
        "troughs": troughs
    }

def estimate_dominant_frequency(
    t_series: np.ndarray,
    val_series: np.ndarray,
    t_start: float,
    t_end: float,
    fs_resamp: float = 4.0,
    f_low: float = FILTER_LOW_HZ,
    f_high: float = FILTER_HIGH_HZ,
    nfft: int = 2048,
    default_rr: float = 17.65
) -> float:
    """
    Estimates dominant respiratory rate (bpm) from an irregularly-sampled modulation series
    within [t_start, t_end] via uniform interpolation and high-resolution FFT.
    """
    if len(t_series) == 0:
        return default_rr

    mask = (t_series >= t_start) & (t_series < t_end)
    t_w = t_series[mask]
    v_w = val_series[mask]

    if len(t_w) < 4:
        return default_rr

    dur = float(t_w[-1] - t_w[0])
    n_pts = int(round(dur * fs_resamp))
    if n_pts < 4:
        return default_rr

    t_u = np.linspace(t_w[0], t_w[-1], n_pts)
    v_u = np.interp(t_u, t_w, v_w)
    v_u = v_u - np.mean(v_u)

    fft_vals = np.abs(np.fft.rfft(v_u * np.hanning(len(v_u)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs_resamp)

    resp_mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(resp_mask):
        return default_rr

    peak_idx = int(np.argmax(fft_vals[resp_mask]))
    dominant_hz = float(freqs[resp_mask][peak_idx])
    return dominant_hz * 60.0

def estimate_spectral_ppg_rr(
    window_segment: np.ndarray,
    fs: float = FS,
    f_low: float = FILTER_LOW_HZ,
    f_high: float = FILTER_HIGH_HZ,
    nfft: int = 4096,
    default_rr: float = 17.65
) -> float:
    """
    Classical spectral baseline (no ML):
    Estimates RR as the dominant peak frequency of the respiratory-band PPG spectrum.
    """
    n = len(window_segment)
    if n < 4:
        return default_rr

    seg = window_segment - np.mean(window_segment)
    fft_vals = np.abs(np.fft.rfft(seg * np.hanning(n), n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    resp_mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(resp_mask):
        return default_rr

    peak_idx = int(np.argmax(fft_vals[resp_mask]))
    dominant_hz = float(freqs[resp_mask][peak_idx])
    return dominant_hz * 60.0

def compute_spectrum_descriptors(
    t_series: np.ndarray,
    val_series: np.ndarray,
    t_start: float,
    t_end: float,
    fs_resamp: float = 4.0,
    f_low: float = FILTER_LOW_HZ,
    f_high: float = FILTER_HIGH_HZ,
    nfft: int = 2048,
    default_rr: float = 17.65
) -> List[float]:
    """
    Extracts 4 spectral descriptors from a modulation series:
    [peak_bpm, resp_power_ratio, resp_power, spectral_centroid_bpm]
    """
    if len(t_series) == 0:
        return [default_rr, 0.0, 0.0, default_rr]

    mask = (t_series >= t_start) & (t_series < t_end)
    t_w = t_series[mask]
    v_w = val_series[mask]

    if len(t_w) < 4:
        return [default_rr, 0.0, 0.0, default_rr]

    dur = float(t_w[-1] - t_w[0])
    n_pts = int(round(dur * fs_resamp))
    if n_pts < 4:
        return [default_rr, 0.0, 0.0, default_rr]

    t_u = np.linspace(t_w[0], t_w[-1], n_pts)
    v_u = np.interp(t_u, t_w, v_w)
    v_u = v_u - np.mean(v_u)

    fft_vals = np.abs(np.fft.rfft(v_u * np.hanning(len(v_u)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs_resamp)

    resp_mask = (freqs >= f_low) & (freqs <= f_high)
    total_power = float(np.sum(fft_vals**2)) + 1e-12
    if not np.any(resp_mask):
        return [default_rr, 0.0, 0.0, default_rr]

    resp_power = float(np.sum(fft_vals[resp_mask]**2))
    peak_idx = int(np.argmax(fft_vals[resp_mask]))
    peak_bpm = float(freqs[resp_mask][peak_idx] * 60.0)
    resp_ratio = float(resp_power / total_power)
    centroid_bpm = float(
        np.sum(freqs[resp_mask] * fft_vals[resp_mask]) / (np.sum(fft_vals[resp_mask]) + 1e-12)
    ) * 60.0

    return [peak_bpm, resp_ratio, resp_power, centroid_bpm]

def compute_ppg_spectrum_descriptors(
    window_segment: np.ndarray,
    fs: float = FS,
    f_low: float = FILTER_LOW_HZ,
    f_high: float = FILTER_HIGH_HZ,
    nfft: int = 4096,
    default_rr: float = 17.65
) -> List[float]:
    """
    Extracts 4 spectral descriptors from the respiratory-band PPG window:
    [peak_bpm, resp_power_ratio, resp_power, spectral_centroid_bpm]
    """
    n = len(window_segment)
    if n < 4:
        return [default_rr, 0.0, 0.0, default_rr]

    seg = window_segment - np.mean(window_segment)
    fft_vals = np.abs(np.fft.rfft(seg * np.hanning(n), n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    resp_mask = (freqs >= f_low) & (freqs <= f_high)
    total_power = float(np.sum(fft_vals**2)) + 1e-12
    if not np.any(resp_mask):
        return [default_rr, 0.0, 0.0, default_rr]

    resp_power = float(np.sum(fft_vals[resp_mask]**2))
    peak_idx = int(np.argmax(fft_vals[resp_mask]))
    peak_bpm = float(freqs[resp_mask][peak_idx] * 60.0)
    resp_ratio = float(resp_power / total_power)
    centroid_bpm = float(
        np.sum(freqs[resp_mask] * fft_vals[resp_mask]) / (np.sum(fft_vals[resp_mask]) + 1e-12)
    ) * 60.0

    return [peak_bpm, resp_ratio, resp_power, centroid_bpm]
