"""
Phase 1: Butterworth Respiratory Filter Module
==============================================
Applies a zero-phase 4th-order Butterworth bandpass filter strictly between
0.1 Hz (6 bpm) and 0.6 Hz (36 bpm) to isolate respiratory modulations from raw PPG.
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple

from config.settings import FS, FILTER_ORDER, FILTER_LOW_HZ, FILTER_HIGH_HZ

class ButterworthRespiratoryFilter:
    """
    Zero-phase digital Butterworth filter in Second-Order Sections (SOS) representation.
    Guarantees numerical stability and prevents phase distortion in sliding windows.
    """

    def __init__(self, fs: float = FS, low: float = FILTER_LOW_HZ, high: float = FILTER_HIGH_HZ, order: int = FILTER_ORDER):
        self.fs = fs
        self.low = low
        self.high = high
        self.order = order
        # Second-order sections representation
        self.sos = signal.butter(
            N=order,
            Wn=[low, high],
            btype="bandpass",
            fs=fs,
            output="sos"
        )

    def apply(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase forward-backward filtering using SOS coefficients.
        Eliminates baseline wander (DC offset) and suppresses high-frequency cardiac pulses.
        """
        # Ensure 64-bit float during filtering to prevent numerical precision loss, return float32
        filtered = signal.sosfiltfilt(self.sos, signal_data.astype(np.float64))
        return filtered.astype(np.float32)

    def get_frequency_response(self, n_points: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """Returns frequency vector (Hz) and complex frequency response array."""
        w, h = signal.sosfreqz(self.sos, worN=n_points, fs=self.fs)
        return w, h

class ButterworthWidebandFilter(ButterworthRespiratoryFilter):
    """
    Zero-phase digital Butterworth filter in Second-Order Sections (SOS) representation
    spanning 0.10 Hz (6 bpm) to 5.00 Hz (300 bpm). Preserves both cardiac pulses and
    respiratory modulations for extracting RIAV, RIIV, and RIFV.
    """

    def __init__(self, fs: float = FS, low: float = 0.10, high: float = 5.00, order: int = FILTER_ORDER):
        super().__init__(fs=fs, low=low, high=high, order=order)

