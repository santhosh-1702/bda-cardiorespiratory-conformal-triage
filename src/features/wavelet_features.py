"""
Phase 2: Discrete Wavelet Transform (DWT) Feature Extraction Module
===================================================================
Applies Daubechies-4 (db4) 4-level wavelet decomposition to extract subband
energy, coefficient statistics, and Shannon entropy for respiratory signal triage.

NOTE: legacy — belongs to the original 6s/55-feature approach, superseded by the 28-feature pipeline in extractor.py; retained for historical reference.
"""

from typing import List
import numpy as np
import pywt

from config.settings import DWT_WAVELET, DWT_LEVEL

def extract_dwt_features(window_segment: np.ndarray) -> List[float]:
    """
    Computes DWT features over a 6-second window.
    Decomposition: [cA4, cD4, cD3, cD2, cD1]

    Returns:
        List of 37 float values representing wavelet subband characteristics.
    """
    coeffs = pywt.wavedec(window_segment, DWT_WAVELET, level=DWT_LEVEL)
    # coeffs = [cA4, cD4, cD3, cD2, cD1]
    total_energy = sum(np.sum(c ** 2) for c in coeffs) + 1e-12

    dwt_feats = []

    for c in coeffs:
        c_energy = float(np.sum(c ** 2))
        dwt_feats.append(float(np.mean(c)))
        dwt_feats.append(float(np.std(c)))
        dwt_feats.append(float(np.max(np.abs(c))))
        dwt_feats.append(c_energy)
        dwt_feats.append(float(c_energy / total_energy))  # Relative subband energy
        dwt_feats.append(float(np.median(c)))

        # Normalized Shannon entropy of squared coefficients
        p = (c ** 2) / (c_energy + 1e-12)
        entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        dwt_feats.append(entropy)

    # Subband energy ratios (low-frequency respiratory band cA4 vs details)
    cA4_energy = float(np.sum(coeffs[0] ** 2))
    cD4_energy = float(np.sum(coeffs[1] ** 2))
    dwt_feats.append(float(cA4_energy / total_energy))
    dwt_feats.append(float(cD4_energy / total_energy))

    return dwt_feats
