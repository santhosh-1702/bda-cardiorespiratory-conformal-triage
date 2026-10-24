"""
Phase 2: Statistical & Time-Domain Feature Extraction Module
============================================================
Extracts time-domain statistical descriptors and waveform morphology metrics
from segmented 6-second PPG windows.

NOTE: legacy — belongs to the original 6s/55-feature approach, superseded by the 28-feature pipeline in extractor.py; retained for historical reference.
"""

from typing import List
import numpy as np
import scipy.stats as stats

def extract_statistical_features(window_segment: np.ndarray) -> List[float]:
    """
    Computes time-domain statistical features from a 6-second window.

    Returns:
        List of 9 float values describing window statistical distribution.
    """
    variance = float(np.var(window_segment))
    std_dev = float(np.std(window_segment))
    peak_to_peak = float(np.ptp(window_segment))
    mean_abs_diff = float(np.mean(np.abs(np.diff(window_segment))))

    # Zero crossing rate (rate of sign transitions around baseline)
    zero_crossings = float(np.sum(np.diff(np.sign(window_segment)) != 0) / len(window_segment))

    skewness = float(stats.skew(window_segment))
    kurtosis_val = float(stats.kurtosis(window_segment))

    rms = float(np.sqrt(np.mean(window_segment ** 2))) + 1e-12
    crest_factor = float(np.max(np.abs(window_segment)) / rms)

    return [
        variance,
        std_dev,
        peak_to_peak,
        mean_abs_diff,
        zero_crossings,
        skewness,
        kurtosis_val,
        rms,
        crest_factor
    ]
