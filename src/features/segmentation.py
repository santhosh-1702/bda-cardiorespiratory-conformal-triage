"""
Phase 2: Sliding Window Signal Segmentation Module
==================================================
Segments continuous 125 Hz filtered PPG signals into 6-second sliding windows
with overlapping strides to capture time-varying cardiorespiratory dynamics.

NOTE: legacy — belongs to the original 6s/55-feature approach, superseded by the 28-feature pipeline in extractor.py; retained for historical reference.
"""

from typing import List, Tuple, Optional, Union
import numpy as np

from config.settings import FS, WINDOW_SAMPLES, STRIDE_SAMPLES

def segment_signal_into_windows(
    signal_array: np.ndarray,
    ground_truth_rr: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    stride_samples: int = STRIDE_SAMPLES,
    fs: float = FS,
    patient_id: Optional[str] = None,
) -> Union[List[Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float, str]]]:
    """
    Slices a continuous filtered PPG signal and maps each window to its
    mean ground-truth respiratory rate (bpm), optionally carrying patient_id.

    Returns:
        If patient_id is None: List of (6s window array, mean ground truth RR)
        If patient_id is provided: List of (6s window array, mean ground truth RR, patient_id)
    """
    n_samples = len(signal_array)
    if n_samples < window_samples:
        return []

    windows = []
    n_steps = (n_samples - window_samples) // stride_samples + 1

    for step in range(n_steps):
        start_idx = step * stride_samples
        end_idx = start_idx + window_samples
        window_slice = signal_array[start_idx:end_idx]

        # Map to corresponding seconds in 1 Hz ground truth vector
        start_sec = int(start_idx / fs)
        end_sec = int(end_idx / fs)
        rr_slice = ground_truth_rr[start_sec:end_sec]

        if len(rr_slice) == 0 or np.all(np.isnan(rr_slice)):
            continue

        mean_rr = float(np.nanmean(rr_slice))
        # Validate that mean RR is physiologically sensible
        if np.isnan(mean_rr) or mean_rr < 6.0 or mean_rr > 45.0:
            continue

        if patient_id is not None:
            windows.append((window_slice, mean_rr, patient_id))
        else:
            windows.append((window_slice, mean_rr))

    return windows
