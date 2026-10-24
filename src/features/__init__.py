"""
Features package for sliding window segmentation and multi-domain feature extraction.
"""

from src.features.segmentation import segment_signal_into_windows
from src.features.wavelet_features import extract_dwt_features
from src.features.spectral_features import extract_spectral_features
from src.features.statistical_features import extract_statistical_features
from src.features.extractor import (
    extract_single_window_features,
    extract_window_features_from_segments,
    generate_patient_folds,
    build_feature_dataset,
    load_feature_dataset,
    get_patient_folds,
)

from src.features.modulations import (
    extract_pulse_modulations,
    estimate_dominant_frequency,
    estimate_spectral_ppg_rr,
    compute_spectrum_descriptors,
    compute_ppg_spectrum_descriptors,
)

__all__ = [
    "segment_signal_into_windows",
    "extract_dwt_features",
    "extract_spectral_features",
    "extract_statistical_features",
    "extract_single_window_features",
    "extract_window_features_from_segments",
    "generate_patient_folds",
    "build_feature_dataset",
    "load_feature_dataset",
    "get_patient_folds",
    "extract_pulse_modulations",
    "estimate_dominant_frequency",
    "estimate_spectral_ppg_rr",
    "compute_spectrum_descriptors",
    "compute_ppg_spectrum_descriptors",
]
