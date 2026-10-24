"""
External Validation Module
==========================
Provides loaders and preprocessing utilities for external benchmark datasets,
such as the CapnoBase IEEE TBME Respiratory Rate Benchmark.
"""

from src.external_validation.capnobase_loader import (
    locate_capnobase_dir,
    get_capnobase_case_ids,
    load_capnobase_record,
    stream_capnobase_records,
    build_capnobase_dataset,
    load_capnobase_dataset,
    resample_signal_125hz,
    parse_capnobase_subject_metadata,
    get_capnobase_age_stratification,
)

__all__ = [
    "locate_capnobase_dir",
    "get_capnobase_case_ids",
    "load_capnobase_record",
    "stream_capnobase_records",
    "build_capnobase_dataset",
    "load_capnobase_dataset",
    "resample_signal_125hz",
    "parse_capnobase_subject_metadata",
    "get_capnobase_age_stratification",
]
