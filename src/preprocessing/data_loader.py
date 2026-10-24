"""
Phase 1: Memory-Efficient Patient Biosignal Data Loader Module
=============================================================
Provides a generator to load, filter, and yield continuous patient records
one at a time, strictly honoring the 8GB RAM / local CPU constraint.
"""

import gc
from typing import Generator, Dict, Any, List, Optional
import numpy as np
import wfdb

from config.settings import DATA_DIR, PATIENT_IDS, FS, ENABLE_GARBAGE_COLLECTION
from src.preprocessing.filter import ButterworthRespiratoryFilter

def stream_patient_records(
    patient_list: Optional[List[str]] = None,
    filter_instance: Optional[ButterworthRespiratoryFilter] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator yielding preprocessed patient dictionary one recording at a time.
    Keeps memory footprint < 100 MB for an 8GB RAM environment.

    Yields:
        dict containing:
            - record_id: str (e.g. 'bidmc01')
            - raw_ppg: np.ndarray (float32 at 125 Hz)
            - filtered_ppg: np.ndarray (filtered float32 at 125 Hz)
            - gt_rr_1hz: np.ndarray (ground-truth RR in bpm at 1 Hz)
            - fs: float (125.0 Hz)
            - duration_sec: float
    """
    if patient_list is None:
        patient_list = PATIENT_IDS
    if filter_instance is None:
        filter_instance = ButterworthRespiratoryFilter(fs=FS)

    for pid in patient_list:
        rec_path = DATA_DIR / pid
        hea_file = DATA_DIR / f"{pid}.hea"
        dat_file = DATA_DIR / f"{pid}.dat"
        num_hea = DATA_DIR / f"{pid}n.hea"

        if not hea_file.exists() or not dat_file.exists():
            continue

        try:
            # 1. Read WFDB record
            record = wfdb.rdrecord(str(rec_path))
            clean_sig_names = [s.strip().replace(",", "").upper() for s in record.sig_name]

            # 2. Extract PLETH (PPG) channel
            pleth_indices = [idx for idx, name in enumerate(clean_sig_names) if "PLETH" in name]
            if not pleth_indices:
                continue
            pleth_channel = pleth_indices[0]
            raw_ppg = record.p_signal[:, pleth_channel].astype(np.float32)

            # 3. Apply Butterworth bandpass filter (0.1 Hz – 0.6 Hz)
            filtered_ppg = filter_instance.apply(raw_ppg)

            # 4. Load ground truth respiratory rate from numerics (1 Hz)
            gt_rr = None
            if num_hea.exists():
                num_record = wfdb.rdrecord(str(DATA_DIR / f"{pid}n"))
                num_names = [s.strip().replace(",", "").upper() for s in num_record.sig_name]
                resp_indices = [idx for idx, name in enumerate(num_names) if "RESP" in name]
                if resp_indices:
                    gt_rr = num_record.p_signal[:, resp_indices[0]].astype(np.float32)

            if gt_rr is None:
                continue

            yield {
                "record_id": pid,
                "raw_ppg": raw_ppg,
                "filtered_ppg": filtered_ppg,
                "gt_rr_1hz": gt_rr,
                "fs": FS,
                "duration_sec": len(raw_ppg) / FS
            }

            # Explicit memory reclamation for 8GB RAM
            del record, raw_ppg, filtered_ppg, gt_rr
            if ENABLE_GARBAGE_COLLECTION:
                gc.collect()

        except Exception as e:
            print(f"  ⚠ Error streaming record {pid}: {e}")
            continue
