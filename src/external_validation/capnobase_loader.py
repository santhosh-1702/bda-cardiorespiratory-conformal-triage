"""
CapnoBase Benchmark Dataset External Validation Loader
======================================================
Parses, resamples, filters, and window-extracts the CapnoBase IEEE TBME
Respiratory Rate Benchmark dataset (42 cases, 8-minute recordings, 300 Hz PPG
and CO2 capnography reference) into the standardized format expected by the
cardiorespiratory triage pipeline.

Key Design Principles:
  1. Genuinely independent external test set: zero overlap with BIDMC cohort.
  2. Preserves frozen BIDMC pipeline contracts:
     - Detects and handles native 300 Hz sampling rate, resampling to 125 Hz.
     - Reuses existing Butterworth filters (src.preprocessing.filter).
     - Calls unmodified extract_window_features_from_segments() (src.features.extractor)
       producing standard 28-dimensional multi-domain feature vectors.
     - Produces ground-truth RR at 1 Hz matching BIDMC's gt_rr_1hz structure.
  3. Caches features and labels to output/cache/capnobase_features.npz for
     subsequent out-of-distribution evaluation tasks.
  4. Respects Intel i3 / 8GB RAM CPU-only constraints via streaming generator
     and explicit garbage collection.

Reference:
  Karlen, W., Raman, S., Ansermino, J.M., Dumont, G.A. (2013).
  "Multiparameter respiratory rate estimation from the photoplethysmogram."
  IEEE Transactions on Biomedical Engineering, 60(7), 1946–1953.
  DOI: 10.1109/TBME.2013.2246160
"""

import gc
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.interpolate import interp1d

from config.settings import (
    FS,
    WINDOW_SEC,
    STRIDE_SEC,
    CACHE_DIR,
    CAPNOBASE_DIR,
    CAPNOBASE_CACHE_FILE,
    CAPNOBASE_TOTAL_CASES,
    ENABLE_GARBAGE_COLLECTION,
    PROJECT_ROOT,
)
from src.features.extractor import (
    extract_window_features_from_segments,
    compute_spectral_peak_sharpness,
)
from src.preprocessing.filter import (
    ButterworthRespiratoryFilter,
    ButterworthWidebandFilter,
)


def locate_capnobase_dir(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Locates the directory containing CapnoBase CSV data files.
    Searches custom_path, environment variable CAPNOBASE_DIR, configured CAPNOBASE_DIR,
    and standard external or gitignored fallback locations.

    Returns:
        Path to the folder containing the dataset CSV files (or the csv/ subfolder).

    Raises:
        FileNotFoundError: If no valid CapnoBase CSV files can be located.
    """
    candidates = []
    if custom_path:
        p = Path(custom_path)
        candidates.extend([p, p / "csv", p / "data" / "csv", p / "data"])

    env_dir = os.environ.get("CAPNOBASE_DIR")
    if env_dir:
        p_env = Path(env_dir)
        candidates.extend([p_env, p_env / "csv", p_env / "data" / "csv", p_env / "data"])

    candidates.extend([
        CAPNOBASE_DIR,
        CAPNOBASE_DIR / "csv",
        CAPNOBASE_DIR / "data" / "csv",
        CAPNOBASE_DIR / "data",
        PROJECT_ROOT.parent / "capnobase_data" / "data" / "csv",
        PROJECT_ROOT.parent / "capnobase_data" / "csv",
        PROJECT_ROOT.parent / "capnobase_data" / "data",
        PROJECT_ROOT.parent / "capnobase_data",
        PROJECT_ROOT / "data" / "capnobase" / "data" / "csv",
        PROJECT_ROOT / "data" / "capnobase" / "csv",
        PROJECT_ROOT / "data" / "capnobase" / "data",
        PROJECT_ROOT / "data" / "capnobase",
        PROJECT_ROOT / "capnobase_data" / "data" / "csv",
        PROJECT_ROOT / "capnobase_data" / "csv",
        PROJECT_ROOT / "capnobase_data",
    ])

    for c in candidates:
        if c.exists() and c.is_dir():
            # Check if this folder or its csv/ subfolder has *_signal.csv or *_param.csv
            if list(c.glob("*_signal.csv")) or list(c.glob("*_param.csv")):
                return c.resolve()
            sub_csv = c / "csv"
            if sub_csv.exists() and (list(sub_csv.glob("*_signal.csv")) or list(sub_csv.glob("*_param.csv"))):
                return sub_csv.resolve()
            sub_data_csv = c / "data" / "csv"
            if sub_data_csv.exists() and (list(sub_data_csv.glob("*_signal.csv")) or list(sub_data_csv.glob("*_param.csv"))):
                return sub_data_csv.resolve()

    raise FileNotFoundError(
        "Could not locate CapnoBase dataset files. "
        "Per CapnoBase terms of use, raw data cannot be redistributed with this repository.\n"
        "Please download the dataset from Borealis Dataverse (https://doi.org/10.5683/SP2/NLB8IT) "
        "and place it at '../capnobase_data' or set the CAPNOBASE_DIR environment variable.\n"
        f"Checked candidates:\n" +
        "\n".join(f"  - {str(c)}" for c in candidates)
    )


def get_capnobase_case_ids(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Returns sorted list of unique CapnoBase case identifiers (e.g. ['0009', '0015', ...]).
    """
    resolved_dir = locate_capnobase_dir(data_dir)
    param_files = list(resolved_dir.glob("*_param.csv"))
    if not param_files:
        param_files = list(resolved_dir.glob("*_signal.csv"))

    case_ids = set()
    for f in param_files:
        # Expected naming: <caseID>_8min_param.csv or <caseID>_param.csv
        parts = f.name.split("_")
        if parts:
            case_ids.add(parts[0])

    return sorted(list(case_ids))


def parse_capnobase_subject_metadata(
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parses subject_age and subject_weight from each case's _meta.csv file.

    CapnoBase's stated population split (from Karlen et al. 2013 and dataset README):
      - Pediatric: age < 18 years
      - Adult: age >= 18 years

    Returns:
        Dict mapping case_id (str) -> {"age": float or NaN, "weight": float or NaN,
                                        "is_pediatric": bool or None (if age is NaN)}
    """
    resolved_dir = locate_capnobase_dir(data_dir)
    case_ids = get_capnobase_case_ids(data_dir)
    metadata: Dict[str, Dict[str, Any]] = {}

    for case_id in case_ids:
        base_prefix = f"{case_id}_8min"
        meta_path = resolved_dir / f"{base_prefix}_meta.csv"
        if not meta_path.exists():
            meta_path = resolved_dir / f"{case_id}_meta.csv"

        age = float("nan")
        weight = float("nan")

        if meta_path.exists():
            try:
                df_m = pd.read_csv(meta_path)
                if "subject_age" in df_m.columns:
                    raw_age = df_m["subject_age"].iloc[0]
                    try:
                        age = float(raw_age)
                    except (ValueError, TypeError):
                        age = float("nan")
                if "subject_weight" in df_m.columns:
                    raw_wt = df_m["subject_weight"].iloc[0]
                    try:
                        weight = float(raw_wt)
                    except (ValueError, TypeError):
                        weight = float("nan")
            except Exception:
                pass

        # Parse ventilation mode from _param.csv
        param_path = resolved_dir / f"{base_prefix}_param.csv"
        if not param_path.exists():
            param_path = resolved_dir / f"{case_id}_param.csv"

        case_vent = "unknown"
        if param_path.exists():
            try:
                df_param = pd.read_csv(param_path)
                if "case_ventilation" in df_param.columns:
                    case_vent = str(df_param["case_ventilation"].iloc[0]).strip().lower()
            except Exception:
                pass

        # CapnoBase adult/pediatric cutoff: age < 18 = pediatric, age >= 18 = adult
        # In Karlen et al. 2013 and CapnoBase TBME documentation, the dataset consists of 29 pediatric and 13 adult cases.
        # Case 0032 has missing age but weight = 26 kg, belonging to the pediatric surgical cohort (BC Children's Hospital).
        if np.isnan(age):
            if not np.isnan(weight) and weight < 40.0:
                is_pediatric = True
            else:
                is_pediatric = None
        else:
            is_pediatric = bool(age < 18)

        metadata[case_id] = {
            "age": age,
            "weight": weight,
            "is_pediatric": is_pediatric,
            "case_ventilation": case_vent,
            "is_ventilated": bool(case_vent == "controlled"),
            "is_spontaneous": bool(case_vent == "spontaneous"),
        }

    # Ensure keys exist both with and without 'capno_' prefix for convenience
    full_metadata = dict(metadata)
    for k, v in metadata.items():
        if not k.startswith("capno_"):
            full_metadata[f"capno_{k}"] = v

    return full_metadata


def get_capnobase_age_stratification(
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """
    Returns a dict with three case-ID lists stratified by age group.

    CapnoBase cutoff (Karlen et al. 2013 dataset documentation):
      - Pediatric: subject_age < 18 years (or weight < 40 kg for case 0032; exactly 29 cases)
      - Adult:     subject_age >= 18 years (exactly 13 cases)
      - Unknown:   subject_age is NaN and weight unknown

    Returns:
        {
          "all":        [sorted list of all 42 case IDs],
          "adult":      [case IDs with age >= 18],
          "pediatric":  [case IDs with age < 18 or pediatric weight],
          "unknown":    [case IDs with missing age/weight],
        }
    """
    subject_meta = parse_capnobase_subject_metadata(data_dir)
    # Use non-prefixed case IDs for the stratification lists
    all_ids = sorted([cid for cid in subject_meta.keys() if not cid.startswith("capno_")])

    adult_ids = [
        cid for cid in all_ids
        if subject_meta[cid]["is_pediatric"] is False
    ]
    pediatric_ids = [
        cid for cid in all_ids
        if subject_meta[cid]["is_pediatric"] is True
    ]
    unknown_ids = [
        cid for cid in all_ids
        if subject_meta[cid]["is_pediatric"] is None
    ]

    return {
        "all": all_ids,
        "adult": adult_ids,
        "pediatric": pediatric_ids,
        "unknown": unknown_ids,
    }


def get_capnobase_ventilation_stratification(
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """
    Returns case ID lists stratified by ventilation mode and adult/pediatric status.

    Ventilation categories:
      - 'controlled': mechanical positive-pressure ventilation (19 cases total, 8 adult, 11 pediatric)
      - 'spontaneous': natural spontaneous breathing (23 cases total, 5 adult, 18 pediatric)

    Returns:
        {
          "all":                   [all 42 case IDs],
          "spontaneous":           [23 spontaneous cases],
          "controlled":            [19 mechanically ventilated cases],
          "adult_spontaneous":     [5 adult spontaneous cases],
          "adult_ventilated":      [8 adult mechanically ventilated cases],
          "pediatric_spontaneous": [18 pediatric spontaneous cases],
          "pediatric_ventilated":  [11 pediatric mechanically ventilated cases],
        }
    """
    subject_meta = parse_capnobase_subject_metadata(data_dir)
    all_ids = sorted([cid for cid in subject_meta.keys() if not cid.startswith("capno_")])

    spont_ids = [cid for cid in all_ids if subject_meta[cid].get("case_ventilation") == "spontaneous"]
    vent_ids = [cid for cid in all_ids if subject_meta[cid].get("case_ventilation") == "controlled"]

    adult_spont_ids = [cid for cid in spont_ids if subject_meta[cid]["is_pediatric"] is False]
    adult_vent_ids = [cid for cid in vent_ids if subject_meta[cid]["is_pediatric"] is False]

    ped_spont_ids = [cid for cid in spont_ids if subject_meta[cid]["is_pediatric"] is True]
    ped_vent_ids = [cid for cid in vent_ids if subject_meta[cid]["is_pediatric"] is True]

    return {
        "all": all_ids,
        "spontaneous": spont_ids,
        "controlled": vent_ids,
        "adult_spontaneous": adult_spont_ids,
        "adult_ventilated": adult_vent_ids,
        "pediatric_spontaneous": ped_spont_ids,
        "pediatric_ventilated": ped_vent_ids,
    }


def resample_signal_125hz(
    sig: np.ndarray,
    orig_fs: float,
    target_fs: float = FS
) -> np.ndarray:
    """
    Resamples a 1D continuous biosignal from orig_fs to target_fs (125.0 Hz).
    Uses polyphase filtering via scipy.signal.resample_poly with rational fraction reduction
    to prevent phase distortion and frequency aliasing.
    """
    if abs(orig_fs - target_fs) < 1e-3:
        return sig.astype(np.float32)

    frac = Fraction(int(round(target_fs * 100)), int(round(orig_fs * 100))).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    resampled = signal.resample_poly(sig.astype(np.float64), up=up, down=down)
    return resampled.astype(np.float32)


def parse_capnobase_reference_rr(
    ref_csv_path: Path,
    labels_csv_path: Optional[Path] = None,
    duration_sec: float = 480.0,
    orig_fs: float = 300.0,
) -> np.ndarray:
    """
    Extracts the capnography-derived reference respiratory rate and maps it onto a
    uniform 1 Hz grid (gt_rr_1hz) spanning duration_sec seconds, mirroring BIDMC's gt_rr_1hz.

    Steps:
      1. Reads instantaneous breath-to-breath reference values (rr_co2_x in seconds,
         rr_co2_y in breaths/minute) from reference.csv.
      2. Cleans non-finite values (inf, NaN) and deduplicates identical timestamps.
      3. Linearly interpolates onto 1-second time points np.arange(int(duration_sec)).
      4. Sets regions outside breath observation boundaries to np.nan.
      5. Parses co2_artif_x in labels.csv and masks expert-annotated artifact intervals as np.nan.

    Returns:
        np.ndarray of shape (int(duration_sec),) float32 containing reference RR in bpm at 1 Hz.
    """
    gt_rr_1hz = np.full(int(duration_sec), np.nan, dtype=np.float32)
    if not ref_csv_path.exists():
        return gt_rr_1hz

    df_ref = pd.read_csv(ref_csv_path)
    if "rr_co2_x" not in df_ref.columns or "rr_co2_y" not in df_ref.columns:
        return gt_rr_1hz

    raw_x = [float(v) for v in str(df_ref["rr_co2_x"].iloc[0]).split() if v]
    raw_y = [float(v) for v in str(df_ref["rr_co2_y"].iloc[0]).split() if v]

    # Filter physiologically plausible finite reference points
    pts = [
        (x, y) for x, y in zip(raw_x, raw_y)
        if np.isfinite(x) and np.isfinite(y) and 4.0 <= y <= 60.0
    ]
    if len(pts) < 2:
        return gt_rr_1hz

    pts.sort(key=lambda p: p[0])
    # Deduplicate timestamps
    unique_x = []
    unique_y = []
    for x, y in pts:
        if not unique_x or x > unique_x[-1]:
            unique_x.append(x)
            unique_y.append(y)

    x_arr = np.array(unique_x)
    y_arr = np.array(unique_y)

    # Linear interpolation onto 1-second grid
    f_interp = interp1d(x_arr, y_arr, kind="linear", bounds_error=False, fill_value=np.nan)
    t_grid = np.arange(int(duration_sec))
    gt_rr_1hz = f_interp(t_grid).astype(np.float32)

    # Mask CO2 artifact periods if annotated in labels.csv
    if labels_csv_path and labels_csv_path.exists():
        try:
            df_lab = pd.read_csv(labels_csv_path)
            if "co2_artif_x" in df_lab.columns:
                art_str = str(df_lab["co2_artif_x"].iloc[0]).strip()
                if art_str and art_str.lower() != "nan":
                    samps = [int(v) for v in art_str.split() if v]
                    for i in range(0, len(samps) - 1, 2):
                        t0 = max(0, int(samps[i] / orig_fs))
                        t1 = min(int(duration_sec), int(np.ceil(samps[i + 1] / orig_fs)) + 1)
                        if t1 > t0:
                            gt_rr_1hz[t0:t1] = np.nan
        except Exception:
            pass

    return gt_rr_1hz


def load_capnobase_record(
    case_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    target_fs: float = FS,
    resp_filter: Optional[ButterworthRespiratoryFilter] = None,
    wide_filter: Optional[ButterworthWidebandFilter] = None,
) -> Dict[str, Any]:
    """
    Parses and standardizes a single CapnoBase recording.

    Returns dictionary matching BIDMC data_loader.stream_patient_records format:
        - record_id: str (e.g. 'capno_0009')
        - raw_ppg: np.ndarray (float32 resampled to 125 Hz)
        - filtered_ppg: np.ndarray (0.10–0.60 Hz Butterworth filtered float32)
        - filtered_wide: np.ndarray (0.10–5.00 Hz Butterworth wideband float32)
        - gt_rr_1hz: np.ndarray (1 Hz ground-truth respiratory rate array)
        - fs: float (target_fs = 125.0 Hz)
        - duration_sec: float
        - meta: dict (demographic and ventilator metadata)
    """
    resolved_dir = locate_capnobase_dir(data_dir)

    # Resolve filename stems
    base_prefix = f"{case_id}_8min"
    sig_path = resolved_dir / f"{base_prefix}_signal.csv"
    if not sig_path.exists():
        sig_path = resolved_dir / f"{case_id}_signal.csv"
        base_prefix = case_id

    param_path = resolved_dir / f"{base_prefix}_param.csv"
    ref_path = resolved_dir / f"{base_prefix}_reference.csv"
    lab_path = resolved_dir / f"{base_prefix}_labels.csv"
    meta_path = resolved_dir / f"{base_prefix}_meta.csv"

    # 1. Read parameters (detect native sampling rate explicitly)
    native_fs = 300.0
    case_vent = "unknown"
    if param_path.exists():
        df_param = pd.read_csv(param_path)
        if "samplingrate_pleth" in df_param.columns:
            native_fs = float(df_param["samplingrate_pleth"].iloc[0])
        if "case_ventilation" in df_param.columns:
            case_vent = str(df_param["case_ventilation"].iloc[0])

    # 2. Read PPG waveform
    df_sig = pd.read_csv(sig_path)
    if "pleth_y" not in df_sig.columns:
        raise ValueError(f"Missing 'pleth_y' in {sig_path}")
    raw_native = df_sig["pleth_y"].to_numpy(dtype=np.float32)

    duration_sec = len(raw_native) / native_fs

    # 3. Resample to target_fs (125 Hz)
    raw_resampled = resample_signal_125hz(raw_native, orig_fs=native_fs, target_fs=target_fs)

    # 4. Filter waveforms using existing filters
    if resp_filter is None:
        resp_filter = ButterworthRespiratoryFilter(fs=target_fs)
    if wide_filter is None:
        wide_filter = ButterworthWidebandFilter(fs=target_fs)

    filtered_ppg = resp_filter.apply(raw_resampled)
    filtered_wide = wide_filter.apply(raw_resampled)

    # 5. Build 1 Hz ground-truth RR array
    gt_rr_1hz = parse_capnobase_reference_rr(
        ref_csv_path=ref_path,
        labels_csv_path=lab_path,
        duration_sec=duration_sec,
        orig_fs=native_fs,
    )

    # 6. Metadata
    meta = {
        "native_fs": native_fs,
        "case_ventilation": case_vent,
        "duration_sec": duration_sec,
    }
    if meta_path.exists():
        try:
            df_m = pd.read_csv(meta_path)
            for col in df_m.columns:
                meta[col] = df_m[col].iloc[0]
        except Exception:
            pass

    return {
        "record_id": f"capno_{case_id}",
        "raw_ppg": raw_resampled,
        "filtered_ppg": filtered_ppg,
        "filtered_wide": filtered_wide,
        "gt_rr_1hz": gt_rr_1hz,
        "fs": target_fs,
        "duration_sec": duration_sec,
        "meta": meta,
    }


def stream_capnobase_records(
    case_list: Optional[List[str]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    target_fs: float = FS,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator yielding standardized CapnoBase records one by one.
    Memory-efficient streaming design tailored for 8GB RAM CPU-only execution.
    """
    if case_list is None:
        case_list = get_capnobase_case_ids(data_dir)

    resp_filter = ButterworthRespiratoryFilter(fs=target_fs)
    wide_filter = ButterworthWidebandFilter(fs=target_fs)

    for cid in case_list:
        rec = load_capnobase_record(
            case_id=cid,
            data_dir=data_dir,
            target_fs=target_fs,
            resp_filter=resp_filter,
            wide_filter=wide_filter,
        )
        yield rec

        del rec
        if ENABLE_GARBAGE_COLLECTION:
            gc.collect()


def build_capnobase_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    cache_path: Optional[Union[str, Path]] = None,
    force_recompute: bool = False,
    window_sec: int = WINDOW_SEC,
    stride_sec: float = STRIDE_SEC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts standard 28-dimensional multi-domain feature vectors from all CapnoBase
    recordings using the unmodified extract_window_features_from_segments() routine
    and caches the resulting matrix.

    Returns:
        Tuple of (X, y, patient_ids):
          - X: np.ndarray of shape (N_windows, 28) float32
          - y: np.ndarray of shape (N_windows,) float32 ground-truth RR
          - patient_ids: np.ndarray of shape (N_windows,) string
    """
    if cache_path is None:
        cache_path = CAPNOBASE_CACHE_FILE
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_recompute:
        data = np.load(cache_path)
        return data["X"], data["y"], data["patient_ids"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    all_X = []
    all_y = []
    all_pids = []
    all_sharpness = []

    case_ids = get_capnobase_case_ids(data_dir)
    print(f"[CapnoBase Import] Beginning extraction across {len(case_ids)} external benchmark cases...")

    w_samples = int(window_sec * FS)
    s_samples = int(stride_sec * FS)

    for record in stream_capnobase_records(case_list=case_ids, data_dir=data_dir, target_fs=FS):
        pid = record["record_id"]
        filt_resp = record["filtered_ppg"]
        filt_wide = record["filtered_wide"]
        gt_rr = record["gt_rr_1hz"]

        n_steps = (len(filt_resp) - w_samples) // s_samples + 1
        case_wins = 0

        for step in range(n_steps):
            start_idx = step * s_samples
            end_idx = start_idx + w_samples

            start_sec = int(start_idx / FS)
            end_sec = int(end_idx / FS)
            rr_slice = gt_rr[start_sec:end_sec]

            # Mirror exact BIDMC validation filters
            if len(rr_slice) == 0 or np.all(np.isnan(rr_slice)):
                continue
            mean_rr = float(np.nanmean(rr_slice))
            if np.isnan(mean_rr) or mean_rr < 6.0 or mean_rr > 45.0:
                continue

            win_resp = filt_resp[start_idx:end_idx]
            win_wide = filt_wide[start_idx:end_idx]

            # Execute EXACT unmodified feature extraction function from BIDMC pipeline
            feats = extract_window_features_from_segments(win_resp, win_wide, fs=FS)
            sharpness = compute_spectral_peak_sharpness(win_resp, fs=FS)

            all_X.append(feats)
            all_y.append(mean_rr)
            all_pids.append(pid)
            all_sharpness.append(sharpness)
            case_wins += 1

        print(f"  Processed {pid}: {case_wins} valid 60s windows extracted")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    patient_ids = np.array(all_pids)
    sharpness_arr = np.array(all_sharpness, dtype=np.float32)

    unique_pids, pid_counts = np.unique(patient_ids, return_counts=True)
    print(f"\n[CapnoBase Import] Completed extraction: {len(y)} windows across {len(unique_pids)} external cases.")
    print(f"  Feature shape: {X.shape} | Windows per case: min={np.min(pid_counts)}, mean={np.mean(pid_counts):.1f}, max={np.max(pid_counts)}")

    # Save format matching features_split.npz conventions
    save_dict = {
        "X": X,
        "y": y,
        "patient_ids": patient_ids,
        "sharpness": sharpness_arr,
        "window_sec": np.array(window_sec),
        "feature_count": np.array(X.shape[1]),
        "X_test": X,
        "y_test": y,
        "test_idx": np.arange(len(y), dtype=np.int64),
        "test_pids": unique_pids,
        "is_external": np.array(True),
    }

    np.savez_compressed(cache_path, **save_dict)
    print(f"[CapnoBase Import] Successfully cached external dataset to {cache_path}")

    return X, y, patient_ids


def load_capnobase_dataset(cache_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads cached CapnoBase feature matrix, target respiratory rates, and patient IDs.
    Builds and caches the dataset if not yet present.
    """
    return build_capnobase_dataset(cache_path=cache_path, force_recompute=False)


if __name__ == "__main__":
    X, y, pids = build_capnobase_dataset(force_recompute=True)
    print(f"\n[Done] CapnoBase external validation set ready: X={X.shape}, y={y.shape}, cases={len(np.unique(pids))}")
