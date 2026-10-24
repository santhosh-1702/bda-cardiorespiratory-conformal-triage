"""
Phase 2 / Phase 3: Unified Feature Extraction & Patient-Grouped Dataset Builder Module
======================================================================================
Coordinates sliding window segmentation, multi-domain feature extraction,
ground-truth mapping, within-window spectral peak sharpness, and 5-fold patient-disjoint
cross-validation splitting (Train, Validation, Calibration, Test) to strictly prevent data leakage.
"""

from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import KFold, GroupKFold

from config.settings import (
    FS,
    FEATURES_CACHE_FILE,
    CACHE_DIR,
    OUTPUT_DIR,
    N_FOLDS,
    VAL_PATIENT_RATIO,
    CAL_PATIENT_RATIO,
    RANDOM_SEED,
    PATIENT_IDS
)
from src.preprocessing.data_loader import stream_patient_records
from src.preprocessing.filter import ButterworthRespiratoryFilter, ButterworthWidebandFilter
from src.features.modulations import (
    extract_pulse_modulations,
    compute_ppg_spectrum_descriptors,
    compute_spectrum_descriptors,
    estimate_dominant_frequency
)

FEATURES_60S_CACHE = CACHE_DIR / "features_60s.npz"
FOLD_MAP_FILE = CACHE_DIR / "fold_map.json"


def create_patient_fold_map(
    patient_ids: Optional[List[str]] = None,
    n_splits: int = N_FOLDS,
    seed: int = RANDOM_SEED,
    save_path: Optional[Path] = None
) -> Dict[str, int]:
    """
    Creates ONE seeded patient->fold map derived strictly from unique patient IDs
    (never window counts). Saves to output/fold_map.json by default.
    """
    if patient_ids is None:
        patient_ids = PATIENT_IDS
    unique_pids = np.array(sorted(list(set(patient_ids))))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_map = {}
    for fold_idx, (_, test_idx) in enumerate(kf.split(unique_pids)):
        for pid in unique_pids[test_idx]:
            fold_map[str(pid)] = int(fold_idx)

    if save_path is None:
        save_path = FOLD_MAP_FILE
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(fold_map, f, indent=2)
    return fold_map


def load_patient_fold_map(
    path: Optional[Path] = None,
    seed: int = RANDOM_SEED
) -> Dict[str, int]:
    """
    Loads existing patient->fold map from JSON or creates and saves a new one.
    """
    if path is None:
        path = FOLD_MAP_FILE
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {str(k): int(v) for k, v in data.items()}
    return create_patient_fold_map(seed=seed, save_path=path)


def extract_single_window_features(window_segment: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    NOTE: legacy — belongs to the original 6s/55-feature approach, superseded by the 28-feature pipeline in extractor.py; retained for historical reference.

    Extracts features from a single window.
    """
    from src.features.wavelet_features import extract_dwt_features
    from src.features.spectral_features import extract_spectral_features
    from src.features.statistical_features import extract_statistical_features
    dwt_feats = extract_dwt_features(window_segment)
    spec_feats = extract_spectral_features(window_segment, fs=fs)
    stat_feats = extract_statistical_features(window_segment)
    all_feats = dwt_feats + spec_feats + stat_feats
    return np.array(all_feats, dtype=np.float32)


def compute_spectral_peak_sharpness(
    win_resp: np.ndarray,
    fs: float = FS,
    f_low: float = 0.10,
    f_high: float = 0.60,
    nfft: int = 4096
) -> float:
    """
    Computes within-window spectral peak sharpness (peak power / mean band power)
    over the respiratory band [f_low, f_high].
    """
    seg = win_resp - np.mean(win_resp)
    fft_vals = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    resp_mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(resp_mask):
        return 1.0
    band_powers = fft_vals[resp_mask] ** 2
    total_band_power = float(np.sum(band_powers))
    if total_band_power <= 1e-12:
        return 1.0
    peak_power = float(np.max(band_powers))
    mean_band_power = total_band_power / len(band_powers)
    return float(peak_power / (mean_band_power + 1e-12))


def extract_window_features_from_segments(
    win_resp: np.ndarray,
    win_wide: Optional[np.ndarray] = None,
    fs: float = FS
) -> np.ndarray:
    """
    Extracts the standard 28-dimensional multi-domain feature vector directly
    from raw 60s window segments (filtered respiratory PPG and optional wideband PPG).
    Enables feature extraction on corrupted raw waveforms before model inference.
    """
    sharpness = compute_spectral_peak_sharpness(win_resp, fs=fs)
    ppg_spec_desc = compute_ppg_spectrum_descriptors(win_resp, fs=fs)

    target_wide = win_wide if (win_wide is not None and len(win_wide) > 0) else win_resp
    mods = extract_pulse_modulations(target_wide, fs=fs)
    t_peaks = mods["t_peaks"]
    riav_series = mods["riav"]
    riiv_series = mods["riiv"]
    rifv_series = mods["rifv"]
    has_mods = len(t_peaks) > 3

    w_t_start = 0.0
    w_t_end = len(win_resp) / fs

    if has_mods:
        riav_spec_desc = compute_spectrum_descriptors(t_peaks, riav_series, w_t_start, w_t_end)
        riiv_spec_desc = compute_spectrum_descriptors(t_peaks, riiv_series, w_t_start, w_t_end)
        rifv_spec_desc = compute_spectrum_descriptors(t_peaks, riiv_series, w_t_start, w_t_end)
        riav_rr = estimate_dominant_frequency(t_peaks, riav_series, w_t_start, w_t_end)
        riiv_rr = estimate_dominant_frequency(t_peaks, riiv_series, w_t_start, w_t_end)
        rifv_rr = estimate_dominant_frequency(t_peaks, rifv_series, w_t_start, w_t_end)
        fusion_rr = float(np.median([riav_rr, riiv_rr, rifv_rr]))

        mask = (t_peaks >= w_t_start) & (t_peaks < w_t_end)
        if np.sum(mask) >= 3:
            riav_sub = riav_series[mask]
            riiv_sub = riiv_series[mask]
            rifv_sub = rifv_series[mask]
            riav_mean, riav_std = float(np.mean(riav_sub)), float(np.std(riav_sub))
            riiv_mean, riiv_std = float(np.mean(riiv_sub)), float(np.std(riiv_sub))
            rifv_mean, rifv_std = float(np.mean(rifv_sub)), float(np.std(rifv_sub))
            mean_hr = float(60.0 / (rifv_mean + 1e-6))
        else:
            riav_mean, riav_std = 0.0, 0.0
            riiv_mean, riiv_std = 0.0, 0.0
            rifv_mean, rifv_std = 0.8, 0.0
            mean_hr = 75.0
    else:
        riav_spec_desc = [17.65, 0.0, 0.0, 17.65]
        riiv_spec_desc = [17.65, 0.0, 0.0, 17.65]
        rifv_spec_desc = [17.65, 0.0, 0.0, 17.65]
        fusion_rr = 17.65
        riav_mean, riav_std = 0.0, 0.0
        riiv_mean, riiv_std = 0.0, 0.0
        rifv_mean, rifv_std = 0.8, 0.0
        mean_hr = 75.0

    sig_std = float(np.std(win_resp))
    sig_ptp = float(np.ptp(win_resp))
    sig_rms = float(np.sqrt(np.mean(win_resp ** 2)))

    feat_vector = (
        ppg_spec_desc +
        riav_spec_desc +
        riiv_spec_desc +
        rifv_spec_desc +
        [fusion_rr, riav_mean, riav_std, riiv_mean, riiv_std, rifv_mean, rifv_std, mean_hr, sig_std, sig_ptp, sig_rms, sharpness]
    )
    return np.array(feat_vector, dtype=np.float32)


def generate_patient_folds(
    patient_ids: np.ndarray,
    n_splits: int = N_FOLDS,
    val_ratio: float = VAL_PATIENT_RATIO,
    cal_ratio: float = CAL_PATIENT_RATIO,
    seed: int = RANDOM_SEED,
    fold_map: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    """
    Generates 5-fold patient-grouped splits using a fixed patient->fold map
    derived strictly from patient IDs (not window counts).
    Within each outer training fold, partitions outer patients into:
      1. Train set: base model fitting
      2. Validation set (~20% of outer patients): hyperparameter selection
      3. Calibration set (~20% of outer patients): conformal prediction calibration
    Outer fold:
      4. Test set: final reporting (held-out patients)
    Guarantees that train, validation, calibration, and test patient sets
    are strictly pairwise disjoint across all folds.
    """
    if fold_map is None:
        fold_map = load_patient_fold_map(seed=seed)

    unique_in_data = set(patient_ids)
    folds = []
    for fold_idx in range(n_splits):
        test_pids = sorted([p for p, f in fold_map.items() if f == fold_idx and p in unique_in_data])
        outer_train_pids = np.array(sorted([p for p, f in fold_map.items() if f != fold_idx and p in unique_in_data]))

        # Stable seeded permutation to partition outer training patients
        rng = np.random.RandomState(seed + fold_idx)
        perm = rng.permutation(len(outer_train_pids))
        shuffled_outer = outer_train_pids[perm]

        n_outer = len(shuffled_outer)
        n_val = int(round(n_outer * val_ratio))
        n_cal = int(round(n_outer * cal_ratio))

        val_pids = sorted(list(shuffled_outer[:n_val]))
        cal_pids = sorted(list(shuffled_outer[n_val : n_val + n_cal]))
        train_pids = sorted(list(shuffled_outer[n_val + n_cal :]))

        # Map patient sets back to window indices
        train_mask = np.isin(patient_ids, train_pids)
        val_mask = np.isin(patient_ids, val_pids)
        cal_mask = np.isin(patient_ids, cal_pids)
        test_mask = np.isin(patient_ids, test_pids)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        cal_indices = np.where(cal_mask)[0]
        test_indices = np.where(test_mask)[0]

        folds.append({
            "fold": fold_idx,
            "train_idx": train_indices,
            "val_idx": val_indices,
            "cal_idx": cal_indices,
            "test_idx": test_indices,
            "train_pids": train_pids,
            "val_pids": val_pids,
            "cal_pids": cal_pids,
            "test_pids": test_pids,
        })

    return folds


def build_feature_dataset(
    force_recompute: bool = False,
    window_sec: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Builds the full feature matrix using the validation-chosen 60s window duration
    across all 53 patients or loads from cache.
    Carries patient_id, ground-truth RR, and within-window spectral peak sharpness.
    Asserts window length and windows-per-patient.

    Returns:
        (X, y, patient_ids, folds)
    """
    cache_path = FEATURES_CACHE_FILE
    if not force_recompute and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cached_w = int(cached.get("window_sec", 60))
            if "patient_ids" in cached and "X" in cached and "y" in cached and cached_w == window_sec:
                X = cached["X"]
                y = cached["y"]
                patient_ids = cached["patient_ids"]

                unique_pids, pid_counts = np.unique(patient_ids, return_counts=True)
                min_wins = int(np.min(pid_counts))
                max_wins = int(np.max(pid_counts))
                mean_wins = float(np.mean(pid_counts))

                print(f"[Dataset Assertion] Window length: {cached_w}s | Windows per patient: min={min_wins}, mean={mean_wins:.1f}, max={max_wins} | Total: {len(y)} windows across {len(unique_pids)} patients.")
                assert cached_w == 60, f"Expected 60s validation-chosen window, found {cached_w}s"
                assert min_wins >= 50 and max_wins == 211, f"Windows-per-patient mismatch: min={min_wins}, max={max_wins}"

                expected_fold_map = load_patient_fold_map()
                expected_f0_test = sorted([p for p, f in expected_fold_map.items() if f == 0])
                n_folds = int(cached.get("n_folds", N_FOLDS))
                folds = []
                has_all_fold_keys = all(f"fold_{k}_train_idx" in cached for k in range(n_folds))
                cached_matches_map = False
                if has_all_fold_keys and "fold_0_test_pids" in cached:
                    cached_f0_test = sorted([str(p) for p in cached["fold_0_test_pids"]])
                    cached_matches_map = (cached_f0_test == expected_f0_test)

                if has_all_fold_keys and cached_matches_map:
                    for k in range(n_folds):
                        folds.append({
                            "fold": k,
                            "train_idx": cached[f"fold_{k}_train_idx"],
                            "val_idx": cached[f"fold_{k}_val_idx"],
                            "cal_idx": cached[f"fold_{k}_cal_idx"],
                            "test_idx": cached[f"fold_{k}_test_idx"],
                            "train_pids": list(cached[f"fold_{k}_train_pids"]),
                            "val_pids": list(cached[f"fold_{k}_val_pids"]),
                            "cal_pids": list(cached[f"fold_{k}_cal_pids"]),
                            "test_pids": list(cached[f"fold_{k}_test_pids"]),
                        })
                else:
                    folds = generate_patient_folds(patient_ids, fold_map=expected_fold_map)
                    save_dict = dict(cached)
                    for k, fold in enumerate(folds):
                        for key in ["train_idx", "val_idx", "cal_idx", "test_idx"]:
                            save_dict[f"fold_{k}_{key}"] = fold[key]
                        for key in ["train_pids", "val_pids", "cal_pids", "test_pids"]:
                            save_dict[f"fold_{k}_{key}"] = np.array(fold[key])
                    np.savez_compressed(cache_path, **save_dict)
                    np.savez_compressed(FEATURES_60S_CACHE, **save_dict)

                return X, y, patient_ids, folds
        except Exception as e:
            print(f"[Feature Cache] Cache load failed ({e}), recomputing...")

    print(f"[Phase Features] Extracting {window_sec}s validation-chosen windows & modulations across 53 BIDMC patients...")
    filt_resp_obj = ButterworthRespiratoryFilter(fs=FS)
    filt_wide_obj = ButterworthWidebandFilter(fs=FS)

    all_X = []
    all_y = []
    all_pids = []
    all_sharpness = []

    for patient in stream_patient_records():
        pid = patient["record_id"]
        raw_ppg = patient["raw_ppg"]
        gt_rr = patient["gt_rr_1hz"]
        filt_resp = filt_resp_obj.apply(raw_ppg)
        filt_wide = filt_wide_obj.apply(raw_ppg)

        mods = extract_pulse_modulations(filt_wide, fs=FS)
        t_peaks = mods["t_peaks"]
        riav_series = mods["riav"]
        riiv_series = mods["riiv"]
        rifv_series = mods["rifv"]
        has_mods = len(t_peaks) > 3

        w_samples = int(window_sec * FS)
        s_samples = int(2.0 * FS)  # 2.0s stride
        n_steps = (len(raw_ppg) - w_samples) // s_samples + 1

        for step in range(n_steps):
            start_idx = step * s_samples
            end_idx = start_idx + w_samples
            start_sec = int(start_idx / FS)
            end_sec = int(end_idx / FS)
            rr_slice = gt_rr[start_sec:end_sec]

            if len(rr_slice) == 0 or np.all(np.isnan(rr_slice)):
                continue
            mean_rr = float(np.nanmean(rr_slice))
            if np.isnan(mean_rr) or mean_rr < 6.0 or mean_rr > 45.0:
                continue

            w_t_start = start_idx / FS
            w_t_end = end_idx / FS
            win_resp = filt_resp[start_idx:end_idx]

            sharpness = compute_spectral_peak_sharpness(win_resp, fs=FS)
            ppg_spec_desc = compute_ppg_spectrum_descriptors(win_resp, fs=FS)

            if has_mods:
                riav_spec_desc = compute_spectrum_descriptors(t_peaks, riav_series, w_t_start, w_t_end)
                riiv_spec_desc = compute_spectrum_descriptors(t_peaks, riiv_series, w_t_start, w_t_end)
                rifv_spec_desc = compute_spectrum_descriptors(t_peaks, riiv_series, w_t_start, w_t_end)
                riav_rr = estimate_dominant_frequency(t_peaks, riav_series, w_t_start, w_t_end)
                riiv_rr = estimate_dominant_frequency(t_peaks, riiv_series, w_t_start, w_t_end)
                rifv_rr = estimate_dominant_frequency(t_peaks, rifv_series, w_t_start, w_t_end)
                fusion_rr = float(np.median([riav_rr, riiv_rr, rifv_rr]))

                mask = (t_peaks >= w_t_start) & (t_peaks < w_t_end)
                if np.sum(mask) >= 3:
                    riav_sub = riav_series[mask]
                    riiv_sub = riiv_series[mask]
                    rifv_sub = rifv_series[mask]
                    riav_mean, riav_std = float(np.mean(riav_sub)), float(np.std(riav_sub))
                    riiv_mean, riiv_std = float(np.mean(riiv_sub)), float(np.std(riiv_sub))
                    rifv_mean, rifv_std = float(np.mean(rifv_sub)), float(np.std(rifv_sub))
                    mean_hr = float(60.0 / (rifv_mean + 1e-6))
                else:
                    riav_mean, riav_std = 0.0, 0.0
                    riiv_mean, riiv_std = 0.0, 0.0
                    rifv_mean, rifv_std = 0.8, 0.0
                    mean_hr = 75.0
            else:
                riav_spec_desc = [17.65, 0.0, 0.0, 17.65]
                riiv_spec_desc = [17.65, 0.0, 0.0, 17.65]
                rifv_spec_desc = [17.65, 0.0, 0.0, 17.65]
                fusion_rr = 17.65
                riav_mean, riav_std = 0.0, 0.0
                riiv_mean, riiv_std = 0.0, 0.0
                rifv_mean, rifv_std = 0.8, 0.0
                mean_hr = 75.0

            sig_std = float(np.std(win_resp))
            sig_ptp = float(np.ptp(win_resp))
            sig_rms = float(np.sqrt(np.mean(win_resp ** 2)))

            feat_vector = (
                ppg_spec_desc +
                riav_spec_desc +
                riiv_spec_desc +
                rifv_spec_desc +
                [fusion_rr, riav_mean, riav_std, riiv_mean, riiv_std, rifv_mean, rifv_std, mean_hr, sig_std, sig_ptp, sig_rms, sharpness]
            )
            all_X.append(feat_vector)
            all_y.append(mean_rr)
            all_pids.append(pid)
            all_sharpness.append(sharpness)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    patient_ids = np.array(all_pids)
    sharpness_arr = np.array(all_sharpness, dtype=np.float32)

    unique_pids, pid_counts = np.unique(patient_ids, return_counts=True)
    min_wins = int(np.min(pid_counts))
    max_wins = int(np.max(pid_counts))
    mean_wins = float(np.mean(pid_counts))

    print(f"[Dataset Assertion] Window length: {window_sec}s | Windows per patient: min={min_wins}, mean={mean_wins:.1f}, max={max_wins} | Total: {len(y)} windows across {len(unique_pids)} patients.")
    assert window_sec == 60, f"Expected 60s validation-chosen window, got {window_sec}s"
    assert min_wins >= 50 and max_wins == 211, f"Windows-per-patient count mismatch: min={min_wins}, max={max_wins}"

    # Generate patient-disjoint 5-fold splits
    folds = generate_patient_folds(patient_ids)

    save_dict = {
        "X": X,
        "y": y,
        "patient_ids": patient_ids,
        "sharpness": sharpness_arr,
        "window_sec": np.array(window_sec),
        "feature_count": np.array(X.shape[1]),
        "n_folds": np.array(len(folds)),
        "X_train": X[folds[0]["train_idx"]],
        "X_test": X[folds[0]["test_idx"]],
        "y_train": y[folds[0]["train_idx"]],
        "y_test": y[folds[0]["test_idx"]],
    }
    for k, fold in enumerate(folds):
        for key in ["train_idx", "val_idx", "cal_idx", "test_idx"]:
            save_dict[f"fold_{k}_{key}"] = fold[key]
        for key in ["train_pids", "val_pids", "cal_pids", "test_pids"]:
            save_dict[f"fold_{k}_{key}"] = np.array(fold[key])

    np.savez_compressed(cache_path, **save_dict)
    np.savez_compressed(FEATURES_60S_CACHE, **save_dict)
    print(f"[Phase Features] Cached {window_sec}s dataset ({X.shape[0]} windows, {X.shape[1]} feats) to {cache_path} and {FEATURES_60S_CACHE}.")

    return X, y, patient_ids, folds


def load_feature_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads cached X, y, and patient_ids array."""
    X, y, pids, _ = build_feature_dataset(force_recompute=False)
    return X, y, pids


def get_patient_folds(force_recompute: bool = False) -> List[Dict[str, Any]]:
    """Returns 5 patient-disjoint folds with attached data views."""
    X, y, patient_ids, folds = build_feature_dataset(force_recompute=force_recompute)
    for f in folds:
        f["X_train"] = X[f["train_idx"]]
        f["y_train"] = y[f["train_idx"]]
        f["X_val"] = X[f["val_idx"]]
        f["y_val"] = y[f["val_idx"]]
        f["X_cal"] = X[f["cal_idx"]]
        f["y_cal"] = y[f["cal_idx"]]
        f["X_test"] = X[f["test_idx"]]
        f["y_test"] = y[f["test_idx"]]
    return folds
