"""
Phase 4: Raw-Signal Artifact Simulation & Signal Quality Index (SQI) Module
==========================================================================
Operates directly on raw filtered PPG windows in the TIME DOMAIN before feature extraction.
Provides four separate composable artifact types across three clinically realistic severities:
  1. High-Frequency Noise (HF noise calibrated by Signal-to-Noise Ratio in dB)
  2. Baseline Drift (low-frequency sinusoidal / respiratory wander distortion)
  3. Motion Burst (transient physical motion artifact with random onset and duration)
  4. Dropout (sensor disconnection / peripheral loss of pulsatile signal)

Severities:
  - 'mild'
  - 'moderate'
  - 'severe'

Rules:
  - Only validation (for threshold selection) and test windows are corrupted.
  - Training and calibration windows are NEVER corrupted.
  - Corrupted windows differ from clean in the time domain (verified by unit test).
  - Provides Signal Quality Index (SQI) baseline (in-band power ratio & spectral flatness).
"""

from typing import Tuple, Dict, Any, Optional, List, Union
from pathlib import Path
import json
import numpy as np

from config.settings import (
    FS,
    CACHE_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
    FILTER_LOW_HZ,
    FILTER_HIGH_HZ,
    ARTIFACT_NOISE_SCALE,
    ARTIFACT_SHIFT_BOUND,
    ARTIFACT_SPIKE_PROBABILITY
)

ARTIFACT_PARAMS_FILE = CACHE_DIR / "artifact_params.json"

ARTIFACT_SEVERITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mild": {
        "description": "Mild clinical disturbance: minor sensor movement, low muscle tremor",
        "hf_snr_db": 20.0,
        "baseline_drift_scale": 0.35,
        "motion_burst_scale": 0.80,
        "motion_burst_duration": [2.0, 4.0],
        "dropout_scale": 0.05,
        "dropout_duration": [2.0, 5.0],
    },
    "moderate": {
        "description": "Moderate clinical disturbance: bed adjustment, patient repositioning, motion transient",
        "hf_snr_db": 10.0,
        "baseline_drift_scale": 0.85,
        "motion_burst_scale": 2.00,
        "motion_burst_duration": [5.0, 10.0],
        "dropout_scale": 0.01,
        "dropout_duration": [6.0, 12.0],
    },
    "severe": {
        "description": "Severe clinical disturbance: intense ambulation, limb tremor, sensor displacement / probe slip",
        "hf_snr_db": 3.0,
        "baseline_drift_scale": 2.00,
        "motion_burst_scale": 4.00,
        "motion_burst_duration": [12.0, 25.0],
        "dropout_scale": 0.00,
        "dropout_duration": [15.0, 30.0],
    },
}


def save_artifact_parameters(path: Optional[Path] = None) -> Path:
    """Saves the official artifact injection parameters to JSON."""
    if path is None:
        path = ARTIFACT_PARAMS_FILE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ARTIFACT_SEVERITY_CONFIGS, f, indent=2)
    return path


def inject_hf_noise(
    signal: np.ndarray,
    snr_db: float = 10.0,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Injects high-frequency noise into a raw PPG window calibrated to a target SNR (dB).
    SNR_dB = 10 * log10(P_signal / P_noise)  =>  P_noise = P_signal / (10^(SNR/10))
    """
    if rng is None:
        rng = np.random.RandomState()
    sig_power = float(np.mean(signal ** 2))
    if sig_power <= 1e-12:
        sig_power = 1.0
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    noise = rng.normal(0, noise_std, size=len(signal)).astype(np.float32)
    return (signal + noise).astype(np.float32)


def inject_baseline_drift(
    signal: np.ndarray,
    drift_scale: float = 0.85,
    fs: float = FS,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Injects low-frequency baseline drift (respiratory baseline wander / sensor shift).
    """
    if rng is None:
        rng = np.random.RandomState()
    n = len(signal)
    t = np.arange(n) / fs
    sig_std = max(1e-4, float(np.std(signal)))
    amp = drift_scale * sig_std

    f1 = rng.uniform(0.02, 0.07)
    f2 = rng.uniform(0.05, 0.12)
    phi1 = rng.uniform(0, 2 * np.pi)
    phi2 = rng.uniform(0, 2 * np.pi)

    drift = (amp * (0.7 * np.sin(2 * np.pi * f1 * t + phi1) +
                    0.3 * np.cos(2 * np.pi * f2 * t + phi2))).astype(np.float32)
    return (signal + drift).astype(np.float32)


def inject_motion_burst(
    signal: np.ndarray,
    burst_scale: float = 2.0,
    duration_range: Tuple[float, float] = (5.0, 10.0),
    fs: float = FS,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Injects transient motion burst with a random onset and duration.
    Returns:
        (corrupted_signal, (start_idx, end_idx))
    """
    if rng is None:
        rng = np.random.RandomState()
    n = len(signal)
    total_dur_sec = n / fs
    dur_sec = rng.uniform(duration_range[0], min(duration_range[1], total_dur_sec * 0.85))
    burst_len = max(int(dur_sec * fs), 4)

    max_start = max(0, n - burst_len)
    start_idx = rng.randint(0, max_start + 1) if max_start > 0 else 0
    end_idx = min(n, start_idx + burst_len)
    actual_len = end_idx - start_idx

    sig_ptp = max(1e-4, float(np.ptp(signal)))
    burst_amp = burst_scale * sig_ptp

    t_burst = np.linspace(0, 1, actual_len, endpoint=False)
    # Hann envelope modulated oscillatory disturbance
    hanning_env = np.hanning(actual_len)
    f_burst = rng.uniform(0.5, 3.0)
    oscillation = burst_amp * np.sin(2 * np.pi * f_burst * t_burst) * hanning_env
    high_freq = rng.normal(0, 0.25 * burst_amp, size=actual_len) * hanning_env

    corrupted = signal.copy()
    corrupted[start_idx:end_idx] += (oscillation + high_freq).astype(np.float32)
    return corrupted.astype(np.float32), (start_idx, end_idx)


def inject_dropout(
    signal: np.ndarray,
    dropout_scale: float = 0.01,
    duration_range: Tuple[float, float] = (6.0, 12.0),
    fs: float = FS,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Simulates optical sensor disconnect / probe displacement / flatline dropout.
    Attenuates signal down to near zero or zero over the dropout duration.
    """
    if rng is None:
        rng = np.random.RandomState()
    n = len(signal)
    total_dur_sec = n / fs
    dur_sec = rng.uniform(duration_range[0], min(duration_range[1], total_dur_sec * 0.9))
    drop_len = max(int(dur_sec * fs), 4)

    max_start = max(0, n - drop_len)
    start_idx = rng.randint(0, max_start + 1) if max_start > 0 else 0
    end_idx = min(n, start_idx + drop_len)
    actual_len = end_idx - start_idx

    corrupted = signal.copy()
    if dropout_scale <= 1e-6:
        corrupted[start_idx:end_idx] = 0.0
    else:
        sig_std = max(1e-4, float(np.std(signal)))
        corrupted[start_idx:end_idx] = (
            dropout_scale * corrupted[start_idx:end_idx] +
            rng.normal(0, dropout_scale * sig_std, size=actual_len)
        ).astype(np.float32)

    return corrupted.astype(np.float32), (start_idx, end_idx)


def corrupt_raw_ppg_window(
    signal: np.ndarray,
    severity: str = "moderate",
    artifact_types: Optional[List[str]] = None,
    fs: float = FS,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Applies composable artifact injection directly to a single raw filtered PPG window
    BEFORE feature extraction.
    Supports individual or composite artifact types:
      ['hf_noise', 'baseline_drift', 'motion_burst', 'dropout']
    """
    if severity not in ARTIFACT_SEVERITY_CONFIGS:
        raise ValueError(f"Unknown severity '{severity}'. Choose from {list(ARTIFACT_SEVERITY_CONFIGS.keys())}")

    cfg = ARTIFACT_SEVERITY_CONFIGS[severity]
    rng = np.random.RandomState(seed)

    if artifact_types is None:
        # Default composite: apply all 4 types
        artifact_types = ["hf_noise", "baseline_drift", "motion_burst", "dropout"]

    corrupted = signal.copy()

    if "baseline_drift" in artifact_types:
        corrupted = inject_baseline_drift(
            corrupted,
            drift_scale=cfg["baseline_drift_scale"],
            fs=fs,
            rng=rng
        )

    if "motion_burst" in artifact_types:
        corrupted, _ = inject_motion_burst(
            corrupted,
            burst_scale=cfg["motion_burst_scale"],
            duration_range=tuple(cfg["motion_burst_duration"]),
            fs=fs,
            rng=rng
        )

    if "dropout" in artifact_types:
        corrupted, _ = inject_dropout(
            corrupted,
            dropout_scale=cfg["dropout_scale"],
            duration_range=tuple(cfg["dropout_duration"]),
            fs=fs,
            rng=rng
        )

    if "hf_noise" in artifact_types:
        corrupted = inject_hf_noise(
            corrupted,
            snr_db=cfg["hf_snr_db"],
            rng=rng
        )

    return corrupted.astype(np.float32)


# ==============================================================================
# Signal Quality Index (SQI) Baselines
# ==============================================================================

def compute_inband_power_ratio(
    signal: np.ndarray,
    fs: float = FS,
    f_low: float = FILTER_LOW_HZ,
    f_high: float = FILTER_HIGH_HZ,
    nfft: int = 4096
) -> float:
    """
    Computes ratio of power in the respiratory band [f_low, f_high]
    to total signal power in [0.01, fs/2].
    Clean respiratory PPG -> ratio near 1.0.
    Corrupted with HF noise, drift, or motion -> ratio drops significantly.
    """
    seg = signal - np.mean(signal)
    fft_vals = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    total_power = float(np.sum(fft_vals))
    if total_power <= 1e-12:
        return 0.0

    resp_mask = (freqs >= f_low) & (freqs <= f_high)
    resp_power = float(np.sum(fft_vals[resp_mask]))
    return float(np.clip(resp_power / total_power, 0.0, 1.0))


def compute_spectral_flatness(
    signal: np.ndarray,
    fs: float = FS,
    nfft: int = 4096
) -> float:
    """
    Computes Wiener spectral flatness (geometric mean / arithmetic mean of power spectrum).
    Pure tones / clean periodic signals -> flatness near 0.
    White noise / motion burst / high disorder -> flatness rises toward 1.
    """
    seg = signal - np.mean(signal)
    psd = (np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=nfft)) ** 2) + 1e-12
    geom = float(np.exp(np.mean(np.log(psd))))
    arith = float(np.mean(psd))
    if arith <= 1e-12:
        return 1.0
    return float(np.clip(geom / arith, 0.0, 1.0))


def compute_sqi_abstention_score(
    signal: np.ndarray,
    fs: float = FS,
    method: str = "inband_ratio"
) -> float:
    """
    Computes SQI-based abstention score where HIGHER score indicates higher corruption/risk.
    For inband_ratio: abstention_score = 1.0 - inband_power_ratio.
    For spectral_flatness: abstention_score = spectral_flatness.
    """
    if method == "inband_ratio":
        ratio = compute_inband_power_ratio(signal, fs=fs)
        return float(1.0 - ratio)
    elif method == "spectral_flatness":
        return float(compute_spectral_flatness(signal, fs=fs))
    else:
        raise ValueError(f"Unknown SQI method: {method}")


# ==============================================================================
# Legacy Backward Compatibility Wrappers
# ==============================================================================

def corrupt_single_ppg_segment(
    clean_segment: np.ndarray,
    t_seconds: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Legacy helper used for 6s visualization snippet."""
    rng = np.random.RandomState(seed)
    noisy_seg = clean_segment.copy()
    clean_std = float(np.std(clean_segment))
    noisy_seg += rng.normal(0, 0.08 * clean_std + 0.04, size=len(t_seconds)).astype(np.float32)
    noisy_seg += (
        0.15 * np.sin(2 * np.pi * 0.08 * t_seconds) +
        0.12 * np.cos(2 * np.pi * 0.03 * t_seconds)
    ).astype(np.float32)
    spike_start = int(0.30 * len(clean_segment))
    spike_end = int(0.58 * len(clean_segment))
    spike_len = spike_end - spike_start
    noisy_seg[spike_start:spike_end] += (
        0.35 * np.sin(np.linspace(0, np.pi, spike_len)) +
        rng.normal(0, 0.15, size=spike_len)
    ).astype(np.float32)
    return noisy_seg, (spike_start, spike_end)


def inject_simulated_motion_artifacts(
    X_clean: np.ndarray,
    seed: int = RANDOM_SEED
) -> np.ndarray:
    """Legacy feature-level corruption for backwards compatibility."""
    rng = np.random.RandomState(seed)
    stds = np.std(X_clean, axis=0) + 1e-6
    gaussian_noise = rng.normal(0, ARTIFACT_NOISE_SCALE, size=X_clean.shape) * stds
    shifts = rng.uniform(-ARTIFACT_SHIFT_BOUND, ARTIFACT_SHIFT_BOUND, size=X_clean.shape) * stds
    spike_mask = rng.binomial(1, ARTIFACT_SPIKE_PROBABILITY, size=(X_clean.shape[0], 1))
    spikes = spike_mask * rng.normal(0, 4.0, size=X_clean.shape) * stds
    return (X_clean + gaussian_noise + shifts + spikes).astype(np.float32)
