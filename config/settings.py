"""
Centralized Configuration & Hyperparameters
===========================================
Defines global constants, paths, hardware constraints, and hyperparameter targets
for the cardiorespiratory triage pipeline.
"""

from pathlib import Path

# ==============================================================================
# 1. Project Filesystem Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "bidmc_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
PAPER_DIR = OUTPUT_DIR / "paper"
PIPELINE_STAGES_DIR = PROJECT_ROOT / "pipeline_stages"
WEB_DIR = PROJECT_ROOT / "web"

FIGURE_EXPORT_PATH = FIGURES_DIR / "abstention_results.png"
ARCHITECTURE_FIGURE_PATH = FIGURES_DIR / "pipeline_architecture.png"


# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)

# Serialized Cache Artifacts
FEATURES_CACHE_FILE = CACHE_DIR / "features_split.npz"
MODEL_CACHE_FILE = CACHE_DIR / "best_xgb_model.json"
CONFORMAL_RESULTS_FILE = CACHE_DIR / "conformal_results.npz"
FINAL_REPORT_FILE = RESULTS_DIR / "final_report.txt"
DEMO_REPORT_FILE = RESULTS_DIR / "demo_window_report.txt"

import os

# CapnoBase External Validation Dataset Paths
# Note: Raw CapnoBase data is governed by terms of use prohibiting redistribution.
# It is stored outside the version-controlled project tree (e.g. ../capnobase_data)
# or in a gitignored local directory (e.g. data/capnobase), configurable via CAPNOBASE_DIR.
_capnobase_env = os.environ.get("CAPNOBASE_DIR")
_default_capnobase_dir = Path(_capnobase_env) if _capnobase_env else (PROJECT_ROOT.parent / "capnobase_data")
if not _default_capnobase_dir.exists():
    for _candidate in [
        PROJECT_ROOT.parent / "capnobase_data",
        PROJECT_ROOT.parent / "capnobase_data" / "data",
        PROJECT_ROOT / "data" / "capnobase",
        PROJECT_ROOT / "data" / "capnobase" / "data",
        PROJECT_ROOT / "capnobase_data",
        PROJECT_ROOT / "capnobase_data" / "data",
    ]:
        if _candidate.exists():
            _default_capnobase_dir = _candidate
            break
CAPNOBASE_DIR = _default_capnobase_dir
CAPNOBASE_CACHE_FILE = CACHE_DIR / "capnobase_features.npz"
CAPNOBASE_TOTAL_CASES = 42

# ==============================================================================
# 2. Hardware Constraints (Intel i3 / 8GB RAM / No GPU)
# ==============================================================================
CPU_THREADS = 2                 # Keep threads restricted to prevent CPU starvation
TREE_METHOD = "hist"            # Memory & compute-efficient histogram gradient boosting
ENABLE_GARBAGE_COLLECTION = True# Aggressive RAM reclamation after generator yields
CHUNK_SIZE_WINDOWS = 500        # Chunk processing batch size for streaming

# ==============================================================================
# 3. Phase 1: Biosignal Acquisition & Preprocessing Settings
# ==============================================================================
PHYSIONET_BASE_URL = "https://physionet.org/files/bidmc/1.0.0/"
TOTAL_PATIENTS = 53
PATIENT_IDS = [f"bidmc{i:02d}" for i in range(1, TOTAL_PATIENTS + 1)]

FS = 125.0                      # BIDMC sampling frequency (Hz)
FILTER_ORDER = 4                # 4th-order Butterworth bandpass filter
FILTER_LOW_HZ = 0.10            # 0.10 Hz = 6 breaths per minute
FILTER_HIGH_HZ = 0.60           # 0.60 Hz = 36 breaths per minute
FILTER_FORM = "sos"             # Second-Order Sections (SOS) for numerical stability

# ==============================================================================
# 4. Phase 2: Segmentation & Feature Extraction Settings
# ==============================================================================
# Validation-chosen window duration across all 5 folds (Phase 1.5 result: 60s is optimal)
WINDOW_SEC = 60.0               # 60.0 seconds duration per segmentation window
WINDOW_SAMPLES = int(FS * WINDOW_SEC)  # 7,500 samples per window
STRIDE_SEC = 2.0                # 2.0 seconds step size
STRIDE_SAMPLES = int(FS * STRIDE_SEC)  # 250 samples per stride

DWT_WAVELET = "db4"             # Daubechies-4 orthogonal wavelet
DWT_LEVEL = 4                   # 4-level decomposition [cA4, cD4, cD3, cD2, cD1]

# Ground Truth Validation Range (physiological breath rate limits)
MIN_VALID_RR = 6.0              # breaths per minute
MAX_VALID_RR = 45.0             # breaths per minute

TRAIN_SPLIT_RATIO = 0.80        # Legacy 80% train, 20% test (superseded by GroupKFold)
RANDOM_SEED = 42

# Patient-Grouped Cross-Validation (disjoint splits by patient)
N_FOLDS = 5                     # 5-fold GroupKFold by patient
VAL_PATIENT_RATIO = 0.20        # ~20% of outer training patients reserved for validation
CAL_PATIENT_RATIO = 0.20        # ~20% of outer training patients reserved for conformal calibration

# ==============================================================================
# 5. Phase 3: Baseline Regression Model Settings
# ==============================================================================
BASELINE_XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.08,
    "n_estimators": 100,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "tree_method": TREE_METHOD,
    "n_jobs": CPU_THREADS,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}

# ==============================================================================
# 6. Phase 4: Conformal Prediction & Clinical Safety Settings
# ==============================================================================
CONFORMAL_CONFIDENCE_LEVEL = 0.90  # 90% target coverage rate (alpha = 0.10)
CALIBRATION_SPLIT_RATIO = 0.25     # 25% of training set reserved for calibration

# Clinical Safety Abstention Rule Threshold
ABSTENTION_THRESHOLD_BPM = 6.00    # Discard prediction if interval width > 6.0 bpm
ABSTENTION_FLAG_LABEL = "Safe Abstention Triggered (Human Review Needed)"

# Motion Artifact Injection Settings
ARTIFACT_NOISE_SCALE = 2.5         # Multiplier on feature standard deviations
ARTIFACT_SHIFT_BOUND = 3.0         # Baseline shift range [-3.0, 3.0] * std
ARTIFACT_SPIKE_PROBABILITY = 0.40  # Probability of transient motion spike burst
