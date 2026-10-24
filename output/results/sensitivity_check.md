# Hyperparameter Sensitivity Check

## 1. Experimental Overview
- **Cohort & Splitting**: PhysioNet BIDMC (53 ICU patients), 5-fold patient-disjoint cross-validation (Seed 42).
- **Evaluation Protocol**: Retrained XGBoost across 5 patient-disjoint folds varying one hyperparameter at a time against the active pipeline default (`max_depth=3`, `learning_rate=0.08`).
- **Hardware & Determinism**: CPU-only execution, `n_jobs=1`, fixed random seeds.

## 2. Sensitivity Benchmark Table

| Configuration | Category | max_depth | learning_rate | Mean Test MAE (bpm) | Std (bpm) | Delta vs Default (bpm) | Per-Fold MAEs (bpm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Default (max_depth=3, lr=0.08)** | Baseline Default | 3 | 0.08 | **1.84** | 0.52 | 0.00 (ref) | [2.33, 2.31, 2.14, 1.08, 1.36] |
| Shallow Trees (max_depth=2) | max_depth variation | 2 | 0.08 | 1.82 | 0.50 | -0.02 | [2.22, 2.37, 2.05, 1.07, 1.40] |
| Deeper Trees (max_depth=4) | max_depth variation | 4 | 0.08 | 1.94 | 0.56 | +0.10 | [2.36, 2.58, 2.23, 1.19, 1.35] |
| Lower Learning Rate (lr=0.05) | learning_rate variation | 3 | 0.05 | 1.81 | 0.50 | -0.03 | [2.25, 2.20, 2.19, 1.08, 1.33] |
| Higher Learning Rate (lr=0.12) | learning_rate variation | 3 | 0.12 | 1.93 | 0.55 | +0.08 | [2.36, 2.45, 2.27, 1.14, 1.39] |

## 3. Plain Robustness Verdict
> **Verdict**: Across all four nearby hyperparameter perturbations (tree depth across {2, 3, 4} and learning rate across {0.05, 0.08, 0.12}), mean test MAE moves by at most 0.10 bpm (ranging from 1.81 to 1.94 bpm around the default 1.84 bpm), confirming that predictive accuracy is robust across reasonable configurations and clearly not an artifact of one lucky hyperparameter setting.
