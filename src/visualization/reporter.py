"""
Phase 5: Clinical Summary Reporter Module
=========================================
Generates and writes the illustrative single-demo-window report with baseline MAE,
empirical conformal coverage, and safe abstention status under motion corruption.
All statements are auto-computed from provided metrics; no hard-coded narrative.
"""

from pathlib import Path
from typing import Dict, Any

from config.settings import DEMO_REPORT_FILE, ABSTENTION_THRESHOLD_BPM


def generate_final_summary_report(
    clean_eval: Dict[str, Any],
    noisy_eval: Dict[str, Any],
    baseline_mae: float,
    model_name: str = "XGBoost (hist, CPU)",
    export_path: Path = DEMO_REPORT_FILE
) -> str:
    """
    Constructs a data-driven demonstration window report (N=1) and writes it to disk.
    Reflects actual single-window inference, coverage, and abstention status without
    fabricating cohort-level percentages.
    """
    lines = [
        "=" * 80,
        "SINGLE-WINDOW CLINICAL DEMONSTRATION REPORT (N=1 DEMO WINDOW)",
        "=" * 80,
        "",
        f"Model: {model_name}",
        f"Clean Test MAE (Cohort 5-Fold CV): {baseline_mae:.4f} bpm",
    ]

    # Clean window section
    lines.extend([
        "",
        "--- 1. Clean PPG Demonstration Window ---",
    ])
    if "predicted_rr" in clean_eval:
        lines.append(f"Clean Predicted RR:          {clean_eval['predicted_rr']:.2f} bpm")
    if "lower" in clean_eval and "upper" in clean_eval:
        w = clean_eval.get("width", clean_eval["upper"] - clean_eval["lower"])
        lines.append(f"Clean 90% Prediction Band:   [{clean_eval['lower']:.2f}, {clean_eval['upper']:.2f}] bpm (Width: {w:.2f} bpm)")
    if "true_rr" in clean_eval:
        lines.append(f"Ground Truth RR:             {clean_eval['true_rr']:.2f} bpm")
    if "is_covered" in clean_eval:
        cov_status = "True (Ground truth within interval)" if clean_eval["is_covered"] else "False (Miscoverage)"
        lines.append(f"Clean Window Covered (N=1):  {cov_status}")
    if "is_abstained" in clean_eval:
        abs_status = "True (Abstention triggered)" if clean_eval["is_abstained"] else "False (Prediction accepted)"
        lines.append(f"Clean Abstention (N=1):      {abs_status}")

    # Noisy window section
    lines.extend([
        "",
        "--- 2. Motion-Corrupted PPG Demonstration Window (Moderate Artifact) ---",
    ])
    if "predicted_rr" in noisy_eval:
        lines.append(f"Noisy Predicted RR:          {noisy_eval['predicted_rr']:.2f} bpm")
    if "lower" in noisy_eval and "upper" in noisy_eval:
        w = noisy_eval.get("width", noisy_eval["upper"] - noisy_eval["lower"])
        lines.append(f"Noisy 90% Prediction Band:   [{noisy_eval['lower']:.2f}, {noisy_eval['upper']:.2f}] bpm (Width: {w:.2f} bpm)")
    if "true_rr" in noisy_eval:
        lines.append(f"Ground Truth RR:             {noisy_eval['true_rr']:.2f} bpm")
    if "is_covered" in noisy_eval:
        cov_status = "True (Ground truth within interval)" if noisy_eval["is_covered"] else "False (Miscoverage)"
        lines.append(f"Noisy Window Covered (N=1):  {cov_status}")
    if "is_abstained" in noisy_eval:
        abs_status = "True (Abstention triggered)" if noisy_eval["is_abstained"] else "False (Prediction accepted / Resilient)"
        lines.append(f"Noisy Abstention (N=1):      {abs_status}")

    # Cohort summary if available
    lines.extend([
        "",
        "--- 3. Cohort Reference Metrics (5-Fold Cross-Validation, 53 Patients) ---",
    ])
    cohort_cov = clean_eval.get("cohort_coverage", clean_eval.get("empirical_coverage", None))
    if cohort_cov is not None:
        lines.append(f"Clean Cohort Empirical Coverage: {cohort_cov:.2%}")
    if "retained_mae" in clean_eval:
        lines.append(f"Clean Cohort Retained MAE:       {clean_eval['retained_mae']:.4f} bpm")

    lines.extend(["", "=" * 80])

    report = "\n".join(lines)

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report
