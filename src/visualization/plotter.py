"""
Phase 5: High-Resolution Comparative Visualization Module
=========================================================
Generates a publication-grade, 300 DPI dual-subplot Matplotlib/Seaborn figure
contrasting clean confident inference against safe abstention under motion artifacts.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config.settings import FIGURE_EXPORT_PATH, WINDOW_SEC, FS, ABSTENTION_THRESHOLD_BPM
from src.visualization.architecture_plotter import render_pipeline_architecture_figure


def render_comparative_triage_figure(
    t_seconds: np.ndarray,
    clean_signal: np.ndarray,
    clean_pred: float,
    clean_lower: float,
    clean_upper: float,
    clean_true: float,
    noisy_signal: np.ndarray,
    noisy_pred: float,
    noisy_lower: float,
    noisy_upper: float,
    noisy_true: float,
    spike_interval: tuple = None,
    threshold_bpm: float = ABSTENTION_THRESHOLD_BPM,
    save_path: Path = FIGURE_EXPORT_PATH
) -> Path:
    """
    Renders and exports the juxtaposed Subplot A & Subplot B comparative graphic.
    Panel styling, banners, cards, and legend for the corrupted window are strictly
    conditional on computed interval width vs threshold_bpm.
    """
    clean_width = clean_upper - clean_lower
    noisy_width = noisy_upper - noisy_lower
    is_abstained = (noisy_width > threshold_bpm)
    half_thresh = threshold_bpm / 2.0

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#2c3e50"
    plt.rcParams["axes.linewidth"] = 1.2

    fig = plt.figure(figsize=(19, 9.5), dpi=300, facecolor="#F8FAFC")
    gs = GridSpec(1, 2, figure=fig, wspace=0.22, left=0.06, right=0.96, top=0.86, bottom=0.12)

    # -------------------------------------------------------------------------
    # SUBPLOT A: Clean PPG Signal — Confident Clinical Prediction
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0], facecolor="#FFFFFF")
    ax_a.plot(t_seconds, clean_signal, color="#0284C7", linewidth=2.0, label="Filtered PPG (0.10–0.60 Hz)")
    ax_a.set_title("Subplot A: Clean PPG Signal — Confident Clinical Prediction",
                   fontsize=14, fontweight="bold", color="#0F172A", pad=14)
    ax_a.set_xlabel("Time (seconds)", fontsize=12, fontweight="semibold", color="#334155")
    ax_a.set_ylabel("PPG Amplitude (arbitrary units)", fontsize=12, fontweight="semibold", color="#334155")
    ax_a.set_xlim(0, WINDOW_SEC)
    ax_a.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")

    # Status banner for Subplot A
    ax_a.text(0.04, 0.94, "● STATUS: PREDICTION ACCEPTED",
              transform=ax_a.transAxes, fontsize=11, fontweight="bold",
              color="#15803D", verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#DCFCE7", edgecolor="#22C55E", linewidth=1.5, alpha=0.95))

    # Metric summary card
    card_clean = (
        f"┌─ Conformal Inference (90% Conf.) ─┐\n"
        f"│ Ground Truth RR:    {clean_true:5.1f} bpm      │\n"
        f"│ Predicted RR (XGB): {clean_pred:5.1f} bpm      │\n"
        f"│ Prediction Interval:[{clean_lower:4.1f}, {clean_upper:4.1f}] bpm │\n"
        f"│ Interval Width:     {clean_width:5.2f} bpm      │\n"
        f"│ Safety Threshold:   < {threshold_bpm:4.2f} bpm      │\n"
        f"└────────────────────────────────────┘"
    )
    ax_a.text(0.96, 0.94, card_clean, transform=ax_a.transAxes, fontsize=9.5,
              verticalalignment="top", horizontalalignment="right", family="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F9FF", edgecolor="#0284C7", linewidth=1.2, alpha=0.95))

    # Secondary twin axis for respiratory rate
    ax_a2 = ax_a.twinx()
    ax_a2.axhspan(clean_lower, clean_upper, color="#22C55E", alpha=0.22,
                  label=f"Tight 90% PI [{clean_lower:.1f}, {clean_upper:.1f}] bpm (Width={clean_width:.1f})")
    ax_a2.axhline(clean_pred, color="#16A34A", linestyle="-", linewidth=2.5,
                  label=f"Predicted RR: {clean_pred:.1f} bpm")
    ax_a2.axhline(clean_true, color="#EA580C", linestyle="--", linewidth=2.0,
                  label=f"Ground Truth RR: {clean_true:.1f} bpm")
    ax_a2.set_ylabel("Estimated Respiratory Rate (breaths/min)", fontsize=12, fontweight="semibold", color="#16A34A")
    ax_a2.tick_params(axis="y", labelcolor="#16A34A")
    ax_a2.set_ylim(clean_pred - 10, clean_pred + 10)
    ax_a2.legend(loc="lower right", fontsize=9.5, framealpha=0.92, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # SUBPLOT B: Motion-Corrupted PPG — Truly Conditional Triage
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1], facecolor="#FFFFFF")
    ax_b.plot(t_seconds, noisy_signal, color="#DC2626", linewidth=1.8, label="Corrupted PPG (Artifact Injected)")
    
    if spike_interval:
        ax_b.axvspan(t_seconds[spike_interval[0]], t_seconds[spike_interval[1]], color="#FEE2E2", alpha=0.55,
                     label="Simulated Motion Artifact Spike")

    # Dynamic titles, banners, cards, and styles based on actual computed width
    if is_abstained:
        title_b = "Subplot B: Motion-Corrupted PPG — Safe Abstention Triggered"
        title_color = "#991B1B"
        banner_text = "⚠ SAFE ABSTENTION TRIGGERED\n   [Human Review Needed]"
        banner_color = "#991B1B"
        banner_face = "#FEE2E2"
        banner_edge = "#DC2626"
        action_text = "ABSTAIN / ALERT "
        thresh_symbol = ">"
        card_face = "#FFF1F2"
        card_edge = "#E11D48"
        span_color = "#EF4444"
        span_alpha = 0.20
        span_label = f"Exploding 90% PI [{noisy_lower:.1f}, {noisy_upper:.1f}] bpm (Width={noisy_width:.1f} > {threshold_bpm:.1f})"
        line_color = "#DC2626"
        bound_color = "#B91C1C"
        bound_label = f"{threshold_bpm:.1f} bpm Safety Cutoff Bound"
    else:
        title_b = "Subplot B: Motion-Corrupted PPG — Resilient Clinical Prediction"
        title_color = "#0F172A"
        banner_text = "● STATUS: PREDICTION ACCEPTED\n   [Artifact-Resilient Inference]"
        banner_color = "#15803D"
        banner_face = "#DCFCE7"
        banner_edge = "#22C55E"
        action_text = "ACCEPTED (RESILIENT)"
        thresh_symbol = "<"
        card_face = "#F0FDF4"
        card_edge = "#16A34A"
        span_color = "#22C55E"
        span_alpha = 0.22
        span_label = f"Controlled 90% PI [{noisy_lower:.1f}, {noisy_upper:.1f}] bpm (Width={noisy_width:.1f} <= {threshold_bpm:.1f})"
        line_color = "#16A34A"
        bound_color = "#15803D"
        bound_label = f"{threshold_bpm:.1f} bpm Safety Bound (Satisfied)"

    ax_b.set_title(title_b, fontsize=14, fontweight="bold", color=title_color, pad=14)
    ax_b.set_xlabel("Time (seconds)", fontsize=12, fontweight="semibold", color="#334155")
    ax_b.set_ylabel("PPG Amplitude (arbitrary units)", fontsize=12, fontweight="semibold", color="#334155")
    ax_b.set_xlim(0, WINDOW_SEC)
    ax_b.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")

    # Status banner for Subplot B
    ax_b.text(0.04, 0.94, banner_text,
              transform=ax_b.transAxes, fontsize=11, fontweight="bold",
              color=banner_color, verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.45", facecolor=banner_face, edgecolor=banner_edge, linewidth=1.5 if not is_abstained else 2.0, alpha=0.98))

    # Metric summary card
    card_noisy = (
        f"┌─ Conformal Inference (90% Conf.) ─┐\n"
        f"│ Ground Truth RR:    {noisy_true:5.1f} bpm      │\n"
        f"│ Predicted RR (XGB): {noisy_pred:5.1f} bpm      │\n"
        f"│ Prediction Interval:[{noisy_lower:4.1f}, {noisy_upper:4.1f}] bpm │\n"
        f"│ Interval Width:     {noisy_width:5.2f} bpm      │\n"
        f"│ Safety Threshold:   {thresh_symbol} {threshold_bpm:4.2f} bpm      │\n"
        f"│ ACTION:             {action_text:<15} │\n"
        f"└────────────────────────────────────┘"
    )
    ax_b.text(0.96, 0.94, card_noisy, transform=ax_b.transAxes, fontsize=9.5,
              verticalalignment="top", horizontalalignment="right", family="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor=card_face, edgecolor=card_edge, linewidth=1.2 if not is_abstained else 1.5, alpha=0.95))

    # Secondary twin axis for respiratory rate
    ax_b2 = ax_b.twinx()
    ax_b2.axhspan(noisy_lower, noisy_upper, color=span_color, alpha=span_alpha, label=span_label)
    ax_b2.axhline(noisy_pred, color=line_color, linestyle="-", linewidth=2.5,
                  label=f"Predicted RR: {noisy_pred:.1f} bpm")
    ax_b2.axhline(noisy_true, color="#EA580C", linestyle="--", linewidth=2.0,
                  label=f"Ground Truth RR: {noisy_true:.1f} bpm")
    # Safety cutoff lines
    ax_b2.axhline(noisy_pred + half_thresh, color=bound_color, linestyle=":", linewidth=2.0, alpha=0.85)
    ax_b2.axhline(noisy_pred - half_thresh, color=bound_color, linestyle=":", linewidth=2.0, alpha=0.85,
                  label=bound_label)
    ax_b2.set_ylabel("Estimated Respiratory Rate (breaths/min)", fontsize=12, fontweight="semibold", color=line_color)
    ax_b2.tick_params(axis="y", labelcolor=line_color)
    ax_b2.set_ylim(noisy_pred - 15, noisy_pred + 15)
    ax_b2.legend(loc="lower right", fontsize=9.5, framealpha=0.92, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # Global Title
    fig.suptitle("Artifact-Resilient Cardiorespiratory Triage using PPG & Conformal Prediction\n"
                 "Real-time Uncertainty Quantification with Safe Abstention for ICU Patient Monitoring",
                 fontsize=16, fontweight="bold", color="#0F172A", y=0.97)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return save_path
