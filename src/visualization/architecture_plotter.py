"""
Pipeline Architecture Overview Visualization Module
===================================================
Renders a publication-grade, 300 DPI schematic diagram (boxes and arrows)
of the end-to-end cardiorespiratory triage pipeline:
  Raw PPG (125 Hz)
  -> Butterworth bandpass filter (0.10-0.60 Hz)
  -> 60 s windows / 2 s stride
  -> 28-feature extraction (PPG spectral, RIAV/RIIV/RIFV modulations, morphology/stats)
  -> XGBoost point prediction
  -> Conformal calibration (active residual_normalised, tested jackknife+/CV+)
  -> Width + SQI combination score
  -> Accept / abstain triage

Consistent with the visual style and 300 DPI standards of the BDA project.
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config.settings import (
    FIGURES_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    PROJECT_ROOT,
    FS,
    WINDOW_SEC,
    STRIDE_SEC,
    FILTER_LOW_HZ,
    FILTER_HIGH_HZ,
    TOTAL_PATIENTS,
    N_FOLDS,
    CONFORMAL_CONFIDENCE_LEVEL,
    ABSTENTION_THRESHOLD_BPM,
)

DEFAULT_ARCH_FIGURE_PATH = FIGURES_DIR / "pipeline_architecture.png"


def _get_coverage_sd_diff_ci() -> Tuple[str, str]:
    """
    Reads the authoritative CV+ vs. difficulty-normalised coverage-spread SD difference
    bootstrap CI directly from output/results/phase7_summary.md.

    Returns:
        Tuple of (ci_string, includes_zero_phrase), e.g. ("[-4.7%, +1.4%]", "includes zero")
    """
    summary_path = RESULTS_DIR / "phase7_summary.md"
    default_ci = "[-4.7%, +1.4%]"
    default_phrase = "includes zero"
    if summary_path.exists():
        try:
            content = summary_path.read_text(encoding="utf-8")
            for raw_line in content.splitlines():
                line = raw_line.strip()
                if line.startswith("- SD Diff:") or line.startswith("  - SD Diff:"):
                    m = re.search(r'95%\s*CI\s*(\[[^\]]+\]),\s*(includes|excludes)\s*zero', line)
                    if m:
                        return m.group(1), f"{m.group(2)} zero"
            m_verdict = re.search(
                r'Bootstrap 95%\s*CI on the SD difference[^:]*:\s*(\[[^\]]+\])\s*\(([^)]+)\)',
                content
            )
            if m_verdict:
                return m_verdict.group(1), m_verdict.group(2)
        except Exception:
            pass
    return default_ci, default_phrase


def _draw_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    box_color: str,
    border_color: str,
    corner_radius: float = 0.012,
    border_width: float = 1.4,
    shadow: bool = True
):
    """Draws a rounded card container with optional subtle drop shadow."""
    if shadow:
        shadow_box = FancyBboxPatch(
            (x + 0.0025, y - 0.0025),
            w,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={corner_radius}",
            facecolor="#0F172A",
            edgecolor="none",
            alpha=0.05,
            zorder=1
        )
        ax.add_patch(shadow_box)

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={corner_radius}",
        facecolor=box_color,
        edgecolor=border_color,
        linewidth=border_width,
        zorder=2
    )
    ax.add_patch(card)
    return card


def _draw_arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    label: str = "",
    label_offset: Tuple[float, float] = (0.0, 0.012),
    color: str = "#475569",
    arrow_style: str = "-|>",
    mutation_scale: float = 14.0,
    line_width: float = 1.6,
    zorder: int = 4,
    label_fontsize: float = 8.0,
    label_color: str = "#1E293B",
    label_bbox: bool = True,
    connectionstyle: Optional[str] = None
):
    """Draws a crisp directed arrow with an optional label badge."""
    patch_kwargs = dict(
        arrowstyle=arrow_style,
        color=color,
        linewidth=line_width,
        mutation_scale=mutation_scale,
        zorder=zorder
    )
    if connectionstyle:
        patch_kwargs["connectionstyle"] = connectionstyle

    arrow = FancyArrowPatch(start, end, **patch_kwargs)
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2.0 + label_offset[0]
        mid_y = (start[1] + end[1]) / 2.0 + label_offset[1]
        bbox_dict = None
        if label_bbox:
            bbox_dict = dict(
                boxstyle="round,pad=0.22",
                facecolor="#FFFFFF",
                edgecolor="#CBD5E1",
                linewidth=0.8,
                alpha=0.96
            )
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold",
            color=label_color,
            bbox=bbox_dict,
            zorder=zorder + 1
        )


def render_pipeline_architecture_figure(
    save_path: Optional[Path] = None,
    secondary_save_path: Optional[Path] = None,
    dpi: int = 300
) -> Path:
    """
    Renders and exports the high-resolution, publication-grade schematic architecture
    overview figure for the end-to-end cardiorespiratory triage pipeline.

    Stages depicted:
      1. Raw PPG Acquisition (125 Hz, PhysioNet BIDMC ICU Cohort, 53 Patients)
      2. Butterworth Bandpass Filtering (0.10-0.60 Hz, 4th-order SOS)
      3. Sliding-Window Segmentation (60 s windows / 2 s stride, 11,026 windows)
      4. 28-Feature Extraction (PPG spectral, RIAV/RIIV/RIFV pulse modulations, morphology/stats)
      5. XGBoost Point Prediction (Histogram tree method, CPU-only Intel i3/8GB RAM)
      6. Split Conformal Calibration (Active: residual_normalised; Tested: Jackknife+/CV+)
      7. Width + SQI Combination Score (Logistic regression on validation clean windows)
      8. Accept / Abstain Clinical Triage (Safe abstention vs confident clinical prediction)

    Parameters:
        save_path: Primary destination path (defaults to output/pipeline_architecture.png).
        secondary_save_path: Optional secondary destination path (defaults to repo root).
        dpi: Export resolution (defaults to 300 DPI).

    Returns:
        Path to the primary exported PNG figure.
    """
    if save_path is None:
        save_path = DEFAULT_ARCH_FIGURE_PATH
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Style configuration matching project standards
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#2c3e50"
    plt.rcParams["axes.linewidth"] = 1.2

    # Canvas dimensions: 18.0 x 12.2 inches for maximum readability at 300 DPI
    fig = plt.figure(figsize=(18.0, 12.2), dpi=dpi, facecolor="#F8FAFC")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#F8FAFC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # =========================================================================
    # HEADER BANNER & SYSTEM METADATA (y: 0.925 to 0.985)
    # =========================================================================
    _draw_card(ax, 0.030, 0.925, 0.940, 0.060, "#FFFFFF", "#CBD5E1", corner_radius=0.008, border_width=1.0)

    # Title & Subtitle on left
    ax.text(
        0.045, 0.963,
        "Figure 1: End-to-End Cardiorespiratory Monitoring & Selective Triage Architecture",
        fontsize=15.5, fontweight="bold", color="#0F172A", va="center"
    )
    ax.text(
        0.045, 0.940,
        "Continuous Photoplethysmography (PPG) to Respiratory Rate Pipeline with Conformal Uncertainty Quantification and Clinical Abstention",
        fontsize=10.0, color="#475569", va="center"
    )

    # Clean 2x2 grid of metadata pills on the right (x: 0.61 to 0.955)
    row1_pills = [
        (0.610, 0.963, "Cohort: BIDMC (53 Patients, 125 Hz)", "#0284C7", "#E0F2FE", "#BAE6FD"),
        (0.795, 0.963, "CV: 5-Fold Patient-Disjoint", "#7C3AED", "#EDE9FE", "#DDD6FE"),
    ]
    row2_pills = [
        (0.610, 0.940, "Hardware: CPU-Only (Intel i3 / 8GB)", "#D97706", "#FEF3C7", "#FDE68A"),
        (0.795, 0.940, "Target Coverage: 90% (α = 0.10)", "#16A34A", "#DCFCE7", "#BBF7D0"),
    ]
    for x_pos, y_pos, text, text_col, bg_col, bdr_col in row1_pills + row2_pills:
        ax.text(
            x_pos, y_pos, text,
            fontsize=8.2, fontweight="bold", color=text_col, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=bg_col, edgecolor=bdr_col, linewidth=1.0)
        )

    # =========================================================================
    # TIER 1: BIOSIGNAL INGESTION & CONDITIONING (y: 0.748 to 0.900)
    # =========================================================================
    ax.text(
        0.032, 0.905,
        "TIER 1: BIOSIGNAL ACQUISITION, FILTERING & SEGMENTATION",
        fontsize=9.2, fontweight="bold", color="#0284C7", va="bottom"
    )

    # Box 1: Raw PPG Ingestion
    _draw_card(ax, 0.030, 0.748, 0.285, 0.150, "#FFFFFF", "#0284C7", corner_radius=0.010, border_width=1.5)
    ax.text(
        0.042, 0.876, "STAGE 1",
        fontsize=7.5, fontweight="bold", color="#FFFFFF", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0284C7", edgecolor="none")
    )
    ax.text(0.082, 0.876, "Raw PPG Acquisition", fontsize=11.2, fontweight="bold", color="#0F172A", va="center")

    stage1_bullets = [
        ("• Cohort:", "PhysioNet BIDMC (53 ICU adult patients, 8-min records)"),
        ("• Sampling Rate:", "fs = 125 Hz continuous fingertip pulse oximeter"),
        ("• Reference GT:", "1 Hz synchronous impedance pneumography RR"),
        ("• Patient Cohort:", "53/53 patients retained (includes bidmc13 flatline)"),
    ]
    cur_y = 0.850
    for label, val in stage1_bullets:
        ax.text(0.042, cur_y, f"{label} {val}", fontsize=8.0, color="#334155", va="center")
        cur_y -= 0.019

    # Mini raw PPG waveform strip at bottom of Card 1 (dedicated clear strip)
    t_mini = np.linspace(0, 4 * np.pi, 140)
    wave_raw = 0.55 * np.sin(t_mini * 2.8) + 0.35 * np.sin(t_mini * 0.4) + 0.08 * np.sin(t_mini * 11)
    wave_x = np.linspace(0.045, 0.295, len(wave_raw))
    wave_y = 0.764 + wave_raw * 0.011
    ax.plot(wave_x, wave_y, color="#0284C7", linewidth=1.2, alpha=0.90, zorder=3)
    ax.text(0.045, 0.776, "Raw PPG Waveform (Cardiac Pulses + Baseline Wander):", fontsize=7.0, color="#64748B", va="bottom")

    # Arrow 1 -> 2
    _draw_arrow(
        ax, (0.315, 0.825), (0.355, 0.825),
        label="Raw PPG\n(125 Hz)",
        label_offset=(0.0, 0.0),
        color="#0284C7"
    )

    # Box 2: Butterworth Bandpass Filter
    _draw_card(ax, 0.355, 0.748, 0.285, 0.150, "#FFFFFF", "#0284C7", corner_radius=0.010, border_width=1.5)
    ax.text(
        0.367, 0.876, "STAGE 2",
        fontsize=7.5, fontweight="bold", color="#FFFFFF", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0284C7", edgecolor="none")
    )
    ax.text(0.407, 0.876, "Butterworth Bandpass Filter", fontsize=11.2, fontweight="bold", color="#0F172A", va="center")

    stage2_bullets = [
        ("• Passband:", "0.10 - 0.60 Hz (6 - 36 breaths/min physiological band)"),
        ("• Filter Form:", "4th-order zero-phase Second-Order Sections (SOS)"),
        ("• Drift Removal:", "Suppresses baseline wander (<0.10 Hz) & pulse (>0.60 Hz)"),
        ("• Wideband Filter:", "Parallel 0.10 - 5.0 Hz filter for systolic peak detection"),
    ]
    cur_y = 0.850
    for label, val in stage2_bullets:
        ax.text(0.367, cur_y, f"{label} {val}", fontsize=8.0, color="#334155", va="center")
        cur_y -= 0.019

    # Mini filtered wave strip at bottom of Card 2
    wave_filt = 0.65 * np.sin(t_mini * 0.4)
    wave_x2 = np.linspace(0.370, 0.620, len(wave_filt))
    wave_y2 = 0.764 + wave_filt * 0.011
    ax.plot(wave_x2, wave_y2, color="#0284C7", linewidth=1.4, alpha=0.90, zorder=3)
    ax.text(0.370, 0.776, "Filtered Respiratory PPG (Clean In-Band Oscillation):", fontsize=7.0, color="#64748B", va="bottom")

    # Arrow 2 -> 3
    _draw_arrow(
        ax, (0.640, 0.825), (0.680, 0.825),
        label="Bandpassed\nWaveform",
        label_offset=(0.0, 0.0),
        color="#0284C7"
    )

    # Box 3: Window Segmentation
    _draw_card(ax, 0.680, 0.748, 0.290, 0.150, "#FFFFFF", "#0284C7", corner_radius=0.010, border_width=1.5)
    ax.text(
        0.692, 0.876, "STAGE 3",
        fontsize=7.5, fontweight="bold", color="#FFFFFF", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0284C7", edgecolor="none")
    )
    ax.text(0.732, 0.876, "60 s Window Segmentation", fontsize=11.2, fontweight="bold", color="#0F172A", va="center")

    stage3_bullets = [
        ("• Window Duration:", "60.0 s (7,500 samples) | Stride: 2.0 s (250 samples)"),
        ("• Window Selection:", "Selected over 6s/15s/30s via validation grid search"),
        ("• Total Dataset:", "11,026 clean windows across 53 patients (mean 208/pt)"),
        ("• Patient Partition:", "Strict 5-Fold GroupKFold (Train/Val/Cal/Test disjoint)"),
    ]
    cur_y = 0.850
    for label, val in stage3_bullets:
        ax.text(0.692, cur_y, f"{label} {val}", fontsize=8.0, color="#334155", va="center")
        cur_y -= 0.019

    # Clean partition summary pill at bottom of Card 3
    ax.text(
        0.825, 0.764,
        "Splits: Train ~64% | Val ~16% | Cal ~20% outer | Test ~20% held-out",
        fontsize=7.2, fontweight="bold", color="#0369A1", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E0F2FE", edgecolor="#BAE6FD", linewidth=0.8)
    )

    # Downward arrow from Tier 1 to Tier 2
    _draw_arrow(
        ax, (0.825, 0.748), (0.825, 0.725),
        label="60 s Windowed PPG Waveforms (7,500 samples/win)",
        label_offset=(-0.16, 0.0),
        color="#0284C7",
        line_width=2.0
    )

    # =========================================================================
    # TIER 2: 28-FEATURE MULTI-DOMAIN EXTRACTION (y: 0.445 to 0.722)
    # =========================================================================
    _draw_card(ax, 0.030, 0.445, 0.940, 0.277, "#FFFFFF", "#6366F1", corner_radius=0.012, border_width=1.6)

    # Header for Stage 4
    ax.text(
        0.045, 0.702, "STAGE 4",
        fontsize=8.0, fontweight="bold", color="#FFFFFF", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#6366F1", edgecolor="none")
    )
    ax.text(
        0.092, 0.702,
        "28-Feature Multi-Domain Extraction (3 Clinically Grounded Groups)",
        fontsize=12.5, fontweight="bold", color="#0F172A", va="center"
    )
    ax.text(
        0.585, 0.702,
        "Physiologically Grounded Modulation Analysis • No Black-Box Wavelets • No Obsolete Proxies",
        fontsize=8.6, fontstyle="italic", color="#4F46E5", va="center"
    )

    # Sub-card 4A: Group 1 - PPG Spectral Descriptors (4 features)
    _draw_card(ax, 0.045, 0.462, 0.285, 0.218, "#F0F9FF", "#0284C7", corner_radius=0.008, border_width=1.2)
    ax.text(
        0.055, 0.658, "GROUP 1: PPG SPECTRAL",
        fontsize=8.8, fontweight="bold", color="#0284C7", va="center"
    )
    ax.text(
        0.315, 0.658, "4 Features",
        fontsize=7.8, fontweight="bold", color="#0369A1", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#BAE6FD", edgecolor="#0284C7", linewidth=0.8)
    )
    ax.text(
        0.055, 0.638, "FFT power spectral density in respiratory band [0.10, 0.60 Hz]:",
        fontsize=7.6, color="#475569", va="center"
    )
    group1_feats = [
        ("1. Dominant Frequency (bpm):", "Peak spectral density argmax frequency"),
        ("2. Spectral Centroid (bpm):", "Power-weighted center of frequency mass"),
        ("3. Spectral Bandwidth:", "Frequency variance / in-band power ratio"),
        ("4. Spectral Roll-off:", "85% cumulative in-band power boundary"),
    ]
    cur_y = 0.610
    for fname, fdesc in group1_feats:
        ax.text(0.055, cur_y, fname, fontsize=8.0, fontweight="bold", color="#0F172A", va="center")
        ax.text(0.055, cur_y - 0.015, fdesc, fontsize=7.4, color="#475569", va="center")
        cur_y -= 0.036

    # Sub-card 4B: Group 2 - Pulse-Modulation Spectral Descriptors (12 features)
    _draw_card(ax, 0.345, 0.462, 0.310, 0.218, "#EEF2FF", "#6366F1", corner_radius=0.008, border_width=1.2)
    ax.text(
        0.355, 0.658, "GROUP 2: PULSE-MODULATION SPECTRAL",
        fontsize=8.8, fontweight="bold", color="#4F46E5", va="center"
    )
    ax.text(
        0.640, 0.658, "12 Features",
        fontsize=7.8, fontweight="bold", color="#4338CA", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#DDD6FE", edgecolor="#6366F1", linewidth=0.8)
    )
    ax.text(
        0.355, 0.638, "4 spectral descriptors each across 3 pulse-modulation series:",
        fontsize=7.6, color="#475569", va="center"
    )
    group2_mods = [
        ("• RIAV (Amplitude Variation):", "Peak-to-peak amplitude modulation series\n  [Dominant Freq, Centroid, Bandwidth, Roll-off]"),
        ("• RIIV (Intensity Variation):", "Diastolic trough / baseline wander modulation\n  [Dominant Freq, Centroid, Bandwidth, Roll-off]"),
        ("• RIFV (Frequency Variation):", "Pulse-to-pulse duration interval series\n  [Dominant Freq, Centroid, Bandwidth, Roll-off]"),
    ]
    cur_y = 0.608
    for mname, mdesc in group2_mods:
        ax.text(0.355, cur_y, mname, fontsize=8.0, fontweight="bold", color="#0F172A", va="center")
        lines = mdesc.split("\n")
        ax.text(0.355, cur_y - 0.014, lines[0], fontsize=7.4, color="#475569", va="center")
        ax.text(0.355, cur_y - 0.027, lines[1], fontsize=7.2, color="#6366F1", fontweight="bold", va="center")
        cur_y -= 0.046

    # Sub-card 4C: Group 3 - Morphology & Statistics (12 features)
    _draw_card(ax, 0.670, 0.462, 0.285, 0.218, "#F0FDF4", "#16A34A", corner_radius=0.008, border_width=1.2)
    ax.text(
        0.680, 0.658, "GROUP 3: MORPHOLOGY / STATISTICS",
        fontsize=8.8, fontweight="bold", color="#15803D", va="center"
    )
    ax.text(
        0.940, 0.658, "12 Features",
        fontsize=7.8, fontweight="bold", color="#15803D", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=0.8)
    )
    ax.text(
        0.680, 0.638, "Cardiorespiratory fusion, moments, & waveform descriptors:",
        fontsize=7.6, color="#475569", va="center"
    )
    group3_feats = [
        ("• Multi-Modulation Fusion RR:", "fusion_rr (median of RIAV, RIIV, RIFV peak freqs)"),
        ("• Modulation Series Moments:", "riav_mean, riav_std, riiv_mean, riiv_std,\nrifv_mean, rifv_std (6 statistical moment features)"),
        ("• Pulse-Derived Heart Rate:", "mean_hr = 60.0 / rifv_mean (cardiac rate proxy)"),
        ("• Waveform Statistics (Time):", "sig_std, sig_ptp, sig_rms on respiratory wave"),
        ("• Within-Window Sharpness:", "sharpness = Peak Power / Mean Band Power"),
    ]
    cur_y = 0.612
    for fname, fdesc in group3_feats:
        ax.text(0.680, cur_y, fname, fontsize=7.8, fontweight="bold", color="#0F172A", va="center")
        if "\n" in fdesc:
            lines = fdesc.split("\n")
            ax.text(0.680, cur_y - 0.012, lines[0], fontsize=7.3, color="#475569", va="center")
            ax.text(0.680, cur_y - 0.024, lines[1], fontsize=7.3, color="#475569", va="center")
            cur_y -= 0.038
        else:
            ax.text(0.680, cur_y - 0.012, fdesc, fontsize=7.3, color="#475569", va="center")
            cur_y -= 0.026

    # Downward arrow from Tier 2 to Tier 3 (placed clearly to avoid overlapping section title)
    _draw_arrow(
        ax, (0.135, 0.445), (0.135, 0.395),
        label="Extracted Feature Vector x ∈ ℝ²⁸",
        label_offset=(0.0, 0.0),
        color="#6366F1",
        line_width=2.0
    )

    # =========================================================================
    # TIER 3: INFERENCE, CONFORMAL CALIBRATION, COMBINATION & TRIAGE (y: 0.080 to 0.395)
    # =========================================================================
    ax.text(
        0.310, 0.405,
        "TIER 3: INFERENCE, CONFORMAL UNCERTAINTY QUANTIFICATION & CLINICAL SELECTIVE TRIAGE",
        fontsize=9.2, fontweight="bold", color="#D97706", va="bottom"
    )

    # Box 5: XGBoost Point Prediction
    _draw_card(ax, 0.030, 0.080, 0.210, 0.315, "#FFFFFF", "#D97706", corner_radius=0.010, border_width=1.5)
    ax.text(
        0.040, 0.370, "STAGE 5",
        fontsize=7.5, fontweight="bold", color="#FFFFFF", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#D97706", edgecolor="none")
    )
    ax.text(0.078, 0.370, "XGBoost Point Prediction", fontsize=10.5, fontweight="bold", color="#0F172A", va="center")

    stage5_bullets = [
        ("• Architecture:", "Extreme Gradient Boosting (XGBoost Regressor)"),
        ("• Hardware Target:", "tree_method = 'hist', n_jobs = 2 (Intel i3 / 8GB)"),
        ("• Hyperparameters:", "max_depth = 3, lr = 0.08, n_estimators = 100"),
        ("• Patient Disjoint:", "5-Fold GroupKFold (zero cross-patient leakage)"),
        ("• Test MAE:", "1.812 ± 0.091 bpm (vs 2.128 bpm median baseline)"),
        ("• Bootstrap Test:", "Δ = -0.307 bpm [-0.505, -0.110] (p < 0.05)"),
    ]
    cur_y = 0.344
    for label, val in stage5_bullets:
        ax.text(0.040, cur_y, label, fontsize=8.0, fontweight="bold", color="#0F172A", va="center")
        ax.text(0.040, cur_y - 0.014, val, fontsize=7.4, color="#475569", va="center")
        cur_y -= 0.034

    # Point estimate badge inside Stage 5
    ax.text(
        0.135, 0.106,
        "Point Estimate: y_hat (bpm)",
        fontsize=8.0, fontweight="bold", color="#B45309", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FEF3C7", edgecolor="#F59E0B", linewidth=1.0)
    )

    # Arrow 5 -> 6
    _draw_arrow(
        ax, (0.240, 0.245), (0.275, 0.245),
        label="Point Pred\ny_hat",
        label_offset=(0.0, 0.0),
        color="#D97706"
    )

    # Box 6: Conformal Calibration
    _draw_card(ax, 0.275, 0.080, 0.245, 0.315, "#FFFFFF", "#7C3AED", corner_radius=0.010, border_width=1.5)
    ax.text(
        0.285, 0.370, "STAGE 6",
        fontsize=7.5, fontweight="bold", color="#FFFFFF", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#7C3AED", edgecolor="none")
    )
    ax.text(0.323, 0.370, "Conformal Calibration", fontsize=10.5, fontweight="bold", color="#0F172A", va="center")

    sd_diff_ci, sd_diff_phrase = _get_coverage_sd_diff_ci()
    stage6_content = [
        ("• Target Confidence:", "1 - α = 90.0% nominal coverage (α = 0.10)"),
        ("• Active Method:", "residual_normalised (difficulty-normalised)"),
        ("  Auxiliary Residual Est:", "s(x) = E[|y - y_hat| | x] (dispersion model)"),
        ("  Adaptive Interval:", "[y_hat - q_hat*s(x), y_hat + q_hat*s(x)]"),
        ("  Empirical Coverage:", "87.4% ± 9.7% over 25 outer test folds"),
        ("• Tested Alternative:", "Jackknife+ / CV+ (cross-fitting ensemble)"),
        ("  Multi-Seed Stability:", f"CV+ does NOT meaningfully narrow spread:\n  SD diff CI {sd_diff_ci} {sd_diff_phrase}"),
    ]
    cur_y = 0.344
    for label, val in stage6_content:
        is_active = "Active" in label
        is_alt = "Alternative" in label
        lbl_col = "#6D28D9" if is_active else ("#7C3AED" if is_alt else "#0F172A")
        ax.text(0.285, cur_y, label, fontsize=8.0, fontweight="bold", color=lbl_col, va="center")
        if "\n" in val:
            lines = val.split("\n")
            ax.text(0.285, cur_y - 0.014, lines[0], fontsize=7.3, color="#475569", va="center")
            ax.text(0.285, cur_y - 0.026, lines[1], fontsize=7.3, color="#6D28D9", fontweight="bold", va="center")
            cur_y -= 0.040
        else:
            ax.text(0.285, cur_y - 0.014, val, fontsize=7.3, color="#475569", va="center")
            cur_y -= 0.030

    # Conformal interval metric badge inside Stage 6
    ax.text(
        0.397, 0.106,
        "Interval Width: W = 2*q_hat*s(x)",
        fontsize=7.8, fontweight="bold", color="#5B21B6", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#EDE9FE", edgecolor="#8B5CF6", linewidth=1.0)
    )

    # Arrow 6 -> 7
    _draw_arrow(
        ax, (0.520, 0.245), (0.555, 0.245),
        label="Interval\nWidth W",
        label_offset=(0.0, 0.0),
        color="#7C3AED"
    )

    # Box 7: Combination Scoring
    _draw_card(ax, 0.555, 0.080, 0.215, 0.315, "#FFFFFF", "#2563EB", corner_radius=0.010, border_width=1.5)
    ax.text(
        0.565, 0.370, "STAGE 7",
        fontsize=7.5, fontweight="bold", color="#FFFFFF", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#2563EB", edgecolor="none")
    )
    ax.text(0.603, 0.370, "Combination Scoring", fontsize=10.5, fontweight="bold", color="#0F172A", va="center")

    stage7_bullets = [
        ("• Uncertainty Input:", "Conformal interval width W = 2*q*s(x)"),
        ("• Signal Quality (SQI):", "In-band spectral ratio P_resp / P_total"),
        ("• Fusion Classifier:", "Logistic Regression fit on validation fold"),
        ("• Zero Test Leakage:", "Weights learned strictly on val patients"),
        ("• Error Target:", "Predicts P(|error| > 3 bpm | W, SQI)"),
        ("• Clean AUROC:", "0.7527 [0.634, 0.856] (combines domains)"),
        ("• Comparison:", "Beats Width alone (0.702) & SQI alone (0.602)"),
    ]
    cur_y = 0.344
    for label, val in stage7_bullets:
        ax.text(0.565, cur_y, label, fontsize=8.0, fontweight="bold", color="#0F172A", va="center")
        ax.text(0.565, cur_y - 0.014, val, fontsize=7.3, color="#475569", va="center")
        cur_y -= 0.031

    # Combination score badge inside Stage 7
    ax.text(
        0.662, 0.106,
        "Score: s = σ(β0 + β1*W + β2*SQI)",
        fontsize=7.6, fontweight="bold", color="#1D4ED8", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#DBEAFE", edgecolor="#3B82F6", linewidth=1.0)
    )

    # Forking arrows from 7 -> 8 (Accept vs Abstain)
    _draw_arrow(
        ax, (0.770, 0.270), (0.805, 0.305),
        label="s <= τ",
        label_offset=(-0.015, 0.012),
        color="#16A34A",
        connectionstyle="arc3,rad=-0.12"
    )
    _draw_arrow(
        ax, (0.770, 0.220), (0.805, 0.175),
        label="s > τ",
        label_offset=(-0.015, -0.012),
        color="#DC2626",
        connectionstyle="arc3,rad=0.12"
    )

    # Box 8: Clinical Safety Triage Decision (Split Cards)
    _draw_card(ax, 0.805, 0.080, 0.165, 0.315, "#FFFFFF", "#334155", corner_radius=0.010, border_width=1.5)
    ax.text(
        0.815, 0.370, "STAGE 8",
        fontsize=7.5, fontweight="bold", color="#FFFFFF", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#334155", edgecolor="none")
    )
    ax.text(0.853, 0.370, "Selective Triage", fontsize=10.5, fontweight="bold", color="#0F172A", va="center")

    # Sub-card 8A: Accept Prediction
    _draw_card(ax, 0.812, 0.245, 0.150, 0.102, "#F0FDF4", "#22C55E", corner_radius=0.008, border_width=1.2, shadow=False)
    ax.text(
        0.818, 0.332, "● ACCEPT PREDICTION",
        fontsize=8.5, fontweight="bold", color="#15803D", va="center"
    )
    ax.text(0.818, 0.312, "• Condition: s <= τ or W <= 6.0 bpm", fontsize=7.2, color="#166534", va="center")
    ax.text(0.818, 0.296, "• High clinical confidence", fontsize=7.2, color="#166534", va="center")
    ax.text(0.818, 0.280, "• Output: Continuous RR to ICU EHR", fontsize=7.2, color="#166534", va="center")
    ax.text(0.818, 0.264, "• Retained MAE: 1.14 - 1.49 bpm", fontsize=7.2, fontweight="bold", color="#15803D", va="center")

    # Sub-card 8B: Safe Abstention Triggered
    _draw_card(ax, 0.812, 0.098, 0.150, 0.130, "#FFF1F2", "#E11D48", corner_radius=0.008, border_width=1.2, shadow=False)
    ax.text(
        0.818, 0.213, "⚠ SAFE ABSTENTION",
        fontsize=8.5, fontweight="bold", color="#991B1B", va="center"
    )
    ax.text(0.818, 0.193, "• Condition: s > τ or W > 6.0 bpm", fontsize=7.2, color="#9F1239", va="center")
    ax.text(0.818, 0.177, "• Motion spike / sensor detachment", fontsize=7.2, color="#9F1239", va="center")
    ax.text(0.818, 0.161, "• Suppress untrustworthy prediction", fontsize=7.2, color="#9F1239", va="center")
    ax.text(0.818, 0.145, "• Action: Flag for bedside ICU review", fontsize=7.2, fontweight="bold", color="#991B1B", va="center")
    ax.text(0.818, 0.129, "• AUROC: 0.99 under motion bursts", fontsize=7.2, color="#9F1239", va="center")
    ax.text(0.818, 0.113, "• Closes 46-54% oracle error gap", fontsize=7.2, color="#9F1239", va="center")

    # =========================================================================
    # BOTTOM FOOTER: SCIENTIFIC GUARANTEES & ARCHITECTURAL SUMMARY (y: 0.016 to 0.062)
    # =========================================================================
    _draw_card(ax, 0.030, 0.016, 0.940, 0.046, "#FFFFFF", "#CBD5E1", corner_radius=0.008, border_width=1.0)

    footer_badges = [
        ("Patient-Disjoint Partitioning:", "Strict 5-fold GroupKFold by patient ID; 0% patient leakage across train, val, cal, and test folds."),
        ("Hardware-Constrained Design:", "Runs on Intel i3, 8GB RAM, no GPU; sub-millisecond per-window inference with streaming memory."),
        ("Conformal Risk Control:", "Distribution-free finite-sample marginal coverage P(y ∈ C(x)) >= 90% guaranteed under exchangeability."),
    ]
    fb_x = 0.042
    for title, desc in footer_badges:
        ax.text(fb_x, 0.046, title, fontsize=8.0, fontweight="bold", color="#0F172A", va="center")
        ax.text(fb_x, 0.030, desc, fontsize=7.2, color="#475569", va="center")
        fb_x += 0.315

    # Save figure to primary and secondary destination paths
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if secondary_save_path:
        secondary_save_path = Path(secondary_save_path)
        secondary_save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(secondary_save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.close(fig)
    return save_path


if __name__ == "__main__":
    out_path = render_pipeline_architecture_figure()
    print(f"[Done] Pipeline architecture overview figure exported to: {out_path}")
