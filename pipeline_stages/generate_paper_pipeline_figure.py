#!/usr/bin/env python3
"""
generate_paper_pipeline_figure.py
=================================
Renders a publication-grade, streamlined IEEE conference architecture diagram
for Figure 1 in main.tex.

Features:
- Clean 2-row modular layout (Data/Features -> Modeling/Conformal/Triage)
- No embedded 'Figure 1:' title bar (handled cleanly by LaTeX caption)
- High-legibility typography matching IEEE font standards (serif / math)
- Outputs both 300 DPI PNG and vector PDF for infinite zoom in LaTeX/Overleaf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# IEEE double-column width
FIG_WIDTH = 7.16
FIG_HEIGHT = 3.65

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def draw_box(ax, x, y, w, h, bg_color, border_color, title, subtitle, bullets, badge=None, badge_color="#1E293B"):
    # Subtle drop shadow
    shadow = FancyBboxPatch((x + 0.003, y - 0.004), w, h,
                            boxstyle="round,pad=0.012,rounding_size=0.02",
                            facecolor="#CBD5E1", edgecolor="none", alpha=0.45, zorder=2)
    ax.add_patch(shadow)

    # Main box
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.02",
                         facecolor=bg_color, edgecolor=border_color, linewidth=1.2, zorder=3)
    ax.add_patch(box)

    # Optional stage badge in top-left
    if badge:
        b_w, b_h = 0.048, 0.038
        b_patch = FancyBboxPatch((x + 0.010, y + h - b_h - 0.008), b_w, b_h,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor=badge_color, edgecolor="none", zorder=4)
        ax.add_patch(b_patch)
        ax.text(x + 0.010 + b_w/2, y + h - b_h/2 - 0.008, badge,
                color="white", fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=5)
        title_x = x + 0.066
    else:
        title_x = x + 0.014

    # Title
    ax.text(title_x, y + h - 0.024, title, fontsize=8.0, fontweight="bold", color="#0F172A", zorder=4)

    # Subtitle / description
    if subtitle:
        ax.text(x + 0.014, y + h - 0.062, subtitle, fontsize=6.8, fontstyle="italic", color="#334155", zorder=4)

    # Bullet points / details
    start_y = y + h - 0.098 if subtitle else y + h - 0.062
    for i, line in enumerate(bullets):
        ax.text(x + 0.014, start_y - i * 0.036, line, fontsize=6.4, color="#1E293B", zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.3, label=None, label_pos=(0, 0), connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            connectionstyle=connectionstyle,
                            arrowstyle="-|>,head_length=5,head_width=3.5",
                            color=color, linewidth=lw, zorder=6)
    ax.add_patch(arrow)
    if label:
        lx = (x1 + x2) / 2 + label_pos[0]
        ly = (y1 + y2) / 2 + label_pos[1]
        ax.text(lx, ly, label, fontsize=6.2, fontweight="bold", color=color,
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.18", fc="#F8FAFC", ec="none", alpha=0.9), zorder=7)


def render_pipeline_figure(out_paths):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # Group backgrounds
    # Top Row: Signal Processing & Feature Engineering
    g1 = FancyBboxPatch((0.008, 0.505), 0.984, 0.465, boxstyle="round,pad=0.01,rounding_size=0.02",
                        facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8, linestyle="--", zorder=1)
    ax.add_patch(g1)
    ax.text(0.022, 0.945, "STAGE I: BIOSIGNAL PREPROCESSING & PHYSIOLOGICAL FEATURE EXTRACTION",
            fontsize=6.8, fontweight="bold", color="#334155", zorder=2)

    # Bottom Row: Modeling, Conformal Uncertainty & Selective Triage
    g2 = FancyBboxPatch((0.008, 0.020), 0.984, 0.465, boxstyle="round,pad=0.01,rounding_size=0.02",
                        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=0.8, linestyle="--", zorder=1)
    ax.add_patch(g2)
    ax.text(0.022, 0.460, "STAGE II: GBDT ESTIMATION, CONFORMAL QUANTIFICATION & SELECTIVE TRIAGE",
            fontsize=6.8, fontweight="bold", color="#334155", zorder=2)

    # Dimensions for 4 boxes per row
    bw = 0.215
    bh = 0.380
    gap = 0.033
    x_starts = [0.022 + i * (bw + gap) for i in range(4)]

    # ---------------- TOP ROW (Left to Right: Steps 1 -> 4) ----------------
    y_top = 0.535

    # Box 1: Raw PPG Acquisition
    draw_box(ax, x_starts[0], y_top, bw, bh,
             "#E0F2FE", "#0284C7",
             "1. Raw PPG Acquisition",
             "Continuous Bedside Pulse Oximetry",
             [r"$\bullet$ In-Domain: BIDMC (ICU, $N{=}53$)",
              r"$\bullet$ Sampling: $f_s = 125\ \mathrm{Hz}$",
              r"$\bullet$ External: CapnoBase ($N{=}42$)",
              r"$\bullet$ Reference: Impedance / $\mathrm{CO}_2$"],
             badge="01", badge_color="#0284C7")

    # Box 2: Bandpass Filtering
    draw_box(ax, x_starts[1], y_top, bw, bh,
             "#EFF6FF", "#3B82F6",
             "2. Dual Bandpass Filtering",
             "Zero-Phase Butterworth SOS",
             [r"$\bullet$ Respiratory: $0.10\text{--}0.60\ \mathrm{Hz}$",
              r"  ($6\text{--}36\ \mathrm{bpm}$ passband)",
              r"$\bullet$ Cardiac: $0.10\text{--}5.00\ \mathrm{Hz}$",
              r"$\bullet$ 4th-order zero-phase filter"],
             badge="02", badge_color="#2563EB")

    # Box 3: Window Segmentation
    draw_box(ax, x_starts[2], y_top, bw, bh,
             "#F0FDF4", "#16A34A",
             "3. Window Segmentation",
             "Sliding Temporal Slicing",
             [r"$\bullet$ Window: $T = 60.0\ \mathrm{s}$",
              r"$\bullet$ Stride: $\Delta t = 2.0\ \mathrm{s}$",
              r"$\bullet$ 11,026 clean in-domain wins",
              r"$\bullet$ Patient-disjoint partitioning"],
             badge="03", badge_color="#16A34A")

    # Box 4: 28-Feature Extraction
    draw_box(ax, x_starts[3], y_top, bw, bh,
             "#FEF3C7", "#D97706",
             "4. Feature Extraction",
             "28 Physiological Descriptors",
             [r"$\bullet$ PPG Spectral (4): Centroid, BW",
              r"$\bullet$ Modulations (12): RIAV, RIIV, RIFV",
              r"$\bullet$ Modulation Stats & Fusion (8)",
              r"$\bullet$ Morphology & Sharpness (4)"],
             badge="04", badge_color="#D97706")

    # Top row connecting arrows
    for i in range(3):
        draw_arrow(ax, x_starts[i] + bw, y_top + bh/2, x_starts[i+1], y_top + bh/2)

    # Connecting curve from Row 1 Box 4 down to Row 2 Box 5 (leftmost of Row 2)
    y_bot = 0.050
    draw_arrow(ax, x_starts[3] + bw/2, y_top - 0.008,
               x_starts[0] + bw/2, y_bot + bh + 0.012,
               color="#64748B", lw=1.4, label="Extracted Feature Vector x",
               connectionstyle="arc3,rad=-0.35", label_pos=(0, 0.02))

    # ---------------- BOTTOM ROW (Left to Right: Steps 5 -> 8) ----------------
    # Box 5: XGBoost Point Prediction
    draw_box(ax, x_starts[0], y_bot, bw, bh,
             "#F5F3FF", "#7C3AED",
             "5. XGBoost Point Regressor",
             "Histogram GBDT (CPU-Only)",
             [r"$\bullet$ 100 Trees, Depth 3, $\eta = 0.08$",
              r"$\bullet$ Point Estimate: $\hat{y} = f_\theta(\mathbf{x})$",
              r"$\bullet$ In-Domain MAE: $1.81\ \mathrm{bpm}$",
              r"$\bullet$ Latency: $\approx 15\ \mathrm{ms}$ / window"],
             badge="05", badge_color="#7C3AED")

    # Box 6: Difficulty-Normalized Conformal Calibration
    draw_box(ax, x_starts[1], y_bot, bw, bh,
             "#EEF2FF", "#4F46E5",
             "6. Conformal Calibration",
             "Difficulty-Normalized Split CP",
             [r"$\bullet$ Difficulty: $s_i = |y_i - \hat{y}_i| / \hat{\sigma}(\mathbf{x}_i)$",
              r"$\bullet$ Conformal Quantile: $\hat{q}_{\alpha}$ (90%)",
              r"$\bullet$ Interval: $\hat{y} \pm \hat{q}\hat{\sigma}(\mathbf{x})$",
              r"$\bullet$ Interval Width: $W(\mathbf{x}) = 2\hat{q}\hat{\sigma}(\mathbf{x})$"],
             badge="06", badge_color="#4F46E5")

    # Box 7: Combination Scoring
    draw_box(ax, x_starts[2], y_bot, bw, bh,
             "#FFF7ED", "#EA580C",
             "7. Combination Scoring",
             "Multi-Domain Risk Predictor",
             [r"$\bullet$ Width: $W(\mathbf{x})$ (conformal scale)",
              r"$\bullet$ $\mathrm{SQI} = 1 - \text{In-band power ratio}$",
              r"$\bullet$ Logistic: $g(\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{z})$",
              r"$\bullet$ AUROC ($|\mathrm{err}| > 3$): $0.75$"],
             badge="07", badge_color="#EA580C")

    # Bottom row connecting arrows (Box 5 -> 6 -> 7)
    draw_arrow(ax, x_starts[0] + bw, y_bot + bh/2, x_starts[1], y_bot + bh/2, label=r"$\hat{y}$")
    draw_arrow(ax, x_starts[1] + bw, y_bot + bh/2, x_starts[2], y_bot + bh/2, label=r"$W(\mathbf{x})$")

    # Box 8: Selective Triage & Decision Output
    # Custom layout for decision branching
    x8 = x_starts[3]
    shadow8 = FancyBboxPatch((x8 + 0.003, y_bot - 0.004), bw, bh,
                             boxstyle="round,pad=0.012,rounding_size=0.02",
                             facecolor="#CBD5E1", edgecolor="none", alpha=0.45, zorder=2)
    ax.add_patch(shadow8)
    box8 = FancyBboxPatch((x8, y_bot), bw, bh,
                          boxstyle="round,pad=0.012,rounding_size=0.02",
                          facecolor="#FFFFFF", edgecolor="#0F172A", linewidth=1.2, zorder=3)
    ax.add_patch(box8)

    # Stage 08 badge
    b_patch8 = FancyBboxPatch((x8 + 0.010, y_bot + bh - 0.046), 0.048, 0.038,
                             boxstyle="round,pad=0.004,rounding_size=0.01",
                             facecolor="#0F172A", edgecolor="none", zorder=4)
    ax.add_patch(b_patch8)
    ax.text(x8 + 0.010 + 0.024, y_bot + bh - 0.027, "08",
            color="white", fontsize=6.2, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(x8 + 0.066, y_bot + bh - 0.024, "8. Selective Triage", fontsize=8.0, fontweight="bold", color="#0F172A", zorder=4)
    ax.text(x8 + 0.014, y_bot + bh - 0.060, r"Decision Rule: Retain if $g(\mathbf{x}) \leq \tau$", fontsize=6.6, fontstyle="italic", color="#334155", zorder=4)

    # Sub-card: Retain / Accept
    r_box = FancyBboxPatch((x8 + 0.012, y_bot + 0.160), bw - 0.024, 0.140,
                           boxstyle="round,pad=0.006,rounding_size=0.01",
                           facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=1.0, zorder=4)
    ax.add_patch(r_box)
    ax.text(x8 + 0.020, y_bot + 0.270, r"ACCEPT / RETAIN ($g \leq \tau$)", fontsize=6.5, fontweight="bold", color="#15803D", zorder=5)
    ax.text(x8 + 0.020, y_bot + 0.228, r"$\bullet$ Output $\hat{y} \pm \hat{q}\hat{\sigma}(\mathbf{x})$ (bpm)", fontsize=6.1, color="#166534", zorder=5)
    ax.text(x8 + 0.020, y_bot + 0.186, r"$\bullet$ 60% Retained MAE: $1.18\ \mathrm{bpm}$", fontsize=6.1, color="#166534", zorder=5)

    # Sub-card: Abstain / Flag
    a_box = FancyBboxPatch((x8 + 0.012, y_bot + 0.012), bw - 0.024, 0.136,
                           boxstyle="round,pad=0.006,rounding_size=0.01",
                           facecolor="#FEE2E2", edgecolor="#DC2626", linewidth=1.0, zorder=4)
    ax.add_patch(a_box)
    ax.text(x8 + 0.020, y_bot + 0.118, r"SAFE ABSTAIN ($g > \tau$)", fontsize=6.5, fontweight="bold", color="#B91C1C", zorder=5)
    ax.text(x8 + 0.020, y_bot + 0.076, r"$\bullet$ High uncertainty / artifact", fontsize=6.1, color="#991B1B", zorder=5)
    ax.text(x8 + 0.020, y_bot + 0.034, r"$\bullet$ Alert clinician to inspect", fontsize=6.1, color="#991B1B", zorder=5)

    # Arrow from Box 7 to Box 8
    draw_arrow(ax, x_starts[2] + bw, y_bot + bh/2, x8, y_bot + bh/2, label=r"$g(\mathbf{x})$")

    plt.subplots_adjust(left=0.005, right=0.995, top=0.990, bottom=0.010)

    for p in out_paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Saved figure to: {p}")
    plt.close(fig)


if __name__ == "__main__":
    targets = [
        Path("output/figures/pipeline_architecture.png"),
        Path("output/figures/pipeline_architecture.pdf"),
        Path("OVERLEAF PAPER/figures/pipeline_architecture.png"),
        Path("OVERLEAF PAPER/figures/pipeline_architecture.pdf"),
        Path("output/pipeline_architecture.png"),
    ]
    render_pipeline_figure(targets)
