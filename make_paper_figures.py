#!/usr/bin/env python3
"""
make_paper_figures.py -- regenerate paper Figures 2-5 from saved project results.

Run from anywhere:
    python make_paper_figures.py --root "BDA PROJECT"
    python make_paper_figures.py --root .

Reads (all already produced by your pipeline):
    output/results/phase7_results.csv                       -> Fig. 2
    output/results/phase5_results.csv   (SECTION 5)         -> Fig. 3
    output/cache/oof_predictions.npz    (optional, see note)-> Fig. 4a
    output/cache/features_60s.npz, capnobase_features.npz   -> Fig. 4b
    output/results/capnobase_external_validation_*.csv      -> Fig. 5

Writes PDF (vector, use this in LaTeX) and PNG into --out (default
<root>/output/figures/paper). Copy the PDFs into overleaf_project/figures/.

NOTE ON FIG. 4a
  Your cache only keeps predictions for the LAST fold (conformal_results.npz,
  fold 4, 10 test patients; its MAE of 1.33 is below the 1.78 average). For an honest
  scatter you want all 53 patients out-of-fold. Save them once, e.g. at the end
  of each fold loop in pipeline_stages/run_phase4.py:
        oof.append((y_test, y_pred_clean, test_patient_ids, fold))
  and after the loop:
        np.savez("output/cache/oof_predictions.npz",
                 y_true=np.concatenate([o[0] for o in oof]),
                 y_pred=np.concatenate([o[1] for o in oof]),
                 patient_ids=np.concatenate([o[2] for o in oof]),
                 fold=np.concatenate([[o[3]] * len(o[0]) for o in oof]))
  This script uses that file automatically if it exists.
"""
import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL_W, FULL_W = 3.5, 7.16          # IEEE single-column / double-column width (inches)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

# CapnoBase adult case IDs, taken from output/paper/PAPER_RESULTS_REFERENCE.md
ADULT_SPONT = {"0322", "0330", "0331", "0332", "0370"}
ADULT_VENT = {"0309", "0311", "0312", "0313", "0325", "0328", "0329", "0333"}

C_BLUE, C_GREEN, C_RED, C_ORANGE = "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"
C_PURPLE, C_GREY, C_TEAL = "#7570b3", "#7f7f7f", "#17a2a2"


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def read_section(path, marker):
    """Read one '[SECTION n: ...]' block from the multi-table results CSV."""
    lines = Path(path).read_text(encoding="utf-8").replace("\r", "").splitlines()
    start = next(i for i, l in enumerate(lines) if marker in l)
    block = []
    for l in lines[start + 1:]:
        if not l.strip() or "[SECTION" in l:
            break
        block.append(l)
    return pd.read_csv(io.StringIO("\n".join(block)))


# ------------------------------------------------------------------ Fig. 2 ----
def fig2_fold_coverage(root, out):
    d = pd.read_csv(root / "output/results/phase7_results.csv")
    d = d[d.section == "DETAILED_RUNS"].copy()
    d["coverage"] = pd.to_numeric(d["coverage"], errors="coerce")   # CSV mixes text rows
    order = ["trivial_median", "absolute_residual", "residual_normalised",
             "mondrian_terciles", "spectral_sharpness", "cv_plus"]
    labels = ["Train\nmedian", "Abs.\nresidual", "Diff.-\nnorm.", "Mondrian\nterciles",
              "Spectral\nsharpness", "CV+"]
    data = [d[d.score_type == s].coverage.dropna().values.astype(float) * 100 for s in order]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.axhline(90, color=C_RED, ls="--", lw=0.9, label="90% target")
    ax.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color="black", lw=1),
               boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=5, alpha=0.55,
                   color=C_BLUE, zorder=3, linewidths=0)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Empirical coverage per fold (%)")
    ax.set_ylim(55, 103)
    ax.legend(loc="lower right", frameon=False)
    save(fig, out, "fig2_fold_coverage")


# ------------------------------------------------------------------ Fig. 3 ----
def fig3_risk_coverage(root, out):
    r = read_section(root / "output/results/phase5_results.csv", "SECTION 5")
    x = r["Retention_Pct"].astype(str).str.rstrip("%").astype(float).values
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot(x, r["Random_MAE_bpm"], color=C_GREY, ls="--", label="Random abstention")
    ax.plot(x, r["Raw_Width_MAE_bpm"], color=C_PURPLE, marker="s", ms=3, label="Raw width")
    ax.plot(x, r["Fold_Norm_MAE_bpm"], color=C_GREEN, marker="^", ms=3, label="Normalized width")
    ax.plot(x, r["Combination_Score_MAE_bpm"], color=C_ORANGE, marker="o", ms=3.5, lw=1.6,
            label="Width + SQI (combination)")
    ax.fill_between(x, r["Combination_95CI_Lo"], r["Combination_95CI_Hi"],
                    color=C_ORANGE, alpha=0.15, lw=0)
    ax.plot(x, r["Oracle_MAE_bpm"], color="black", ls=":", label="Oracle")
    ax.set_xlabel("Windows retained (%)")
    ax.set_ylabel("Retained MAE (bpm)")
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 2.7)
    ax.legend(loc="upper left", ncol=2, frameon=False, columnspacing=1.0, handlelength=1.8)
    save(fig, out, "fig3_risk_coverage")


# ------------------------------------------------------------------ Fig. 4 ----
def fig4_pred_vs_ref(root, out):
    cache = root / "output/cache"
    oof = cache / "oof_predictions.npz"
    if oof.exists():
        z = np.load(oof, allow_pickle=True)
        yt, yp, tag = z["y_true"], z["y_pred"], "all 53 patients, out-of-fold"
    else:
        z = np.load(cache / "conformal_results.npz")
        yt, yp, tag = z["y_test"], z["y_clean"] if "y_clean" in z.files else z["y_pred_clean"], \
            "last fold only (fold 4, 10 patients) -- replace with out-of-fold predictions"
    r = np.corrcoef(yt, yp)[0, 1]
    mae = np.mean(np.abs(yt - yp))

    bid = np.load(cache / "features_60s.npz", allow_pickle=True)["y"]
    cz = np.load(cache / "capnobase_features.npz", allow_pickle=True)
    cy = cz["y"]
    case = np.array([p.split("_")[-1] for p in cz["patient_ids"]])
    groups = [
        ("BIDMC (ICU)", bid, C_BLUE, "-"),
        ("CapnoBase adult, spontaneous", cy[np.isin(case, list(ADULT_SPONT))], C_GREEN, "-"),
        ("CapnoBase adult, ventilated", cy[np.isin(case, list(ADULT_VENT))], C_RED, "-"),
        ("CapnoBase pediatric", cy[~np.isin(case, list(ADULT_SPONT | ADULT_VENT))], C_ORANGE, "--"),
    ]

    lo, hi = min(yt.min(), yp.min()) - 1, max(yt.max(), yp.max()) + 1
    bins = np.arange(4, 47, 1.0)

    # ------------------ Fig. 4a (Standalone) ------------------
    fig_a, ax_a = plt.subplots(figsize=(COL_W, 2.7))
    ax_a.scatter(yt, yp, s=3.5, alpha=0.22, color=C_BLUE, linewidths=0, rasterized=True)
    ax_a.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
    ax_a.set_xlim(lo, hi)
    ax_a.set_ylim(lo, hi)
    ax_a.set_xlabel("Reference respiratory rate (bpm)")
    ax_a.set_ylabel("Predicted respiratory rate (bpm)")
    ax_a.set_title(f"(a) Predicted vs. reference ({tag})", fontsize=7)
    ax_a.text(0.04, 0.95, f"$r$ = {r:.2f}\nMAE = {mae:.2f} bpm\n$n$ = {len(yt):,} windows",
              transform=ax_a.transAxes, va="top", fontsize=7)
    save(fig_a, out, "fig4a_pred_vs_ref")

    # ------------------ Fig. 4b (Standalone) ------------------
    fig_b, ax_b = plt.subplots(figsize=(COL_W, 2.6))
    for name, v, c, ls in groups:
        ax_b.hist(v, bins=bins, density=True, histtype="step", color=c, ls=ls, lw=1.2,
                  label=f"{name} (n={len(v):,})")
    ax_b.set_xlabel("Reference respiratory rate (bpm)")
    ax_b.set_ylabel("Density")
    ax_b.set_title("(b) Reference-rate distribution per cohort", fontsize=7)
    ax_b.legend(frameon=False, loc="upper right", fontsize=6.5)
    save(fig_b, out, "fig4b_cohort_dist")

    # ------------------ Fig. 4 (Combined 2-panel) ------------
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.45), gridspec_kw={"wspace": 0.28})
    a.scatter(yt, yp, s=3, alpha=0.18, color=C_BLUE, linewidths=0, rasterized=True)
    a.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("Reference respiratory rate (bpm)")
    a.set_ylabel("Predicted respiratory rate (bpm)")
    a.set_title(f"(a) Predicted vs. reference ({tag})", fontsize=7)
    a.text(0.04, 0.95, f"$r$ = {r:.2f}\nMAE = {mae:.2f} bpm\n$n$ = {len(yt):,} windows",
           transform=a.transAxes, va="top", fontsize=7)

    for name, v, c, ls in groups:
        b.hist(v, bins=bins, density=True, histtype="step", color=c, ls=ls, lw=1.2,
               label=f"{name} (n={len(v):,})")
    b.set_xlabel("Reference respiratory rate (bpm)")
    b.set_ylabel("Density")
    b.set_title("(b) Reference-rate distribution per cohort", fontsize=7)
    b.legend(frameon=False, loc="upper right", fontsize=6)
    save(fig, out, "fig4_pred_vs_ref")
    print(f"  Fig.4a source: {tag}; r={r:.3f}, MAE={mae:.3f}")



# ------------------------------------------------------------------ Fig. 5 ----
def fig5_external(root, out):
    res = root / "output/results"
    strat = pd.read_csv(res / "capnobase_external_validation_stratified_results.csv")
    main = pd.read_csv(res / "capnobase_external_validation_results.csv")
    if "score_type" in main.columns:
        pref = "residual_normalised" if "residual_normalised" in set(main.score_type) else main.score_type.iloc[0]
        main = main[main.score_type == pref]
    stats = {}
    stats["BIDMC"] = {m: (main[f"bidmc_{k}"].mean(), main[f"bidmc_{k}"].std(ddof=1))
                      for m, k in [("mae", "mae"), ("cov", "coverage"), ("wid", "width")]}
    for cohort, name in [("adult_spontaneous", "Adult\nspont."), ("adult_ventilated", "Adult\nvent."),
                         ("adult", "Adult\nall"), ("pediatric", "Pedi-\natric"), ("all", "All\n42")]:
        s = strat[strat.cohort == cohort]
        stats[name] = {m: (s[k].mean(), s[k].std(ddof=1))
                       for m, k in [("mae", "mae"), ("cov", "coverage"), ("wid", "width")]}
    stats = {("BIDMC\n(ICU)" if k == "BIDMC" else k): v for k, v in stats.items()}
    names = list(stats)
    colors = [C_BLUE, C_GREEN, C_RED, C_TEAL, C_ORANGE, C_GREY]
    panels = [("mae", "(a) MAE (bpm)", 1.0), ("cov", "(b) Coverage, 90% target (%)", 100.0),
              ("wid", "(c) Mean interval width (bpm)", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3), gridspec_kw={"wspace": 0.32})
    for ax, (m, title, scale) in zip(axes, panels):
        mu = np.array([stats[n][m][0] for n in names]) * scale
        sd = np.array([stats[n][m][1] for n in names]) * scale
        ax.bar(range(len(names)), mu, yerr=sd, color=colors, width=0.62,
               error_kw=dict(lw=0.7, capsize=2))
        for i, (a_, s_) in enumerate(zip(mu, sd)):
            ax.text(i, a_ + s_ + 0.02 * (mu + sd).max(), f"{a_:.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=6.3)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylim(0, (mu + sd).max() * 1.18)
        if m == "cov":
            ax.axhline(90, color="black", ls="--", lw=0.8)
            ax.set_ylim(0, 112)
    save(fig, out, "fig5_external_validation")
    for n in names:
        print("  ", n.replace("\n", " "), {m: f"{stats[n][m][0]:.3f}+/-{stats[n][m][1]:.3f}" for m in stats[n]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root folder (default: .)")
    ap.add_argument("--out", default=None, help="output folder (default <root>/output/figures/paper)")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists() and (Path(".") / "output").exists():
        root = Path(".")
    out = Path(args.out) if args.out else root / "output/figures/paper"
    out.mkdir(parents=True, exist_ok=True)
    fig2_fold_coverage(root, out)
    fig3_risk_coverage(root, out)
    fig4_pred_vs_ref(root, out)
    fig5_external(root, out)
