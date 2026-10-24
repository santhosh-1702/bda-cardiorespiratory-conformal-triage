#!/usr/bin/env python3
"""
Pipeline Architecture Overview Figure Generator
================================================
Generates the publication-grade Figure 1 overview schematic diagram (300 DPI)
illustrating the end-to-end cardiorespiratory triage pipeline:
  Raw PPG (125 Hz)
  -> Butterworth bandpass filter (0.10-0.60 Hz)
  -> 60 s windows / 2 s stride
  -> 28-feature extraction (PPG spectral, RIAV/RIIV/RIFV modulations, morphology/stats)
  -> XGBoost point prediction
  -> Conformal calibration (active residual_normalised, tested jackknife+/CV+)
  -> Width + SQI combination score
  -> Accept / abstain triage

Usage:
  python generate_architecture_figure.py
  python generate_architecture_figure.py --output output/pipeline_architecture.png --dpi 300
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ARCHITECTURE_FIGURE_PATH, PROJECT_ROOT
from src.visualization.architecture_plotter import render_pipeline_architecture_figure


def main():
    parser = argparse.ArgumentParser(
        description="Generates publication Figure 1 pipeline architecture diagram (300 DPI)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARCHITECTURE_FIGURE_PATH,
        help="Primary output path for the figure (defaults to output/figures/pipeline_architecture.png)"
    )
    parser.add_argument(
        "--secondary-output",
        type=Path,
        default=None,
        help="Optional secondary output path"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution in DPI (defaults to 300)"
    )
    args = parser.parse_args()

    print(f"[Architecture Overview] Generating Figure 1 at {args.dpi} DPI...")
    out_path = render_pipeline_architecture_figure(
        save_path=args.output,
        secondary_save_path=args.secondary_output,
        dpi=args.dpi
    )
    print(f"[Architecture Overview] Saved primary figure to: {out_path}")
    if args.secondary_output:
        print(f"[Architecture Overview] Saved companion figure to: {args.secondary_output}")


if __name__ == "__main__":
    main()
