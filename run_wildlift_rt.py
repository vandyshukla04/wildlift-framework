#!/usr/bin/env python3
"""
WildLIFT-RT: 3D Wildlife Reconstruction and Tracking Pipeline

Convenience wrapper — runs the full inference + tracking pipeline.

Usage:
    python run_wildlift_rt.py --seq_path <images_dir> --mask_dir <masks_dir> \\
        --model_path <model.pth> --device cuda --size 512

    # Use Kalman tracker (default):
    python run_wildlift_rt.py --seq_path ... --tracker kalman

    # Use simple online tracker:
    python run_wildlift_rt.py --seq_path ... --tracker simple

    # Enable GPS pose refinement:
    python run_wildlift_rt.py --seq_path ... --dji_log flight.srt --gps_refine
"""

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wildlift.rt.pipeline import main

if __name__ == "__main__":
    main()
