#!/usr/bin/env python3
"""
WildLIFT-V: Viewpoint Analysis and Visualization

Convenience wrapper — runs the viewpoint analyzer.

Usage:
    # Basic viewpoint analysis:
    python run_wildlift_v.py --annotator_output <result_dir>

    # With semantic face propagation:
    python run_wildlift_v.py --annotator_output <result_dir> --semantic

    # Aggregate PDF with saved selections:
    python run_wildlift_v.py --annotator_output <result_dir> --aggregate --load_saved
"""

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wildlift.viewpoint.analyzer import main

if __name__ == "__main__":
    sys.exit(main())
