#!/usr/bin/env python3
"""
WildLIFT-A: 3D Bounding Box Annotation Tool

Convenience wrapper — launches the interactive annotation tool.

Usage:
    python run_wildlift_a.py --auto_bboxes <bbox_dir> --images <images_dir> \\
        --output <output_dir> --port 8080
"""

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import and run bbox editor
import runpy
runpy.run_module('wildlift.annotator.bbox_editor', run_name='__main__')
