#!/usr/bin/env python3
"""Create a video from a directory of image frames."""

import argparse
import cv2
import os
import re
from pathlib import Path


def natural_sort_key(filename):
    """Sort filenames by embedded numbers so frame ordering is correct."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', filename)]


def frames_to_video(src_dir, dst_dir, filename="output.mp4", fps=15):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    frames = sorted(
        [f for f in os.listdir(src_dir) if Path(f).suffix.lower() in exts],
        key=natural_sort_key,
    )

    if not frames:
        print(f"No image files found in {src_dir}")
        return

    # Read first frame to get dimensions
    sample = cv2.imread(str(src_dir / frames[0]))
    h, w = sample.shape[:2]

    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / filename

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for f in frames:
        img = cv2.imread(str(src_dir / f))
        if img is not None:
            writer.write(img)

    writer.release()
    print(f"Saved video ({len(frames)} frames, {fps} fps) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from image frames")
    parser.add_argument("src", help="Directory containing image frames")
    parser.add_argument("dst", help="Destination directory for the output video")
    parser.add_argument("--filename", default="output.mp4", help="Output filename (default: output.mp4)")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    args = parser.parse_args()

    frames_to_video(args.src, args.dst, args.filename, args.fps)
