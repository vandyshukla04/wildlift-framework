#!/usr/bin/env python3
"""
Overlay instance masks with track-consistent colors on source images.

Usage:
    python visualize_mask_tracks.py \
        --images examples/wd_data/gira/gira-1_1 \
        --masks results/.../instance_labels \
        --mapping results/.../mask_track_mapping.json \
        --output results/.../mask_vis
"""

import argparse
import json
import os
import glob
import numpy as np
import cv2

COLOR_PALETTE = [
    (255, 50, 50),    # Red
    (50, 255, 50),    # Green
    (50, 50, 255),    # Blue
    (255, 255, 50),   # Yellow
    (255, 50, 255),   # Magenta
    (50, 255, 255),   # Cyan
    (255, 150, 50),   # Orange
    (150, 50, 255),   # Purple
    (50, 200, 50),    # Forest Green
    (200, 50, 150),   # Pink
]


def main():
    parser = argparse.ArgumentParser(description='Visualize masks with track-consistent colors')
    parser.add_argument('--images', required=True, help='Source image folder')
    parser.add_argument('--masks', required=True, help='Instance labels folder (.npy)')
    parser.add_argument('--mapping', required=True, help='mask_track_mapping.json')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--alpha', type=float, default=0.5, help='Mask overlay alpha')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.mapping) as f:
        mapping = json.load(f)  # {frame_name: {track_id_str: mask_index}}

    # Collect all track IDs for consistent coloring
    all_track_ids = sorted({int(tid) for frame_map in mapping.values() for tid in frame_map})

    def track_color(tid):
        idx = all_track_ids.index(tid) if tid in all_track_ids else tid
        return COLOR_PALETTE[idx % len(COLOR_PALETTE)]

    # Build image lookup
    img_files = sorted(
        glob.glob(os.path.join(args.images, '*.jpg')) +
        glob.glob(os.path.join(args.images, '*.png')) +
        glob.glob(os.path.join(args.images, '*.jpeg'))
    )
    img_by_name = {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}

    count = 0
    for frame_name, frame_map in sorted(mapping.items()):
        # Load image
        img_path = img_by_name.get(frame_name)
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Load instance labels
        mask_path = os.path.join(args.masks, f"{frame_name}.npy")
        if not os.path.exists(mask_path):
            continue
        labels = np.load(mask_path)  # (H, W), values: 0=bg, 1,2,...=instances

        # Resize image to match mask if needed
        mh, mw = labels.shape[:2]
        ih, iw = img.shape[:2]
        if (ih, iw) != (mh, mw):
            img = cv2.resize(img, (mw, mh))

        overlay = img.copy()

        for tid_str, mask_idx in frame_map.items():
            tid = int(tid_str)
            instance_val = mask_idx + 1  # mask_index is 0-based, labels are 1-indexed
            mask = (labels == instance_val)
            if not mask.any():
                continue

            color_bgr = track_color(tid)
            overlay[mask] = color_bgr

            # Label at mask centroid
            ys, xs = np.where(mask)
            cy, cx = int(ys.mean()), int(xs.mean())
            cv2.putText(img, f"T{tid}", (cx - 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

        # Blend
        result = cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0)

        cv2.imwrite(os.path.join(args.output, f"{frame_name}_tracked.png"), result)
        count += 1

    print(f"Saved {count} annotated frames to {args.output}")


if __name__ == '__main__':
    main()
