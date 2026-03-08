#!/usr/bin/env python3
"""
Run 2D baseline trackers (ByteTrack, BotSORT, StrongSORT) on the same
detections used by our pipeline, then write mask_track_mapping.json
so eval_tracking.py can evaluate them.

For each frame we derive 2D bounding boxes from instance_labels + the
original mask_track_mapping.json (same detections the original tracker saw).
We feed [x1,y1,x2,y2,conf,cls] to each boxmot tracker, collect assigned
track IDs, and write a new mask_track_mapping.json in a subfolder.

Usage:
    python run_baseline_trackers.py \
        --result_dir results/paper_final/thursday/giraffes/gira-1_1 \
        --source_images examples/wd_data/gira/gira-1_1 \
        --trackers bytetrack botsort strongsort
"""

import argparse
import json
import os
import glob
import numpy as np
import cv2

from boxmot import ByteTrack, BotSort, OcSort
import torch


TRACKER_MAP = {
    'bytetrack': ByteTrack,
    'botsort': BotSort,
    'ocsort': OcSort,
}


def load_detections(result_dir):
    """Load per-frame 2D detections from instance_labels + original mask_track_mapping.
    Returns list of (frame_name, detections) where detections is list of
    (x1, y1, x2, y2, conf, cls_id, instance_label_value).
    """
    mapping_path = os.path.join(result_dir, "mask_track_mapping.json")
    label_dir = os.path.join(result_dir, "instance_labels")
    bbox_dir = os.path.join(result_dir, "bounding_boxes")

    with open(mapping_path) as f:
        orig_mapping = json.load(f)

    bbox_files = sorted(glob.glob(os.path.join(bbox_dir, "*.json")))
    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in bbox_files]

    frames = []
    for fname, bbox_file in zip(frame_names, bbox_files):
        with open(bbox_file) as f:
            bboxes = json.load(f)

        label_path = os.path.join(label_dir, f"{fname}.npy")
        if not os.path.exists(label_path):
            frames.append((fname, []))
            continue
        labels = np.load(label_path)

        frame_map = orig_mapping.get(fname, {})
        # Build track_id -> mask_index mapping, then bbox index -> instance val
        bbox_to_instance = {}
        for tid_str, mask_idx in frame_map.items():
            orig_tid = int(tid_str)
            for bi, b in enumerate(bboxes):
                if b.get('track_id') == orig_tid:
                    bbox_to_instance[bi] = mask_idx + 1  # 0-based -> 1-indexed
                    break

        dets = []
        for bi, bbox in enumerate(bboxes):
            inst_val = bbox_to_instance.get(bi, bi + 1)
            mask = (labels == inst_val)
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())
            conf = bbox.get('confidence', 0.9)
            dets.append((x1, y1, x2, y2, conf, 0, inst_val))

        frames.append((fname, dets))

    return frames, frame_names


def run_tracker(tracker_name, frames, source_images_dir):
    """Run a boxmot tracker on the detections. Returns {frame_name: {tid_str: mask_idx}}."""
    TrackerCls = TRACKER_MAP[tracker_name]

    if tracker_name == 'botsort':
        device = torch.device('cpu')
        tracker = TrackerCls(reid_weights='none', device=device, half=False, with_reid=False)
    else:
        tracker = TrackerCls()

    results = {}
    for fname, dets in frames:
        # Load image for tracker (some need it for ReID)
        img = None
        if source_images_dir:
            for ext in ['.jpg', '.png', '.jpeg']:
                p = os.path.join(source_images_dir, fname + ext)
                if os.path.exists(p):
                    img = cv2.imread(p)
                    break
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        if not dets:
            results[fname] = {}
            continue

        # Build Nx6 detection array: [x1, y1, x2, y2, conf, cls]
        det_array = np.array([[d[0], d[1], d[2], d[3], d[4], d[5]] for d in dets],
                             dtype=np.float32)
        inst_vals = [d[6] for d in dets]

        # Run tracker
        tracks = tracker.update(det_array, img)  # Nx8: [x1,y1,x2,y2,tid,conf,cls,?]

        # Match tracks back to detections by IoU to get instance_label_value
        mapping = {}
        if len(tracks) > 0:
            for t in tracks:
                tx1, ty1, tx2, ty2 = t[0], t[1], t[2], t[3]
                tid = int(t[4])

                # Find best matching detection by IoU
                best_iou = 0
                best_inst = None
                for d in dets:
                    dx1, dy1, dx2, dy2 = d[0], d[1], d[2], d[3]
                    ix1 = max(tx1, dx1)
                    iy1 = max(ty1, dy1)
                    ix2 = min(tx2, dx2)
                    iy2 = min(ty2, dy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_t = (tx2 - tx1) * (ty2 - ty1)
                    area_d = (dx2 - dx1) * (dy2 - dy1)
                    union = area_t + area_d - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_inst = d[6]  # instance_label_value

                if best_inst is not None:
                    mapping[str(tid)] = best_inst - 1  # convert to 0-based mask index

        results[fname] = mapping

    return results


def save_results(output_dir, mapping, frames):
    """Save mask_track_mapping.json and tracking_summary.json."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'mask_track_mapping.json'), 'w') as f:
        json.dump(mapping, f, indent=2)

    # Build tracking summary
    track_frames = {}
    track_classes = {}
    track_confs = {}
    for fname, frame_map in mapping.items():
        for tid_str in frame_map:
            tid = int(tid_str)
            if tid not in track_frames:
                track_frames[tid] = []
                track_confs[tid] = []
                track_classes[tid] = 'animal'
            track_frames[tid].append(fname)
            track_confs[tid].append(0.9)

    # Convert frame names to indices
    all_frame_names = [f[0] for f in frames]
    tracks_info = {}
    for tid in sorted(track_frames.keys()):
        fnames = track_frames[tid]
        frame_indices = sorted([all_frame_names.index(fn) for fn in fnames if fn in all_frame_names])
        tracks_info[str(tid)] = {
            "length": len(frame_indices),
            "class_name": track_classes[tid],
            "first_frame": frame_indices[0] if frame_indices else 0,
            "last_frame": frame_indices[-1] if frame_indices else 0,
            "avg_confidence": float(np.mean(track_confs[tid])),
            "frames": frame_indices,
        }

    summary = {
        "total_tracks": len(tracks_info),
        "frames_processed": len(all_frame_names),
        "total_detections": sum(len(m) for m in mapping.values()),
        "avg_tracklet_length": float(np.mean([t["length"] for t in tracks_info.values()])) if tracks_info else 0,
        "max_tracklet_length": max([t["length"] for t in tracks_info.values()]) if tracks_info else 0,
        "tracks": tracks_info,
    }

    with open(os.path.join(output_dir, 'tracking_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved to {output_dir}: {summary['total_tracks']} tracks, "
          f"avg length {summary['avg_tracklet_length']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--source_images', required=True)
    parser.add_argument('--trackers', nargs='+', default=['bytetrack', 'botsort', 'ocsort'],
                        choices=list(TRACKER_MAP.keys()))
    args = parser.parse_args()

    frames, frame_names = load_detections(args.result_dir)
    print(f"Loaded {len(frames)} frames from {args.result_dir}")

    for tracker_name in args.trackers:
        print(f"\nRunning {tracker_name}...")
        mapping = run_tracker(tracker_name, frames, args.source_images)
        output_dir = os.path.join(args.result_dir, tracker_name)
        save_results(output_dir, mapping, frames)


if __name__ == '__main__':
    main()
