#!/usr/bin/env python3
"""
Evaluate tracking performance against GT annotations (MOTChallenge format).

Computes: MOTA, IDF1, ID switches, FP, FN, precision, recall, track fragmentation.

Usage:
    python eval_tracking.py \
        --gt results/paper_final/gt_annotations/giraffes/gira-1_1 \
        --pred results/paper_final/thursday/giraffes/gira-1_1 \
        --pred_retracked results/paper_final/thursday/giraffes/gira-1_1/retracked
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


def load_gt(gt_dir):
    """Load GT in MOTChallenge format: frame,id,x,y,w,h,conf,...
    Returns: {frame_1indexed: [(id, x, y, w, h), ...]}
    """
    gt_path = os.path.join(gt_dir, "gt.txt")
    meta_path = os.path.join(gt_dir, "metadata.json")

    with open(meta_path) as f:
        meta = json.load(f)
    frame_names = meta.get("frame_names", [])

    gt = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[0])  # 1-indexed
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt[frame].append((tid, x, y, w, h))

    return gt, frame_names


def load_pred(pred_dir, frame_names):
    """Load predictions from mask_track_mapping + instance_labels.
    Returns: {frame_1indexed: [(track_id, x, y, w, h), ...]}
    """
    mapping_path = os.path.join(pred_dir, "mask_track_mapping.json")
    label_dir_candidates = [
        os.path.join(pred_dir, "instance_labels"),
        os.path.join(os.path.dirname(pred_dir), "instance_labels"),
        # For retracked subfolder, labels are in parent
        os.path.join(pred_dir, "..", "instance_labels"),
    ]

    label_dir = None
    for c in label_dir_candidates:
        if os.path.isdir(c):
            label_dir = os.path.abspath(c)
            break

    if label_dir is None:
        raise ValueError(f"Cannot find instance_labels relative to {pred_dir}")

    with open(mapping_path) as f:
        mapping = json.load(f)

    pred = defaultdict(list)

    for frame_idx, fname in enumerate(frame_names):
        frame_1idx = frame_idx + 1
        frame_map = mapping.get(fname, {})
        if not frame_map:
            continue

        label_path = os.path.join(label_dir, f"{fname}.npy")
        if not os.path.exists(label_path):
            continue
        labels = np.load(label_path)

        for tid_str, mask_idx in frame_map.items():
            tid = int(tid_str)
            instance_val = mask_idx + 1  # 0-based index -> 1-indexed label
            mask = (labels == instance_val)
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            x, y = float(xs.min()), float(ys.min())
            w, h = float(xs.max() - xs.min()), float(ys.max() - ys.min())
            pred[frame_1idx].append((tid, x, y, w, h))

    return pred


def bbox_iou(a, b):
    """IoU between two (x, y, w, h) bboxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(gt, pred, iou_threshold=0.5):
    """Compute MOT metrics."""
    all_frames = sorted(set(gt.keys()) | set(pred.keys()))

    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_id_sw = 0
    total_matches = 0

    # For IDF1: track true positives, false positives, false negatives per ID
    gt_id_frames = defaultdict(set)    # gt_id -> set of frames present
    pred_id_frames = defaultdict(set)  # pred_id -> set of frames present
    match_pairs = defaultdict(lambda: defaultdict(int))  # (gt_id, pred_id) -> match count

    # For fragmentation
    gt_id_last_matched = {}  # gt_id -> last pred_id matched
    fragments = 0

    prev_match = {}  # gt_id -> pred_id from previous frame

    for frame in all_frames:
        gt_dets = gt.get(frame, [])
        pred_dets = pred.get(frame, [])
        total_gt += len(gt_dets)

        for g in gt_dets:
            gt_id_frames[g[0]].add(frame)
        for p in pred_dets:
            pred_id_frames[p[0]].add(frame)

        if not gt_dets or not pred_dets:
            total_fn += len(gt_dets)
            total_fp += len(pred_dets)
            continue

        # Cost matrix (negative IoU)
        n_gt = len(gt_dets)
        n_pred = len(pred_dets)
        cost = np.zeros((n_gt, n_pred))
        for i, g in enumerate(gt_dets):
            for j, p in enumerate(pred_dets):
                cost[i, j] = -bbox_iou(g[1:], p[1:])

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pred = set()
        curr_match = {}

        for r, c in zip(row_ind, col_ind):
            iou = -cost[r, c]
            if iou >= iou_threshold:
                matched_gt.add(r)
                matched_pred.add(c)
                gt_id = gt_dets[r][0]
                pred_id = pred_dets[c][0]
                curr_match[gt_id] = pred_id
                total_matches += 1
                match_pairs[gt_id][pred_id] += 1

                # ID switch
                if gt_id in prev_match and prev_match[gt_id] != pred_id:
                    total_id_sw += 1

                # Fragmentation: GT was unmatched last frame but matched now to different pred
                if gt_id in gt_id_last_matched and gt_id_last_matched[gt_id] != pred_id:
                    fragments += 1
                gt_id_last_matched[gt_id] = pred_id

        total_fn += n_gt - len(matched_gt)
        total_fp += n_pred - len(matched_pred)
        prev_match = curr_match

    # MOTA
    mota = 1.0 - (total_fn + total_fp + total_id_sw) / total_gt if total_gt > 0 else 0.0

    # IDF1: find best matching between GT IDs and pred IDs
    all_gt_ids = sorted(gt_id_frames.keys())
    all_pred_ids = sorted(pred_id_frames.keys())

    if all_gt_ids and all_pred_ids:
        # Build cost matrix for ID matching
        id_cost = np.zeros((len(all_gt_ids), len(all_pred_ids)))
        for i, gid in enumerate(all_gt_ids):
            for j, pid in enumerate(all_pred_ids):
                id_cost[i, j] = -match_pairs[gid][pid]

        id_row, id_col = linear_sum_assignment(id_cost)

        idtp = 0
        for r, c in zip(id_row, id_col):
            idtp += -id_cost[r, c]

        idfn = total_gt - idtp
        idfp = sum(len(pred.get(f, [])) for f in all_frames) - idtp
        # Wait, let me recount properly
        total_pred = sum(len(pred.get(f, [])) for f in all_frames)
        idf1 = 2 * idtp / (total_gt + total_pred) if (total_gt + total_pred) > 0 else 0.0
    else:
        idf1 = 0.0
        total_pred = sum(len(pred.get(f, [])) for f in all_frames)

    precision = total_matches / (total_matches + total_fp) if (total_matches + total_fp) > 0 else 0.0
    recall = total_matches / total_gt if total_gt > 0 else 0.0

    return {
        "MOTA": mota,
        "IDF1": idf1,
        "ID_Sw": total_id_sw,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "Matches": total_matches,
        "GT_Dets": total_gt,
        "Pred_Dets": total_matches + total_fp,
        "Fragments": fragments,
        "GT_IDs": len(all_gt_ids),
        "Pred_IDs": len(all_pred_ids),
    }


def print_results(name, metrics):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  MOTA:       {metrics['MOTA']:.4f}")
    print(f"  IDF1:       {metrics['IDF1']:.4f}")
    print(f"  ID Sw:      {metrics['ID_Sw']}")
    print(f"  Fragments:  {metrics['Fragments']}")
    print(f"  Precision:  {metrics['Precision']:.4f}")
    print(f"  Recall:     {metrics['Recall']:.4f}")
    print(f"  FP:         {metrics['FP']}")
    print(f"  FN:         {metrics['FN']}")
    print(f"  GT dets:    {metrics['GT_Dets']}")
    print(f"  Pred dets:  {metrics['Pred_Dets']}")
    print(f"  GT IDs:     {metrics['GT_IDs']}")
    print(f"  Pred IDs:   {metrics['Pred_IDs']}")


def print_comparison(m_orig, m_retr):
    def arrow(old, new, higher_better=True):
        if new > old:
            return "+" if higher_better else "!"
        elif new < old:
            return "-" if higher_better else "+"
        return "="

    print(f"\n{'='*60}")
    print(f"  COMPARISON: Original vs Retracked")
    print(f"{'='*60}")
    print(f"  {'Metric':<14} {'Original':>10} {'Retracked':>10} {'Delta':>10}")
    print(f"  {'-'*44}")

    for key, hb in [("MOTA", True), ("IDF1", True), ("ID_Sw", False),
                     ("Fragments", False), ("Precision", True), ("Recall", True),
                     ("FP", False), ("FN", False), ("Pred_IDs", False)]:
        o, r = m_orig[key], m_retr[key]
        if isinstance(o, float):
            delta = r - o
            tag = arrow(o, r, hb)
            print(f"  {key:<14} {o:>10.4f} {r:>10.4f} {delta:>+10.4f} {tag}")
        else:
            delta = r - o
            tag = arrow(o, r, hb)
            print(f"  {key:<14} {o:>10d} {r:>10d} {delta:>+10d} {tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True, help="Original prediction dir")
    parser.add_argument("--pred_retracked", default=None, help="Retracked prediction dir")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    gt, frame_names = load_gt(args.gt)
    print(f"Loaded GT: {sum(len(v) for v in gt.values())} detections across {len(gt)} frames, "
          f"{len(set(d[0] for dets in gt.values() for d in dets))} IDs")

    pred_orig = load_pred(args.pred, frame_names)
    m_orig = evaluate(gt, pred_orig, args.iou_threshold)
    print_results("Original Tracking", m_orig)

    if args.pred_retracked:
        pred_retr = load_pred(args.pred_retracked, frame_names)
        m_retr = evaluate(gt, pred_retr, args.iou_threshold)
        print_results("Retracked (Kalman + Dormant Re-ID)", m_retr)
        print_comparison(m_orig, m_retr)


if __name__ == "__main__":
    main()
