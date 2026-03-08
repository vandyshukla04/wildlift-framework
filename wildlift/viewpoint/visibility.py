#!/usr/bin/env python3
"""
Viewpoint Visibility Evaluation Tool for WildLIFT Pipeline

Two-mode tool:
  1. label  — tkinter GUI for human labeling of visible animal sides per frame
  2. evaluate — confusion matrix + metrics comparing human labels vs algorithm

Runs evaluation with BOTH score types (geometric visibility and occlusion-adjusted
effective score) to quantify how much inter-animal occlusion affects accuracy.

Usage:
    # Human labeling
    python visibility_eval.py label \
        --annotator_output results/.../corrected_bboxes \
        --images_dir examples/wd_data/zebras/zebr-3 \
        --output human_visibility_labels.json

    # Evaluation
    python visibility_eval.py evaluate \
        --annotator_output results/.../corrected_bboxes \
        --images_dir examples/wd_data/zebras/zebr-3 \
        --human_labels human_visibility_labels.json \
        --threshold 0.1 \
        --output_dir eval_results/
"""

import os
import sys
import json
import tempfile
import argparse
import re
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for PDF generation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

from wildlift.viewpoint.analyzer import ViewpointAnalyzer, MaskCropExtractor, CONFIG

try:
    from wildlift.viewpoint.occlusion import OcclusionAnalyzer
    OCCLUSION_AVAILABLE = True
except ImportError:
    OCCLUSION_AVAILABLE = False

SIDES = CONFIG['VISIBLE_FACES']  # ['front', 'back', 'left', 'right', 'top']
SIDE_COLORS = CONFIG['NATURE_STYLE']['ORIENTATION_COLORS']


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PerSideMetrics:
    side: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    kappa: float = 0.0
    auc: float = 0.0


@dataclass
class EvaluationResult:
    score_type: str
    threshold: float
    per_side: Dict[str, PerSideMetrics] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    hamming_loss: float = 0.0
    subset_accuracy: float = 0.0
    mean_iou: float = 0.0
    macro_f1: float = 0.0
    macro_kappa: float = 0.0
    roc_curves: Dict[str, Tuple] = field(default_factory=dict)
    calibration_data: Optional[Tuple] = None
    per_track_metrics: Dict = field(default_factory=dict)
    temporal_data: Dict = field(default_factory=dict)
    n_frames_evaluated: int = 0
    # Raw data for advanced visualizations
    pairs: List = field(default_factory=list)  # (tid, fname, human_set, algo_dict)
    optimal_thresholds: Dict = field(default_factory=dict)  # {side: (threshold, f1)}


# =============================================================================
# ALGORITHM LABEL EXTRACTOR
# =============================================================================

class AlgorithmLabelExtractor:
    """Extract per-side visibility scores from viewpoint/occlusion analysis."""

    def __init__(self, annotator_output, images_dir, occlusion_json=None):
        self.vp = ViewpointAnalyzer(annotator_output, images_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.annotator_output = Path(annotator_output)

        self.occlusion_data = None
        if occlusion_json and Path(occlusion_json).exists():
            self._load_occlusion_data(occlusion_json)

        self._occlusion_analyzer = None

    def _load_occlusion_data(self, json_path):
        """Load pre-computed occlusion_summary.json."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.occlusion_data = {}
        for tid_str, track_data in data.get('tracks', {}).items():
            tid = int(tid_str)
            self.occlusion_data[tid] = {}
            for frame_entry in track_data.get('per_frame', []):
                frame = str(frame_entry['frame'])
                self.occlusion_data[tid][frame] = {}
                for face_name, face_info in frame_entry.get('faces', {}).items():
                    self.occlusion_data[tid][frame][face_name] = {
                        'visibility_score': face_info.get('visibility_score', 0.0),
                        'effective_score': face_info.get('effective_score', 0.0),
                        'ray_occlusion_pct': face_info.get('ray_occlusion_pct', 0.0),
                    }

    def _ensure_occlusion_analyzer(self):
        """Lazily create OcclusionAnalyzer if needed."""
        if self._occlusion_analyzer is None:
            if not OCCLUSION_AVAILABLE:
                raise ImportError("occlusion_analyzer not available")
            self._occlusion_analyzer = OcclusionAnalyzer(
                str(self.annotator_output),
                str(self.images_dir) if self.images_dir else None,
                ray_samples=8,
            )
        return self._occlusion_analyzer

    def get_visibility_scores(self, track_id, frame_name):
        """Get geometric visibility scores (dot product only) for all sides."""
        scores = {}
        if track_id not in self.vp.semantic_faces:
            return {s: 0.0 for s in SIDES}
        if frame_name not in self.vp.semantic_faces[track_id]:
            return {s: 0.0 for s in SIDES}

        camera_params = self.vp._load_camera_params(frame_name)
        if camera_params is None:
            return {s: 0.0 for s in SIDES}

        semantic_faces = self.vp.semantic_faces[track_id][frame_name]
        for side in SIDES:
            if side not in semantic_faces:
                scores[side] = 0.0
                continue
            face_data = semantic_faces[side]
            is_visible, vis_score = self.vp._calculate_face_visibility(
                face_data, camera_params
            )
            scores[side] = max(vis_score, 0.0)
        return scores

    def get_effective_scores(self, track_id, frame_name):
        """Get effective scores (visibility * (1 - occlusion%)) for all sides."""
        # Try pre-computed data first
        if self.occlusion_data:
            frame_data = self.occlusion_data.get(track_id, {}).get(frame_name, {})
            if frame_data:
                return {side: frame_data.get(side, {}).get('effective_score', 0.0)
                        for side in SIDES}

        # Compute on the fly
        oa = self._ensure_occlusion_analyzer()
        records = oa.compute_frame_occlusion(frame_name)
        if track_id not in records:
            return {s: 0.0 for s in SIDES}

        record = records[track_id]
        scores = {}
        for side in SIDES:
            detail = record.face_details.get(side)
            scores[side] = detail.effective_score if detail else 0.0
        return scores

    def get_occlusion_pct(self, track_id, frame_name):
        """Get per-side ray occlusion percentages."""
        if self.occlusion_data:
            frame_data = self.occlusion_data.get(track_id, {}).get(frame_name, {})
            if frame_data:
                return {side: frame_data.get(side, {}).get('ray_occlusion_pct', 0.0)
                        for side in SIDES}

        oa = self._ensure_occlusion_analyzer()
        records = oa.compute_frame_occlusion(frame_name)
        if track_id not in records:
            return {s: 0.0 for s in SIDES}

        record = records[track_id]
        return {side: record.face_details.get(side, type('', (), {'ray_occlusion_pct': 0.0})).ray_occlusion_pct
                for side in SIDES}

    def extract_all(self, track_frames, score_type='visibility'):
        """
        Extract scores for a list of (track_id, frame_name) pairs.

        Args:
            track_frames: list of (track_id, frame_name) tuples
            score_type: 'visibility' or 'effective'

        Returns:
            dict: {track_id: {frame_name: {side: float_score}}}
        """
        results = defaultdict(dict)
        get_fn = (self.get_visibility_scores if score_type == 'visibility'
                  else self.get_effective_scores)

        for i, (tid, fname) in enumerate(track_frames):
            if (i + 1) % 50 == 0:
                print(f"  Extracting {score_type} scores: {i+1}/{len(track_frames)}")
            results[tid][fname] = get_fn(tid, fname)

        return dict(results)


# =============================================================================
# HUMAN LABELING TOOL (tkinter)
# =============================================================================

class HumanLabelingTool:
    """Tkinter GUI for labeling visible sides per frame per track."""

    def __init__(self, annotator_output, images_dir, output_path, annotator_name=""):
        self.annotator_output = Path(annotator_output)
        self.images_dir = Path(images_dir)
        self.output_path = Path(output_path)
        self.annotator_name = annotator_name

        # Load viewpoint analyzer for geometry + projections
        self.vp = ViewpointAnalyzer(str(annotator_output), str(images_dir))

        # Load mask extractor for overlays
        results_dir = self.annotator_output.parent
        self.mask_extractor = MaskCropExtractor(
            images_dir=self.images_dir,
            results_dir=results_dir,
        )

        # Build (track_id, frame_name) iteration list
        self.pairs = self._build_pair_list()
        print(f"Total (track, frame) pairs to label: {len(self.pairs)}")

        # Load existing labels or start fresh
        self.labels_data = self._load_existing_labels()
        self.current_idx = self._find_first_unlabeled()

        # Track colors for mask overlay
        self.track_colors = {}
        for i, tid in enumerate(sorted(self.vp.all_bbox_data.keys())):
            hue = (i * 0.618033988749895) % 1.0
            r, g, b = [int(c * 255) for c in __import__('colorsys').hsv_to_rgb(hue, 0.8, 0.9)]
            self.track_colors[tid] = (b, g, r)  # BGR for OpenCV

    def _build_pair_list(self):
        """Build sorted list of (track_id, frame_name) pairs."""
        pairs = []
        for tid in sorted(self.vp.all_bbox_data.keys()):
            for fname in self.vp.frame_order:
                if fname in self.vp.all_bbox_data[tid]:
                    pairs.append((tid, fname))
        return pairs

    def _load_existing_labels(self):
        """Load existing labels file or create fresh."""
        if self.output_path.exists():
            with open(self.output_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded existing labels: {data['metadata'].get('total_labeled', 0)} labeled")
            return data

        return {
            "metadata": {
                "scene": self.annotator_output.parent.name,
                "annotator": self.annotator_name,
                "created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "total_labeled": 0,
                "total_skipped": 0,
            },
            "labels": {},
            "skipped": {},
        }

    def _find_first_unlabeled(self):
        """Find the index of the first unlabeled pair."""
        for i, (tid, fname) in enumerate(self.pairs):
            tid_str = str(tid)
            if tid_str not in self.labels_data["labels"]:
                return i
            if fname not in self.labels_data["labels"][tid_str]:
                # Also check if it's in skipped
                if fname not in self.labels_data.get("skipped", {}).get(tid_str, []):
                    return i
        return len(self.pairs)  # all labeled

    def _save_labels(self):
        """Atomic save of labels."""
        self.labels_data["metadata"]["last_modified"] = datetime.now().isoformat()

        # Count totals
        total_labeled = sum(
            len(frames) for frames in self.labels_data["labels"].values()
        )
        total_skipped = sum(
            len(frames) for frames in self.labels_data.get("skipped", {}).values()
        )
        self.labels_data["metadata"]["total_labeled"] = total_labeled
        self.labels_data["metadata"]["total_skipped"] = total_skipped

        # Atomic write
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.output_path.parent, suffix='.json'
        )
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(self.labels_data, f, indent=2)
            os.replace(tmp_path, self.output_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _get_prev_label(self, track_id, current_idx):
        """Get visible_sides from the most recent labeled frame of the same track."""
        tid_str = str(track_id)
        # Walk backwards through pairs to find the previous frame for this track
        for i in range(current_idx - 1, -1, -1):
            prev_tid, prev_fname = self.pairs[i]
            if prev_tid != track_id:
                continue
            # Check if this frame was labeled
            if (tid_str in self.labels_data["labels"] and
                    prev_fname in self.labels_data["labels"][tid_str]):
                return set(self.labels_data["labels"][tid_str][prev_fname].get("visible_sides", []))
        return set()

    def _render_frame(self, track_id, frame_name):
        """Render frame image with target track mask highlighted (no bounding boxes)."""
        image = self.mask_extractor._load_image(frame_name)
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)

        overlay = image.copy()

        # Dim other tracks' masks
        for other_tid in self.vp.all_bbox_data:
            if other_tid == track_id:
                continue
            if frame_name not in self.vp.all_bbox_data[other_tid]:
                continue
            mask = self.mask_extractor._load_mask(frame_name, other_tid)
            if mask is not None:
                overlay[mask > 0] = (overlay[mask > 0] * 0.4).astype(np.uint8)

        # Highlight target track mask
        mask = self.mask_extractor._load_mask(frame_name, track_id)
        if mask is not None:
            color = self.track_colors.get(track_id, (0, 255, 0))
            color_overlay = np.zeros_like(overlay)
            color_overlay[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 0.7, color_overlay, 0.3, 0)

        return overlay

    def run(self):
        """Launch the tkinter labeling GUI."""
        import tkinter as tk
        from PIL import Image, ImageTk

        self.root = tk.Tk()
        self.root.title("Visibility Labeling Tool")

        # State
        self.side_vars = {}
        self.flagged_var = tk.BooleanVar(value=False)

        # Main layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: image
        self.image_label = tk.Label(main_frame)
        self.image_label.pack(side=tk.LEFT, padx=5)

        # Right: controls
        ctrl_frame = tk.Frame(main_frame)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)

        tk.Label(ctrl_frame, text="VISIBLE SIDES:", font=("Helvetica", 11, "bold")).pack(anchor='w')
        tk.Label(ctrl_frame, text="").pack()  # spacer

        for i, side in enumerate(SIDES):
            var = tk.BooleanVar(value=False)
            self.side_vars[side] = var
            cb = tk.Checkbutton(ctrl_frame, text=f"{side.capitalize()}  ({i+1})",
                                variable=var, font=("Helvetica", 10))
            cb.pack(anchor='w', pady=2)

        tk.Label(ctrl_frame, text="").pack()  # spacer

        flag_cb = tk.Checkbutton(ctrl_frame, text="Flag for review (F)",
                                 variable=self.flagged_var, font=("Helvetica", 9))
        flag_cb.pack(anchor='w', pady=2)

        tk.Label(ctrl_frame, text="").pack()

        # Buttons
        btn_frame = tk.Frame(ctrl_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        tk.Button(btn_frame, text="<< Prev", command=self._prev, width=8).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Skip (S)", command=self._skip, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Next >>", command=self._next, width=8).pack(side=tk.LEFT)

        tk.Label(ctrl_frame, text="").pack()
        tk.Button(ctrl_frame, text="Confirm & Advance (Enter)",
                  command=self._confirm, width=25,
                  bg='#4CAF50', fg='white', font=("Helvetica", 10, "bold")).pack(pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Helvetica", 9), anchor='w').pack(fill=tk.X, padx=5, pady=3)

        # Progress bar
        self.progress_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.progress_var,
                 font=("Helvetica", 9), anchor='w').pack(fill=tk.X, padx=5, pady=3)

        # Key bindings
        self.root.bind('1', lambda e: self._toggle_side(0))
        self.root.bind('2', lambda e: self._toggle_side(1))
        self.root.bind('3', lambda e: self._toggle_side(2))
        self.root.bind('4', lambda e: self._toggle_side(3))
        self.root.bind('5', lambda e: self._toggle_side(4))
        self.root.bind('<Return>', lambda e: self._confirm())
        self.root.bind('<space>', lambda e: self._confirm())
        self.root.bind('<Left>', lambda e: self._prev())
        self.root.bind('<Right>', lambda e: self._next())
        self.root.bind('s', lambda e: self._skip())
        self.root.bind('f', lambda e: self.flagged_var.set(not self.flagged_var.get()))

        # Display first frame
        self._display_current()

        self.root.mainloop()

    def _toggle_side(self, idx):
        side = SIDES[idx]
        self.side_vars[side].set(not self.side_vars[side].get())

    def _display_current(self):
        from PIL import Image, ImageTk

        if self.current_idx >= len(self.pairs):
            self.status_var.set("All frames labeled! Close window when done.")
            self.progress_var.set(f"Progress: {len(self.pairs)}/{len(self.pairs)} (100%)")
            return

        tid, fname = self.pairs[self.current_idx]

        # Render frame
        frame_img = self._render_frame(tid, fname)

        # Resize for display (max 700px wide)
        h, w = frame_img.shape[:2]
        max_w = 700
        if w > max_w:
            scale = max_w / w
            frame_img = cv2.resize(frame_img, (max_w, int(h * scale)))

        # Convert BGR -> RGB -> PIL -> Tk
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self._tk_image)

        # Load existing labels if this pair was already labeled
        tid_str = str(tid)
        if (tid_str in self.labels_data["labels"] and
                fname in self.labels_data["labels"][tid_str]):
            existing = self.labels_data["labels"][tid_str][fname]
            visible = existing.get("visible_sides", [])
            for side in SIDES:
                self.side_vars[side].set(side in visible)
            self.flagged_var.set(existing.get("flagged", False))
        else:
            # For unlabeled frames: carry forward from previous frame of same track.
            # Check if previous pair in the list is the same track — if so,
            # checkboxes already hold the right state from the last confirm/display.
            prev_tid = self.pairs[self.current_idx - 1][0] if self.current_idx > 0 else None
            if prev_tid != tid:
                # Track changed — reset checkboxes, try to load from saved labels
                prev_visible = self._get_prev_label(tid, self.current_idx)
                for side in SIDES:
                    self.side_vars[side].set(side in prev_visible)
            # else: same track — leave checkboxes as-is (carried from previous frame)
            self.flagged_var.set(False)

        # Update status
        self.root.title(f"Visibility Labeler - Track {tid}, Frame {fname} "
                        f"({self.current_idx + 1}/{len(self.pairs)})")
        labeled = self.labels_data["metadata"].get("total_labeled", 0)
        skipped = self.labels_data["metadata"].get("total_skipped", 0)
        self.status_var.set(f"Track {tid} | Frame {fname} | "
                            f"Labeled: {labeled} | Skipped: {skipped}")
        pct = (self.current_idx / len(self.pairs)) * 100
        self.progress_var.set(f"Progress: {self.current_idx}/{len(self.pairs)} ({pct:.1f}%)")

    def _confirm(self):
        """Save current label and advance."""
        if self.current_idx >= len(self.pairs):
            return

        tid, fname = self.pairs[self.current_idx]
        tid_str = str(tid)

        visible_sides = [side for side in SIDES if self.side_vars[side].get()]

        if tid_str not in self.labels_data["labels"]:
            self.labels_data["labels"][tid_str] = {}

        self.labels_data["labels"][tid_str][fname] = {
            "visible_sides": visible_sides,
            "flagged": self.flagged_var.get(),
        }

        self._save_labels()
        self.current_idx += 1
        self._display_current()

    def _skip(self):
        """Skip current pair."""
        if self.current_idx >= len(self.pairs):
            return

        tid, fname = self.pairs[self.current_idx]
        tid_str = str(tid)

        if tid_str not in self.labels_data.get("skipped", {}):
            self.labels_data.setdefault("skipped", {})[tid_str] = []
        if fname not in self.labels_data["skipped"][tid_str]:
            self.labels_data["skipped"][tid_str].append(fname)

        self._save_labels()
        self.current_idx += 1
        self._display_current()

    def _next(self):
        if self.current_idx < len(self.pairs) - 1:
            self.current_idx += 1
            self._display_current()

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._display_current()


# =============================================================================
# CONFUSION MATRIX EVALUATOR
# =============================================================================

class ConfusionMatrixEvaluator:
    """Compute all metrics comparing human labels vs algorithm scores."""

    def __init__(self, human_labels, algo_scores, threshold=0.1, score_type='visibility'):
        """
        Args:
            human_labels: dict from labels JSON {track_id_str: {frame: {visible_sides: [...]}}}
            algo_scores: dict {track_id: {frame: {side: float_score}}}
            threshold: score threshold for binary classification
            score_type: label for this evaluation run
        """
        self.human_labels = human_labels
        self.algo_scores = algo_scores
        self.threshold = threshold
        self.score_type = score_type

        # Build aligned pairs
        self.pairs = []  # (track_id, frame_name, human_visible_set, algo_score_dict)
        for tid_str, frames in human_labels.items():
            tid = int(tid_str)
            for fname, ldata in frames.items():
                if tid in algo_scores and fname in algo_scores[tid]:
                    human_set = set(ldata.get("visible_sides", []))
                    algo_dict = algo_scores[tid][fname]
                    self.pairs.append((tid, fname, human_set, algo_dict))

        print(f"ConfusionMatrixEvaluator: {len(self.pairs)} aligned (track, frame) pairs "
              f"[score_type={score_type}, threshold={threshold}]")

    def _algo_binary(self, algo_dict, threshold=None):
        """Convert continuous scores to binary set of visible sides."""
        if threshold is None:
            threshold = self.threshold
        return {side for side in SIDES if algo_dict.get(side, 0.0) >= threshold}

    def compute_per_side_confusion(self):
        """Compute TP/FP/FN/TN per side."""
        metrics = {}
        for side in SIDES:
            tp = fp = fn = tn = 0
            for tid, fname, human_set, algo_dict in self.pairs:
                h = side in human_set
                a = algo_dict.get(side, 0.0) >= self.threshold
                if h and a:
                    tp += 1
                elif a and not h:
                    fp += 1
                elif h and not a:
                    fn += 1
                else:
                    tn += 1

            n = tp + fp + fn + tn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            # Cohen's Kappa
            if n > 0:
                p_o = (tp + tn) / n
                p_yes = ((tp + fp) / n) * ((tp + fn) / n)
                p_no = ((tn + fn) / n) * ((tn + fp) / n)
                p_e = p_yes + p_no
                kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0
            else:
                kappa = 0.0

            metrics[side] = PerSideMetrics(
                side=side, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, kappa=kappa,
            )
        return metrics

    def compute_accuracy_hamming(self):
        """Overall accuracy and Hamming loss across all (side, frame) pairs."""
        correct = 0
        total = 0
        for tid, fname, human_set, algo_dict in self.pairs:
            algo_set = self._algo_binary(algo_dict)
            for side in SIDES:
                total += 1
                if (side in human_set) == (side in algo_set):
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        hamming = 1.0 - accuracy
        return accuracy, hamming

    def compute_subset_accuracy(self):
        """Fraction of frames where human and algo sets match exactly."""
        exact = 0
        for tid, fname, human_set, algo_dict in self.pairs:
            algo_set = self._algo_binary(algo_dict)
            if human_set == algo_set:
                exact += 1
        return exact / len(self.pairs) if self.pairs else 0.0

    def compute_iou(self):
        """Mean IoU of visible-side sets."""
        ious = []
        for tid, fname, human_set, algo_dict in self.pairs:
            algo_set = self._algo_binary(algo_dict)
            intersection = len(human_set & algo_set)
            union = len(human_set | algo_set)
            iou = intersection / union if union > 0 else 1.0  # both empty = perfect match
            ious.append(iou)
        mean_iou = np.mean(ious) if ious else 0.0
        return float(mean_iou), ious

    def compute_roc_auc(self):
        """Per-side ROC/AUC curves by sweeping threshold."""
        roc_data = {}
        for side in SIDES:
            y_true = []
            y_score = []
            for tid, fname, human_set, algo_dict in self.pairs:
                y_true.append(1 if side in human_set else 0)
                y_score.append(algo_dict.get(side, 0.0))

            y_true = np.array(y_true)
            y_score = np.array(y_score)

            if len(np.unique(y_true)) < 2:
                # Only one class — AUC undefined
                roc_data[side] = (np.array([0, 1]), np.array([0, 1]), 0.5)
                continue

            # Sort by descending score
            sorted_idx = np.argsort(-y_score)
            y_true_sorted = y_true[sorted_idx]
            y_score_sorted = y_score[sorted_idx]

            # Compute ROC at unique thresholds
            thresholds = np.unique(y_score_sorted)[::-1]
            total_pos = y_true.sum()
            total_neg = len(y_true) - total_pos

            fprs = [0.0]
            tprs = [0.0]
            for t in thresholds:
                predicted = y_score >= t
                tp = np.sum(predicted & (y_true == 1))
                fp = np.sum(predicted & (y_true == 0))
                tprs.append(tp / max(total_pos, 1))
                fprs.append(fp / max(total_neg, 1))
            fprs.append(1.0)
            tprs.append(1.0)

            fprs = np.array(fprs)
            tprs = np.array(tprs)

            # Sort by FPR for proper AUC
            sort_idx = np.argsort(fprs)
            fprs = fprs[sort_idx]
            tprs = tprs[sort_idx]

            auc = float(np.trapz(tprs, fprs))
            roc_data[side] = (fprs, tprs, auc)

        return roc_data

    def compute_calibration(self, n_bins=10):
        """Calibration: bin scores, compute fraction of human 'visible' labels per bin."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_counts = np.zeros(n_bins)
        bin_positives = np.zeros(n_bins)

        for tid, fname, human_set, algo_dict in self.pairs:
            for side in SIDES:
                score = algo_dict.get(side, 0.0)
                is_visible = 1 if side in human_set else 0
                bin_idx = min(int(score * n_bins), n_bins - 1)
                bin_counts[bin_idx] += 1
                bin_positives[bin_idx] += is_visible

        with np.errstate(divide='ignore', invalid='ignore'):
            fractions = np.where(bin_counts > 0, bin_positives / bin_counts, np.nan)
        return bin_centers, fractions, bin_counts

    def compute_per_track(self):
        """Metrics broken down per track."""
        track_groups = defaultdict(list)
        for tid, fname, human_set, algo_dict in self.pairs:
            track_groups[tid].append((tid, fname, human_set, algo_dict))

        per_track = {}
        for tid, pairs in sorted(track_groups.items()):
            sub_eval = ConfusionMatrixEvaluator.__new__(ConfusionMatrixEvaluator)
            sub_eval.human_labels = self.human_labels
            sub_eval.algo_scores = self.algo_scores
            sub_eval.threshold = self.threshold
            sub_eval.score_type = self.score_type
            sub_eval.pairs = pairs

            metrics = sub_eval.compute_per_side_confusion()
            accuracy, hamming = sub_eval.compute_accuracy_hamming()
            subset = sub_eval.compute_subset_accuracy()
            mean_iou, _ = sub_eval.compute_iou()

            macro_f1 = np.mean([m.f1 for m in metrics.values()])
            macro_kappa = np.mean([m.kappa for m in metrics.values()])

            per_track[tid] = {
                'n_frames': len(pairs),
                'accuracy': accuracy,
                'hamming_loss': hamming,
                'subset_accuracy': subset,
                'mean_iou': mean_iou,
                'macro_f1': float(macro_f1),
                'macro_kappa': float(macro_kappa),
                'per_side': {s: {'f1': m.f1, 'precision': m.precision,
                                  'recall': m.recall, 'kappa': m.kappa}
                              for s, m in metrics.items()},
            }
        return per_track

    def compute_temporal(self):
        """Temporal analysis: disagreement rate over frame sequence."""
        frame_disagreements = defaultdict(int)
        frame_total = defaultdict(int)

        for tid, fname, human_set, algo_dict in self.pairs:
            algo_set = self._algo_binary(algo_dict)
            disagreements = sum(1 for s in SIDES if (s in human_set) != (s in algo_set))
            frame_disagreements[fname] += disagreements
            frame_total[fname] += len(SIDES)

        # Sort by frame number
        frames_sorted = sorted(frame_disagreements.keys(),
                               key=lambda x: int(re.findall(r'\d+', x)[0])
                               if re.findall(r'\d+', x) else 0)

        rates = [frame_disagreements[f] / frame_total[f]
                 for f in frames_sorted if frame_total[f] > 0]

        # Rolling mean (window=5 or fewer)
        window = min(5, len(rates))
        if window > 0 and len(rates) >= window:
            rolling = np.convolve(rates, np.ones(window) / window, mode='valid')
        else:
            rolling = np.array(rates)

        return {
            'frames': frames_sorted,
            'disagreement_rates': rates,
            'rolling_mean': rolling.tolist() if len(rolling) > 0 else [],
            'window': window,
        }

    def compute_optimal_thresholds(self):
        """Find per-side threshold that maximizes F1 by sweeping 0-1."""
        thresholds = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00
        optimal = {}

        for side in SIDES:
            y_true = []
            y_score = []
            for tid, fname, human_set, algo_dict in self.pairs:
                y_true.append(side in human_set)
                y_score.append(algo_dict.get(side, 0.0))

            y_true = np.array(y_true)
            y_score = np.array(y_score)

            best_t, best_f1 = self.threshold, 0.0
            f1_curve = []

            for t in thresholds:
                predicted = y_score >= t
                tp = np.sum(predicted & y_true)
                fp = np.sum(predicted & ~y_true)
                fn = np.sum(~predicted & y_true)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
                f1_curve.append(f1)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)

            optimal[side] = {
                'threshold': best_t,
                'f1': best_f1,
                'f1_curve': f1_curve,
                'thresholds': thresholds.tolist(),
            }

        return optimal

    def run_all(self):
        """Run all metrics and return EvaluationResult."""
        per_side = self.compute_per_side_confusion()
        accuracy, hamming = self.compute_accuracy_hamming()
        subset = self.compute_subset_accuracy()
        mean_iou, iou_list = self.compute_iou()
        roc = self.compute_roc_auc()
        calibration = self.compute_calibration()
        per_track = self.compute_per_track()
        temporal = self.compute_temporal()
        optimal = self.compute_optimal_thresholds()

        # Update AUC in per_side metrics
        for side in SIDES:
            if side in roc:
                per_side[side].auc = roc[side][2]

        macro_f1 = float(np.mean([m.f1 for m in per_side.values()]))
        macro_kappa = float(np.mean([m.kappa for m in per_side.values()]))

        return EvaluationResult(
            score_type=self.score_type,
            threshold=self.threshold,
            per_side=per_side,
            overall_accuracy=accuracy,
            hamming_loss=hamming,
            subset_accuracy=subset,
            mean_iou=mean_iou,
            macro_f1=macro_f1,
            macro_kappa=macro_kappa,
            roc_curves=roc,
            calibration_data=calibration,
            per_track_metrics=per_track,
            temporal_data=temporal,
            n_frames_evaluated=len(self.pairs),
            pairs=self.pairs,
            optimal_thresholds=optimal,
        )


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def _apply_nature_rcparams():
    """Set matplotlib rcParams for Nature Methods publication style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 6,
        'axes.labelsize': 6,
        'axes.titlesize': 7,
        'axes.titleweight': 'bold',
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'legend.fontsize': 5,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'lines.linewidth': 0.8,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.25,
    })


class EvaluationReportGenerator:
    """Generate PDF report and JSON summary from evaluation results."""

    def __init__(self, results: List[EvaluationResult], scene_name: str = ""):
        """
        Args:
            results: list of EvaluationResult (one per score type)
            scene_name: name of the scene
        """
        self.results = results
        self.scene_name = scene_name

    def save_json(self, output_path):
        """Save all metrics as JSON."""
        data = {
            'scene': self.scene_name,
            'generated': datetime.now().isoformat(),
            'evaluations': [],
        }

        for r in self.results:
            eval_data = {
                'score_type': r.score_type,
                'threshold': r.threshold,
                'n_frames_evaluated': r.n_frames_evaluated,
                'overall_accuracy': r.overall_accuracy,
                'hamming_loss': r.hamming_loss,
                'subset_accuracy': r.subset_accuracy,
                'mean_iou': r.mean_iou,
                'macro_f1': r.macro_f1,
                'macro_kappa': r.macro_kappa,
                'per_side': {},
                'per_track': r.per_track_metrics,
                'temporal': r.temporal_data,
            }
            for side, m in r.per_side.items():
                eval_data['per_side'][side] = {
                    'tp': m.tp, 'fp': m.fp, 'fn': m.fn, 'tn': m.tn,
                    'precision': m.precision, 'recall': m.recall,
                    'f1': m.f1, 'kappa': m.kappa, 'auc': m.auc,
                }
            data['evaluations'].append(eval_data)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved metrics JSON: {output_path}")

    def generate_pdf(self, output_path):
        """Generate multi-page PDF report."""
        _apply_nature_rcparams()

        with PdfPages(output_path) as pdf:
            self._page_executive_summary(pdf)
            for r in self.results:
                self._page_confusion_matrices(pdf, r)
            self._page_roc_curves(pdf)
            self._page_calibration(pdf)
            self._page_per_track(pdf)
            if any(r.temporal_data.get('frames') for r in self.results):
                self._page_temporal(pdf)
            # New visualizations: binary-vs-continuous analysis
            for r in self.results:
                if r.pairs:
                    self._page_score_distributions(pdf, r)
                    self._page_temporal_heatmap(pdf, r)
                if r.optimal_thresholds:
                    self._page_optimal_thresholds(pdf, r)

        print(f"Saved PDF report: {output_path}")

    def _page_executive_summary(self, pdf):
        """Page 1: Summary metrics for all score types."""
        fig = plt.figure(figsize=(8.5, 11))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 2, 2], hspace=0.35)

        # Title
        ax_title = fig.add_subplot(gs[0])
        ax_title.axis('off')
        ax_title.text(0.5, 0.8, f"Viewpoint Visibility Evaluation: {self.scene_name}",
                      ha='center', fontsize=10, fontweight='bold',
                      transform=ax_title.transAxes)
        ax_title.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      ha='center', fontsize=6, color='gray',
                      transform=ax_title.transAxes)

        # Summary table
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis('off')

        headers = ['Metric'] + [r.score_type for r in self.results]
        table_data = []
        metric_names = [
            ('Overall Accuracy', 'overall_accuracy'),
            ('Hamming Loss', 'hamming_loss'),
            ('Subset Accuracy', 'subset_accuracy'),
            ('Mean IoU', 'mean_iou'),
            ('Macro F1', 'macro_f1'),
            ("Macro Cohen's Kappa", 'macro_kappa'),
            ('Frames Evaluated', 'n_frames_evaluated'),
        ]
        for label, attr in metric_names:
            row = [label]
            for r in self.results:
                val = getattr(r, attr)
                if isinstance(val, float):
                    row.append(f"{val:.3f}")
                else:
                    row.append(str(val))
            table_data.append(row)

        table = ax_table.table(cellText=table_data, colLabels=headers,
                               loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.4)

        # Color header
        for j in range(len(headers)):
            table[0, j].set_facecolor('#E8EAF6')
            table[0, j].set_text_props(fontweight='bold')

        # Per-side F1 bar chart
        ax_f1 = fig.add_subplot(gs[2])
        n_results = len(self.results)
        bar_width = 0.35
        x = np.arange(len(SIDES))

        for i, r in enumerate(self.results):
            f1_values = [r.per_side[s].f1 for s in SIDES]
            colors = [SIDE_COLORS.get(s, '#999999') for s in SIDES]
            offset = (i - (n_results - 1) / 2) * bar_width
            bars = ax_f1.bar(x + offset, f1_values, bar_width,
                             label=r.score_type, color=colors,
                             alpha=0.6 + 0.3 * i, edgecolor='black', linewidth=0.3)
            for bar, val in zip(bars, f1_values):
                ax_f1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=5)

        ax_f1.set_xticks(x)
        ax_f1.set_xticklabels([s.capitalize() for s in SIDES])
        ax_f1.set_ylabel('F1 Score')
        ax_f1.set_title('Per-Side F1 Scores')
        ax_f1.set_ylim(0, 1.15)
        ax_f1.legend(fontsize=6)

        fig.subplots_adjust(hspace=0.35)
        pdf.savefig(fig)
        plt.close(fig)

    def _page_confusion_matrices(self, pdf, result):
        """One page per score type: 5 confusion matrix heatmaps."""
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 11))
        fig.suptitle(f"Confusion Matrices — {result.score_type} (threshold={result.threshold})",
                     fontsize=9, fontweight='bold', y=0.98)

        for idx, side in enumerate(SIDES):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            m = result.per_side[side]
            matrix = np.array([[m.tn, m.fp], [m.fn, m.tp]])
            total = matrix.sum()

            im = ax.imshow(matrix, cmap='Blues', aspect='auto',
                           vmin=0, vmax=max(total, 1))

            for i in range(2):
                for j in range(2):
                    val = matrix[i, j]
                    pct = val / total * 100 if total > 0 else 0
                    ax.text(j, i, f'{val}\n({pct:.1f}%)',
                            ha='center', va='center', fontsize=7,
                            color='white' if val > total * 0.5 else 'black')

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Not Vis', 'Visible'], fontsize=6)
            ax.set_yticklabels(['Not Vis', 'Visible'], fontsize=6)
            ax.set_xlabel('Algorithm', fontsize=6)
            ax.set_ylabel('Human', fontsize=6)

            color = SIDE_COLORS.get(side, '#333333')
            ax.set_title(f"{side.capitalize()}\nP={m.precision:.2f} R={m.recall:.2f} "
                         f"F1={m.f1:.2f} K={m.kappa:.2f}",
                         fontsize=6, color=color, fontweight='bold')

        # Hide 6th subplot
        axes[1, 2].axis('off')
        # Add summary text in the empty cell
        txt_lines = [
            f"Score type: {result.score_type}",
            f"Threshold: {result.threshold}",
            f"Accuracy: {result.overall_accuracy:.3f}",
            f"Hamming: {result.hamming_loss:.3f}",
            f"Subset Acc: {result.subset_accuracy:.3f}",
            f"Mean IoU: {result.mean_iou:.3f}",
        ]
        axes[1, 2].text(0.1, 0.7, '\n'.join(txt_lines), fontsize=7,
                        transform=axes[1, 2].transAxes, va='top',
                        family='monospace')

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

    def _page_roc_curves(self, pdf):
        """ROC/AUC curves for all score types."""
        fig, axes = plt.subplots(1, len(self.results), figsize=(8.5, 5),
                                 squeeze=False)

        for i, r in enumerate(self.results):
            ax = axes[0, i]
            for side in SIDES:
                fprs, tprs, auc = r.roc_curves.get(side, (np.array([0, 1]),
                                                            np.array([0, 1]), 0.5))
                color = SIDE_COLORS.get(side, '#999')
                ax.plot(fprs, tprs, color=color, linewidth=1,
                        label=f'{side} (AUC={auc:.2f})')

            ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC — {r.score_type}', fontsize=8)
            ax.legend(fontsize=5, loc='lower right')
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect('equal')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_calibration(self, pdf):
        """Calibration plot."""
        fig, axes = plt.subplots(1, len(self.results), figsize=(8.5, 5),
                                 squeeze=False)

        for i, r in enumerate(self.results):
            ax = axes[0, i]
            if r.calibration_data:
                bin_centers, fractions, counts = r.calibration_data
                valid = ~np.isnan(fractions)
                ax.plot(bin_centers[valid], fractions[valid], 'o-',
                        color='#1E88E5', linewidth=1, markersize=3)
                ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)

                # Show bin counts as bar chart on secondary axis
                ax2 = ax.twinx()
                ax2.bar(bin_centers, counts, width=0.08, alpha=0.15, color='gray')
                ax2.set_ylabel('Count', fontsize=5, color='gray')
                ax2.tick_params(axis='y', labelsize=4, colors='gray')

            ax.set_xlabel('Algorithm Score')
            ax.set_ylabel('Fraction Human "Visible"')
            ax.set_title(f'Calibration — {r.score_type}', fontsize=8)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_per_track(self, pdf):
        """Per-track performance table."""
        fig = plt.figure(figsize=(8.5, 11))

        for ri, r in enumerate(self.results):
            ax = fig.add_subplot(len(self.results), 1, ri + 1)
            ax.axis('off')
            ax.set_title(f'Per-Track Metrics — {r.score_type}', fontsize=8,
                         fontweight='bold', pad=10)

            if not r.per_track_metrics:
                ax.text(0.5, 0.5, 'No per-track data', ha='center', fontsize=7)
                continue

            headers = ['Track', 'N', 'Acc', 'Hamming', 'Subset', 'IoU', 'F1', 'Kappa']
            rows = []
            for tid in sorted(r.per_track_metrics.keys()):
                pt = r.per_track_metrics[tid]
                rows.append([
                    str(tid),
                    str(pt['n_frames']),
                    f"{pt['accuracy']:.3f}",
                    f"{pt['hamming_loss']:.3f}",
                    f"{pt['subset_accuracy']:.3f}",
                    f"{pt['mean_iou']:.3f}",
                    f"{pt['macro_f1']:.3f}",
                    f"{pt['macro_kappa']:.3f}",
                ])

            table = ax.table(cellText=rows, colLabels=headers,
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.scale(1, 1.3)
            for j in range(len(headers)):
                table[0, j].set_facecolor('#E8EAF6')
                table[0, j].set_text_props(fontweight='bold')

        pdf.savefig(fig)
        plt.close(fig)

    def _page_temporal(self, pdf):
        """Temporal analysis: disagreement rate over time."""
        fig, axes = plt.subplots(len(self.results), 1, figsize=(8.5, 4 * len(self.results)),
                                 squeeze=False)

        for i, r in enumerate(self.results):
            ax = axes[i, 0]
            td = r.temporal_data
            if not td.get('frames'):
                ax.text(0.5, 0.5, 'No temporal data', ha='center', fontsize=7)
                continue

            frames = td['frames']
            rates = td['disagreement_rates']
            rolling = td['rolling_mean']
            window = td.get('window', 5)

            x = np.arange(len(frames))
            ax.bar(x, rates, color='#90CAF9', alpha=0.6, width=0.8, label='Per-frame')

            if len(rolling) > 0:
                offset = (window - 1) // 2
                rx = np.arange(offset, offset + len(rolling))
                ax.plot(rx, rolling, color='#E53935', linewidth=1.2,
                        label=f'Rolling mean (w={window})')

            ax.set_xlabel('Frame')
            ax.set_ylabel('Disagreement Rate')
            ax.set_title(f'Temporal Disagreement — {r.score_type}', fontsize=8)
            ax.legend(fontsize=5)

            # Label some x-ticks
            if len(frames) > 10:
                step = max(len(frames) // 10, 1)
                tick_positions = list(range(0, len(frames), step))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([frames[j] for j in tick_positions],
                                   rotation=45, fontsize=4)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(frames, rotation=45, fontsize=5)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _page_score_distributions(self, pdf, result):
        """Score distribution histograms split by human label (visible vs not visible)."""
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 7))
        fig.suptitle(f"Score Distribution by Human Label — {result.score_type}",
                     fontsize=9, fontweight='bold', y=0.98)

        for idx, side in enumerate(SIDES):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            # Collect scores split by human label
            scores_visible = []
            scores_not_visible = []
            for tid, fname, human_set, algo_dict in result.pairs:
                score = algo_dict.get(side, 0.0)
                if side in human_set:
                    scores_visible.append(score)
                else:
                    scores_not_visible.append(score)

            bins = np.linspace(0, 1, 26)  # 25 bins

            if scores_not_visible:
                ax.hist(scores_not_visible, bins=bins, alpha=0.6, color='#E53935',
                        label=f'Not visible (n={len(scores_not_visible)})',
                        edgecolor='white', linewidth=0.3, density=True)
            if scores_visible:
                ax.hist(scores_visible, bins=bins, alpha=0.6, color='#43A047',
                        label=f'Visible (n={len(scores_visible)})',
                        edgecolor='white', linewidth=0.3, density=True)

            # Current threshold
            ax.axvline(x=result.threshold, color='gray', linestyle='--',
                       linewidth=0.8, label=f'Threshold={result.threshold}')

            # Optimal threshold
            opt = result.optimal_thresholds.get(side, {})
            if opt:
                ax.axvline(x=opt['threshold'], color='#1565C0', linestyle='-',
                           linewidth=1.0, label=f"Optimal={opt['threshold']:.2f}")

            color = SIDE_COLORS.get(side, '#333333')
            ax.set_title(f'{side.capitalize()}', fontsize=7, color=color, fontweight='bold')
            ax.set_xlabel('Algorithm Score', fontsize=5)
            ax.set_ylabel('Density', fontsize=5)
            ax.legend(fontsize=4, loc='upper right')
            ax.set_xlim(-0.02, 1.02)

        # Hide 6th subplot, add explanation
        axes[1, 2].axis('off')
        explanation = (
            "Interpretation:\n"
            "• Good separation between green\n"
            "  (visible) and red (not visible)\n"
            "  distributions means the algorithm\n"
            "  score is discriminative.\n\n"
            "• Overlap zone = gray area where\n"
            "  binary labeling is ambiguous.\n\n"
            "• Dashed gray = current threshold.\n"
            "• Solid blue = optimal threshold\n"
            "  (maximizes F1)."
        )
        axes[1, 2].text(0.05, 0.85, explanation, fontsize=5.5,
                        transform=axes[1, 2].transAxes, va='top',
                        family='sans-serif', linespacing=1.4)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

    def _page_temporal_heatmap(self, pdf, result):
        """Per-track temporal heatmap: algorithm scores as color, human labels overlaid."""
        # Group pairs by track
        track_data = defaultdict(list)
        for tid, fname, human_set, algo_dict in result.pairs:
            track_data[tid].append((fname, human_set, algo_dict))

        # Sort frames within each track
        for tid in track_data:
            track_data[tid].sort(
                key=lambda x: int(re.findall(r'\d+', x[0])[0])
                if re.findall(r'\d+', x[0]) else 0
            )

        n_tracks = len(track_data)
        if n_tracks == 0:
            return

        fig, axes = plt.subplots(n_tracks, 1,
                                 figsize=(8.5, max(3, 1.8 * n_tracks)),
                                 squeeze=False)
        fig.suptitle(f"Temporal Heatmap — {result.score_type} (threshold={result.threshold})",
                     fontsize=9, fontweight='bold', y=0.98)

        for ti, tid in enumerate(sorted(track_data.keys())):
            ax = axes[ti, 0]
            frames = track_data[tid]
            n_frames = len(frames)

            # Build score matrix (n_sides x n_frames)
            score_matrix = np.zeros((len(SIDES), n_frames))
            human_matrix = np.zeros((len(SIDES), n_frames), dtype=bool)

            for fi, (fname, human_set, algo_dict) in enumerate(frames):
                for si, side in enumerate(SIDES):
                    score_matrix[si, fi] = algo_dict.get(side, 0.0)
                    human_matrix[si, fi] = side in human_set

            # Heatmap of algorithm scores
            im = ax.imshow(score_matrix, aspect='auto', cmap='YlOrRd',
                           vmin=0, vmax=1, interpolation='nearest')

            # Overlay markers for disagreements
            for si in range(len(SIDES)):
                for fi in range(n_frames):
                    algo_vis = score_matrix[si, fi] >= result.threshold
                    human_vis = human_matrix[si, fi]
                    if algo_vis != human_vis:
                        # Disagreement: red border marker
                        ax.plot(fi, si, 's', markersize=3, markerfacecolor='none',
                                markeredgecolor='blue', markeredgewidth=0.8)
                    if human_vis:
                        # Human said visible: small white checkmark
                        ax.plot(fi, si, '.', markersize=1.5, color='white')

            ax.set_yticks(range(len(SIDES)))
            ax.set_yticklabels([s.capitalize() for s in SIDES], fontsize=5)
            ax.set_title(f'Track {tid} ({n_frames} frames)', fontsize=7, fontweight='bold')

            # X-axis: show sparse frame labels
            if n_frames > 20:
                step = max(n_frames // 10, 1)
                tick_pos = list(range(0, n_frames, step))
                ax.set_xticks(tick_pos)
                ax.set_xticklabels([frames[j][0] for j in tick_pos],
                                   rotation=45, fontsize=3)
            else:
                ax.set_xticks(range(n_frames))
                ax.set_xticklabels([f[0] for f in frames], rotation=45, fontsize=3)

        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Algorithm Score', fontsize=5)
        cbar.ax.tick_params(labelsize=4)

        # Legend note
        fig.text(0.02, 0.01,
                 "White dot = human 'visible' | Blue square = human-algorithm disagreement",
                 fontsize=5, color='gray')

        fig.subplots_adjust(left=0.08, right=0.88, top=0.94, bottom=0.06,
                            hspace=0.4)
        pdf.savefig(fig)
        plt.close(fig)

    def _page_optimal_thresholds(self, pdf, result):
        """F1-vs-threshold curves per side + summary table."""
        fig = plt.figure(figsize=(8.5, 9))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.5], hspace=0.3)

        # Top: F1 vs threshold curves
        ax_curves = fig.add_subplot(gs[0])
        for side in SIDES:
            opt = result.optimal_thresholds.get(side, {})
            if not opt or 'f1_curve' not in opt:
                continue
            thresholds = opt['thresholds']
            f1_curve = opt['f1_curve']
            color = SIDE_COLORS.get(side, '#999')
            ax_curves.plot(thresholds, f1_curve, color=color, linewidth=1.0,
                           label=f"{side} (opt={opt['threshold']:.2f}, F1={opt['f1']:.2f})")
            # Mark optimal point
            ax_curves.plot(opt['threshold'], opt['f1'], 'o', color=color,
                           markersize=4, markeredgecolor='black', markeredgewidth=0.4)

        # Mark current threshold
        ax_curves.axvline(x=result.threshold, color='gray', linestyle='--',
                          linewidth=0.8, label=f'Current threshold={result.threshold}')

        ax_curves.set_xlabel('Threshold', fontsize=7)
        ax_curves.set_ylabel('F1 Score', fontsize=7)
        ax_curves.set_title(f'Optimal Threshold Analysis — {result.score_type}',
                            fontsize=9, fontweight='bold')
        ax_curves.legend(fontsize=5, loc='lower left')
        ax_curves.set_xlim(-0.02, 1.02)
        ax_curves.set_ylim(0, 1.05)
        ax_curves.grid(True, alpha=0.3)

        # Bottom: summary table
        ax_table = fig.add_subplot(gs[1])
        ax_table.axis('off')

        headers = ['Side', 'Default Threshold', 'F1 @ Default',
                   'Optimal Threshold', 'F1 @ Optimal', 'F1 Improvement']
        rows = []
        for side in SIDES:
            opt = result.optimal_thresholds.get(side, {})
            m = result.per_side.get(side)
            default_f1 = m.f1 if m else 0.0
            opt_t = opt.get('threshold', result.threshold)
            opt_f1 = opt.get('f1', 0.0)
            improvement = opt_f1 - default_f1
            rows.append([
                side.capitalize(),
                f"{result.threshold:.2f}",
                f"{default_f1:.3f}",
                f"{opt_t:.2f}",
                f"{opt_f1:.3f}",
                f"{improvement:+.3f}",
            ])

        table = ax_table.table(cellText=rows, colLabels=headers,
                               loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1, 1.4)
        for j in range(len(headers)):
            table[0, j].set_facecolor('#E8EAF6')
            table[0, j].set_text_props(fontweight='bold')

        # Color improvement cells
        for i in range(len(rows)):
            improvement = float(rows[i][5])
            cell = table[i + 1, 5]
            if improvement > 0.01:
                cell.set_facecolor('#C8E6C9')  # green
            elif improvement < -0.01:
                cell.set_facecolor('#FFCDD2')  # red

        pdf.savefig(fig)
        plt.close(fig)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def cmd_label(args):
    """Run the human labeling tool."""
    # Switch to interactive backend for tkinter
    matplotlib.use('TkAgg')

    tool = HumanLabelingTool(
        annotator_output=args.annotator_output,
        images_dir=args.images_dir,
        output_path=args.output,
        annotator_name=args.annotator or "",
    )
    tool.run()


def cmd_evaluate(args):
    """Run evaluation: extract algorithm labels, compute metrics, generate report."""
    # Load human labels
    with open(args.human_labels, 'r') as f:
        raw = json.load(f)
    human_labels = raw.get('labels', {})

    n_labeled = sum(len(frames) for frames in human_labels.values())
    print(f"Loaded {n_labeled} human labels from {args.human_labels}")

    # Build (track, frame) pair list from human labels
    track_frames = []
    for tid_str, frames in human_labels.items():
        tid = int(tid_str)
        for fname in frames:
            track_frames.append((tid, fname))

    # Initialize extractor
    extractor = AlgorithmLabelExtractor(
        annotator_output=args.annotator_output,
        images_dir=args.images_dir,
        occlusion_json=getattr(args, 'occlusion_json', None),
    )

    # Run evaluation for both score types
    results = []
    score_types = ['visibility']
    if OCCLUSION_AVAILABLE:
        score_types.append('effective')
    else:
        print("Warning: occlusion_analyzer not available, skipping effective score evaluation")

    for score_type in score_types:
        print(f"\n--- Evaluating with score_type={score_type} ---")
        algo_scores = extractor.extract_all(track_frames, score_type=score_type)

        evaluator = ConfusionMatrixEvaluator(
            human_labels=human_labels,
            algo_scores=algo_scores,
            threshold=args.threshold,
            score_type=score_type,
        )
        result = evaluator.run_all()
        results.append(result)

        # Print summary
        print(f"\n  Results ({score_type}):")
        print(f"    Overall Accuracy: {result.overall_accuracy:.3f}")
        print(f"    Hamming Loss:     {result.hamming_loss:.3f}")
        print(f"    Subset Accuracy:  {result.subset_accuracy:.3f}")
        print(f"    Mean IoU:         {result.mean_iou:.3f}")
        print(f"    Macro F1:         {result.macro_f1:.3f}")
        print(f"    Macro Kappa:      {result.macro_kappa:.3f}")
        for side in SIDES:
            m = result.per_side[side]
            print(f"    {side:>6s}: P={m.precision:.2f} R={m.recall:.2f} "
                  f"F1={m.f1:.2f} K={m.kappa:.2f} AUC={m.auc:.2f}")

    # Generate outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_name = Path(args.annotator_output).parent.name

    report_gen = EvaluationReportGenerator(results, scene_name=scene_name)
    report_gen.save_json(output_dir / "evaluation_metrics.json")
    report_gen.generate_pdf(output_dir / "evaluation_report.pdf")

    print(f"\nOutputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Viewpoint Visibility Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Human labeling
  python visibility_eval.py label \\
      --annotator_output results/zebr-3/corrected_bboxes \\
      --images_dir examples/wd_data/zebras/zebr-3 \\
      --output human_visibility_labels.json

  # Evaluation
  python visibility_eval.py evaluate \\
      --annotator_output results/zebr-3/corrected_bboxes \\
      --images_dir examples/wd_data/zebras/zebr-3 \\
      --human_labels human_visibility_labels.json \\
      --threshold 0.1 --output_dir eval_results/
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Mode')

    # Label subcommand
    p_label = subparsers.add_parser('label', help='Human labeling GUI')
    p_label.add_argument('--annotator_output', required=True,
                         help='Path to corrected_bboxes directory')
    p_label.add_argument('--images_dir', required=True,
                         help='Path to image frames directory')
    p_label.add_argument('--output', default='human_visibility_labels.json',
                         help='Output JSON file for labels')
    p_label.add_argument('--annotator', default='',
                         help='Annotator name')

    # Evaluate subcommand
    p_eval = subparsers.add_parser('evaluate', help='Run evaluation')
    p_eval.add_argument('--annotator_output', required=True,
                        help='Path to corrected_bboxes directory')
    p_eval.add_argument('--images_dir', required=True,
                        help='Path to image frames directory')
    p_eval.add_argument('--human_labels', required=True,
                        help='Path to human labels JSON')
    p_eval.add_argument('--threshold', type=float, default=0.1,
                        help='Visibility score threshold for binary classification')
    p_eval.add_argument('--occlusion_json', default=None,
                        help='Path to pre-computed occlusion_summary.json (optional)')
    p_eval.add_argument('--output_dir', default='eval_results',
                        help='Output directory for report and metrics')

    args = parser.parse_args()

    if args.command == 'label':
        cmd_label(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
