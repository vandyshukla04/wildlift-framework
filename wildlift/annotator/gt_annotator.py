#!/usr/bin/env python3
"""
MOT Ground Truth Annotator

Tool to review detections and assign/correct ground truth track IDs
for multi-object tracking evaluation.

Features:
    - Load 2D bounding boxes from instance masks (primary) or Grounded-SAM results
    - Assign correct GT track IDs (possibly different from predicted)
    - Mark false positives
    - Save in MOT Challenge format

Usage:
    python tools/mot_gt_annotator.py \
        --results_dir results/wildlift/tmp-zebr-14_2-revisit-1 \
        --images_dir examples/wd_data/zebras/zebr-14_2 \
        --output_dir gt_annotations/zebra_14_2

    # With Grounded-SAM results:
    python tools/mot_gt_annotator.py \
        --results_dir results/wildlift/tmp-zebr-14_2-revisit-1 \
        --images_dir examples/wd_data/zebras/zebr-14_2 \
        --gsam_dir examples/wd_data/zebras/zebr-14_2/grounded-sam \
        --output_dir gt_annotations/zebra_14_2

Controls:
    - Left/Right Arrow: Navigate frames
    - Click on bbox: Select for editing
    - Click track in sidebar panel: Toggle entire track as FP
    - Number keys 0-9: Assign GT track ID
    - 'd': Toggle entire track of selected bbox as FP
    - 'f': Mark single detection as false positive
    - 's': Save annotations
    - 'q': Quit
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.wildlife_tracking.utils.data_loaders import SequenceLoader


@dataclass
class GTAnnotation:
    """Ground truth annotation for a single detection."""
    frame_idx: int              # Frame index (0-based)
    frame_name: str             # Original frame name
    gt_track_id: int            # Ground truth track ID
    pred_track_id: int          # Predicted track ID
    bbox_2d: List[float]        # [x, y, w, h] 2D bounding box
    class_name: str
    is_false_positive: bool = False
    is_manually_added: bool = False


class MOTGTAnnotator:
    """OpenCV-based GT annotation tool."""

    # Colors for different tracks (BGR)
    TRACK_COLORS = [
        (255, 0, 0),     # Blue
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 128),   # Purple
        (0, 128, 128),   # Olive
        (128, 128, 0),   # Teal
        (255, 165, 0),   # Orange
    ]

    def __init__(self, results_dir: Path, images_dir: Optional[Path],
                 output_dir: Path, gsam_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.output_dir = Path(output_dir)
        self.gsam_dir = Path(gsam_dir) if gsam_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.loader = SequenceLoader(results_dir)
        self.images: Dict[int, np.ndarray] = {}
        self.gsam_bboxes: Dict[str, List[dict]] = {}  # frame_name -> list of detections

        # Annotations
        self.annotations: Dict[int, List[GTAnnotation]] = {}  # frame_idx -> annotations

        # State
        self.current_frame_idx = 0
        self.selected_ann_idx: Optional[int] = None
        self.gt_id_counter = 0
        self.pred_to_gt_mapping: Dict[int, int] = {}  # pred_track_id -> gt_track_id
        self.modified = False

        # Target image size (matches masks)
        self.target_size: Optional[Tuple[int, int]] = None  # (height, width)

        # Image display
        self.window_name = "MOT GT Annotator"
        self.image_scale = 1.0

        # Track panel (sidebar)
        self.fp_tracks: Set[int] = set()  # pred_track_ids marked as whole-track FP
        self.panel_track_rects: List[Tuple[int, int, int, int, int]] = []  # (y1, y2, x1, x2, pred_track_id)
        self.panel_width = 200

    def load_data(self):
        """Load masks, tracking data, and images."""
        print("Loading data...")
        self.loader.load_bounding_boxes()  # For class names and track info
        self.loader.load_instance_masks()
        self.loader.load_mask_track_mapping()
        self.loader.load_tracking_summary()

        # Get target size from masks
        self.target_size = self.loader.get_image_shape()
        if self.target_size:
            print(f"Target image size: {self.target_size[1]}x{self.target_size[0]}")
        else:
            self.target_size = (288, 512)
            print(f"Using default size: 512x288")

        # Load Grounded-SAM results if provided
        if self.gsam_dir and self.gsam_dir.exists():
            print(f"Loading Grounded-SAM results from {self.gsam_dir}...")
            self._load_gsam_bboxes()

        # Try to load images
        if self.images_dir and self.images_dir.exists():
            print(f"Loading images from {self.images_dir}...")
            self._load_images()

        # Fallback: try annotated_2d directory in results
        if not self.images:
            annotated_dir = self.results_dir / "annotated_2d"
            if annotated_dir.exists():
                print(f"Using annotated images from {annotated_dir}...")
                self._load_annotated_images(annotated_dir)

        if not self.images:
            print("No images found, will use blank frames")

        # Initialize annotations from masks
        self._initialize_annotations()

        print(f"Loaded {len(self.loader.frame_indices)} frames")
        print(f"Found {len(self.pred_to_gt_mapping)} unique tracks")

    def _load_images(self):
        """Load images from directory and resize to target size."""
        for idx, frame_name in enumerate(self.loader.frame_names):
            # Try various extensions and patterns
            patterns = [
                f"{frame_name}.jpg",
                f"{frame_name}.png",
                f"{frame_name}.jpeg",
                f"{int(frame_name):06d}.jpg",
                f"{int(frame_name):06d}.png",
            ]

            for pattern in patterns:
                img_path = self.images_dir / pattern
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Resize to target size if needed
                        if self.target_size and img.shape[:2] != self.target_size:
                            img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
                        self.images[idx] = img
                    break

        print(f"Loaded {len(self.images)} images")

    def _load_annotated_images(self, annotated_dir: Path):
        """Load annotated images (with bboxes already drawn)."""
        for idx, frame_name in enumerate(self.loader.frame_names):
            # Try various patterns for annotated images
            patterns = [
                f"{frame_name}_tracked.png",
                f"{frame_name}_tracked.jpg",
                f"{frame_name}.png",
                f"{frame_name}.jpg",
            ]

            for pattern in patterns:
                img_path = annotated_dir / pattern
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Resize to target size if needed
                        if self.target_size and img.shape[:2] != self.target_size:
                            img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
                        self.images[idx] = img
                    break

        print(f"Loaded {len(self.images)} annotated images")

    def _load_gsam_bboxes(self):
        """Load Grounded-SAM detection results."""
        for frame_name in self.loader.frame_names:
            json_path = self.gsam_dir / f"{frame_name}_results.json"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)

                # Get scale factors
                gsam_w = data.get('img_width', 768)
                gsam_h = data.get('img_height', 432)
                scale_x = self.target_size[1] / gsam_w
                scale_y = self.target_size[0] / gsam_h

                # Process annotations
                detections = []
                for ann in data.get('annotations', []):
                    bbox = ann.get('bbox', [])
                    if len(bbox) == 4:
                        # bbox is in xyxy format, scale to target size
                        x1, y1, x2, y2 = bbox
                        x1 = x1 * scale_x
                        y1 = y1 * scale_y
                        x2 = x2 * scale_x
                        y2 = y2 * scale_y
                        detections.append({
                            'bbox': [x1, y1, x2 - x1, y2 - y1],  # Convert to xywh
                            'class_name': ann.get('class_name', 'unknown'),
                            'score': ann.get('score', [1.0])[0] if isinstance(ann.get('score'), list) else ann.get('score', 1.0)
                        })

                self.gsam_bboxes[frame_name] = detections

        print(f"Loaded Grounded-SAM results for {len(self.gsam_bboxes)} frames")

    def _bbox_from_mask(self, mask: np.ndarray, mask_value: int) -> Optional[List[float]]:
        """Compute [x, y, w, h] bbox from instance mask."""
        binary = (mask == mask_value)
        coords = np.where(binary)
        if len(coords[0]) == 0:
            return None
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]

    def _get_class_for_track(self, track_id: int) -> str:
        """Get class name for a track from bounding boxes."""
        for bboxes in self.loader.bboxes.values():
            for bbox in bboxes:
                if bbox.track_id == track_id:
                    return bbox.class_name
        return "unknown"

    def _initialize_annotations(self):
        """Initialize annotations from instance masks."""
        for frame_idx, frame_name in enumerate(self.loader.frame_names):
            self.annotations[frame_idx] = []

            mask = self.loader.instance_masks.get(frame_idx)
            mapping = self.loader.mask_track_mapping.get(frame_name, {})

            if mask is None:
                continue

            for track_id_str, mask_instance_id in mapping.items():
                track_id = int(track_id_str)

                # Assign GT ID (initially same as predicted)
                if track_id not in self.pred_to_gt_mapping:
                    self.pred_to_gt_mapping[track_id] = self.gt_id_counter
                    self.gt_id_counter += 1

                gt_id = self.pred_to_gt_mapping[track_id]

                # Compute bbox from mask (mask values are 1-indexed, 0 is background)
                mask_value = mask_instance_id + 1
                bbox_2d = self._bbox_from_mask(mask, mask_value)

                if bbox_2d is not None:
                    ann = GTAnnotation(
                        frame_idx=frame_idx,
                        frame_name=frame_name,
                        gt_track_id=gt_id,
                        pred_track_id=track_id,
                        bbox_2d=bbox_2d,
                        class_name=self._get_class_for_track(track_id)
                    )
                    self.annotations[frame_idx].append(ann)

    def _build_track_summary(self) -> List[dict]:
        """Build summary of all tracks across the entire sequence."""
        track_info: Dict[int, dict] = {}
        for anns in self.annotations.values():
            for ann in anns:
                tid = ann.pred_track_id
                if tid not in track_info:
                    track_info[tid] = {
                        'pred_track_id': tid,
                        'class_name': ann.class_name,
                        'frame_count': 0,
                    }
                track_info[tid]['frame_count'] += 1
        for info in track_info.values():
            info['is_fp'] = info['pred_track_id'] in self.fp_tracks
        return sorted(track_info.values(), key=lambda t: t['pred_track_id'])

    def _sync_fp_tracks(self):
        """Sync is_false_positive on all annotations based on self.fp_tracks."""
        for anns in self.annotations.values():
            for ann in anns:
                ann.is_false_positive = ann.pred_track_id in self.fp_tracks

    def _get_frame_image(self, frame_idx: int) -> np.ndarray:
        """Get or create image for frame."""
        if frame_idx in self.images:
            return self.images[frame_idx].copy()
        else:
            # Create blank image
            h, w = self.target_size if self.target_size else (288, 512)
            return np.zeros((h, w, 3), dtype=np.uint8) + 50

    def _draw_frame(self) -> np.ndarray:
        """Draw current frame with annotations."""
        img = self._get_frame_image(self.current_frame_idx)
        h, w = img.shape[:2]

        annotations = self.annotations.get(self.current_frame_idx, [])

        for i, ann in enumerate(annotations):
            if ann.is_false_positive:
                color = (128, 128, 128)  # Gray for FP
                thickness = 1
            else:
                color = self.TRACK_COLORS[ann.gt_track_id % len(self.TRACK_COLORS)]
                thickness = 2

            # Highlight selected
            if i == self.selected_ann_idx:
                thickness = 3

            # Draw bbox
            x, y, bw, bh = ann.bbox_2d
            x, y, bw, bh = int(x), int(y), int(bw), int(bh)

            if bw > 0 and bh > 0:
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, thickness)

                # Draw label
                label = f"GT:{ann.gt_track_id}"
                if ann.is_false_positive:
                    label = "FP"

                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x, y - lh - 5), (x + lw + 4, y), color, -1)
                text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                cv2.putText(img, label, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Draw Grounded-SAM bboxes if available (in cyan, dashed style simulation)
        frame_name = self.loader.frame_names[self.current_frame_idx]
        gsam_dets = self.gsam_bboxes.get(frame_name, [])
        for det in gsam_dets:
            x, y, bw, bh = det['bbox']
            x, y, bw, bh = int(x), int(y), int(bw), int(bh)
            # Draw with dotted effect (every 5 pixels)
            for px in range(x, x + bw, 10):
                cv2.line(img, (px, y), (min(px + 5, x + bw), y), (255, 255, 0), 1)
                cv2.line(img, (px, y + bh), (min(px + 5, x + bw), y + bh), (255, 255, 0), 1)
            for py in range(y, y + bh, 10):
                cv2.line(img, (x, py), (x, min(py + 5, y + bh)), (255, 255, 0), 1)
                cv2.line(img, (x + bw, py), (x + bw, min(py + 5, y + bh)), (255, 255, 0), 1)

        # Draw frame info
        info = f"Frame {self.current_frame_idx + 1}/{len(self.loader.frame_indices)} ({frame_name})"
        cv2.putText(img, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw instructions
        instructions = "Arrow:Nav | Click:Select | 0-9:ID | d:TrackFP | f:FP | s:Save | q:Quit"
        cv2.putText(img, instructions, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # --- Sidebar track panel ---
        pw = self.panel_width
        canvas = np.zeros((h, w + pw, 3), dtype=np.uint8)
        canvas[:, :w] = img
        # Panel background
        canvas[:, w:] = (30, 30, 30)
        cv2.line(canvas, (w, 0), (w, h), (100, 100, 100), 1)

        # Title
        cv2.putText(canvas, "TRACKS", (w + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, "(click to toggle FP)", (w + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

        # Build track list and draw rows
        track_summary = self._build_track_summary()
        self.panel_track_rects = []
        row_y = 58
        row_h = 28
        for info in track_summary:
            tid = info['pred_track_id']
            is_fp = info['is_fp']
            color = (128, 128, 128) if is_fp else self.TRACK_COLORS[tid % len(self.TRACK_COLORS)]

            # Highlight row on hover-style with background
            ry1, ry2 = row_y - 2, row_y + row_h - 4
            self.panel_track_rects.append((ry1, ry2, w, w + pw, tid))

            # Color swatch
            cv2.rectangle(canvas, (w + 8, row_y), (w + 22, row_y + 14), color, -1)

            # Label
            label = f"T{tid}: {info['class_name']} ({info['frame_count']}f)"
            cv2.putText(canvas, label, (w + 28, row_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

            # FP badge
            if is_fp:
                cv2.putText(canvas, "[FP]", (w + pw - 40, row_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

            row_y += row_h

        return canvas

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check track panel clicks first
            for ry1, ry2, rx1, rx2, tid in self.panel_track_rects:
                if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                    if tid in self.fp_tracks:
                        self.fp_tracks.discard(tid)
                        print(f"Unmarked track {tid} as FP")
                    else:
                        self.fp_tracks.add(tid)
                        print(f"Marked entire track {tid} as FP")
                    self._sync_fp_tracks()
                    self.modified = True
                    return

            # Check if click is on any bbox
            annotations = self.annotations.get(self.current_frame_idx, [])

            for i, ann in enumerate(annotations):
                bx, by, bw, bh = ann.bbox_2d
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.selected_ann_idx = i
                    return

            self.selected_ann_idx = None

    def run(self):
        """Run the annotation interface."""
        print("\nStarting MOT GT Annotator...")
        print("Controls:")
        print("  Left/Right Arrow: Navigate frames")
        print("  Click on bbox: Select for editing")
        print("  Click track in panel: Toggle entire track as FP")
        print("  Number keys 0-9: Assign GT track ID to selected bbox")
        print("  d: Toggle entire track of selected bbox as FP")
        print("  f: Mark single selected detection as FP")
        print("  s: Save annotations")
        print("  q: Quit")
        if self.gsam_bboxes:
            print("  (Yellow dotted boxes = Grounded-SAM detections for reference)")
        print()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            img = self._draw_frame()
            cv2.imshow(self.window_name, img)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                if self.modified:
                    print("\nUnsaved changes! Press 's' to save or 'q' again to quit without saving.")
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('s'):
                        self.save_annotations()
                    elif key2 != ord('q'):
                        continue
                break

            elif key == 81 or key == 2:  # Left arrow
                self.current_frame_idx = max(0, self.current_frame_idx - 1)
                self.selected_ann_idx = None

            elif key == 83 or key == 3:  # Right arrow
                self.current_frame_idx = min(len(self.loader.frame_indices) - 1, self.current_frame_idx + 1)
                self.selected_ann_idx = None

            elif ord('0') <= key <= ord('9'):
                # Assign GT track ID
                if self.selected_ann_idx is not None:
                    new_id = key - ord('0')
                    annotations = self.annotations.get(self.current_frame_idx, [])
                    if self.selected_ann_idx < len(annotations):
                        annotations[self.selected_ann_idx].gt_track_id = new_id
                        annotations[self.selected_ann_idx].is_false_positive = False
                        self.modified = True
                        print(f"Assigned GT ID {new_id} to detection")

            elif key == ord('f'):
                # Mark single detection as false positive
                if self.selected_ann_idx is not None:
                    annotations = self.annotations.get(self.current_frame_idx, [])
                    if self.selected_ann_idx < len(annotations):
                        annotations[self.selected_ann_idx].is_false_positive = True
                        self.modified = True
                        print("Marked detection as false positive")

            elif key == ord('d'):
                # Mark entire track as FP (toggle)
                if self.selected_ann_idx is not None:
                    annotations = self.annotations.get(self.current_frame_idx, [])
                    if self.selected_ann_idx < len(annotations):
                        tid = annotations[self.selected_ann_idx].pred_track_id
                        if tid in self.fp_tracks:
                            self.fp_tracks.discard(tid)
                            print(f"Unmarked entire track {tid} as FP")
                        else:
                            self.fp_tracks.add(tid)
                            print(f"Marked entire track {tid} as FP")
                        self._sync_fp_tracks()
                        self.modified = True

            elif key == ord('s'):
                self.save_annotations()

        cv2.destroyAllWindows()

    def save_annotations(self):
        """Save annotations in MOT Challenge format."""
        # gt.txt format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        gt_file = self.output_dir / "gt.txt"

        with open(gt_file, 'w') as f:
            for frame_idx in sorted(self.annotations.keys()):
                frame_name = self.loader.frame_names[frame_idx]
                for ann in self.annotations[frame_idx]:
                    if not ann.is_false_positive:
                        x, y, w, h = ann.bbox_2d
                        # Use 1-based frame index for MOT format
                        f.write(f"{frame_idx + 1},{ann.gt_track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

        # Save metadata
        metadata = {
            'source_results_dir': str(self.results_dir),
            'frame_count': len(self.loader.frame_indices),
            'total_gt_ids': max(
                (ann.gt_track_id for anns in self.annotations.values() for ann in anns if not ann.is_false_positive),
                default=0
            ) + 1,
            'pred_to_gt_mapping': self.pred_to_gt_mapping,
            'frame_names': self.loader.frame_names,
        }
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save detailed annotations
        detailed = {}
        for frame_idx, anns in self.annotations.items():
            detailed[frame_idx] = [asdict(ann) for ann in anns]

        with open(self.output_dir / "annotations_detailed.json", 'w') as f:
            json.dump(detailed, f, indent=2)

        self.modified = False
        print(f"\nAnnotations saved to: {self.output_dir}")
        print(f"  - gt.txt (MOT format)")
        print(f"  - metadata.json")
        print(f"  - annotations_detailed.json")


def main():
    parser = argparse.ArgumentParser(description="MOT Ground Truth Annotator")
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to CUT3R results directory"
    )
    parser.add_argument(
        "--images_dir", type=str, default=None,
        help="Path to original images directory (optional)"
    )
    parser.add_argument(
        "--gsam_dir", type=str, default=None,
        help="Path to Grounded-SAM results directory (optional, for reference)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="gt_annotations",
        help="Output directory for GT annotations"
    )
    args = parser.parse_args()

    annotator = MOTGTAnnotator(
        results_dir=Path(args.results_dir),
        images_dir=Path(args.images_dir) if args.images_dir else None,
        gsam_dir=Path(args.gsam_dir) if args.gsam_dir else None,
        output_dir=Path(args.output_dir)
    )

    annotator.load_data()
    annotator.run()


if __name__ == "__main__":
    main()
