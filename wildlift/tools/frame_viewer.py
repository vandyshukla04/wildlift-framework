#!/usr/bin/env python3
"""
Interactive Frame Viewer v3 for Ecology Visualizations

New features (v3):
- Fixed: All instances now show labels in motion trajectories column
- Labels include ID numbers (e.g., "zebra 1" instead of just "zebra")
- Saving creates individual column images for table creation

Features (from v2):
- First column shows images with SAM masks (instead of bounding boxes)
- Trajectories use gradient shading from start to finish
- Consistent track colors across all frames and columns

Controls:
    Left/Right arrows: Previous/Next frame
    Up/Down arrows: Jump 10 frames
    Home/End: First/Last frame
    S: Save current figure (also saves individual column images)
    Q/Escape: Quit

Batch mode (no display):
    python -m tools.frame_viewer_v3 --result_dir ... --batch --frames 0,10,20,30
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Set up matplotlib backend BEFORE importing pyplot
import matplotlib

# Check if we have a display (including WSLg)
_display = os.environ.get('DISPLAY')
_wayland = os.environ.get('WAYLAND_DISPLAY')
_wslg = os.path.exists('/mnt/wslg/.X11-unix')

HAS_DISPLAY = bool(_display or _wayland or _wslg)

if HAS_DISPLAY:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        try:
            matplotlib.use('Qt5Agg')
        except Exception:
            matplotlib.use('Agg')
            HAS_DISPLAY = False
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button
import matplotlib.colors as mcolors

# Import from ecology_visualizer
from tools.ecology_visualizer import (
    EcologyVisualizer,
    TrackingRenderer,
    get_instance_color,
    get_track_color,
    load_frame_data,
    unproject_depth,
    TRACK_COLORS,
)


def decode_rle_mask(rle: dict, height: int, width: int) -> np.ndarray:
    """Decode RLE mask to binary numpy array."""
    try:
        from pycocotools import mask as mask_utils
        # RLE format from COCO
        rle_obj = {
            'size': [height, width],
            'counts': rle['counts'].encode() if isinstance(rle['counts'], str) else rle['counts']
        }
        mask = mask_utils.decode(rle_obj)
        return mask
    except ImportError:
        # Fallback: manual decoding
        counts = rle['counts']
        if isinstance(counts, str):
            # Compressed RLE string - need pycocotools
            print("Warning: pycocotools needed for compressed RLE. Install with: pip install pycocotools")
            return np.zeros((height, width), dtype=np.uint8)

        # Uncompressed counts
        mask = np.zeros(height * width, dtype=np.uint8)
        pos = 0
        val = 0
        for count in counts:
            mask[pos:pos+count] = val
            pos += count
            val = 1 - val
        return mask.reshape((height, width), order='F')


# Valid animal class names for labeling (case-insensitive matching)
VALID_ANIMAL_CLASSES = {
    'zebra', 'rhino', 'rhinoceros', 'elephant', 'giraffe',
    'lion', 'leopard', 'cheetah', 'buffalo', 'hippo', 'hippopotamus',
    'wildebeest', 'antelope', 'gazelle', 'impala', 'warthog',
    'hyena', 'jackal', 'wild dog', 'crocodile', 'bird', 'ostrich',
    'horse', 'cow', 'cattle', 'deer', 'elk', 'moose', 'bear',
    'wolf', 'fox', 'rabbit', 'hare', 'monkey', 'baboon', 'gorilla',
}


def is_valid_animal(class_name: str) -> bool:
    """Check if a class name is a valid animal for labeling."""
    if not class_name:
        return False
    # Case-insensitive check
    name_lower = class_name.lower().strip()
    # Check exact match or if any valid animal name is contained in the class name
    if name_lower in VALID_ANIMAL_CLASSES:
        return True
    # Also check if class_name contains any animal name (e.g., "african elephant" contains "elephant")
    for animal in VALID_ANIMAL_CLASSES:
        if animal in name_lower:
            return True
    return False


def lighten_color(color: Tuple[float, float, float], amount: float = 0.5) -> Tuple[float, float, float]:
    """Lighten a color by mixing with white."""
    return tuple(c + (1.0 - c) * amount for c in color)


def darken_color(color: Tuple[float, float, float], amount: float = 0.3) -> Tuple[float, float, float]:
    """Darken a color by reducing values."""
    return tuple(c * (1.0 - amount) for c in color)


class InteractiveFrameViewerV3:
    """Interactive viewer with SAM masks, gradient trajectories, and improved labeling."""

    def __init__(self, result_dir: str, srt_file: Optional[str] = None,
                 output_dir: str = "saved_frames",
                 image_dir: Optional[str] = None,
                 mask_dir: Optional[str] = None):
        self.result_dir = Path(result_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths for images and masks
        self.image_dir = Path(image_dir) if image_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None

        # Initialize visualizer
        self.viz = EcologyVisualizer(result_dir, srt_file)
        self.viz.load_all_frames()
        self.viz.compute_canonical()
        self.viz.load_tracking()

        self.frame_ids = self.viz.frame_ids
        self.current_idx = 0
        self.fig = None
        self.axes = None

        # Load mask-track mapping
        self.mask_track_mapping = self._load_mask_track_mapping()

        # Precompute per-frame data
        self._frame_data = {}
        self._frame_points = {}
        self._sam_data = {}
        self._load_frame_points()

    def _load_mask_track_mapping(self) -> Dict:
        """Load mapping from track IDs to SAM detection indices."""
        mapping_file = self.result_dir / "mask_track_mapping.json"
        if mapping_file.exists():
            with open(mapping_file) as f:
                return json.load(f)
        return {}

    def _get_mask_to_track_mapping(self, fid: int) -> Dict[int, int]:
        """Get inverted mapping: mask_idx -> track_id for a given frame.

        The stored mapping is {frame_id: {track_id: mask_idx}}.
        We need {mask_idx: track_id} to convert instance_labels (which are mask_idx+1)
        to track_ids for consistent coloring.
        """
        frame_mapping = self.mask_track_mapping.get(str(fid), {})
        # Invert: {track_id: mask_idx} -> {mask_idx: track_id}
        return {int(v): int(k) for k, v in frame_mapping.items()}

    def _load_frame_points(self):
        """Load per-frame point clouds and SAM data."""
        print("Pre-loading frame point data...")
        for i, fid in enumerate(self.frame_ids):
            try:
                data = load_frame_data(self.result_dir, fid)
                pts = unproject_depth(data['depth'], data['intrinsics'], data['pose'])
                pts_canonical = self.viz.aligner.transform(pts)

                self._frame_data[fid] = data
                self._frame_points[fid] = {
                    'points': pts_canonical,
                    'labels': data['instance_labels'].flatten(),
                    'conf': data['conf'].flatten(),
                    'image': data.get('image')
                }

                # Load SAM data if mask_dir is provided
                if self.mask_dir:
                    sam_json = self.mask_dir / f"{fid}_results.json"
                    if sam_json.exists():
                        with open(sam_json) as f:
                            self._sam_data[fid] = json.load(f)

            except Exception as e:
                print(f"Warning: Could not load frame {fid}: {e}")

            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{len(self.frame_ids)} frames")
        print(f"Loaded {len(self._frame_points)} frame point clouds")
        if self.mask_dir:
            print(f"Loaded {len(self._sam_data)} SAM annotation files")

    def _load_original_image(self, fid: int) -> Optional[np.ndarray]:
        """Load original image from image_dir."""
        if self.image_dir is None:
            return None

        # Try common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path = self.image_dir / f"{fid}{ext}"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def _get_single_frame_data(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get points, labels, and confidence for a single frame only."""
        fid = self.frame_ids[frame_idx]
        if fid not in self._frame_points:
            return np.array([]), np.array([]), np.array([])

        fp = self._frame_points[fid]
        return fp['points'], fp['labels'], fp['conf']

    def _get_trajectories_up_to(self, up_to_idx: int) -> Dict[int, List]:
        """Get trajectories only up to current frame index."""
        trajectories = {}
        for tid, traj in self.viz._trajectories.items():
            filtered = [p for p in traj if p.get('frame', 0) <= up_to_idx]
            if filtered:
                trajectories[tid] = filtered

        return trajectories

    def _get_instance_centroids(self, frame_idx: int, conf_threshold: float = 1.5) -> Dict[int, np.ndarray]:
        """Compute centroids for each instance in the current frame from point cloud data.

        Returns dict mapping track_id -> centroid position (X, Y in canonical coords).
        This is used as fallback when trajectory data is missing for visible instances.
        """
        fid = self.frame_ids[frame_idx]
        if fid not in self._frame_points:
            return {}

        fp = self._frame_points[fid]
        pts = fp['points']
        lbls = fp['labels']
        conf = fp['conf']

        # Filter by confidence
        valid = conf > conf_threshold
        pts = pts[valid]
        lbls = lbls[valid]

        # Get mask_idx -> track_id mapping
        mask_to_track = self._get_mask_to_track_mapping(fid)

        centroids = {}
        for inst_id in np.unique(lbls):
            if inst_id == 0:
                continue
            mask = lbls == inst_id
            if not np.any(mask):
                continue

            # inst_id is mask_idx + 1
            mask_idx = int(inst_id) - 1
            track_id = mask_to_track.get(mask_idx, mask_idx)

            # Compute centroid (X, Y for top-down view)
            centroid = pts[mask].mean(axis=0)[:2]
            centroids[track_id] = centroid

        return centroids

    def _render_image_with_masks(self, ax, frame_idx: int):
        """Render input image with SAM masks colored by track ID."""
        fid = self.frame_ids[frame_idx]

        # Try to load original high-res image
        image = self._load_original_image(fid)

        # Fall back to stored image if no original
        if image is None and fid in self._frame_points:
            image = self._frame_points[fid].get('image')

        if image is None:
            ax.text(0.5, 0.5, 'Image not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return

        img_h, img_w = image.shape[:2]

        # Get SAM data and mask-track mapping for this frame
        sam_data = self._sam_data.get(fid)
        frame_mapping = self.mask_track_mapping.get(str(fid), {})

        if sam_data is None or not frame_mapping:
            # No SAM data - just show the image
            ax.imshow(image)
            return

        # Create overlay image
        overlay = image.copy().astype(np.float32)

        # Get scale factors if resolutions differ
        sam_h = sam_data.get('img_height', img_h)
        sam_w = sam_data.get('img_width', img_w)
        scale_h = img_h / sam_h
        scale_w = img_w / sam_w

        # Invert mapping: track_id -> detection_idx
        # frame_mapping is {track_id: detection_idx}
        track_to_det = {int(k): int(v) for k, v in frame_mapping.items()}

        # Draw masks for each tracked object
        for track_id, det_idx in track_to_det.items():
            if det_idx >= len(sam_data['annotations']):
                continue

            ann = sam_data['annotations'][det_idx]

            # Decode mask
            if 'segmentation' in ann:
                seg = ann['segmentation']
                mask = decode_rle_mask(seg, sam_h, sam_w)

                # Resize mask if needed
                if scale_h != 1.0 or scale_w != 1.0:
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                # Get track color (consistent across all frames)
                color = get_track_color(track_id)
                color_rgb = np.array(color) * 255

                # Apply colored mask overlay
                mask_bool = mask > 0
                alpha = 0.45  # Mask transparency

                for c in range(3):
                    overlay[:, :, c] = np.where(
                        mask_bool,
                        overlay[:, :, c] * (1 - alpha) + color_rgb[c] * alpha,
                        overlay[:, :, c]
                    )

                # Draw mask contour
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                # Convert overlay to uint8 for contour drawing
                overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
                for contour in contours:
                    if scale_h != 1.0 or scale_w != 1.0:
                        contour = (contour * np.array([scale_w, scale_h])).astype(np.int32)
                    cv2.drawContours(overlay_uint8, [contour], -1,
                                    (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])),
                                    thickness=2)
                overlay = overlay_uint8.astype(np.float32)

        # Display
        ax.imshow(overlay.astype(np.uint8))

    def _render_point_cloud(self, ax, points: np.ndarray, labels: np.ndarray,
                            conf: np.ndarray, frame_id: Optional[int] = None,
                            conf_threshold: float = 1.5,
                            point_size: float = 0.5, bg_alpha: float = 0.15,
                            inst_alpha: float = 0.8, use_track_colors: bool = False):
        """Render point cloud with instance/track colors.

        Args:
            frame_id: Required when use_track_colors=True to look up the correct
                     mask_idx -> track_id mapping for this frame.
        """
        valid = conf > conf_threshold
        pts = points[valid]
        lbls = labels[valid]

        if len(pts) == 0:
            return

        # Top-down view: X, Y
        xi, yi = 0, 1

        # Background
        bg = lbls == 0
        if np.any(bg):
            ax.scatter(pts[bg, xi], pts[bg, yi], c='lightgray',
                      s=point_size, alpha=bg_alpha, rasterized=True)

        # Get mask_idx -> track_id mapping for this frame (if using track colors)
        mask_to_track = {}
        if use_track_colors and frame_id is not None:
            mask_to_track = self._get_mask_to_track_mapping(frame_id)

        # Instances - use track colors for consistency
        for inst_id in np.unique(lbls):
            if inst_id == 0:
                continue
            mask = lbls == inst_id

            if use_track_colors:
                # inst_id is mask_idx + 1, so mask_idx = inst_id - 1
                mask_idx = int(inst_id) - 1
                # Look up actual track_id from mapping
                track_id = mask_to_track.get(mask_idx, mask_idx)  # fallback to mask_idx if not found
                color = get_track_color(track_id)
            else:
                color = get_instance_color(inst_id)

            ax.scatter(pts[mask, xi], pts[mask, yi], c=[color],
                      s=point_size * 1.5, alpha=inst_alpha, rasterized=True)

    def _render_trajectories_gradient(self, ax, trajectories: Dict[int, List],
                                      up_to_idx: int, linewidth: float = 2.5,
                                      instance_centroids: Optional[Dict[int, np.ndarray]] = None):
        """Render motion trails with start-to-finish color gradient shading.

        Args:
            ax: Matplotlib axis to render on
            trajectories: Dict of track_id -> list of trajectory points
            up_to_idx: Current frame index
            linewidth: Base line width for trajectories
            instance_centroids: Optional fallback centroids for tracks without trajectory data
        """
        # Track which track_ids we've rendered labels for
        rendered_track_ids = set()

        for tid, traj in trajectories.items():
            # Get class name and check if it's a valid animal
            class_name = self.viz._track_info.get(tid, {}).get('class_name', 'unknown')

            # Skip non-animal tracks (e.g., ground, unknown, etc.)
            if not is_valid_animal(class_name):
                continue

            base_color = get_track_color(tid)
            start_color = lighten_color(base_color, 0.6)
            end_color = base_color

            # Sort by frame index
            traj_sorted = sorted(traj, key=lambda x: x.get('frame', 0))

            # Get positions (canonical coords, top-down = X, Y)
            positions = []
            for p in traj_sorted:
                if 'center_canonical' in p:
                    positions.append(p['center_canonical'][:2])
                elif 'center' in p:
                    c = np.array(p['center']).reshape(1, 3)
                    c_can = self.viz.aligner.transform(c)[0]
                    positions.append(c_can[:2])

            if len(positions) == 0:
                continue

            positions = np.array(positions)
            n_points = len(positions)

            # Get label text with ID
            label_text = f"{class_name} {tid}"

            if n_points == 1:
                # Single point: just show marker and label (no line)
                ax.scatter(positions[0, 0], positions[0, 1], c=[end_color],
                          s=60, marker='^', edgecolors='white', linewidths=1, zorder=11)
                ax.annotate(label_text, positions[0], fontsize=7,
                           xytext=(5, 5), textcoords='offset points',
                           color=end_color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    alpha=0.8, edgecolor=end_color, linewidth=0.5))
                rendered_track_ids.add(tid)
                continue

            # Multiple points: draw gradient trajectory

            # Create line segments for gradient coloring
            segments = []
            colors = []
            alphas = []
            widths = []

            for i in range(n_points - 1):
                # Progress from 0 (start) to 1 (end)
                progress = i / max(n_points - 2, 1)

                # Interpolate color from start to end
                r = start_color[0] + (end_color[0] - start_color[0]) * progress
                g = start_color[1] + (end_color[1] - start_color[1]) * progress
                b = start_color[2] + (end_color[2] - start_color[2]) * progress

                segment = [positions[i], positions[i+1]]
                segments.append(segment)
                colors.append((r, g, b))

                # Alpha: start faint, end solid
                alpha = 0.4 + 0.6 * progress
                alphas.append(alpha)

                # Width: start thin, end thick
                width = linewidth * (0.5 + 0.5 * progress)
                widths.append(width)

            # Draw segments with outline for better visibility
            # First pass: draw dark outline (slightly thicker)
            for seg, col, alp, w in zip(segments, colors, alphas, widths):
                seg = np.array(seg)
                # Dark outline for contrast
                ax.plot(seg[:, 0], seg[:, 1], color='black', linewidth=w + 1.5,
                       alpha=alp * 0.5, solid_capstyle='round', zorder=5)
            # Second pass: draw colored trajectory on top
            for seg, col, alp, w in zip(segments, colors, alphas, widths):
                seg = np.array(seg)
                ax.plot(seg[:, 0], seg[:, 1], color=col, linewidth=w,
                       alpha=alp, solid_capstyle='round', zorder=6)

            # Start marker (small circle, light color)
            ax.scatter(positions[0, 0], positions[0, 1], c=[start_color],
                      s=25, marker='o', edgecolors='white', linewidths=0.5,
                      zorder=10, alpha=0.7)

            # End marker (larger triangle, full color)
            ax.scatter(positions[-1, 0], positions[-1, 1], c=[end_color],
                      s=60, marker='^', edgecolors='white', linewidths=1, zorder=11)

            # Label at end position with ID number
            ax.annotate(label_text, positions[-1], fontsize=7,
                       xytext=(5, 5), textcoords='offset points',
                       color=end_color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                alpha=0.8, edgecolor=end_color, linewidth=0.5))
            rendered_track_ids.add(tid)

        # Fallback: render labels for instances visible in point cloud but not in trajectories
        if instance_centroids:
            for tid, centroid in instance_centroids.items():
                if tid in rendered_track_ids:
                    continue  # Already rendered

                # Get class name and check if it's a valid animal
                class_name = self.viz._track_info.get(tid, {}).get('class_name', 'unknown')

                # Skip non-animal tracks
                if not is_valid_animal(class_name):
                    continue

                base_color = get_track_color(tid)
                label_text = f"{class_name} {tid}"

                # Just show marker and label at centroid position
                ax.scatter(centroid[0], centroid[1], c=[base_color],
                          s=60, marker='^', edgecolors='white', linewidths=1, zorder=11)
                ax.annotate(label_text, centroid, fontsize=7,
                           xytext=(5, 5), textcoords='offset points',
                           color=base_color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    alpha=0.8, edgecolor=base_color, linewidth=0.5))

    def _draw_frame(self):
        """Draw the current frame's comparison figure."""
        if self.fig is None:
            return

        # Clear all axes
        for ax in self.axes:
            ax.clear()

        fid = self.frame_ids[self.current_idx]
        fp = self._frame_points.get(fid)

        if fp is None:
            self.fig.suptitle(f"Frame {fid} - Data not available", fontsize=12)
            self.fig.canvas.draw_idle()
            return

        # Get single frame data (not cumulative)
        pts, lbls, conf = self._get_single_frame_data(self.current_idx)
        # But trajectories accumulate up to current frame
        trajectories = self._get_trajectories_up_to(self.current_idx)

        # Panel A: Input image with SAM masks
        ax1 = self.axes[0]
        self._render_image_with_masks(ax1, self.current_idx)
        ax1.set_title('(a) Input + Tracking Masks', fontsize=11, fontweight='bold')
        ax1.axis('off')

        # Panel B: Instance segmentation (single frame, track colors)
        ax2 = self.axes[1]
        if len(pts) > 0:
            self._render_point_cloud(ax2, pts, lbls, conf, frame_id=fid,
                                    point_size=0.5, use_track_colors=True)
        ax2.set_title('(b) Instance Segmentation', fontsize=11, fontweight='bold')
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Panel C: Single frame point cloud + accumulated trajectories with gradient
        ax3 = self.axes[2]
        if len(pts) > 0:
            self._render_point_cloud(ax3, pts, lbls, conf, frame_id=fid,
                                    point_size=0.3, bg_alpha=0.1, inst_alpha=0.4,
                                    use_track_colors=True)
        # Get instance centroids as fallback for missing trajectory data
        instance_centroids = self._get_instance_centroids(self.current_idx)
        self._render_trajectories_gradient(ax3, trajectories, self.current_idx,
                                          instance_centroids=instance_centroids)
        ax3.set_title('(c) Motion Trajectories', fontsize=11, fontweight='bold')
        ax3.invert_yaxis()
        ax3.set_aspect('equal')
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Update title
        self.fig.suptitle(f"Frame {fid} ({self.current_idx + 1}/{len(self.frame_ids)}) - "
                         f"Press S to save, Q to quit", fontsize=10)

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key in ('right', 'd'):
            self.current_idx = min(self.current_idx + 1, len(self.frame_ids) - 1)
            self._draw_frame()
        elif event.key in ('left', 'a'):
            self.current_idx = max(self.current_idx - 1, 0)
            self._draw_frame()
        elif event.key in ('up', 'w'):
            self.current_idx = min(self.current_idx + 10, len(self.frame_ids) - 1)
            self._draw_frame()
        elif event.key in ('down', 'x'):
            self.current_idx = max(self.current_idx - 10, 0)
            self._draw_frame()
        elif event.key == 'home':
            self.current_idx = 0
            self._draw_frame()
        elif event.key == 'end':
            self.current_idx = len(self.frame_ids) - 1
            self._draw_frame()
        elif event.key == 's':
            self._save_current()
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)

    def _save_current(self, event=None):
        """Save current figure and individual column images."""
        fid = self.frame_ids[self.current_idx]

        # Create visualizations/{fid}/ directory
        frame_dir = self.output_dir / "visualizations" / str(fid)
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save combined comparison figure
        output_path = frame_dir / f"comparison_frame_{fid}.png"
        self.fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

        # Save individual column images
        self._save_individual_panels(fid)

    def _save_individual_panels(self, fid: int):
        """Save each column as a separate high-resolution image (content only, no decorations)."""
        frame_dir = self.output_dir / "visualizations" / str(fid)
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Get data for this frame
        frame_idx = self.frame_ids.index(fid)
        pts, lbls, conf = self._get_single_frame_data(frame_idx)
        trajectories = self._get_trajectories_up_to(frame_idx)
        instance_centroids = self._get_instance_centroids(frame_idx)

        panel_names = [
            ('input_masks.png', 'masks'),
            ('instance_segmentation.png', 'segmentation'),
            ('motion_trajectories.png', 'trajectories')
        ]

        for (filename, panel_type) in panel_names:
            # Create a fresh figure for each panel
            fig, ax = plt.subplots(figsize=(8, 8))

            if panel_type == 'masks':
                # Panel A: Input image with SAM masks
                self._render_image_with_masks(ax, frame_idx)

            elif panel_type == 'segmentation':
                # Panel B: Instance segmentation point cloud
                if len(pts) > 0:
                    self._render_point_cloud(ax, pts, lbls, conf, frame_id=fid,
                                            point_size=1.0, use_track_colors=True)
                ax.invert_yaxis()
                ax.set_aspect('equal')

            elif panel_type == 'trajectories':
                # Panel C: Point cloud + trajectories
                if len(pts) > 0:
                    self._render_point_cloud(ax, pts, lbls, conf, frame_id=fid,
                                            point_size=0.6, bg_alpha=0.1, inst_alpha=0.4,
                                            use_track_colors=True)
                self._render_trajectories_gradient(ax, trajectories, frame_idx,
                                                  instance_centroids=instance_centroids)
                ax.invert_yaxis()
                ax.set_aspect('equal')

            # Remove all decorations
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

            # Save with tight layout and no padding
            panel_path = frame_dir / filename
            fig.savefig(panel_path, dpi=300, bbox_inches='tight', pad_inches=0,
                       facecolor='white', edgecolor='none')
            plt.close(fig)

        print(f"Saved individual panels to: {frame_dir}/")

    def run(self):
        """Start the interactive viewer."""
        if not HAS_DISPLAY:
            print("\n" + "="*60)
            print("ERROR: No display available!")
            print("="*60)
            print("Options:")
            print("  1. Set up X11 forwarding (export DISPLAY=:0)")
            print("  2. Use WSLg (Windows 11 with WSL2)")
            print("  3. Use batch mode to export specific frames:")
            print(f"     python -m tools.frame_viewer_v3 --result_dir {self.result_dir} --batch --frames 0,10,20")
            print("  4. Export all frames:")
            print(f"     python -m tools.frame_viewer_v3 --result_dir {self.result_dir} --batch --all")
            print("="*60 + "\n")
            return

        # Create figure
        self.fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)

        self.axes = [
            self.fig.add_subplot(gs[0]),
            self.fig.add_subplot(gs[1]),
            self.fig.add_subplot(gs[2])
        ]

        # Connect keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Add Save button
        button_ax = self.fig.add_axes([0.85, 0.02, 0.12, 0.05])  # [left, bottom, width, height]
        self.save_button = Button(button_ax, 'Save Figure', color='lightgreen', hovercolor='green')
        self.save_button.on_clicked(self._save_current)

        # Draw initial frame
        self._draw_frame()

        print("\n" + "="*60)
        print("Interactive Frame Viewer v3")
        print("="*60)
        print("Features:")
        print("  - First column: Image with SAM masks (track-colored)")
        print("  - Trajectories: Gradient shading from start to end")
        print("  - Consistent track colors across all columns")
        print("  - Labels include ID numbers (e.g., 'zebra 1')")
        print("  - All instances show labels in trajectories column")
        print("")
        print("Controls:")
        print("  Left/Right (or A/D): Previous/Next frame")
        print("  Up/Down (or W/X): Jump 10 frames")
        print("  Home/End: First/Last frame")
        print("  S or Save button: Save comparison + individual column images")
        print("  Q/Escape: Quit")
        print("="*60 + "\n")

        plt.show()

    def export_frame(self, frame_idx: int) -> str:
        """Export a single frame to file (comparison + individual panels)."""
        # Create figure
        self.fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)

        self.axes = [
            self.fig.add_subplot(gs[0]),
            self.fig.add_subplot(gs[1]),
            self.fig.add_subplot(gs[2])
        ]

        self.current_idx = frame_idx
        self._draw_frame()

        fid = self.frame_ids[frame_idx]
        frame_dir = self.output_dir / "visualizations" / str(fid)
        frame_dir.mkdir(parents=True, exist_ok=True)
        output_path = frame_dir / f"comparison_frame_{fid}.png"
        self.fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(self.fig)

        # Also save individual panels
        self._save_individual_panels(fid)

        return str(output_path)

    def export_batch(self, frame_indices: List[int], show_progress: bool = True):
        """Export multiple frames."""
        print(f"\nExporting {len(frame_indices)} frames to {self.output_dir}/")
        saved = []

        for i, idx in enumerate(frame_indices):
            if idx < 0 or idx >= len(self.frame_ids):
                print(f"  Skipping invalid frame index: {idx}")
                continue

            path = self.export_frame(idx)
            saved.append(path)

            if show_progress and (i + 1) % 5 == 0:
                print(f"  Exported {i + 1}/{len(frame_indices)} frames")

        print(f"\nDone! Exported {len(saved)} frames to {self.output_dir}/")
        return saved


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Frame Viewer v3 with SAM masks, gradient trajectories, and improved labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode with masks:
    python -m tools.frame_viewer_v3 --result_dir results/zebras/ \\
        --image_dir /path/to/images --mask_dir /path/to/grounded-sam

  Batch export specific frames:
    python -m tools.frame_viewer_v3 --result_dir results/zebras/ \\
        --image_dir /path/to/images --mask_dir /path/to/grounded-sam \\
        --batch --frames 0,10,20,30

  Export all frames:
    python -m tools.frame_viewer_v3 --result_dir results/zebras/ \\
        --batch --all

  Full example:
    python -m tools.frame_viewer_v3 \\
        --result_dir /home/shuklva/CUT3R/results/paper_final/thursday/zebras/zebr-14_2/ \\
        --srt_file /home/shuklva/CUT3R/examples/wd_data/zebras/zebr-14_2/DJI_20250802125512_0006_V.SRT \\
        --image_dir /home/shuklva/CUT3R/examples/wd_data/zebras/zebr-14_2 \\
        --mask_dir /home/shuklva/CUT3R/examples/wd_data/zebras/zebr-14_2/grounded-sam \\
        --output_dir saved_frames --start_frame 40

New in v3:
  - All instances show labels in motion trajectories column
  - Labels include ID numbers (e.g., 'zebra 1' instead of 'zebra')
  - Saving creates individual column images in frame_<fid>/ folder
        """
    )
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--srt_file', type=str, default=None,
                       help='Optional SRT file for gimbal alignment')
    parser.add_argument('--output_dir', type=str, default='saved_frames',
                       help='Directory to save frames')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame index (interactive mode)')

    # New arguments for masks
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Path to original images directory')
    parser.add_argument('--mask_dir', type=str, default=None,
                       help='Path to grounded-sam masks directory')

    # Batch mode options
    parser.add_argument('--batch', action='store_true',
                       help='Run in batch mode (no display needed)')
    parser.add_argument('--frames', type=str, default=None,
                       help='Comma-separated frame indices to export (e.g., 0,10,20,30)')
    parser.add_argument('--all', action='store_true',
                       help='Export all frames')
    parser.add_argument('--every', type=int, default=None,
                       help='Export every Nth frame')

    args = parser.parse_args()

    viewer = InteractiveFrameViewerV3(
        result_dir=args.result_dir,
        srt_file=args.srt_file,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir
    )

    # Batch mode
    if args.batch:
        n_frames = len(viewer.frame_ids)

        if args.frames:
            # Parse comma-separated frame indices
            frame_indices = [int(x.strip()) for x in args.frames.split(',')]
        elif args.all:
            frame_indices = list(range(n_frames))
        elif args.every:
            frame_indices = list(range(0, n_frames, args.every))
        else:
            print("Batch mode requires --frames, --all, or --every option")
            print(f"Total frames available: {n_frames}")
            sys.exit(1)

        viewer.export_batch(frame_indices)
    else:
        # Interactive mode
        viewer.current_idx = args.start_frame
        viewer.run()


if __name__ == '__main__':
    main()
