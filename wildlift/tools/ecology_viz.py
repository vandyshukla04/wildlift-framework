#!/usr/bin/env python3
"""
Ecology Paper Visualization System for CUT3R

Creates publication-quality visualizations for ecology/applied tech papers:
1. PCA/Gimbal-aligned canonical views (true top-down, side, front)
2. Motion trajectory trails with temporal color gradients
3. Instance segmentation with convex hulls and callout annotations
4. Multi-panel comparison layouts for paper figures
5. Interactive Plotly HTML for supplementary materials
6. GIF/Video animations for presentations

Usage:
    python -m tools.ecology_visualizer --result_dir results/wildlift/zebra-scene1 --output_dir figures/
    python -m tools.ecology_visualizer --result_dir results/cami/rhino-35 --srt_file data/DJI_0035.srt --output_dir figures/
"""

import os
import sys
import json
import argparse
import re
import os
import numpy as np
import matplotlib
# Only set Agg if no interactive backend is already configured
if matplotlib.get_backend().lower() not in ('tkagg', 'qt5agg', 'qtagg', 'gtk3agg', 'wxagg', 'macosx'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# Optional dependencies
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


# ============================================================================
# COLOR PALETTES
# ============================================================================

TRACK_COLORS = [
    (0.90, 0.30, 0.25),   # Red
    (0.25, 0.70, 0.40),   # Green
    (0.30, 0.50, 0.85),   # Blue
    (0.95, 0.70, 0.15),   # Yellow/Gold
    (0.65, 0.35, 0.75),   # Purple
    (0.20, 0.70, 0.70),   # Cyan
    (1.00, 0.50, 0.20),   # Orange
    (0.80, 0.40, 0.60),   # Pink
    (0.50, 0.65, 0.30),   # Lime
    (0.75, 0.30, 0.45),   # Rose
]

INSTANCE_COLORS = {
    0: (0.75, 0.75, 0.75),  # Gray background
    1: (0.90, 0.30, 0.25),
    2: (0.25, 0.70, 0.40),
    3: (0.30, 0.50, 0.85),
    4: (0.95, 0.70, 0.15),
    5: (0.65, 0.35, 0.75),
    6: (0.20, 0.70, 0.70),
    7: (1.00, 0.50, 0.20),
    8: (0.80, 0.40, 0.60),
}


def get_track_color(track_id: int) -> Tuple[float, float, float]:
    """Get consistent color for a track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def get_instance_color(instance_id: int) -> Tuple[float, float, float]:
    """Get consistent color for an instance ID."""
    if instance_id in INSTANCE_COLORS:
        return INSTANCE_COLORS[instance_id]
    np.random.seed(instance_id * 42)
    return tuple(np.random.rand(3).tolist())


# ============================================================================
# GIMBAL DATA PARSER
# ============================================================================

def parse_dji_logs(log_file_path: str, frame_indices: Optional[List[int]] = None) -> Optional[Dict]:
    """Parse DJI SRT log file to extract gimbal data."""
    if not os.path.exists(log_file_path):
        print(f"Warning: SRT file not found: {log_file_path}")
        return None

    with open(log_file_path, 'r') as f:
        content = f.read()

    gimbal_data = {}

    # Pattern 1: gb_yaw, gb_pitch, gb_roll format
    pattern1 = r'FrameCnt: (\d+).*?\[rel_alt: ([\d.]+).*?\[gb_yaw: ([-\d.]+) gb_pitch: ([-\d.]+) gb_roll: ([-\d.]+)\]'
    matches = re.findall(pattern1, content, re.DOTALL)

    # Pattern 2: gimbal_heading, gimbal_pitch, gimbal_roll format (CSV_DATA)
    if not matches:
        pattern2 = r'FrameCnt: (\d+).*?\[rel_alt: ([\d.]+).*?gimbal_heading\(degrees\): ([-\d.]+)\].*?gimbal_pitch\(degrees\): ([-\d.]+)\].*?gimbal_roll\(degrees\): ([-\d.]+)\]'
        matches = re.findall(pattern2, content, re.DOTALL)

    if not matches:
        print("No gimbal data found in SRT file")
        return None

    srt_frames = set(int(m[0]) for m in matches)
    target_frames = srt_frames if not frame_indices else (set(frame_indices) & srt_frames) or srt_frames

    for frame_str, altitude, yaw, pitch, roll in matches:
        frame = int(frame_str)
        if frame in target_frames:
            gimbal_data[frame] = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'altitude': float(altitude)
            }

    print(f"Parsed gimbal data for {len(gimbal_data)} frames")
    return gimbal_data if gimbal_data else None


# ============================================================================
# DATA LOADING
# ============================================================================

def get_available_frames(result_dir: str) -> List[str]:
    """Get list of available frame IDs."""
    result_path = Path(result_dir)
    depth_files = sorted(result_path.glob('depth/*.npy'))
    return [f.stem for f in depth_files]


def load_frame_data(result_dir: str, frame_id: str) -> Dict[str, Any]:
    """Load all data for a single frame."""
    result_path = Path(result_dir)
    data = {}

    # Depth
    depth_path = result_path / f'depth/{frame_id}.npy'
    if depth_path.exists():
        data['depth'] = np.load(depth_path)
    else:
        raise FileNotFoundError(f"Depth not found: {depth_path}")

    # Confidence
    conf_path = result_path / f'conf/{frame_id}.npy'
    data['conf'] = np.load(conf_path) if conf_path.exists() else np.ones_like(data['depth']) * 5.0

    # Camera
    cam_path = result_path / f'camera/{frame_id}.npz'
    if cam_path.exists():
        cam = np.load(cam_path)
        data['pose'] = cam['pose']
        data['intrinsics'] = cam['intrinsics']
    else:
        raise FileNotFoundError(f"Camera not found: {cam_path}")

    # Instance labels
    inst_path = result_path / f'instance_labels/{frame_id}.npy'
    data['instance_labels'] = np.load(inst_path) if inst_path.exists() else np.zeros_like(data['depth'], dtype=np.int32)

    # Bounding boxes
    bbox_path = result_path / f'bounding_boxes/{frame_id}.json'
    if bbox_path.exists():
        with open(bbox_path) as f:
            data['bboxes'] = json.load(f)
    else:
        data['bboxes'] = []

    # Image
    for ext in ['.png', '.jpg']:
        for subdir in ['annotated_2d', 'images']:
            img_path = result_path / subdir / f'{frame_id}{ext}'
            if not img_path.exists() and subdir == 'annotated_2d':
                img_path = result_path / subdir / f'{frame_id}_tracked{ext}'
            if img_path.exists():
                data['image'] = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                break
        if 'image' in data:
            break

    return data


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Convert depth map to 3D world points."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = depth.flatten()
    x = (u.flatten() - cx) * z / fx
    y = (v.flatten() - cy) * z / fy

    pts_cam = np.stack([x, y, z], axis=-1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
    pts_world = (pose @ pts_cam_h.T).T[:, :3]

    return pts_world


# ============================================================================
# SCENE ALIGNER
# ============================================================================

def rotation_matrix_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class SceneAligner:
    """Compute canonical coordinate system for meaningful views."""

    def __init__(self, gimbal_data: Optional[Dict] = None):
        self.gimbal_data = gimbal_data
        self.basis = None
        self.centroid = None

    def compute_basis(self, points: np.ndarray) -> Dict:
        """Compute canonical axes from point cloud."""
        self.centroid = np.mean(points, axis=0)

        if self.gimbal_data and len(self.gimbal_data) > 0:
            return self._from_gimbal()
        return self._from_pca(points)

    def _from_gimbal(self) -> Dict:
        """Use gimbal pitch/roll for ground plane."""
        pitches = [g['pitch'] for g in self.gimbal_data.values()]
        rolls = [g['roll'] for g in self.gimbal_data.values()]

        pitch = np.radians(np.median(pitches))
        roll = np.radians(np.median(rolls))

        up = np.array([0, 0, 1])
        up = rotation_matrix_z(-roll) @ rotation_matrix_x(-pitch) @ up
        up = up / np.linalg.norm(up)

        forward = np.array([1, 0, 0])
        forward = forward - np.dot(forward, up) * up
        forward = forward / np.linalg.norm(forward)

        right = np.cross(up, forward)

        self.basis = {
            'forward': forward,
            'right': right,
            'up': up,
            'rotation': np.stack([forward, right, up], axis=1),
            'source': 'gimbal'
        }
        return self.basis

    def _from_pca(self, points: np.ndarray) -> Dict:
        """PCA-based ground plane estimation."""
        sample = points[np.random.choice(len(points), min(50000, len(points)), replace=False)]

        pca = PCA(n_components=3)
        pca.fit(sample)

        # PC3 (smallest variance) is vertical
        up = pca.components_[2]
        if up[2] < 0:
            up = -up
        up = up / np.linalg.norm(up)

        forward = pca.components_[0]
        forward = forward - np.dot(forward, up) * up
        forward = forward / np.linalg.norm(forward)

        right = np.cross(up, forward)

        self.basis = {
            'forward': forward,
            'right': right,
            'up': up,
            'rotation': np.stack([forward, right, up], axis=1),
            'source': 'pca'
        }
        return self.basis

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Transform points to canonical coordinates."""
        if self.basis is None or self.centroid is None:
            self.compute_basis(points)
        return (points - self.centroid) @ self.basis['rotation']


# ============================================================================
# TRACKING RENDERER
# ============================================================================

class TrackingRenderer:
    """Render tracking trajectories with temporal coloring."""

    def __init__(self, trajectories: Dict, colors: Dict, info: Dict):
        self.trajectories = trajectories
        self.colors = colors
        self.info = info

    def render_trail(self, ax, track_id: int, view: str = 'top_down',
                     transform: Optional[np.ndarray] = None,
                     centroid: Optional[np.ndarray] = None,
                     line_width: float = 2.5, alpha_gradient: bool = True):
        """Render motion trail with temporal gradient."""
        if track_id not in self.trajectories or len(self.trajectories[track_id]) < 2:
            return

        traj = sorted(self.trajectories[track_id], key=lambda x: x['frame'])
        centers = np.array([p['center'] for p in traj])

        if transform is not None and centroid is not None:
            centers = (centers - centroid) @ transform

        # View projection
        if view == 'top_down':
            x, y = centers[:, 0], centers[:, 1]
        elif view == 'front':
            x, y = centers[:, 1], centers[:, 2]
        else:  # side
            x, y = centers[:, 0], centers[:, 2]

        color = np.array(self.colors.get(track_id, (0.5, 0.5, 0.5)))
        n = len(centers)

        # Draw segments with gradient
        for i in range(n - 1):
            t = i / max(n - 1, 1)
            seg_color = color * (0.4 + 0.6 * t) if alpha_gradient else color
            seg_alpha = 0.3 + 0.7 * t if alpha_gradient else 0.8
            seg_width = line_width * (0.5 + 0.5 * t)

            ax.plot([x[i], x[i+1]], [y[i], y[i+1]],
                   color=np.clip(seg_color, 0, 1),
                   alpha=seg_alpha, linewidth=seg_width,
                   solid_capstyle='round')

        # Start marker (circle)
        ax.scatter(x[0], y[0], c=[color * 0.6], s=60, marker='o',
                  edgecolors='white', linewidths=1.5, zorder=100)

        # End marker (triangle)
        ax.scatter(x[-1], y[-1], c=[color], s=100, marker='^',
                  edgecolors='black', linewidths=1.5, zorder=101)

    def render_all(self, ax, view: str = 'top_down',
                   transform: Optional[np.ndarray] = None,
                   centroid: Optional[np.ndarray] = None):
        """Render all trails."""
        for track_id in self.trajectories:
            self.render_trail(ax, track_id, view, transform, centroid)


# ============================================================================
# INSTANCE RENDERER
# ============================================================================

class InstanceRenderer:
    """Render instance segmentation with visual clarity."""

    def render_hull(self, ax, points: np.ndarray, color: Tuple, alpha: float = 0.2, view: str = 'top_down'):
        """Draw convex hull around instance."""
        if len(points) < 3:
            return

        if view == 'top_down':
            pts_2d = points[:, :2]
        elif view == 'front':
            pts_2d = points[:, [1, 2]]
        else:
            pts_2d = points[:, [0, 2]]

        try:
            hull = ConvexHull(pts_2d)
            hull_pts = pts_2d[hull.vertices]
            poly = MplPolygon(hull_pts, closed=True,
                             facecolor=(*color, alpha),
                             edgecolor=color, linewidth=2)
            ax.add_patch(poly)
        except Exception:
            pass

    def add_callout(self, ax, center: np.ndarray, track_id: int,
                    class_name: str, color: Tuple, view: str = 'top_down'):
        """Add annotation callout."""
        if view == 'top_down':
            x, y = center[0], center[1]
        elif view == 'front':
            x, y = center[1], center[2]
        else:
            x, y = center[0], center[2]

        label = f"{class_name.title()}\nT{track_id}"
        ax.annotate(label, xy=(x, y), xytext=(x + 0.3, y + 0.3),
                   fontsize=7, fontweight='bold', color='white',
                   arrowprops=dict(arrowstyle='->', color=color, lw=1),
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85),
                   zorder=200)


# ============================================================================
# FIGURE BUILDER
# ============================================================================

class FigureBuilder:
    """Create publication-quality figures."""

    DPI = 300

    def render_point_cloud(self, ax, points: np.ndarray, labels: np.ndarray,
                           conf: np.ndarray, view: str = 'top_down',
                           conf_threshold: float = 1.5, point_size: float = 0.3,
                           bg_alpha: float = 0.15, inst_alpha: float = 0.8):
        """Render point cloud with instance colors."""
        labels_flat = labels.flatten()
        conf_flat = conf.flatten()

        valid = conf_flat > conf_threshold
        pts = points[valid]
        lbls = labels_flat[valid]

        if len(pts) == 0:
            return

        # View projection indices
        if view == 'top_down':
            xi, yi = 0, 1
        elif view == 'front':
            xi, yi = 1, 2
        else:
            xi, yi = 0, 2

        # Background
        bg = lbls == 0
        if np.any(bg):
            ax.scatter(pts[bg, xi], pts[bg, yi], c='lightgray',
                      s=point_size, alpha=bg_alpha, rasterized=True)

        # Instances
        for inst_id in np.unique(lbls):
            if inst_id == 0:
                continue
            mask = lbls == inst_id
            color = get_instance_color(inst_id)
            ax.scatter(pts[mask, xi], pts[mask, yi], c=[color],
                      s=point_size * 1.5, alpha=inst_alpha, rasterized=True)

    def create_comparison_figure(self, input_image: Optional[np.ndarray],
                                  points: np.ndarray, labels: np.ndarray,
                                  conf: np.ndarray,
                                  track_renderer: Optional[TrackingRenderer] = None,
                                  output_path: Optional[str] = None,
                                  conf_threshold: float = 1.5) -> plt.Figure:
        """Create 3-panel comparison: [Input | Segmentation | Trajectories]"""
        fig = plt.figure(figsize=(12, 4), dpi=self.DPI)
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)

        # Panel A: Input
        ax1 = fig.add_subplot(gs[0])
        if input_image is not None:
            ax1.imshow(input_image)
        ax1.set_title('(a) Input Frame', fontsize=11, fontweight='bold')
        ax1.axis('off')

        # Panel B: Segmentation
        ax2 = fig.add_subplot(gs[1])
        self.render_point_cloud(ax2, points, labels, conf,
                               conf_threshold=conf_threshold, point_size=0.5)
        ax2.set_title('(b) Instance Segmentation', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Forward (m)', fontsize=9)
        ax2.set_ylabel('Right (m)', fontsize=9)
        ax2.invert_yaxis()  # Flip so forward is up
        ax2.set_aspect('equal')
        ax2.tick_params(labelsize=8)

        # Panel C: Trajectories
        ax3 = fig.add_subplot(gs[2])
        self.render_point_cloud(ax3, points, labels, conf,
                               conf_threshold=conf_threshold, point_size=0.3,
                               bg_alpha=0.1, inst_alpha=0.4)
        if track_renderer:
            track_renderer.render_all(ax3, view='top_down')
        ax3.set_title('(c) Motion Trajectories', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Forward (m)', fontsize=9)
        ax3.set_ylabel('Right (m)', fontsize=9)
        ax3.invert_yaxis()  # Flip so forward is up
        ax3.set_aspect('equal')
        ax3.tick_params(labelsize=8)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.DPI, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved comparison figure: {output_path}")

        return fig

    def create_multiview_figure(self, points: np.ndarray, labels: np.ndarray,
                                 conf: np.ndarray,
                                 input_image: Optional[np.ndarray] = None,
                                 output_path: Optional[str] = None,
                                 conf_threshold: float = 1.5) -> plt.Figure:
        """Create 4-panel canonical views."""
        fig = plt.figure(figsize=(10, 8), dpi=self.DPI)
        gs = GridSpec(2, 2, wspace=0.2, hspace=0.25)

        # Panel A: Input
        ax1 = fig.add_subplot(gs[0, 0])
        if input_image is not None:
            ax1.imshow(input_image)
        ax1.set_title('(a) Input Frame', fontweight='bold')
        ax1.axis('off')

        # Panel B: Top-Down
        ax2 = fig.add_subplot(gs[0, 1])
        self.render_point_cloud(ax2, points, labels, conf, 'top_down', conf_threshold)
        ax2.set_title('(b) Top-Down View', fontweight='bold')
        ax2.set_xlabel('Forward (m)')
        ax2.set_ylabel('Right (m)')
        ax2.invert_yaxis()  # Flip so forward is up
        ax2.set_aspect('equal')

        # Panel C: Front
        ax3 = fig.add_subplot(gs[1, 0])
        self.render_point_cloud(ax3, points, labels, conf, 'front', conf_threshold)
        ax3.set_title('(c) Front View', fontweight='bold')
        ax3.set_xlabel('Right (m)')
        ax3.set_ylabel('Up (m)')
        ax3.set_aspect('equal')

        # Panel D: Side
        ax4 = fig.add_subplot(gs[1, 1])
        self.render_point_cloud(ax4, points, labels, conf, 'side', conf_threshold)
        ax4.set_title('(d) Side View', fontweight='bold')
        ax4.set_xlabel('Forward (m)')
        ax4.set_ylabel('Up (m)')
        ax4.set_aspect('equal')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.DPI, bbox_inches='tight', facecolor='white')
            print(f"Saved multiview figure: {output_path}")

        return fig

    def create_temporal_strip(self, frames: List[Dict], output_path: Optional[str] = None,
                              n_cols: int = 5) -> plt.Figure:
        """Create temporal sequence strip."""
        n_frames = len(frames)
        n_rows = (n_frames + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), dpi=self.DPI)
        axes = np.atleast_2d(axes)

        for idx, frame in enumerate(frames):
            r, c = idx // n_cols, idx % n_cols
            ax = axes[r, c]
            if 'image' in frame and frame['image'] is not None:
                ax.imshow(frame['image'])
            if 'frame_id' in frame:
                ax.text(0.02, 0.98, f"t={frame['frame_id']}", transform=ax.transAxes,
                       fontsize=8, va='top', color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
            ax.axis('off')

        # Hide empty
        for idx in range(n_frames, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.DPI, bbox_inches='tight')
            print(f"Saved temporal strip: {output_path}")

        return fig


# ============================================================================
# INTERACTIVE EXPORTER
# ============================================================================

class InteractiveExporter:
    """Export interactive Plotly HTML."""

    def export(self, points: np.ndarray, labels: np.ndarray,
               trajectories: Optional[Dict] = None,
               track_colors: Optional[Dict] = None,
               output_path: str = "interactive.html",
               max_points: int = 500000,
               conf: Optional[np.ndarray] = None,
               conf_threshold: float = 1.5) -> Optional[str]:
        """Export interactive HTML viewer."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping HTML export.")
            return None

        labels_flat = labels.flatten()

        # Filter by confidence if available
        if conf is not None:
            conf_flat = conf.flatten()
            valid = conf_flat > conf_threshold
            points = points[valid]
            labels_flat = labels_flat[valid]

        # Subsample if too many points
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            labels_flat = labels_flat[idx]

        fig = go.Figure()

        # Background
        bg = labels_flat == 0
        if np.any(bg):
            fig.add_trace(go.Scatter3d(
                x=points[bg, 0], y=points[bg, 1], z=points[bg, 2],
                mode='markers', marker=dict(size=1, color='gray', opacity=0.2),
                name='Background'
            ))

        # Instances
        for inst_id in np.unique(labels_flat):
            if inst_id == 0:
                continue
            mask = labels_flat == inst_id
            color = get_instance_color(inst_id)
            color_str = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            fig.add_trace(go.Scatter3d(
                x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
                mode='markers', marker=dict(size=2, color=color_str, opacity=0.8),
                name=f'Instance {inst_id}'
            ))

        # Trajectories
        if trajectories and track_colors:
            for tid, traj in trajectories.items():
                if len(traj) < 2:
                    continue
                centers = np.array([p['center'] for p in sorted(traj, key=lambda x: x['frame'])])
                color = track_colors.get(tid, (0.5, 0.5, 0.5))
                color_str = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
                fig.add_trace(go.Scatter3d(
                    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                    mode='lines+markers', line=dict(width=4, color=color_str),
                    marker=dict(size=4), name=f'Track {tid}'
                ))

        # Layout with view buttons
        fig.update_layout(
            title='Interactive 3D Point Cloud',
            scene=dict(xaxis_title='Forward', yaxis_title='Right', zaxis_title='Up', aspectmode='data'),
            updatemenus=[dict(
                type="buttons", showactive=True, y=1.0, x=0.0,
                buttons=[
                    dict(label="Top-Down", method="relayout", args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2}}]),
                    dict(label="Front", method="relayout", args=[{"scene.camera.eye": {"x": 0, "y": -2, "z": 0.3}}]),
                    dict(label="Side", method="relayout", args=[{"scene.camera.eye": {"x": 2, "y": 0, "z": 0.3}}]),
                    dict(label="3/4 View", method="relayout", args=[{"scene.camera.eye": {"x": 1.5, "y": -1.5, "z": 1}}]),
                ]
            )]
        )

        fig.write_html(output_path)
        print(f"Saved interactive HTML: {output_path}")
        return output_path


# ============================================================================
# ANIMATION EXPORTER
# ============================================================================

class AnimationExporter:
    """Export GIF/video animations."""

    def create_rotating_gif(self, points: np.ndarray, labels: np.ndarray,
                            output_path: str = "rotating.gif",
                            n_frames: int = 24, fps: int = 15,
                            figsize: Tuple = (6, 6),
                            max_points: int = 100000) -> Optional[str]:
        """Create rotating point cloud GIF."""
        if not IMAGEIO_AVAILABLE:
            print("imageio not available. Skipping GIF export.")
            return None

        labels_flat = labels.flatten()

        # Subsample
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            labels_flat = labels_flat[idx]

        frames = []
        print(f"Generating {n_frames} frames for rotating GIF...")

        for i in range(n_frames):
            angle = (i / n_frames) * 360
            fig = plt.figure(figsize=figsize, dpi=100)
            ax = fig.add_subplot(111, projection='3d')

            # Background
            bg = labels_flat == 0
            if np.any(bg):
                ax.scatter(points[bg, 0], points[bg, 1], points[bg, 2],
                          c='lightgray', s=0.3, alpha=0.1)

            # Instances
            for inst_id in np.unique(labels_flat):
                if inst_id == 0:
                    continue
                mask = labels_flat == inst_id
                color = get_instance_color(inst_id)
                ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                          c=[color], s=1, alpha=0.8)

            ax.view_init(elev=30, azim=angle)
            ax.set_xlabel('Forward')
            ax.set_ylabel('Right')
            ax.set_zlabel('Up')

            # Render to array
            fig.canvas.draw()
            try:
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

            frames.append(frame)
            plt.close(fig)

            if (i + 1) % 8 == 0:
                print(f"  Frame {i + 1}/{n_frames}")

        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"Saved rotating GIF: {output_path}")
        return output_path

    def create_temporal_video(self, result_dir: str, frame_ids: List[str],
                              output_path: str = "temporal.gif",
                              fps: int = 10) -> Optional[str]:
        """Create temporal sequence video/GIF."""
        if not IMAGEIO_AVAILABLE:
            print("imageio not available. Skipping video export.")
            return None

        result_path = Path(result_dir)
        frames = []

        print(f"Loading {len(frame_ids)} frames for temporal video...")
        for fid in frame_ids:
            for subdir in ['annotated_2d', 'images']:
                for ext in ['.png', '.jpg']:
                    img_path = result_path / subdir / f'{fid}{ext}'
                    if not img_path.exists() and subdir == 'annotated_2d':
                        img_path = result_path / subdir / f'{fid}_tracked{ext}'
                    if img_path.exists():
                        frame = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        break
                if frames and len(frames) > 0:
                    break

        if not frames:
            print("No frames found for video")
            return None

        # Always save as GIF for compatibility
        if not output_path.endswith('.gif'):
            output_path = output_path.rsplit('.', 1)[0] + '.gif'

        imageio.mimsave(output_path, frames, duration=1000 // fps, loop=0)
        print(f"Saved temporal video: {output_path}")
        return output_path


# ============================================================================
# MAIN VISUALIZER
# ============================================================================

class EcologyVisualizer:
    """Main facade for ecology paper visualizations."""

    def __init__(self, result_dir: str, srt_file: Optional[str] = None):
        self.result_dir = result_dir
        self.frame_ids = get_available_frames(result_dir)

        if not self.frame_ids:
            raise ValueError(f"No frames found in {result_dir}")

        print(f"Found {len(self.frame_ids)} frames")

        # Parse gimbal data if available
        frame_nums = [int(f) for f in self.frame_ids if f.isdigit()]
        self.gimbal_data = parse_dji_logs(srt_file, frame_nums) if srt_file else None

        # Initialize components
        self.aligner = SceneAligner(self.gimbal_data)
        self.figure_builder = FigureBuilder()
        self.interactive_exporter = InteractiveExporter()
        self.animation_exporter = AnimationExporter()
        self.instance_renderer = InstanceRenderer()

        # Cached data
        self._points = None
        self._labels = None
        self._conf = None
        self._points_canonical = None
        self._trajectories = None
        self._track_colors = None
        self._track_info = None

    def load_all_frames(self, conf_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and combine all frame data."""
        if self._points is not None:
            return self._points, self._labels, self._conf

        all_pts, all_lbls, all_conf = [], [], []

        print(f"Loading {len(self.frame_ids)} frames...")
        for i, fid in enumerate(self.frame_ids):
            try:
                data = load_frame_data(self.result_dir, fid)
                pts = unproject_depth(data['depth'], data['intrinsics'], data['pose'])
                all_pts.append(pts)
                all_lbls.append(data['instance_labels'].flatten())
                all_conf.append(data['conf'].flatten())
            except Exception as e:
                print(f"Warning: Failed to load frame {fid}: {e}")

            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{len(self.frame_ids)} frames")

        self._points = np.concatenate(all_pts, axis=0)
        self._labels = np.concatenate(all_lbls, axis=0)
        self._conf = np.concatenate(all_conf, axis=0)

        print(f"Loaded {len(self._points):,} total points")
        return self._points, self._labels, self._conf

    def compute_canonical(self) -> np.ndarray:
        """Compute canonical coordinates."""
        if self._points_canonical is not None:
            return self._points_canonical

        pts, _, _ = self.load_all_frames()
        self.aligner.compute_basis(pts)
        self._points_canonical = self.aligner.transform(pts)

        print(f"Canonical coordinates computed using {self.aligner.basis['source']} alignment")
        return self._points_canonical

    def load_tracking(self) -> Tuple[Dict, Dict, Dict]:
        """Load tracking data."""
        if self._trajectories is not None:
            return self._trajectories, self._track_colors, self._track_info

        self._trajectories = {}
        self._track_info = {}

        # Load from tracking_summary.json first
        summary_path = Path(self.result_dir) / 'tracking_summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

            tracks = summary.get('tracks', {})
            if isinstance(tracks, dict):
                for tid_str, tdata in tracks.items():
                    tid = int(tid_str)
                    self._track_info[tid] = {
                        'class_name': tdata.get('class_name', 'unknown'),
                        'first_frame': tdata.get('first_frame', 0),
                        'last_frame': tdata.get('last_frame', 0),
                        'length': tdata.get('length', 0)
                    }
                    self._trajectories[tid] = []

        # Build trajectories from bounding boxes
        print(f"Loading tracking data from {len(self.frame_ids)} frames...")
        for i, fid in enumerate(self.frame_ids):
            try:
                data = load_frame_data(self.result_dir, fid)
                for bbox in data.get('bboxes', []):
                    # Use explicit None checks to handle track_id=0 correctly
                    tid = bbox.get('track_id')
                    if tid is None:
                        tid = bbox.get('persistent_instance_id')
                    if tid is None:
                        tid = bbox.get('instance_id')
                    if tid is None:
                        continue
                    tid = int(tid)

                    if tid not in self._trajectories:
                        self._trajectories[tid] = []
                    if tid not in self._track_info:
                        self._track_info[tid] = {
                            'class_name': bbox.get('class_name', 'unknown'),
                            'first_frame': i, 'last_frame': i, 'length': 0
                        }

                    self._trajectories[tid].append({
                        'frame': i,
                        'center': bbox.get('center', [0, 0, 0]),
                        'confidence': bbox.get('confidence', 1.0)
                    })
                    self._track_info[tid]['last_frame'] = i
                    self._track_info[tid]['length'] += 1
            except Exception:
                pass

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(self.frame_ids)} frames")

        # Generate colors
        self._track_colors = {tid: get_track_color(i) for i, tid in enumerate(sorted(self._trajectories.keys()))}

        print(f"Loaded {len(self._trajectories)} tracks")
        return self._trajectories, self._track_colors, self._track_info

    def create_publication_figure(self, frame_idx: int = 0, output_path: Optional[str] = None) -> plt.Figure:
        """Generate 3-panel comparison figure."""
        fid = self.frame_ids[frame_idx]
        data = load_frame_data(self.result_dir, fid)

        pts = unproject_depth(data['depth'], data['intrinsics'], data['pose'])
        self.aligner.compute_basis(pts)
        pts_canonical = self.aligner.transform(pts)

        traj, colors, info = self.load_tracking()
        track_renderer = TrackingRenderer(traj, colors, info)

        return self.figure_builder.create_comparison_figure(
            input_image=data.get('image'),
            points=pts_canonical,
            labels=data['instance_labels'],
            conf=data['conf'],
            track_renderer=track_renderer,
            output_path=output_path
        )

    def create_multiview_figure(self, frame_idx: int = 0, output_path: Optional[str] = None) -> plt.Figure:
        """Generate multi-view figure."""
        fid = self.frame_ids[frame_idx]
        data = load_frame_data(self.result_dir, fid)

        pts = unproject_depth(data['depth'], data['intrinsics'], data['pose'])
        self.aligner.compute_basis(pts)
        pts_canonical = self.aligner.transform(pts)

        return self.figure_builder.create_multiview_figure(
            points=pts_canonical,
            labels=data['instance_labels'],
            conf=data['conf'],
            input_image=data.get('image'),
            output_path=output_path
        )

    def export_interactive_html(self, output_path: str = "interactive.html",
                                 max_points: int = 300000) -> Optional[str]:
        """Export interactive Plotly HTML."""
        pts_canonical = self.compute_canonical()
        _, labels, conf = self.load_all_frames()
        traj, colors, _ = self.load_tracking()

        return self.interactive_exporter.export(
            points=pts_canonical,
            labels=labels.reshape(-1),
            trajectories=traj,
            track_colors=colors,
            output_path=output_path,
            max_points=max_points,
            conf=conf,
            conf_threshold=1.5
        )

    def export_rotating_gif(self, output_path: str = "rotating.gif",
                            n_frames: int = 24, max_points: int = 80000) -> Optional[str]:
        """Export rotating point cloud GIF."""
        pts_canonical = self.compute_canonical()
        _, labels, conf = self.load_all_frames()

        # Filter by confidence
        valid = conf > 1.5
        pts_valid = pts_canonical[valid]
        labels_valid = labels[valid]

        return self.animation_exporter.create_rotating_gif(
            points=pts_valid,
            labels=labels_valid.reshape(-1),
            output_path=output_path,
            n_frames=n_frames,
            max_points=max_points
        )

    def export_temporal_video(self, output_path: str = "temporal.gif") -> Optional[str]:
        """Export temporal sequence video."""
        return self.animation_exporter.create_temporal_video(
            result_dir=self.result_dir,
            frame_ids=self.frame_ids,
            output_path=output_path
        )

    def run_all(self, output_dir: str):
        """Generate all visualizations."""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 50)
        print("ECOLOGY VISUALIZER - Generating All Outputs")
        print("=" * 50)

        print("\n[1/5] Creating publication figure...")
        self.create_publication_figure(0, os.path.join(output_dir, "comparison_figure.png"))

        print("\n[2/5] Creating multiview figure...")
        self.create_multiview_figure(0, os.path.join(output_dir, "multiview_figure.png"))

        print("\n[3/5] Creating interactive HTML...")
        self.export_interactive_html(os.path.join(output_dir, "interactive_3d.html"))

        print("\n[4/5] Creating rotating GIF...")
        self.export_rotating_gif(os.path.join(output_dir, "rotating_cloud.gif"))

        print("\n[5/5] Creating temporal video...")
        self.export_temporal_video(os.path.join(output_dir, "temporal_sequence.gif"))

        print("\n" + "=" * 50)
        print(f"All visualizations saved to: {output_dir}")
        print("=" * 50)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ecology Paper Visualization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.ecology_visualizer --result_dir results/wildlift/zebra-1 --output_dir figures/
  python -m tools.ecology_visualizer --result_dir results/cami/rhino-35 --srt_file data/DJI.SRT --output_dir figures/
  python -m tools.ecology_visualizer --result_dir results/wildlift/elephant-1 --mode interactive
        """
    )

    parser.add_argument('--result_dir', type=str, required=True, help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default='ecology_figures', help='Output directory')
    parser.add_argument('--srt_file', type=str, default=None, help='DJI SRT file for gimbal data')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'publication', 'multiview', 'interactive', 'animation'],
                        help='Visualization mode')
    parser.add_argument('--frame', type=int, default=0, help='Frame index for single-frame outputs')

    args = parser.parse_args()

    viz = EcologyVisualizer(args.result_dir, args.srt_file)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'all':
        viz.run_all(args.output_dir)
    elif args.mode == 'publication':
        viz.create_publication_figure(args.frame, os.path.join(args.output_dir, "comparison_figure.png"))
    elif args.mode == 'multiview':
        viz.create_multiview_figure(args.frame, os.path.join(args.output_dir, "multiview_figure.png"))
    elif args.mode == 'interactive':
        viz.export_interactive_html(os.path.join(args.output_dir, "interactive_3d.html"))
    elif args.mode == 'animation':
        viz.export_rotating_gif(os.path.join(args.output_dir, "rotating_cloud.gif"))
        viz.export_temporal_video(os.path.join(args.output_dir, "temporal_sequence.gif"))


if __name__ == '__main__':
    main()
