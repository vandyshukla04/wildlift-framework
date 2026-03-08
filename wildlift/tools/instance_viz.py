#!/usr/bin/env python3
"""
Enhanced Instance Visualization Tool for CUT3R

Produces publication-quality visualizations for animal instance isolation and tracking:
1. Top-down point cloud view with highlighted instances (distance-colored, with legend)
2. Multi-view instance visualization (original image + top/side/front orthographic views)
3. Full scene with all tracked instances highlighted

Usage:
    python tools/instance_visualizer.py --result_dir results/wildlift/tmp-zebr-3-revisit-1 --output_dir visualizations/
    python tools/instance_visualizer.py --result_dir results/cami/tmp-rhin-35_1-revisit-1 --frame 1680 --output_dir visualizations/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional, Any
import glob as glob_module


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_frame_data(result_dir: str, frame_id: str) -> Dict[str, Any]:
    """
    Load all data for a single frame from the results directory.

    Args:
        result_dir: Path to results directory (e.g., results/wildlift/tmp-zebr-3-revisit-1)
        frame_id: Frame identifier (e.g., '6520', '1680')

    Returns:
        Dictionary containing:
        - depth: [H, W] depth map
        - conf: [H, W] confidence scores
        - pose: [4, 4] camera-to-world transformation
        - intrinsics: [3, 3] camera intrinsic matrix
        - instance_labels: [H, W] per-pixel instance IDs
        - bboxes: list of bounding box dictionaries
        - image: [H, W, 3] original RGB image (if available)
        - annotated_image: [H, W, 3] image with tracking annotations (if available)
    """
    result_path = Path(result_dir)
    data = {}

    # Load depth map
    depth_path = result_path / f'depth/{frame_id}.npy'
    if depth_path.exists():
        data['depth'] = np.load(depth_path)
    else:
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    # Load confidence scores
    conf_path = result_path / f'conf/{frame_id}.npy'
    if conf_path.exists():
        data['conf'] = np.load(conf_path)
    else:
        data['conf'] = np.ones_like(data['depth']) * 5.0  # Default high confidence

    # Load camera parameters
    cam_path = result_path / f'camera/{frame_id}.npz'
    if cam_path.exists():
        cam_data = np.load(cam_path)
        data['pose'] = cam_data['pose']
        data['intrinsics'] = cam_data['intrinsics']
    else:
        raise FileNotFoundError(f"Camera file not found: {cam_path}")

    # Load instance labels
    instance_path = result_path / f'instance_labels/{frame_id}.npy'
    if instance_path.exists():
        data['instance_labels'] = np.load(instance_path)
    else:
        data['instance_labels'] = np.zeros_like(data['depth'], dtype=np.int32)

    # Load bounding boxes
    bbox_path = result_path / f'bounding_boxes/{frame_id}.json'
    if bbox_path.exists():
        with open(bbox_path) as f:
            data['bboxes'] = json.load(f)
    else:
        data['bboxes'] = []

    # Load original image if available
    # Try to find original image from various source locations
    img_patterns = [
        result_path / f'images/{frame_id}.png',
        result_path / f'images/{frame_id}.jpg',
        result_path.parent.parent / f'images/{frame_id}.png',
        result_path.parent.parent / f'images/{frame_id}.jpg',
    ]

    # Also search in the wd_data source directories based on result folder name
    # e.g., tmp-zebr-14_2-revisit-1 -> zebras/zebr-14_2
    result_name = result_path.name  # e.g., "tmp-zebr-14_2-revisit-1"
    wd_data_base = Path('/home/shuklva/CUT3R/examples/wd_data')

    # Parse the result folder name to find source data
    if 'zebr' in result_name:
        # Extract zebra folder name (e.g., "zebr-14_2" from "tmp-zebr-14_2-revisit-1")
        parts = result_name.replace('tmp-', '').split('-revisit')[0]
        img_patterns.extend([
            wd_data_base / f'zebras/{parts}/{frame_id}.jpg',
            wd_data_base / f'zebras/{parts}/{frame_id}.png',
        ])
    elif 'rhin' in result_name:
        parts = result_name.replace('tmp-', '').split('-revisit')[0]
        img_patterns.extend([
            wd_data_base / f'rhinos/{parts}/{frame_id}.jpg',
            wd_data_base / f'rhinos/{parts}/{frame_id}.png',
            wd_data_base / f'rhinos_cami/{parts}/{frame_id}.jpg',
            wd_data_base / f'rhinos_cami/{parts}/{frame_id}.png',
        ])
    elif 'eleph' in result_name:
        parts = result_name.replace('tmp-', '').split('-revisit')[0]
        img_patterns.extend([
            wd_data_base / f'elephants/{parts}/{frame_id}.jpg',
            wd_data_base / f'elephants/{parts}/{frame_id}.png',
        ])

    for img_path in img_patterns:
        if img_path.exists():
            data['image'] = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            break

    # Load annotated (tracked) image if available
    annotated_path = result_path / f'annotated_2d/{frame_id}_tracked.png'
    if annotated_path.exists():
        data['annotated_image'] = cv2.cvtColor(cv2.imread(str(annotated_path)), cv2.COLOR_BGR2RGB)

    return data


def get_available_frames(result_dir: str) -> List[str]:
    """Get list of available frame IDs in the results directory."""
    result_path = Path(result_dir)
    depth_files = sorted(result_path.glob('depth/*.npy'))
    return [f.stem for f in depth_files]


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Convert depth map to 3D points in world frame.

    Args:
        depth: [H, W] depth map
        intrinsics: [3, 3] camera intrinsic matrix
        pose: [4, 4] camera-to-world transformation

    Returns:
        points: [N, 3] 3D points in world coordinates where N = H * W
    """
    H, W = depth.shape

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u.flatten().astype(np.float64)
    v_flat = v.flatten().astype(np.float64)
    z = depth.flatten().astype(np.float64)

    # Extract camera parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Unproject to camera space
    x_cam = (u_flat - cx) * z / fx
    y_cam = (v_flat - cy) * z / fy
    z_cam = z

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [N, 3]

    # Transform to world space
    points_cam_h = np.concatenate(
        [points_cam, np.ones((len(points_cam), 1))],
        axis=-1
    )  # [N, 4]
    points_world = (pose @ points_cam_h.T).T[:, :3]  # [N, 3]

    return points_world


def get_instance_points(
    points: np.ndarray,
    instance_labels: np.ndarray,
    instance_id: int,
    conf: Optional[np.ndarray] = None,
    conf_threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 3D points belonging to a specific instance.

    Args:
        points: [N, 3] 3D point cloud
        instance_labels: [H, W] instance label map
        instance_id: target instance ID
        conf: [H, W] confidence scores (optional)
        conf_threshold: minimum confidence (optional)

    Returns:
        instance_points: [M, 3] points for this instance
        mask: [N] boolean mask
    """
    instance_labels_flat = instance_labels.flatten()
    mask = instance_labels_flat == instance_id

    # Apply confidence threshold if provided
    if conf is not None:
        conf_flat = conf.flatten()
        mask = mask & (conf_flat > conf_threshold)

    instance_points = points[mask]
    return instance_points, mask


# ============================================================================
# COLOR UTILITIES
# ============================================================================

# Distinct colors for instances (RGB tuples, 0-1 range)
INSTANCE_COLORS = {
    0: (0.7, 0.7, 0.7),    # Gray (background)
    1: (1.0, 0.2, 0.2),    # Red
    2: (0.2, 0.8, 0.2),    # Green
    3: (0.3, 0.5, 1.0),    # Blue
    4: (1.0, 0.8, 0.0),    # Yellow/Gold
    5: (0.8, 0.2, 0.8),    # Magenta
    6: (0.0, 0.8, 0.8),    # Cyan
    7: (1.0, 0.5, 0.0),    # Orange
    8: (0.5, 0.0, 0.8),    # Purple
}

def get_instance_color(instance_id: int) -> Tuple[float, float, float]:
    """Get consistent color for an instance ID."""
    if instance_id in INSTANCE_COLORS:
        return INSTANCE_COLORS[instance_id]
    # Generate deterministic color for unknown IDs
    np.random.seed(instance_id * 42)
    return tuple(np.random.rand(3).tolist())


# ============================================================================
# VISUALIZATION 1: TOP-DOWN VIEW WITH HIGHLIGHTED INSTANCES
# ============================================================================

def create_topdown_view_with_instances(
    result_dir: str,
    frame_ids: Optional[List[str]] = None,
    conf_threshold: float = 1.5,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    point_size: float = 0.3,
    show_legend: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create a top-down (bird's eye) view of the point cloud with highlighted instances.

    The background is colored by distance (using viridis colormap), and each tracked
    animal instance is highlighted with a distinct color and marker in the legend.

    Args:
        result_dir: Path to results directory
        frame_ids: List of frame IDs to include (None = all frames)
        conf_threshold: Minimum confidence to include points
        output_path: Path to save the figure (optional)
        figsize: Figure size in inches
        point_size: Size of points in scatter plot
        show_legend: Whether to show the legend
        title: Custom title (optional)

    Returns:
        matplotlib Figure object
    """
    result_path = Path(result_dir)

    # Get available frames
    available_frames = get_available_frames(result_dir)
    if frame_ids is None:
        frame_ids = available_frames
    else:
        frame_ids = [f for f in frame_ids if f in available_frames]

    if not frame_ids:
        raise ValueError(f"No valid frames found in {result_dir}")

    print(f"Loading {len(frame_ids)} frames for top-down visualization...")

    # Collect all points and instance information
    all_points = []
    all_instance_ids = []
    all_conf = []
    instance_info = {}  # track_id -> {class_name, confidence, ...}

    for frame_id in frame_ids:
        try:
            data = load_frame_data(result_dir, frame_id)
            points = unproject_depth(data['depth'], data['intrinsics'], data['pose'])

            all_points.append(points)
            all_instance_ids.append(data['instance_labels'].flatten())
            all_conf.append(data['conf'].flatten())

            # Collect instance info from bounding boxes
            for bbox in data['bboxes']:
                track_id = bbox.get('track_id', bbox.get('instance_id', 0))
                if track_id not in instance_info:
                    instance_info[track_id] = {
                        'class_name': bbox.get('class_name', 'unknown'),
                        'confidence': bbox.get('confidence', 0.0),
                        'instance_id': bbox.get('instance_id', track_id)
                    }
                else:
                    # Update with max confidence
                    instance_info[track_id]['confidence'] = max(
                        instance_info[track_id]['confidence'],
                        bbox.get('confidence', 0.0)
                    )
        except Exception as e:
            print(f"  Warning: Failed to load frame {frame_id}: {e}")
            continue

    if not all_points:
        raise ValueError("No valid frames loaded")

    # Concatenate all data
    points = np.vstack(all_points)
    instance_ids = np.concatenate(all_instance_ids)
    conf = np.concatenate(all_conf)

    # Filter by confidence
    conf_mask = conf > conf_threshold
    points = points[conf_mask]
    instance_ids = instance_ids[conf_mask]

    print(f"Total points after filtering: {len(points):,}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique instances (excluding background = 0)
    unique_instances = sorted(set(instance_ids) - {0})

    # First, plot background points (instance_id == 0) colored by distance
    bg_mask = instance_ids == 0
    bg_points = points[bg_mask]

    if len(bg_points) > 0:
        # Color by Z (depth/distance)
        z_values = bg_points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        z_norm = (z_values - z_min) / (z_max - z_min + 1e-8)

        # Plot background with viridis colormap
        scatter_bg = ax.scatter(
            bg_points[:, 0], bg_points[:, 2],  # X-Z plane (top view)
            c=z_norm, cmap='viridis', s=point_size, alpha=0.6,
            rasterized=True  # Better for large point clouds
        )

        # Add colorbar for distance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(scatter_bg, cax=cax)
        cbar.set_label('Distance', fontsize=11)

    # Plot each instance with distinct color
    legend_elements = []

    for inst_id in unique_instances:
        inst_mask = instance_ids == inst_id
        inst_points = points[inst_mask]

        if len(inst_points) < 10:  # Skip very small instances
            continue

        color = get_instance_color(inst_id)

        # Get instance info
        info = instance_info.get(inst_id, {'class_name': 'unknown', 'confidence': 0.0})

        # Plot instance points (highlighted, larger)
        ax.scatter(
            inst_points[:, 0], inst_points[:, 2],  # X-Z plane
            c=[color], s=point_size * 8, alpha=0.9,
            edgecolors='white', linewidths=0.3,
            zorder=10  # On top
        )

        # Plot instance center marker
        center = inst_points.mean(axis=0)
        ax.scatter(
            center[0], center[2],
            c=[color], s=150, marker='*',
            edgecolors='black', linewidths=1.5,
            zorder=20
        )

        # Add to legend
        label = f"{info['class_name']} ({info['confidence']:.2f})"
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor='black', label=label)
        )

    # Formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Z', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        dataset_name = result_path.parent.name
        ax.set_title(f'Highlighted Instances - Top View', fontsize=14, fontweight='bold')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    if show_legend and legend_elements:
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=9,
            framealpha=0.9,
            edgecolor='black'
        )

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved top-down view to: {output_path}")

    return fig


# ============================================================================
# VISUALIZATION 2: MULTI-VIEW INSTANCE VISUALIZATION
# ============================================================================

def create_instance_multiview(
    result_dir: str,
    frame_id: str,
    instance_id: int,
    conf_threshold: float = 1.5,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create a multi-view visualization for a single instance showing:
    - Top-left: Original image with instance mask highlighted
    - Top-right: Top view (X-Z plane)
    - Bottom-left: Side view (X-Y plane)
    - Bottom-right: Front view (Z-Y plane)

    Args:
        result_dir: Path to results directory
        frame_id: Frame identifier
        instance_id: Instance ID to visualize
        conf_threshold: Minimum confidence threshold
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Load frame data
    data = load_frame_data(result_dir, frame_id)
    points = unproject_depth(data['depth'], data['intrinsics'], data['pose'])

    # Get instance points - try with given threshold first, then relax if needed
    instance_points, mask = get_instance_points(
        points, data['instance_labels'], instance_id,
        data['conf'], conf_threshold
    )

    # If no points found, try progressively lower thresholds
    if len(instance_points) < 10:
        for fallback_threshold in [1.0, 0.5, 0.0]:
            instance_points, mask = get_instance_points(
                points, data['instance_labels'], instance_id,
                data['conf'], fallback_threshold
            )
            if len(instance_points) >= 10:
                print(f"  Note: Used lower confidence threshold ({fallback_threshold}) for instance {instance_id}")
                break

    if len(instance_points) < 10:
        raise ValueError(f"Instance {instance_id} has too few points ({len(instance_points)}) even with no confidence filtering")

    # Get instance info
    instance_info = None
    for bbox in data['bboxes']:
        if bbox.get('instance_id') == instance_id or bbox.get('track_id') == instance_id:
            instance_info = bbox
            break

    if instance_info is None:
        instance_info = {'class_name': 'Unknown', 'confidence': 0.0}

    class_name = instance_info.get('class_name', 'Unknown').capitalize()
    confidence = instance_info.get('confidence', 0.0)

    # Create figure with 2x3 subplots (added one more column for original+mask)
    fig = plt.figure(figsize=(figsize[0] * 1.5, figsize[1]))  # Wider figure
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)

    # Color instance points by depth
    z_values = instance_points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_norm = (z_values - z_min) / (z_max - z_min + 1e-8)
    cmap = plt.get_cmap('viridis')
    colors = cmap(z_norm)

    instance_color = get_instance_color(instance_id)

    # Get original image (without bounding boxes)
    original_img = None
    if 'image' in data:
        original_img = data['image'].copy().astype(np.float32) / 255.0

    # Get annotated image (with bounding boxes) as fallback
    annotated_img = None
    if 'annotated_image' in data:
        annotated_img = data['annotated_image'].copy().astype(np.float32) / 255.0

    # Create the instance mask (at model resolution)
    mask_2d_model = data['instance_labels'] == instance_id

    # Create a bright green color for the highlighted instance
    highlight_color = np.array([0.0, 1.0, 0.0])  # Bright green

    # ===================
    # Top-left: Original image with mask overlay ONLY (no bounding boxes)
    # ===================
    ax1 = fig.add_subplot(gs[0, 0])

    if original_img is not None:
        # Resize mask to match original image size if different
        img_h, img_w = original_img.shape[:2]
        mask_h, mask_w = mask_2d_model.shape

        if (img_h, img_w) != (mask_h, mask_w):
            # Resize mask to match image
            mask_2d = cv2.resize(
                mask_2d_model.astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_2d = mask_2d_model

        # Use original image (no bounding boxes) with mask overlay
        blended = original_img.copy()

        # Fill the instance region with semi-transparent green
        alpha = 0.7
        for c in range(3):
            blended[:, :, c] = np.where(
                mask_2d,
                (1 - alpha) * original_img[:, :, c] + alpha * highlight_color[c],
                original_img[:, :, c]
            )

        # Add edge outline
        mask_uint8 = mask_2d.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_uint8, kernel, iterations=2)
        edge_mask = (dilated > 0) & (~mask_2d)
        for c in range(3):
            blended[:, :, c] = np.where(edge_mask, highlight_color[c], blended[:, :, c])

        ax1.imshow(np.clip(blended, 0, 1))
        ax1.set_title(f"{class_name} (score: {confidence:.2f})", fontsize=12, fontweight='bold')
    elif annotated_img is not None:
        # Resize mask to match annotated image size if different
        img_h, img_w = annotated_img.shape[:2]
        mask_h, mask_w = mask_2d_model.shape

        if (img_h, img_w) != (mask_h, mask_w):
            mask_2d = cv2.resize(
                mask_2d_model.astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_2d = mask_2d_model

        # Fallback: use annotated but apply mask
        blended = annotated_img.copy()
        alpha = 0.7
        for c in range(3):
            blended[:, :, c] = np.where(
                mask_2d,
                (1 - alpha) * annotated_img[:, :, c] + alpha * highlight_color[c],
                annotated_img[:, :, c]
            )
        ax1.imshow(np.clip(blended, 0, 1))
        ax1.set_title(f"{class_name} (score: {confidence:.2f})", fontsize=12, fontweight='bold')
    else:
        # No image - show mask only
        mask_2d = mask_2d_model
        label_vis = np.zeros((*data['instance_labels'].shape, 3))
        label_vis[mask_2d] = [0, 1, 0]
        ax1.imshow(label_vis)
        ax1.set_title(f"{class_name} - Mask Only", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # ===================
    # Top-middle: Annotated image with bounding boxes (if available)
    # ===================
    ax1b = fig.add_subplot(gs[0, 1])

    if annotated_img is not None:
        ax1b.imshow(annotated_img)
        ax1b.set_title("With Tracking Boxes", fontsize=11, fontweight='bold')
    elif original_img is not None:
        ax1b.imshow(original_img)
        ax1b.set_title("Original Image", fontsize=11, fontweight='bold')
    else:
        ax1b.text(0.5, 0.5, 'No image available', ha='center', va='center', fontsize=12)
        ax1b.set_title("Image", fontsize=11)
    ax1b.axis('off')

    # ===================
    # Top-right: Top View (X-Z plane, looking down Y-axis)
    # ===================
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(instance_points[:, 0], instance_points[:, 2], c=colors, s=1, alpha=0.8)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Z', fontsize=10)
    ax2.set_title('Top View (X-Z)', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, linestyle=':')

    # ===================
    # Bottom-left: Side View (X-Y plane, looking along Z-axis)
    # ===================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(instance_points[:, 0], instance_points[:, 1], c=colors, s=1, alpha=0.8)
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    ax3.set_title('Side View (X-Y)', fontsize=11, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3, linestyle=':')

    # ===================
    # Bottom-middle: Front View (Z-Y plane, looking along X-axis)
    # ===================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(instance_points[:, 2], instance_points[:, 1], c=colors, s=1, alpha=0.8)
    ax4.set_xlabel('Z', fontsize=10)
    ax4.set_ylabel('Y', fontsize=10)
    ax4.set_title('Front View (Z-Y)', fontsize=11, fontweight='bold')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3, linestyle=':')

    # ===================
    # Bottom-right: Instance statistics
    # ===================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    # Instance statistics text
    stats_text = f"Instance Statistics\n"
    stats_text += f"{'─' * 25}\n"
    stats_text += f"Class: {class_name}\n"
    stats_text += f"Confidence: {confidence:.3f}\n"
    stats_text += f"Points: {len(instance_points):,}\n"
    center = instance_points.mean(axis=0)
    stats_text += f"Center:\n"
    stats_text += f"  X: {center[0]:.3f}\n"
    stats_text += f"  Y: {center[1]:.3f}\n"
    stats_text += f"  Z: {center[2]:.3f}\n"
    extents = instance_points.max(axis=0) - instance_points.min(axis=0)
    stats_text += f"Extents:\n"
    stats_text += f"  W: {extents[0]:.3f}\n"
    stats_text += f"  H: {extents[1]:.3f}\n"
    stats_text += f"  D: {extents[2]:.3f}"

    ax5.text(
        0.1, 0.95, stats_text,
        transform=ax5.transAxes,
        fontfamily='monospace',
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    )

    # Overall title
    fig.suptitle(
        f'Instance {instance_id}: {class_name} - Multi-View Visualization',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved multi-view visualization to: {output_path}")

    return fig


# ============================================================================
# VISUALIZATION 3: ALL INSTANCES IN A SINGLE FRAME
# ============================================================================

def create_all_instances_view(
    result_dir: str,
    frame_id: str,
    conf_threshold: float = 1.5,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive view showing all instances in a single frame.

    Layout:
    - Left column: Original image with all masks
    - Right column: 3D orthographic views (top, side, front)

    Args:
        result_dir: Path to results directory
        frame_id: Frame identifier
        conf_threshold: Minimum confidence threshold
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Load frame data
    data = load_frame_data(result_dir, frame_id)
    points = unproject_depth(data['depth'], data['intrinsics'], data['pose'])

    # Filter by confidence
    conf_flat = data['conf'].flatten()
    conf_mask = conf_flat > conf_threshold
    points_filt = points[conf_mask]
    labels_filt = data['instance_labels'].flatten()[conf_mask]

    # Get unique instances
    unique_instances = sorted(set(labels_filt) - {0})

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2, height_ratios=[1.2, 1, 1])

    # ===================
    # Top-left: Original image with all instance masks
    # ===================
    ax_img = fig.add_subplot(gs[0, 0])

    if 'annotated_image' in data:
        ax_img.imshow(data['annotated_image'])
    elif 'image' in data:
        img = data['image'].copy().astype(np.float32) / 255.0

        # Overlay each instance
        for inst_id in unique_instances:
            mask_2d = data['instance_labels'] == inst_id
            color = get_instance_color(inst_id)

            overlay = np.zeros_like(img)
            overlay[mask_2d] = color
            img[mask_2d] = 0.6 * img[mask_2d] + 0.4 * overlay[mask_2d]

        ax_img.imshow(img)
    else:
        ax_img.imshow(data['instance_labels'], cmap='tab10')

    ax_img.set_title(f'Frame {frame_id} - All Instances', fontsize=12, fontweight='bold')
    ax_img.axis('off')

    # Build legend
    legend_elements = []
    for inst_id in unique_instances:
        color = get_instance_color(inst_id)
        # Find bbox info
        info = None
        for bbox in data['bboxes']:
            if bbox.get('instance_id') == inst_id:
                info = bbox
                break

        if info:
            label = f"{info['class_name']} #{inst_id} ({info['confidence']:.2f})"
        else:
            label = f"Instance #{inst_id}"

        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=label))

    if legend_elements:
        ax_img.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

    # ===================
    # Top-right: Top view (X-Z plane)
    # ===================
    ax_top = fig.add_subplot(gs[0, 1])

    # Plot background
    bg_mask = labels_filt == 0
    if bg_mask.sum() > 0:
        bg_pts = points_filt[bg_mask]
        z_norm = (bg_pts[:, 2] - bg_pts[:, 2].min()) / (bg_pts[:, 2].max() - bg_pts[:, 2].min() + 1e-8)
        ax_top.scatter(bg_pts[:, 0], bg_pts[:, 2], c=z_norm, cmap='gray', s=0.2, alpha=0.4)

    # Plot instances
    for inst_id in unique_instances:
        inst_mask = labels_filt == inst_id
        inst_pts = points_filt[inst_mask]
        color = get_instance_color(inst_id)
        ax_top.scatter(inst_pts[:, 0], inst_pts[:, 2], c=[color], s=2, alpha=0.8)

    ax_top.set_xlabel('X', fontsize=10)
    ax_top.set_ylabel('Z', fontsize=10)
    ax_top.set_title('Top View (X-Z)', fontsize=11, fontweight='bold')
    ax_top.set_aspect('equal')
    ax_top.grid(True, alpha=0.3)

    # ===================
    # Middle-left: Side view (X-Y plane)
    # ===================
    ax_side = fig.add_subplot(gs[1, 0])

    if bg_mask.sum() > 0:
        ax_side.scatter(bg_pts[:, 0], bg_pts[:, 1], c='lightgray', s=0.2, alpha=0.4)

    for inst_id in unique_instances:
        inst_mask = labels_filt == inst_id
        inst_pts = points_filt[inst_mask]
        color = get_instance_color(inst_id)
        ax_side.scatter(inst_pts[:, 0], inst_pts[:, 1], c=[color], s=2, alpha=0.8)

    ax_side.set_xlabel('X', fontsize=10)
    ax_side.set_ylabel('Y', fontsize=10)
    ax_side.set_title('Side View (X-Y)', fontsize=11, fontweight='bold')
    ax_side.set_aspect('equal')
    ax_side.grid(True, alpha=0.3)

    # ===================
    # Middle-right: Front view (Z-Y plane)
    # ===================
    ax_front = fig.add_subplot(gs[1, 1])

    if bg_mask.sum() > 0:
        ax_front.scatter(bg_pts[:, 2], bg_pts[:, 1], c='lightgray', s=0.2, alpha=0.4)

    for inst_id in unique_instances:
        inst_mask = labels_filt == inst_id
        inst_pts = points_filt[inst_mask]
        color = get_instance_color(inst_id)
        ax_front.scatter(inst_pts[:, 2], inst_pts[:, 1], c=[color], s=2, alpha=0.8)

    ax_front.set_xlabel('Z', fontsize=10)
    ax_front.set_ylabel('Y', fontsize=10)
    ax_front.set_title('Front View (Z-Y)', fontsize=11, fontweight='bold')
    ax_front.set_aspect('equal')
    ax_front.grid(True, alpha=0.3)

    # ===================
    # Bottom row: Instance statistics
    # ===================
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')

    # Create statistics table
    stats_text = "Instance Statistics:\n"
    stats_text += "-" * 60 + "\n"
    stats_text += f"{'ID':<6} {'Class':<12} {'Confidence':<12} {'Points':<12} {'Center (X,Y,Z)':<30}\n"
    stats_text += "-" * 60 + "\n"

    for inst_id in unique_instances:
        inst_mask = labels_filt == inst_id
        inst_pts = points_filt[inst_mask]
        center = inst_pts.mean(axis=0)

        # Find bbox info
        info = None
        for bbox in data['bboxes']:
            if bbox.get('instance_id') == inst_id:
                info = bbox
                break

        class_name = info['class_name'] if info else 'unknown'
        confidence = info['confidence'] if info else 0.0

        stats_text += f"{inst_id:<6} {class_name:<12} {confidence:<12.3f} {len(inst_pts):<12} ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n"

    ax_stats.text(
        0.05, 0.95, stats_text,
        transform=ax_stats.transAxes,
        fontfamily='monospace',
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved all-instances view to: {output_path}")

    return fig


# ============================================================================
# BATCH VISUALIZATION FOR ALL INSTANCES IN A DATASET
# ============================================================================

def visualize_all_instances_in_dataset(
    result_dir: str,
    output_dir: str,
    conf_threshold: float = 1.5,
    max_frames: Optional[int] = None
):
    """
    Generate multi-view visualizations for all instances across all frames.

    Args:
        result_dir: Path to results directory
        output_dir: Directory to save visualizations
        conf_threshold: Minimum confidence threshold
        max_frames: Maximum number of frames to process (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all frames
    frames = get_available_frames(result_dir)
    if max_frames:
        frames = frames[:max_frames]

    print(f"Processing {len(frames)} frames from {result_dir}")

    # Track which instances we've visualized
    visualized_instances = set()

    for frame_id in frames:
        try:
            data = load_frame_data(result_dir, frame_id)

            # Get unique instances in this frame
            unique_instances = set(np.unique(data['instance_labels'])) - {0}

            for inst_id in unique_instances:
                if inst_id in visualized_instances:
                    continue

                # Find instance info
                inst_info = None
                for bbox in data['bboxes']:
                    if bbox.get('instance_id') == inst_id:
                        inst_info = bbox
                        break

                if inst_info:
                    class_name = inst_info['class_name']
                else:
                    class_name = 'unknown'

                # Generate multi-view for this instance
                output_path = os.path.join(
                    output_dir,
                    f"{class_name}_{inst_id}_views.png"
                )

                try:
                    fig = create_instance_multiview(
                        result_dir, frame_id, inst_id,
                        conf_threshold=conf_threshold,
                        output_path=output_path
                    )
                    plt.close(fig)
                    visualized_instances.add(inst_id)
                    print(f"  Created visualization for {class_name} #{inst_id}")
                except Exception as e:
                    print(f"  Warning: Failed to visualize instance {inst_id}: {e}")

        except Exception as e:
            print(f"  Warning: Failed to process frame {frame_id}: {e}")

    # Also create the full top-down view
    topdown_path = os.path.join(output_dir, "full_point_cloud_with_instances_top_colored.png")
    fig = create_topdown_view_with_instances(
        result_dir,
        frame_ids=frames,
        conf_threshold=conf_threshold,
        output_path=topdown_path
    )
    plt.close(fig)

    print(f"\nGenerated {len(visualized_instances)} instance visualizations")
    print(f"Output saved to: {output_dir}")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate enhanced visualizations for animal instance isolation and tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate top-down view with all instances highlighted
  python tools/instance_visualizer.py --result_dir results/wildlift/tmp-zebr-3-revisit-1 --mode topdown

  # Generate multi-view for a specific instance
  python tools/instance_visualizer.py --result_dir results/cami/tmp-rhin-35_1-revisit-1 --frame 1680 --instance 1 --mode multiview

  # Generate all visualizations for a frame
  python tools/instance_visualizer.py --result_dir results/cami/tmp-rhin-35_1-revisit-1 --frame 1680 --mode all

  # Batch generate for all instances in dataset
  python tools/instance_visualizer.py --result_dir results/wildlift/tmp-zebr-3-revisit-1 --mode batch --output_dir visualizations/zebra/
        """
    )

    parser.add_argument('--result_dir', type=str, required=True,
                        help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--frame', type=str, default=None,
                        help='Specific frame ID to visualize')
    parser.add_argument('--instance', type=int, default=None,
                        help='Specific instance ID to visualize')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['topdown', 'multiview', 'all', 'batch'],
                        help='Visualization mode')
    parser.add_argument('--conf_threshold', type=float, default=1.0,
                        help='Confidence threshold for filtering points (default: 1.0, auto-lowers if needed)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')

    args = parser.parse_args()

    # Ensure result directory exists
    if not os.path.exists(args.result_dir):
        print(f"Error: Result directory not found: {args.result_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get dataset name for output naming
    result_path = Path(args.result_dir)
    dataset_name = result_path.name

    if args.mode == 'batch':
        # Batch mode: generate all visualizations
        visualize_all_instances_in_dataset(
            args.result_dir,
            args.output_dir,
            conf_threshold=args.conf_threshold
        )

    elif args.mode == 'topdown':
        # Top-down view only
        output_path = os.path.join(args.output_dir, f"{dataset_name}_topdown.png")
        fig = create_topdown_view_with_instances(
            args.result_dir,
            frame_ids=[args.frame] if args.frame else None,
            conf_threshold=args.conf_threshold,
            output_path=output_path
        )
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    elif args.mode == 'multiview':
        # Multi-view for specific instance
        if args.frame is None:
            # Use first available frame
            frames = get_available_frames(args.result_dir)
            if not frames:
                print("Error: No frames found")
                sys.exit(1)
            args.frame = frames[0]

        if args.instance is None:
            # Get first instance in the frame
            data = load_frame_data(args.result_dir, args.frame)
            instances = sorted(set(np.unique(data['instance_labels'])) - {0})
            if not instances:
                print("Error: No instances found in frame")
                sys.exit(1)
            args.instance = instances[0]

        output_path = os.path.join(args.output_dir, f"{dataset_name}_{args.instance}_views.png")
        fig = create_instance_multiview(
            args.result_dir,
            args.frame,
            args.instance,
            conf_threshold=args.conf_threshold,
            output_path=output_path
        )
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    elif args.mode == 'all':
        # All visualizations for a frame
        if args.frame is None:
            frames = get_available_frames(args.result_dir)
            if not frames:
                print("Error: No frames found")
                sys.exit(1)
            args.frame = frames[0]

        # Create all-instances view
        output_path = os.path.join(args.output_dir, f"{dataset_name}_{args.frame}_all_instances.png")
        fig = create_all_instances_view(
            args.result_dir,
            args.frame,
            conf_threshold=args.conf_threshold,
            output_path=output_path
        )
        if args.show:
            plt.show()
        else:
            plt.close(fig)

        # Also create individual instance views
        data = load_frame_data(args.result_dir, args.frame)
        instances = sorted(set(np.unique(data['instance_labels'])) - {0})

        for inst_id in instances:
            try:
                # Get class name
                class_name = 'unknown'
                for bbox in data['bboxes']:
                    if bbox.get('instance_id') == inst_id:
                        class_name = bbox['class_name']
                        break

                output_path = os.path.join(
                    args.output_dir,
                    f"{class_name}_{inst_id}_views.png"
                )
                fig = create_instance_multiview(
                    args.result_dir,
                    args.frame,
                    inst_id,
                    conf_threshold=args.conf_threshold,
                    output_path=output_path
                )
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Failed to create view for instance {inst_id}: {e}")

    print(f"\nVisualization complete! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
