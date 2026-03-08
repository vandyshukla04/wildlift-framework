#!/usr/bin/env python3
"""
WildLIFT-RT: 3D Wildlife Reconstruction and Tracking Pipeline

Runs CUT3R 3D reconstruction on image sequences with instance segmentation
masks, performs multi-object tracking, and generates 3D bounding boxes.
"""

import os
import sys
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
import json
from copy import deepcopy
from pathlib import Path

# Add CUT3R backend to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CUT3R_ROOT = _REPO_ROOT / "backends" / "cut3r"
sys.path.insert(0, str(_CUT3R_ROOT))


def _resolve_model_path():
    """Find model checkpoint, searching common locations."""
    CKPT_NAME = "cut3r_512_dpt_4_64.pth"
    candidates = [
        _CUT3R_ROOT / "src" / CKPT_NAME,
        _REPO_ROOT / "checkpoints" / CKPT_NAME,
        Path.home() / ".cache" / "wildlift" / CKPT_NAME,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    default = _CUT3R_ROOT / "src" / CKPT_NAME
    print(f"\nWARNING: Model checkpoint not found in any of:")
    for p in candidates:
        print(f"  - {p}")
    print(f"\nDownload it with:")
    print(f"  cd backends/cut3r/src && gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view")
    print(f"Or pass --model_path /path/to/{CKPT_NAME}\n")
    return str(default)

from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

# Import extracted modules
from wildlift.rt.tracker import AnimalTracker
from wildlift.rt.dji_utils import parse_dji_logs, parse_dji_logs_with_gps, refine_poses_with_gps

# Set random seed for reproducibility.
random.seed(42)

# NOTE: parse_dji_logs, AnimalTracker imported from wildlift.rt.dji_utils and wildlift.rt.tracker
# The following inline definitions have been removed and replaced by imports above.

def compute_track_trajectories_and_colors(bounding_boxes):
    """
    Compute track trajectories and assign consistent colors to tracks
    
    Returns:
        - track_trajectories: dict[track_id] -> list of (frame_idx, center_3d)
        - track_colors: dict[track_id] -> RGB color tuple
        - track_info: dict[track_id] -> {class_name, first_frame, last_frame, etc.}
    """
    track_trajectories = {}
    track_info = {}
    
    # Collect all track data
    for frame_idx, frame_bboxes in enumerate(bounding_boxes):
        for bbox in frame_bboxes:
            if hasattr(bbox, 'track_id') and bbox.track_id is not None:
                track_id = bbox.track_id
                
                if track_id not in track_trajectories:
                    track_trajectories[track_id] = []
                    track_info[track_id] = {
                        'class_name': bbox.class_name,
                        'first_frame': frame_idx,
                        'last_frame': frame_idx,
                        'confidence_sum': bbox.confidence,
                        'detection_count': 1
                    }
                
                track_trajectories[track_id].append({
                    'frame': frame_idx,
                    'center': bbox.center.copy(),
                    'confidence': bbox.confidence
                })
                
                # Update track info
                track_info[track_id]['last_frame'] = frame_idx
                track_info[track_id]['confidence_sum'] += bbox.confidence
                track_info[track_id]['detection_count'] += 1
    
    # Generate consistent colors for tracks
    track_colors = {}
    color_palette = [
        (1.0, 0.2, 0.2),   # Red
        (0.2, 1.0, 0.2),   # Green  
        (0.2, 0.2, 1.0),   # Blue
        (1.0, 1.0, 0.2),   # Yellow
        (1.0, 0.2, 1.0),   # Magenta
        (0.2, 1.0, 1.0),   # Cyan
        (1.0, 0.6, 0.2),   # Orange
        (0.6, 0.2, 1.0),   # Purple
        (0.2, 0.8, 0.2),   # Forest Green
        (0.8, 0.2, 0.6),   # Pink
    ]
    
    for i, track_id in enumerate(sorted(track_trajectories.keys())):
        track_colors[track_id] = color_palette[i % len(color_palette)]
    
    return track_trajectories, track_colors, track_info

def add_track_visualization_to_viewer(viewer, bounding_boxes, track_trajectories, track_colors, track_info):
    """
    Simplified track visualization that works with any viser version
    """
    print(f"\n🎨 Adding track visualization to 3D viewer...")
    
    # 1. Print detailed track information
    print(f"\n📊 === DETAILED TRACK ANALYSIS ===")
    for track_id, trajectory in track_trajectories.items():
        info = track_info[track_id]
        color = track_colors[track_id]
        
        # Sort trajectory by frame
        trajectory_sorted = sorted(trajectory, key=lambda x: x['frame'])
        centers = [point['center'] for point in trajectory_sorted]
        frames = [point['frame'] for point in trajectory_sorted]
        
        print(f"\n🔴 Track {track_id} ({info['class_name']}):")
        print(f"   Color: RGB{color}")
        print(f"   Duration: frames {frames[0]} → {frames[-1]} ({len(frames)} detections)")
        print(f"   Avg confidence: {info['confidence_sum'] / info['detection_count']:.3f}")
        
        # Print trajectory summary
        if len(centers) >= 2:
            start_pos = centers[0]
            end_pos = centers[-1]
            total_distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            print(f"   Start position: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
            print(f"   End position:   [{end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}]")
            print(f"   Total movement: {total_distance:.2f} units")
            
            # Compute movement per frame
            if len(centers) > 1:
                movements = []
                for i in range(1, len(centers)):
                    movement = np.linalg.norm(np.array(centers[i]) - np.array(centers[i-1]))
                    movements.append(movement)
                avg_movement = np.mean(movements)
                max_movement = max(movements)
                print(f"   Avg movement/frame: {avg_movement:.3f}")
                print(f"   Max movement/frame: {max_movement:.3f}")
    
    # 2. Try simple 3D visualization (that might work with your viser)
    try:
        # Simple approach: just add text labels at track positions
        for frame_idx, frame_bboxes in enumerate(bounding_boxes):
            for bbox in frame_bboxes:
                if hasattr(bbox, 'track_id') and bbox.track_id is not None:
                    track_id = bbox.track_id
                    color = track_colors.get(track_id, (1.0, 1.0, 1.0))
                    
                    # Try to add a simple text label (most basic viser feature)
                    try:
                        label_position = bbox.center + np.array([0, 0, 0.5])  # Slightly above center
                        viewer.server.scene.add_text(
                            name=f"track_label_{frame_idx}_{track_id}",
                            text=f"T{track_id}",
                            position=label_position,
                            color=color
                        )
                    except:
                        # If that fails, try even simpler
                        try:
                            viewer.server.scene.add_point_cloud(
                                name=f"track_center_{frame_idx}_{track_id}",
                                points=bbox.center.reshape(1, 3),
                                colors=(np.array(color) * 255).astype(np.uint8).reshape(1, 3),
                                point_size = 0.02
                            )
                        except:
                            pass  # Give up on 3D visualization
        
        print("✅ Added simple 3D track markers")
        
    except Exception as e:
        print(f"⚠️ 3D track visualization not supported by your viser version")
        print(f"   (This is normal - check the 2D projections instead)")
    
    # 3. Create a trajectory analysis file
    try:
        import json
        trajectory_analysis = {
            'tracks': {},
            'summary': {
                'total_tracks': len(track_trajectories),
                'total_frames': max(info['last_frame'] for info in track_info.values()) + 1,
                'avg_track_length': np.mean([len(traj) for traj in track_trajectories.values()])
            }
        }
        
        for track_id, trajectory in track_trajectories.items():
            info = track_info[track_id]
            trajectory_sorted = sorted(trajectory, key=lambda x: x['frame'])
            
            trajectory_analysis['tracks'][track_id] = {
                'class_name': info['class_name'],
                'color_rgb': track_colors[track_id],
                'frames': [point['frame'] for point in trajectory_sorted],
                'centers': [point['center'].tolist() for point in trajectory_sorted],
                'confidences': [point['confidence'] for point in trajectory_sorted]
            }
        
        # Save trajectory analysis
        # with open('tmp/trajectory_analysis.json', 'w') as f:
        #     json.dump(trajectory_analysis, f, indent=2)
        
        # print(f"💾 Saved detailed trajectory analysis to: tmp/trajectory_analysis.json")
        
    except Exception as e:
        print(f"⚠️ Could not save trajectory analysis: {e}")

def create_trajectory_plot(bounding_boxes, track_trajectories, track_colors, output_dir):
    """
    Create a matplotlib plot of track trajectories (bird's eye view)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        print(f"\n📈 Creating trajectory plots...")
        
        # Create 2D top-down view
        plt.figure(figsize=(12, 8))
        
        for track_id, trajectory in track_trajectories.items():
            trajectory_sorted = sorted(trajectory, key=lambda x: x['frame'])
            centers = np.array([point['center'] for point in trajectory_sorted])
            color = track_colors[track_id]
            
            # Plot trajectory
            plt.plot(centers[:, 0], centers[:, 2], 'o-', color=color, 
                    label=f'Track {track_id}', linewidth=2, markersize=4)
            
            # Add start and end markers
            plt.plot(centers[0, 0], centers[0, 2], 's', color=color, markersize=8, label=f'Start T{track_id}')
            plt.plot(centers[-1, 0], centers[-1, 2], '^', color=color, markersize=8, label=f'End T{track_id}')
        
        plt.xlabel('X Position')
        plt.ylabel('Z Position') 
        plt.title('Animal Tracking - Top Down View')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plot_path = os.path.join(output_dir, 'track_trajectories_2d.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for track_id, trajectory in track_trajectories.items():
            trajectory_sorted = sorted(trajectory, key=lambda x: x['frame'])
            centers = np.array([point['center'] for point in trajectory_sorted])
            color = track_colors[track_id]
            
            # Plot 3D trajectory
            ax.plot(centers[:, 0], centers[:, 2], centers[:, 1], 'o-', color=color,
                   label=f'Track {track_id}', linewidth=2, markersize=4)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_zlabel('Y Position (Height)')
        ax.set_title('Animal Tracking - 3D View')
        ax.legend()
        
        plot_path_3d = os.path.join(output_dir, 'track_trajectories_3d.png')
        plt.savefig(plot_path_3d, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved trajectory plots:")
        print(f"   2D: {plot_path}")
        print(f"   3D: {plot_path_3d}")
        
    except ImportError:
        print("⚠️ matplotlib not available for trajectory plots")
    except Exception as e:
        print(f"⚠️ Could not create trajectory plots: {e}")

def enhanced_track_analysis(bounding_boxes, track_trajectories, track_colors, track_info, output_dir):
    """
    Complete track analysis with plots and detailed output
    """
    # Create trajectory plots
    create_trajectory_plot(bounding_boxes, track_trajectories, track_colors, output_dir)
    
    # Print movement analysis
    print(f"\n🏃 === MOVEMENT ANALYSIS ===")
    for track_id, trajectory in track_trajectories.items():
        trajectory_sorted = sorted(trajectory, key=lambda x: x['frame'])
        
        if len(trajectory_sorted) < 2:
            continue
            
        centers = np.array([point['center'] for point in trajectory_sorted])
        frames = [point['frame'] for point in trajectory_sorted]
        
        # Calculate speeds (movement per frame)
        speeds = []
        for i in range(1, len(centers)):
            distance = np.linalg.norm(centers[i] - centers[i-1])
            frame_gap = frames[i] - frames[i-1]
            speed = distance / frame_gap if frame_gap > 0 else 0
            speeds.append(speed)
        
        if speeds:
            avg_speed = np.mean(speeds)
            max_speed = max(speeds)
            print(f"Track {track_id}: avg speed = {avg_speed:.3f}, max speed = {max_speed:.3f} units/frame")

def add_2d_projection_with_tracks(original_images, bounding_boxes, cam_dict, output_dir, 
                                track_colors=None, track_info=None, frame_names=None):
    """
    Enhanced 2D projection that includes track IDs and consistent track colors
    This replaces your existing add_2d_projection_to_demo function
    """
    print(f"\n📸 Creating 2D projections with track visualization...")
    
    if track_colors is None:
        _, track_colors, track_info = compute_track_trajectories_and_colors(bounding_boxes)
    
    annotated_images = []
    
    def draw_3d_bbox_wireframe(img, corners_2d, color_bgr, thickness=2):
        """Draw 3D bounding box wireframe using OpenCV"""
        try:
            corners_2d = corners_2d.astype(int)
            
            # Define the 12 edges of a 3D bounding box
            # Bottom face (indices 0,1,2,3)
            cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[1]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[1]), tuple(corners_2d[2]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[2]), tuple(corners_2d[3]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[3]), tuple(corners_2d[0]), color_bgr, thickness)
            
            # Top face (indices 4,5,6,7)
            cv2.line(img, tuple(corners_2d[4]), tuple(corners_2d[5]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[5]), tuple(corners_2d[6]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[6]), tuple(corners_2d[7]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[7]), tuple(corners_2d[4]), color_bgr, thickness)
            
            # Vertical edges connecting bottom and top
            cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[4]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[1]), tuple(corners_2d[5]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[2]), tuple(corners_2d[6]), color_bgr, thickness)
            cv2.line(img, tuple(corners_2d[3]), tuple(corners_2d[7]), color_bgr, thickness)
            
        except Exception as e:
            print(f"      ⚠️ Failed to draw wireframe: {e}")
    
    for frame_idx, (original_img, frame_bboxes) in enumerate(zip(original_images, bounding_boxes)):
        if len(frame_bboxes) == 0:
            annotated_images.append(original_img)
            continue
        
        # Convert image format if needed
        # if torch.is_tensor(original_img):
        #     if original_img.dim() == 4:  # [B, C, H, W]
        #         img_np = original_img[0].permute(1, 2, 0).cpu().numpy()
        #     else:  # [C, H, W]
        #         img_np = original_img.permute(1, 2, 0).cpu().numpy()
        # else:
        #     img_np = original_img
        
        # # Ensure image is in [0, 1] range
        # if img_np.max() <= 1.0:
        #     img_display = (img_np * 255).astype(np.uint8)
        # else:
        #     img_display = img_np.astype(np.uint8)
        
        # Convert image format if needed
        if torch.is_tensor(original_img):
            if original_img.dim() == 4:  # [B, C, H, W]
                img_np = original_img[0].permute(1, 2, 0).cpu().numpy()
            else:  # [C, H, W]
                img_np = original_img.permute(1, 2, 0).cpu().numpy()
            
            # Convert from RGB to BGR for OpenCV
            if img_np.max() <= 1.0:
                img_display = (img_np * 255).astype(np.uint8)
            else:
                img_display = img_np.astype(np.uint8)
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        else:
            # Already a numpy array (from cv2.imread, which is BGR)
            img_np = original_img
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_display = (img_np * 255).astype(np.uint8)
                else:
                    img_display = img_np.astype(np.uint8)
            else:
                img_display = img_np.copy()  # Already uint8 BGR from cv2.imread

        # Get camera parameters for this frame
        focal = cam_dict["focal"][frame_idx]
        pp = cam_dict["pp"][frame_idx]
        R = cam_dict["R"][frame_idx]
        t = cam_dict["t"][frame_idx]

        # Create camera intrinsic matrix (no scaling needed - images resized to model size)
        K = np.array([
            [focal, 0, pp[0]],
            [0, focal, pp[1]],
            [0, 0, 1]
        ])


        # Create camera pose matrix
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = t
        
        # Project each bounding box
        annotated_img = img_display.copy()
        
        for bbox in frame_bboxes:
            # Get track-specific color
            if hasattr(bbox, 'track_id') and bbox.track_id is not None:
                track_id = bbox.track_id
                color_rgb = track_colors.get(track_id, (1.0, 1.0, 1.0))
                color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])  # Convert to BGR for OpenCV
                track_label = f"T{track_id}"
            else:
                color_bgr = (255, 255, 255)  # White for untracked
                track_label = "?"
            
            # Project 3D bounding box to 2D
            corners_3d = bbox.get_corners()
            
            try:
                # Transform to camera coordinates
                corners_cam = (np.linalg.inv(camera_pose) @ 
                             np.concatenate([corners_3d, np.ones((8, 1))], axis=1).T)[:3].T
                
                # Check if points are in front of camera
                if np.any(corners_cam[:, 2] <= 0):
                    print(f"      ⚠️ Track {track_label}: Some points behind camera")
                    continue
                
                # Project to image coordinates
                corners_2d_hom = (K @ corners_cam.T).T
                corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
                
                # Check if projected points are reasonable
                img_h, img_w = img_display.shape[:2]
                if (np.any(corners_2d < -img_w) or np.any(corners_2d > 2*img_w) or 
                    np.any(corners_2d[:, 1] < -img_h) or np.any(corners_2d[:, 1] > 2*img_h)):
                    print(f"      ⚠️ Track {track_label}: Projected points outside reasonable bounds")
                    continue
                
                # Draw 3D bounding box wireframe
                draw_3d_bbox_wireframe(annotated_img, corners_2d, color_bgr, thickness=2)
                
                # Add track label
                center_2d = np.mean(corners_2d, axis=0).astype(int)
                label_text = f"{track_label}: {bbox.class_name}"
                
                # Make sure label position is on screen
                center_2d[0] = max(10, min(center_2d[0], img_w - 100))
                center_2d[1] = max(20, min(center_2d[1], img_h - 10))
                
                print(f"      ✓ Track {track_label}: Projected successfully")
                
            except Exception as e:
                print(f"      ❌ Track {track_label}: Projection failed - {e}")
                continue
        
        annotated_images.append(annotated_img)
        
        # Save annotated image with original frame name
        if frame_names and frame_idx < len(frame_names):
            frame_name = frame_names[frame_idx]
            output_path = os.path.join(output_dir, "annotated_2d", f"{frame_name}_tracked.png")
        else:
            output_path = os.path.join(output_dir, "annotated_2d", f"frame_{frame_idx:06d}_tracked.png")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_img)
    
    print(f"✅ Saved {len(annotated_images)} tracked 2D projections")
    return annotated_images

class BoundingBox3D:
    """3D Bounding Box representation"""
    
    def __init__(self, center, dimensions, rotation_matrix, class_name, confidence, instance_id):
        self.center = center  # [x, y, z]
        self.dimensions = dimensions  # [length, width, height]
        self.rotation_matrix = rotation_matrix  # 3x3 rotation matrix
        self.class_name = class_name
        self.confidence = confidence
        self.instance_id = instance_id
        self.track_id = None
        self.persistent_instance_id = None
        self.mask = None  # Will store 2D mask for tracking
    
    def get_corners(self):
        """Get 8 corner points of the bounding box"""
        # Define box corners in local coordinate system
        l, w, h = self.dimensions
        corners_local = np.array([
            [-l/2, -w/2, -h/2],
            [+l/2, -w/2, -h/2],
            [+l/2, +w/2, -h/2],
            [-l/2, +w/2, -h/2],
            [-l/2, -w/2, +h/2],
            [+l/2, -w/2, +h/2],
            [+l/2, +w/2, +h/2],
            [-l/2, +w/2, +h/2],
        ])
        
        # Transform to world coordinates
        corners_world = (self.rotation_matrix @ corners_local.T).T + self.center
        return corners_world

    def get_wireframe_edges(self):
        """Get edges for wireframe visualization"""
        corners = self.get_corners()
        edges = [
            # Bottom face
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face  
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        return corners, edges
    
class Enhanced3DBBoxBackProjector:
    """Enhanced mask backprojection with 3D bounding box computation"""

    def __init__(self):
        self.class_colors = {
            'zebra': np.array([1.0, 0.6, 0.2]),    # Bright orange
            'ground': np.array([0.2, 1.0, 0.2]),   # Bright green
            'sky': np.array([0.3, 0.7, 1.0]),      # Bright blue
            'person': np.array([1.0, 0.2, 0.6]),   # Bright pink
            'car': np.array([0.8, 0.2, 1.0]),      # Purple
            'building': np.array([1.0, 1.0, 0.2]), # Yellow
            'tree': np.array([0.0, 0.8, 0.4]),     # Forest green
            'rhino': np.array([0.9, 0.5, 0.1]),    # Orange-brown
            'rhinoceros': np.array([0.9, 0.5, 0.1]), # Same as rhino
            'elephant': np.array([0.6, 0.6, 0.6]), # Gray
        }

        # Classes to exclude from bounding boxes
        self.excluded_classes = {'ground', 'sky', 'background'}

        # PHASE 1A: Species-specific dimensions (in meters)
        # Format: {'length': L, 'width': W, 'height': H}
        # Length = nose to tail, Width = side to side, Height = ground to top
        self.animal_dimensions = {
            'zebra': {
                'length': 2.5,   # ~2.3-2.7m body length
                'width': 0.7,    # ~0.6-0.8m width at shoulders
                'height': 1.4,   # ~1.3-1.5m shoulder height
            },
            'rhino': {
                'length': 3.8,   # ~3.5-4.0m body length (white rhino)
                'width': 1.5,    # ~1.4-1.8m width
                'height': 1.8,   # ~1.6-2.0m shoulder height
            },
            'rhinoceros': {  # Alias for rhino
                'length': 3.8,
                'width': 1.5,
                'height': 1.8,
            },
            'elephant': {
                'length': 5.5,   # ~5.0-6.5m body length (African elephant)
                'width': 2.5,    # ~2.0-3.0m width
                'height': 3.2,   # ~3.0-4.0m shoulder height
            },
            # Generic quadruped fallback (medium-sized animal)
            'animal': {
                'length': 2.0,
                'width': 0.6,
                'height': 1.2,
            }
        }

        # PHASE 1A: Enable/disable species-constrained fitting
        # DISABLED: Scale issue - CUT3R units don't match expected metric scale
        self.use_species_constraints = False  # Set to True once scale recovery is implemented
        self.species_constraint_strength = 0.7  # How much to trust species dimensions vs PCA (0-1)

        # PHASE 1B: Temporal point accumulation
        self.use_temporal_accumulation = False  # Set to False to disable
        self.temporal_window_size = 2  # Number of frames before/after (±N frames)
        self.drift_threshold = 0.5  # Pose drift threshold for adaptive window

        # PHASE 1C: Temporal smoothing
        self.use_temporal_smoothing = False  # Set to False to disable
        self.smoothing_alpha = 0.3  # EMA factor (lower = more smoothing)
        self.bbox_history = {}  # track_id -> previous bbox for smoothing
    
    def compute_oriented_bbox_pca(self, points_3d):
        """Compute oriented bounding box using PCA"""
        if len(points_3d) < 3:
            return None
            
        # Center the points
        centroid = np.mean(points_3d, axis=0)
        centered_points = points_3d - centroid
        
        # Apply PCA to find principal axes
        pca = PCA(n_components=3)
        pca.fit(centered_points)
        
        # Transform points to PCA space
        transformed_points = pca.transform(centered_points)
        
        # Find min/max in each PCA axis
        min_vals = np.min(transformed_points, axis=0)
        max_vals = np.max(transformed_points, axis=0)
        dimensions = max_vals - min_vals
        
        # The center in PCA space
        center_pca = (min_vals + max_vals) / 2
        
        # Transform center back to world space
        center_world = pca.inverse_transform(center_pca.reshape(1, -1))[0] + centroid
        
        # Rotation matrix (PCA components)
        rotation_matrix = pca.components_.T

        return center_world, dimensions, rotation_matrix

    def get_species_dimensions(self, class_name):
        """
        PHASE 1A: Get species-specific dimensions with fallback to generic

        Args:
            class_name: Name of the animal class (e.g., 'zebra', 'rhino')

        Returns:
            dict: {'length', 'width', 'height'} in meters
        """
        class_name_lower = class_name.lower()

        if class_name_lower in self.animal_dimensions:
            return self.animal_dimensions[class_name_lower]
        else:
            # Fallback to generic quadruped
            print(f"⚠️  Unknown species '{class_name}', using generic quadruped dimensions")
            return self.animal_dimensions['animal']

    def compute_species_constrained_bbox(self, points_3d, class_name, ground_normal=None):
        """
        PHASE 1A: Compute bbox with species-specific dimension constraints

        This method:
        1. Uses PCA to find orientation and approximate scale
        2. Constrains dimensions to match known animal proportions
        3. Blends PCA dimensions with species dimensions based on constraint_strength

        Args:
            points_3d: (N, 3) array of 3D points
            class_name: Animal class name for dimension lookup
            ground_normal: Optional (3,) ground plane normal for height constraint

        Returns:
            (center, dimensions, rotation_matrix) or None if failed
        """
        # Step 1: Standard PCA bbox
        pca_result = self.compute_oriented_bbox_pca(points_3d)
        if pca_result is None:
            return None

        center_pca, dimensions_pca, rotation_pca = pca_result

        # If species constraints disabled, return PCA result
        if not self.use_species_constraints:
            return center_pca, dimensions_pca, rotation_pca

        # Step 2: Get species dimensions
        species_dims = self.get_species_dimensions(class_name)
        target_dims = np.array([
            species_dims['length'],
            species_dims['width'],
            species_dims['height']
        ])

        # Step 3: Estimate scale from PCA dimensions
        # Use ratio of PCA dimensions to species dimensions
        # Filter out very small PCA dimensions to avoid division issues
        scale_factors = []
        for i, (pca_dim, target_dim) in enumerate(zip(dimensions_pca, target_dims)):
            if pca_dim > 0.1:  # Only use dimensions > 10cm
                scale_factors.append(pca_dim / target_dim)

        if len(scale_factors) == 0:
            # All dimensions too small, use species dimensions as-is
            scale = 1.0
        else:
            # Take median scale factor (robust to outliers from partial views)
            scale = np.median(scale_factors)

        # Step 4: Compute constrained dimensions
        # Blend between PCA dimensions and scaled species dimensions
        species_dims_scaled = target_dims * scale
        alpha = self.species_constraint_strength

        dimensions_constrained = (
            alpha * species_dims_scaled +
            (1 - alpha) * dimensions_pca
        )

        # Step 5: Adjust center if ground plane is available
        center_constrained = center_pca.copy()

        if ground_normal is not None:
            # Project points onto ground plane to find "floor"
            # Ground normal points up, so we dot product with vertical
            ground_normal_unit = ground_normal / np.linalg.norm(ground_normal)

            # Find minimum height (closest to ground)
            heights = np.dot(points_3d - center_pca, ground_normal_unit)
            min_height = np.min(heights)

            # Adjust center so bbox bottom is at minimum height
            # This assumes ground_normal points upward
            center_offset = min_height + dimensions_constrained[2] / 2
            center_constrained = center_pca + ground_normal_unit * center_offset

        print(f"    📏 Species: {class_name} | Scale: {scale:.2f}x | "
              f"PCA dims: [{dimensions_pca[0]:.2f}, {dimensions_pca[1]:.2f}, {dimensions_pca[2]:.2f}] → "
              f"Constrained: [{dimensions_constrained[0]:.2f}, {dimensions_constrained[1]:.2f}, {dimensions_constrained[2]:.2f}]")

        return center_constrained, dimensions_constrained, rotation_pca

    def estimate_pose_drift(self, camera_poses, frame_idx, window=3):
        """
        PHASE 1B: Estimate camera pose drift around current frame
        Returns drift score (higher = more drift)
        """
        if camera_poses is None or len(camera_poses) < 2:
            return float('inf')  # No poses = assume high drift

        start = max(0, frame_idx - window)
        end = min(len(camera_poses), frame_idx + window + 1)

        if end - start < 2:
            return 0.0  # Not enough frames

        drifts = []
        for i in range(start + 1, end):
            pose_prev = camera_poses[i-1]
            pose_curr = camera_poses[i]

            if torch.is_tensor(pose_prev):
                pose_prev = pose_prev.cpu().numpy()
            if torch.is_tensor(pose_curr):
                pose_curr = pose_curr.cpu().numpy()

            # Translation drift
            t_diff = np.linalg.norm(pose_curr[:3, 3] - pose_prev[:3, 3])

            # Rotation drift (Frobenius norm)
            R_diff = np.linalg.norm(pose_curr[:3, :3] - pose_prev[:3, :3])

            drift = t_diff + R_diff * 0.1  # Weight rotation less
            drifts.append(drift)

        return np.mean(drifts) if drifts else 0.0

    def accumulate_temporal_points(self, frame_idx, all_pts3d, all_instance_labels,
                                     all_masks_data, instance_id, class_name,
                                     camera_poses, use_pose_transform):
        """
        PHASE 1B: Accumulate points from nearby frames for more stable bbox fitting

        Args:
            frame_idx: Current frame index
            all_pts3d: List of point clouds for all frames
            all_instance_labels: List of instance labels for all frames
            all_masks_data: Dict of mask data for all frames
            instance_id: Instance ID to accumulate
            class_name: Class name for matching
            camera_poses: List of camera poses (for transformation)
            use_pose_transform: Whether to use pose transformation

        Returns:
            Accumulated points in current frame's coordinate system
        """
        if not self.use_temporal_accumulation:
            # Just return current frame points (transformed to world coords if needed)
            instance_mask = all_instance_labels[frame_idx] == instance_id
            pts3d = all_pts3d[frame_idx]
            if torch.is_tensor(pts3d):
                pts3d = pts3d.cpu().numpy()
            instance_points = pts3d.reshape(-1, 3)[instance_mask.reshape(-1)]

            # Transform to world coordinates (matching OG behavior)
            if use_pose_transform and camera_poses is not None and frame_idx < len(camera_poses):
                pose = camera_poses[frame_idx]
                if torch.is_tensor(pose):
                    pose = pose.cpu().numpy()
                if not np.allclose(pose, np.eye(4)):
                    # Add homogeneous coordinate and transform
                    instance_points_h = np.concatenate([
                        instance_points,
                        np.ones((len(instance_points), 1))
                    ], axis=1)
                    instance_points = (pose @ instance_points_h.T).T[:, :3]

            return instance_points

        # Estimate drift to determine window size
        drift = self.estimate_pose_drift(camera_poses, frame_idx)

        # Adaptive window based on drift
        if drift > self.drift_threshold:
            window = 1  # High drift: only use ±1 frame
        else:
            window = self.temporal_window_size  # Low drift: use full window

        accumulated_points = []
        current_pose = camera_poses[frame_idx] if camera_poses else np.eye(4)
        if torch.is_tensor(current_pose):
            current_pose = current_pose.cpu().numpy()

        # Collect from nearby frames
        start_frame = max(0, frame_idx - window)
        end_frame = min(len(all_pts3d), frame_idx + window + 1)

        for f_idx in range(start_frame, end_frame):
            # Skip if no mask data for this frame
            if f_idx not in all_masks_data:
                continue

            # Find matching instance in this frame (same class)
            instance_labels_f = all_instance_labels[f_idx]
            if torch.is_tensor(instance_labels_f):
                instance_labels_f = instance_labels_f.cpu().numpy()

            # Look for instances of same class
            unique_instances = np.unique(instance_labels_f)
            unique_instances = unique_instances[unique_instances > 0]

            for inst_id in unique_instances:
                if inst_id - 1 >= len(all_masks_data[f_idx]):
                    continue

                mask_info = all_masks_data[f_idx][inst_id - 1]
                if mask_info['class_name'].lower() != class_name:
                    continue

                # Extract points for this instance
                inst_mask = instance_labels_f == inst_id
                pts3d_f = all_pts3d[f_idx]
                if torch.is_tensor(pts3d_f):
                    pts3d_f = pts3d_f.cpu().numpy()

                # Ensure pts3d_f is 2D array (might be 3D or 4D depending on input)
                if pts3d_f.ndim > 2:
                    pts3d_f = pts3d_f.reshape(-1, 3)

                points_f = pts3d_f[inst_mask.reshape(-1)]

                if len(points_f) < 5:
                    continue

                # Transform to current frame if needed
                if use_pose_transform and camera_poses is not None:
                    pose_f = camera_poses[f_idx]
                    if torch.is_tensor(pose_f):
                        pose_f = pose_f.cpu().numpy()

                    # Transform from frame f to current frame
                    # points_f are in world coords, transform to current camera
                    if not np.allclose(pose_f, np.eye(4)) or not np.allclose(current_pose, np.eye(4)):
                        # points_f is already in world coords
                        # Transform to current frame: inv(current_pose) @ points_world
                        points_h = np.concatenate([points_f, np.ones((len(points_f), 1))], axis=1)
                        points_current = (np.linalg.inv(current_pose) @ points_h.T).T[:, :3]
                        accumulated_points.append(points_current)
                    else:
                        accumulated_points.append(points_f)
                else:
                    accumulated_points.append(points_f)

                # Only take first matching instance per frame
                break

        if not accumulated_points:
            # Fallback to current frame only
            instance_mask = all_instance_labels[frame_idx] == instance_id
            pts3d = all_pts3d[frame_idx]
            if torch.is_tensor(pts3d):
                pts3d = pts3d.cpu().numpy()
            return pts3d.reshape(-1, 3)[instance_mask.reshape(-1)]

        return np.concatenate(accumulated_points, axis=0)

    def apply_temporal_smoothing(self, bbox, track_id):
        """
        PHASE 1C: Apply exponential moving average smoothing to bbox parameters

        Args:
            bbox: Current BoundingBox3D object
            track_id: Track ID for history lookup

        Returns:
            Smoothed BoundingBox3D object
        """
        if not self.use_temporal_smoothing or track_id is None or track_id not in self.bbox_history:
            # First occurrence or smoothing disabled
            if track_id is not None:
                self.bbox_history[track_id] = bbox
            return bbox

        prev_bbox = self.bbox_history[track_id]
        alpha = self.smoothing_alpha

        # Smooth center
        smoothed_center = alpha * bbox.center + (1 - alpha) * prev_bbox.center

        # Smooth dimensions
        smoothed_dims = alpha * bbox.dimensions + (1 - alpha) * prev_bbox.dimensions

        # Smooth rotation using simple linear interpolation (good enough for small changes)
        smoothed_rotation = alpha * bbox.rotation_matrix + (1 - alpha) * prev_bbox.rotation_matrix

        # Re-orthogonalize rotation matrix (ensure it's still a valid rotation)
        U, _, Vt = np.linalg.svd(smoothed_rotation)
        smoothed_rotation = U @ Vt

        # Create smoothed bbox
        smoothed_bbox = BoundingBox3D(
            center=smoothed_center,
            dimensions=smoothed_dims,
            rotation_matrix=smoothed_rotation,
            class_name=bbox.class_name,
            confidence=bbox.confidence,
            instance_id=bbox.instance_id
        )
        smoothed_bbox.track_id = track_id
        smoothed_bbox.mask = bbox.mask

        # Update history
        self.bbox_history[track_id] = smoothed_bbox

        return smoothed_bbox

    def filter_outliers(self, points_3d, outlier_factor=1.5, use_iqr=True):
        """Remove outlier points using statistical filtering.

        Args:
            points_3d: (N, 3) array of 3D points
            outlier_factor: For std-based filtering, keep points within mean + outlier_factor * std
                           For IQR-based filtering, keep points within Q1 - factor*IQR to Q3 + factor*IQR
            use_iqr: If True, use IQR-based filtering (more robust to extreme outliers)
        """
        if len(points_3d) < 10:
            return points_3d

        # Ensure points_3d is the right shape
        points_3d = np.array(points_3d)
        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            print(f"🔍 DEBUG: Unexpected points_3d shape: {points_3d.shape}")
            if points_3d.size % 3 == 0:
                points_3d = points_3d.reshape(-1, 3)
            else:
                print(f"❌ Cannot reshape points_3d to (N, 3): size={points_3d.size}")
                return points_3d

        # Compute distances from centroid
        centroid = np.mean(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - centroid, axis=1)

        if use_iqr:
            # IQR-based filtering (more robust to extreme outliers)
            q1 = np.percentile(distances, 25)
            q3 = np.percentile(distances, 75)
            iqr = q3 - q1
            lower_bound = max(0, q1 - outlier_factor * iqr)
            upper_bound = q3 + outlier_factor * iqr
            inliers = (distances >= lower_bound) & (distances <= upper_bound)
        else:
            # Standard deviation-based filtering
            threshold = np.mean(distances) + outlier_factor * np.std(distances)
            inliers = distances < threshold

        # Ensure inliers is 1D boolean array
        inliers = np.array(inliers).flatten().astype(bool)

        # Debug output
        n_original = points_3d.shape[0]
        n_kept = np.sum(inliers)
        pct_kept = 100 * n_kept / n_original
        print(f"🔍 Filtering outliers: {n_original} -> {n_kept} points ({pct_kept:.1f}% kept)")

        # Apply filter
        try:
            filtered_points = points_3d[inliers]
            return filtered_points
        except Exception as e:
            print(f"❌ Error in outlier filtering: {e}")
            print(f"   Falling back to original points")
            return points_3d

    def get_ground_plane_from_gimbal(self, gimbal_data):
        """Extract ground plane from gimbal data"""
        if not gimbal_data:
            return None
            
        # Average gimbal angles across all frames for stability
        pitches = [data['pitch'] for data in gimbal_data.values()]
        rolls = [data['roll'] for data in gimbal_data.values()]
        altitudes = [data['altitude'] for data in gimbal_data.values()]
        
        avg_pitch = np.mean(pitches)
        avg_roll = np.mean(rolls)
        avg_altitude = np.mean(altitudes)
        
        print(f"🚁 Gimbal summary: pitch={avg_pitch:.1f}°, roll={avg_roll:.1f}°, alt={avg_altitude:.1f}m")
        
        # Convert to radians
        pitch_rad = np.radians(avg_pitch)
        roll_rad = np.radians(avg_roll)
        
        # Compute ground normal from gimbal orientation
        # For DJI: pitch negative means camera pointing down
        # Ground normal points UP from the ground plane
        ground_normal = np.array([
            np.sin(roll_rad),                          # X component from roll
            np.cos(pitch_rad) * np.cos(roll_rad),      # Y component (up direction)
            -np.sin(pitch_rad) * np.cos(roll_rad)      # Z component from pitch
        ])
        
        # Normalize to unit vector
        ground_normal = ground_normal / np.linalg.norm(ground_normal)
        
        print(f"📐 Computed ground normal: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")
        
        return ground_normal, avg_altitude

    def align_bbox_to_ground_plane(self, bbox, ground_normal):
        """Align bounding box orientation to ground plane (TEST VERSION)"""
        print(f"  🎯 Aligning {bbox.class_name} bbox to ground plane...")
        
        # Current bbox has rotation matrix from PCA
        original_rotation = bbox.rotation_matrix.copy()
        
        # Compute ground-aligned axes
        # Ground normal is our "up" direction (Y-axis)
        up_axis = ground_normal / np.linalg.norm(ground_normal)
        
        # Find the PCA axis most aligned with ground normal (this becomes our height)
        pca_axes = original_rotation  # PCA components are columns
        
        # Find which PCA axis is most aligned with up direction
        dots = [abs(np.dot(up_axis, axis)) for axis in pca_axes.T]
        height_axis_idx = np.argmax(dots)
        
        print(f"    📏 PCA axis {height_axis_idx} most aligned with ground (dot={dots[height_axis_idx]:.3f})")
        
        # Reorder dimensions: height axis becomes the Y (up) axis
        old_dims = bbox.dimensions.copy()
        old_axes = pca_axes.T.copy()
        
        # Create new aligned rotation matrix
        # Y-axis: align with ground normal (up)
        new_y = up_axis
        
        # X-axis: project one of the remaining PCA axes onto ground plane
        remaining_axes = [i for i in range(3) if i != height_axis_idx]
        candidate_x = old_axes[remaining_axes[0]]
        
        # Project onto ground plane (remove component along ground normal)
        new_x = candidate_x - np.dot(candidate_x, up_axis) * up_axis
        new_x = new_x / np.linalg.norm(new_x)
        
        # Z-axis: cross product to complete right-handed system
        new_z = np.cross(new_x, new_y)
        new_z = new_z / np.linalg.norm(new_z)
        
        # Construct aligned rotation matrix
        aligned_rotation = np.column_stack([new_x, new_y, new_z])
        
        # Reorder dimensions to match new axes
        # Height dimension goes to Y, others distributed to X and Z
        aligned_dims = bbox.dimensions.copy()
        aligned_dims[1] = old_dims[height_axis_idx]  # Height -> Y
        
        remaining_dims = [old_dims[i] for i in remaining_axes]
        aligned_dims[0] = remaining_dims[0]  # -> X
        aligned_dims[2] = remaining_dims[1] if len(remaining_dims) > 1 else remaining_dims[0]  # -> Z
        
        print(f"    📦 Dimensions: {old_dims} -> {aligned_dims}")
        print(f"    🔄 Rotation: PCA -> Ground-aligned")
        
        # Create aligned bounding box
        aligned_bbox = BoundingBox3D(
            center=bbox.center.copy(),  # Keep same center
            dimensions=aligned_dims,
            rotation_matrix=aligned_rotation,
            class_name=bbox.class_name,
            confidence=bbox.confidence,
            instance_id=bbox.instance_id
        )
        
        return aligned_bbox
    
    def check_rotation_consistency(self, prev_rotation, curr_rotation, threshold=0.7):
        """
        Check if the box has rotated by 90 degrees unexpectedly
        Returns: corrected_rotation matrix
        """
        # Check how well the axes align
        x_alignment = np.dot(curr_rotation[:, 0], prev_rotation[:, 0])
        y_alignment = np.dot(curr_rotation[:, 1], prev_rotation[:, 1])
        z_alignment = np.dot(curr_rotation[:, 2], prev_rotation[:, 2])
        
        # Check if axes have swapped (90-degree rotation)
        xy_swap = abs(np.dot(curr_rotation[:, 0], prev_rotation[:, 1])) > abs(x_alignment)
        xz_swap = abs(np.dot(curr_rotation[:, 0], prev_rotation[:, 2])) > abs(x_alignment)
        
        if xy_swap or xz_swap:
            print(f"⚠️ Detected 90-degree rotation jump!")
            
            # Find the best 90-degree rotation to align with previous
            candidates = [
                curr_rotation,  # Original
                curr_rotation @ np.array([[0,-1,0],[1,0,0],[0,0,1]]),  # 90° around Z
                curr_rotation @ np.array([[0,1,0],[-1,0,0],[0,0,1]]),  # -90° around Z
                curr_rotation @ np.array([[1,0,0],[0,0,-1],[0,1,0]]),  # 90° around X
                curr_rotation @ np.array([[1,0,0],[0,0,1],[0,-1,0]]),  # -90° around X
            ]
            
            best_rotation = curr_rotation
            best_score = -1
            
            for candidate in candidates:
                score = (np.dot(candidate[:, 0], prev_rotation[:, 0]) +
                        np.dot(candidate[:, 1], prev_rotation[:, 1]) +
                        np.dot(candidate[:, 2], prev_rotation[:, 2]))
                if score > best_score:
                    best_score = score
                    best_rotation = candidate
            
            return best_rotation
        
        return curr_rotation

    def compute_instance_bboxes_unified(self, pts3ds_list, instance_labels_list, masks_data, 
                                    camera_poses=None, gimbal_data=None, use_pose_transform=True):
        """
        MODIFIED: Unified method with tracking integration
        """
        all_bboxes = []
        
        # Extract ground plane from gimbal data if available and pose transform is enabled
        ground_normal = None
        if gimbal_data is not None and use_pose_transform:
            print("\n🌍 === GROUND PLANE ALIGNMENT ===")
            ground_info = self.get_ground_plane_from_gimbal(gimbal_data)
            if ground_info is not None:
                ground_normal, avg_altitude = ground_info
                print(f"✅ Ground plane detected! Normal: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")
            else:
                print("⚠️ Could not extract ground plane from gimbal data")
        
        # STEP 1: First pass - create all bounding boxes without tracking
        print("\n🔍 === PASS 1: Creating Bounding Boxes ===")
        untracked_bboxes_sequence = []
        
        for frame_idx, (pts3d, instance_labels) in enumerate(zip(pts3ds_list, instance_labels_list)):
            
            if torch.is_tensor(pts3d):
                pts3d = pts3d.cpu().numpy()
            if torch.is_tensor(instance_labels):
                instance_labels = instance_labels.cpu().numpy()
            frame_bboxes = []
            
            if frame_idx not in masks_data:
                untracked_bboxes_sequence.append(frame_bboxes)
                continue
            
            # Get camera pose for this frame if pose transformation is enabled
            pose = np.eye(4)  # Default identity
            if use_pose_transform and camera_poses is not None and frame_idx < len(camera_poses):
                pose = camera_poses[frame_idx]
                if torch.is_tensor(pose):
                    pose = pose.cpu().numpy()
            
            # Get unique instance IDs (excluding background)
            unique_instances = np.unique(instance_labels)
            unique_instances = unique_instances[unique_instances > 0]
            
            for instance_id in unique_instances:
                # Get mask data for this instance
                if instance_id - 1 < len(masks_data[frame_idx]):
                    mask_info = masks_data[frame_idx][instance_id - 1]
                    class_name = mask_info['class_name'].lower()
                    confidence = mask_info['score']
                    
                    # Skip excluded classes
                    if class_name in self.excluded_classes:
                        continue

                    # PHASE 1B: Accumulate points from temporal window
                    # Note: accumulate_temporal_points already handles world transform internally
                    accumulated_points = self.accumulate_temporal_points(
                        frame_idx=frame_idx,
                        all_pts3d=pts3ds_list,
                        all_instance_labels=instance_labels_list,
                        all_masks_data=masks_data,
                        instance_id=instance_id,
                        class_name=class_name,
                        camera_poses=camera_poses,
                        use_pose_transform=use_pose_transform
                    )

                    if len(accumulated_points) < 5:  # Skip if not enough points
                        continue

                    # Filter outliers (points already in appropriate coordinate frame)
                    filtered_points = self.filter_outliers(accumulated_points)
                    
                    if len(filtered_points) < 3:
                        continue
                    
                    # Compute oriented bounding box
                    try:
                        # PHASE 1A: Use species-constrained bbox fitting
                        bbox_result = self.compute_species_constrained_bbox(
                            filtered_points,
                            class_name,
                            ground_normal=ground_normal
                        )
                        if bbox_result is None:
                            continue

                        center, dimensions, rotation = bbox_result

                        # Create bounding box object
                        pca_bbox = BoundingBox3D(
                            center=center,
                            dimensions=dimensions,
                            rotation_matrix=rotation,
                            class_name=class_name,
                            confidence=confidence,
                            instance_id=instance_id
                        )
                        

                        # NEW: Check rotation consistency with previous frame
                        if frame_idx > 0 and len(untracked_bboxes_sequence[frame_idx - 1]) > 0:
                            prev_frame_bboxes = untracked_bboxes_sequence[frame_idx - 1]
                            
                            # Find matching box from previous frame (same class, nearby position)
                            for prev_bbox in prev_frame_bboxes:
                                if prev_bbox.class_name == class_name:
                                    distance = np.linalg.norm(prev_bbox.center - center)
                                    if distance < 8.0:  # Within 8 units (same as tracking threshold)
                                        # Apply rotation consistency check
                                        corrected_rotation = self.check_rotation_consistency(
                                            prev_bbox.rotation_matrix,
                                            rotation
                                        )
                                        pca_bbox.rotation_matrix = corrected_rotation
                                        print(f"  🔄 Applied rotation correction for {class_name}")
                                        break
                                    
                        # NEW: Store the 2D mask for tracking
                        pca_bbox.mask = mask_info['mask']
                        
                        # Apply ground plane alignment if available and pose transform is enabled
                        if ground_normal is not None and use_pose_transform:
                            aligned_bbox = self.align_bbox_to_ground_plane(pca_bbox, ground_normal)
                            # Copy mask to aligned bbox
                            aligned_bbox.mask = pca_bbox.mask
                            frame_bboxes.append(aligned_bbox)
                        else:
                            frame_bboxes.append(pca_bbox)
                        
                    except Exception as e:
                        print(f"  ✗ Failed to create bbox for {class_name}: {e}")
                        continue
            
            untracked_bboxes_sequence.append(frame_bboxes)
            print(f"Frame {frame_idx}: Created {len(frame_bboxes)} untracked bounding boxes")
        
        # STEP 2: Apply tracking to the sequence
        tracker_type = getattr(self, '_tracker_type', 'kalman')
        print(f"\n🔗 === PASS 2: Applying Tracking (method: {tracker_type}) ===")
        tracker = AnimalTracker(
            max_distance_threshold=8.0,  # Adjust based on your scene scale
            mask_iou_threshold=0.15,     # Lower for robustness
            max_missing_frames=2         # Allow 2 frames of missing detections
        )
        
        tracked_bboxes_sequence = []
        
        for frame_idx, frame_bboxes in enumerate(untracked_bboxes_sequence):
            if frame_bboxes:  # Only process frames with detections
                tracked_bboxes = tracker.update(frame_bboxes, frame_idx)

                # PHASE 1C: Apply temporal smoothing
                smoothed_bboxes = []
                for bbox in tracked_bboxes:
                    smoothed_bbox = self.apply_temporal_smoothing(bbox, bbox.track_id)
                    smoothed_bboxes.append(smoothed_bbox)

                tracked_bboxes_sequence.append(smoothed_bboxes)

                # Print tracking info
                active_tracks = len(tracker.active_tracks)
                print(f"Frame {frame_idx}: {len(smoothed_bboxes)} detections, {active_tracks} active tracks")

                # Show track assignments
                for bbox in smoothed_bboxes:
                    track_info = f"Track {bbox.track_id}" if hasattr(bbox, 'track_id') and bbox.track_id is not None else "No track"
                    print(f"  {bbox.class_name} -> {track_info}")
            else:
                tracked_bboxes_sequence.append([])
        
        # Print tracking summary
        stats = tracker.get_tracking_statistics()
        print(f"\n📊 === TRACKING SUMMARY ===")
        print(f"Total tracks created: {stats['total_tracks']}")
        print(f"Active tracks: {stats['active_tracks']}")
        print(f"Average tracklet length: {stats['avg_tracklet_length']:.1f} frames")
        
        for class_name, class_stats in stats['class_stats'].items():
            avg_length = np.mean(class_stats['lengths'])
            print(f"  {class_name}: {class_stats['count']} tracks, avg {avg_length:.1f} frames")
        
        return tracked_bboxes_sequence
    
    def decode_rle_mask(self, rle_data):
        """Decode RLE mask using pycocotools"""
        try:
            from pycocotools import mask as mask_utils
            if isinstance(rle_data, dict) and 'size' in rle_data and 'counts' in rle_data:
                decoded = mask_utils.decode(rle_data)
                return decoded.astype(bool)
        except ImportError:
            print("    pycocotools not available - install with: pip install pycocotools")
            return None
        except Exception as e:
            print(f"    RLE decode failed: {e}")
            return None
    
    def generate_instance_color(self, class_name, instance_idx):
        """Generate consistent colors for instances"""
        if class_name.lower() in self.class_colors:
            base_color = self.class_colors[class_name.lower()]
        else:
            # Generate color based on hash of class name
            hash_val = hash(class_name.lower()) % 1000
            base_color = np.array([
                (hash_val * 0.618) % 1.0,
                ((hash_val * 0.618) * 2) % 1.0,
                ((hash_val * 0.618) * 3) % 1.0
            ])
            # Make it brighter for visibility
            base_color = np.clip(base_color + 0.3, 0.3, 1.0)
        
        # Vary the color slightly for different instances of the same class
        variation = 0.15 * (instance_idx % 3) / 3.0
        varied_color = np.clip(base_color + variation, 0.0, 1.0)
        
        return varied_color
    
    def get_coordinate_mapping(self, original_shape, model_shape, resize_method='letterbox'):
        """Get the mapping between original image coordinates and model coordinates"""
        orig_h, orig_w = original_shape
        model_h, model_w = model_shape
        
        if resize_method == 'letterbox':
            # Calculate letterbox scaling (preserves aspect ratio)
            scale = min(model_w / orig_w, model_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # Calculate padding
            pad_x = (model_w - new_w) // 2
            pad_y = (model_h - new_h) // 2
            
            return {
                'scale': scale,
                'new_size': (new_h, new_w),
                'padding': (pad_y, pad_x),
                'method': 'letterbox'
            }
        else:
            # Simple resize (changes aspect ratio)
            scale_x = model_w / orig_w
            scale_y = model_h / orig_h
            
            return {
                'scale_x': scale_x,
                'scale_y': scale_y,
                'method': 'resize'
            }
    
    def transform_mask_to_model_coordinates(self, mask, original_shape, model_shape):
        """Transform mask from original image coordinates to model coordinates"""
        mapping = self.get_coordinate_mapping(original_shape, model_shape, 'letterbox')
        
        if mapping['method'] == 'letterbox':
            # First resize with aspect ratio preserved
            scale = mapping['scale']
            new_h, new_w = mapping['new_size']
            
            # Resize mask
            mask_resized = cv2.resize(
                mask.astype(np.uint8), 
                (new_w, new_h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # Add padding to match model input
            pad_y, pad_x = mapping['padding']
            mask_padded = np.zeros(model_shape, dtype=bool)
            
            # Place resized mask in center
            end_y = pad_y + new_h
            end_x = pad_x + new_w
            mask_padded[pad_y:end_y, pad_x:end_x] = mask_resized
            
            return mask_padded
        else:
            # Simple resize
            mask_resized = cv2.resize(
                mask.astype(np.uint8), 
                model_shape[::-1],  # cv2 uses (width, height)
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            return mask_resized
    
    def blend_colors(self, original_colors, mask_colors, mask_labels, blend_mode='overlay', alpha=0.6):
        """
        Blend original colors with mask colors using different modes
        """
        H, W, C = original_colors.shape
        blended = original_colors.copy()
        
        # Only blend where there are masks
        mask_areas = mask_labels > 0
        
        if not np.any(mask_areas):
            return blended, np.zeros((H, W), dtype=bool)
        
        if blend_mode == 'overlay':
            # Classic overlay blend: highlights bright areas, multiplies dark areas
            for c in range(3):
                orig_channel = original_colors[mask_areas, c]
                mask_channel = mask_colors[mask_areas, c]
                
                # Overlay formula
                bright_mask = orig_channel >= 0.5
                dark_mask = ~bright_mask
                
                result = orig_channel.copy()
                if np.any(bright_mask):
                    result[bright_mask] = 1 - 2 * (1 - orig_channel[bright_mask]) * (1 - mask_channel[bright_mask])
                if np.any(dark_mask):
                    result[dark_mask] = 2 * orig_channel[dark_mask] * mask_channel[dark_mask]
                
                # Alpha blend with original
                blended[mask_areas, c] = (1 - alpha) * orig_channel + alpha * result
                
        elif blend_mode == 'highlight':
            # Additive highlighting - brightens the original colors
            for c in range(3):
                highlighted = original_colors[mask_areas, c] + alpha * mask_colors[mask_areas, c]
                blended[mask_areas, c] = np.clip(highlighted, 0, 1)
                
        elif blend_mode == 'multiply':
            # Multiply blend - darkens with mask color
            for c in range(3):
                multiplied = original_colors[mask_areas, c] * mask_colors[mask_areas, c]
                blended[mask_areas, c] = (1 - alpha) * original_colors[mask_areas, c] + alpha * multiplied
                
        elif blend_mode == 'screen':
            # Screen blend - lightens with mask color
            for c in range(3):
                screened = 1 - (1 - original_colors[mask_areas, c]) * (1 - mask_colors[mask_areas, c])
                blended[mask_areas, c] = (1 - alpha) * original_colors[mask_areas, c] + alpha * screened
                
        elif blend_mode == 'replace':
            # Simply replace original colors in mask areas
            blended[mask_areas] = mask_colors[mask_areas]
        
        return blended, mask_areas
    
    def backproject_masks_overlay(self, pts3ds_list, original_colors_list, masks_data, original_images, model_size):
        """
        Backproject masks as overlays on original colors with multiple visualization modes
        """
        
        # Storage for different visualization modes
        results = {
            'original_colors': [],          # Just original colors
            'overlay_colors': [],           # Original + overlay blend
            'highlight_colors': [],         # Original + additive highlights  
            'mask_only_colors': [],         # Just mask colors
            'instance_labels': [],          # Instance ID per pixel
            'mask_boundaries': [],          # Boolean mask showing where instances are
        }
        
        print(f"Starting mask overlay backprojection for {len(pts3ds_list)} frames...")
        
        for frame_idx, (pts3d, original_colors) in enumerate(zip(pts3ds_list, original_colors_list)):
            # Handle batch dimension
            if len(pts3d.shape) == 4:
                pts3d = pts3d.squeeze(0)
            if len(original_colors.shape) == 4:
                original_colors = original_colors.squeeze(0)
            
            # Convert to numpy
            if torch.is_tensor(original_colors):
                original_colors = original_colors.cpu().numpy()
            if torch.is_tensor(pts3d):
                pts3d = pts3d.cpu().numpy()
            
            H, W, _ = pts3d.shape
            print(f"\nFrame {frame_idx}: Processing {H}x{W} points")
            
            # Initialize outputs
            instance_labels = np.zeros((H, W), dtype=np.int32)
            mask_colors = np.zeros((H, W, 3))  # For overlay colors
            
            if frame_idx not in masks_data or len(masks_data[frame_idx]) == 0:
                print(f"Frame {frame_idx}: No masks available")
                # Store original colors only
                results['original_colors'].append(original_colors)
                results['overlay_colors'].append(original_colors)
                results['highlight_colors'].append(original_colors)
                results['mask_only_colors'].append(np.ones_like(original_colors) * 0.7)  # Gray
                results['instance_labels'].append(instance_labels)
                results['mask_boundaries'].append(np.zeros((H, W), dtype=bool))
                continue
            
            # Get original image shape
            if frame_idx < len(original_images):
                orig_image = original_images[frame_idx]
                if torch.is_tensor(orig_image):
                    if orig_image.dim() == 4:  # [B, C, H, W]
                        orig_h, orig_w = orig_image.shape[2], orig_image.shape[3]
                    else:  # [C, H, W]
                        orig_h, orig_w = orig_image.shape[1], orig_image.shape[2]
                else:
                    orig_h, orig_w = orig_image.shape[:2]
            else:
                orig_h, orig_w = H, W  # Fallback
            
            # Process each mask
            for mask_idx, mask_data in enumerate(masks_data[frame_idx]):
                mask = mask_data['mask']
                class_name = mask_data['class_name']
                score = mask_data['score']
                
                print(f"  Processing mask {mask_idx}: {class_name} (score: {score:.3f})")
                
                if np.sum(mask) == 0:
                    continue
                
                # Apply light erosion for cleaner boundaries
                kernel = np.ones((3, 3), np.uint8)
                eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                eroded_mask = eroded_mask.astype(bool)
                
                if np.sum(eroded_mask) == 0:
                    eroded_mask = mask  # Fall back if erosion removes everything
                
                # Transform mask to model coordinates
                mask_transformed = self.transform_mask_to_model_coordinates(
                    eroded_mask, (orig_h, orig_w), (H, W)
                )
                
                num_masked_points = np.sum(mask_transformed)
                print(f"    Transformed to {num_masked_points} points")
                
                if num_masked_points > 0:
                    # Assign instance ID
                    instance_id = mask_idx + 1
                    instance_labels[mask_transformed] = instance_id
                    
                    # Generate bright mask color for this instance
                    mask_color = self.generate_instance_color(class_name, mask_idx)
                    mask_colors[mask_transformed] = mask_color
                    
                    print(f"    ✓ Applied {class_name} mask with color {mask_color}")
            
            # Create different visualization modes
            
            # 1. Original colors (unchanged)
            results['original_colors'].append(original_colors.copy())
            
            # 2. Overlay blend
            overlay_colors, mask_boundary = self.blend_colors(
                original_colors, mask_colors, instance_labels, 
                blend_mode='overlay', alpha=0.7
            )
            results['overlay_colors'].append(overlay_colors)
            
            # 3. Highlight blend  
            highlight_colors, _ = self.blend_colors(
                original_colors, mask_colors, instance_labels,
                blend_mode='highlight', alpha=0.5
            )
            results['highlight_colors'].append(highlight_colors)
            
            # 4. Mask colors only (like the old approach)
            mask_only_colors = original_colors.copy()
            mask_only_colors[instance_labels == 0] = [0.7, 0.7, 0.7]  # Gray background
            mask_only_colors[instance_labels > 0] = mask_colors[instance_labels > 0]
            results['mask_only_colors'].append(mask_only_colors)
            
            # 5. Store labels and boundaries
            results['instance_labels'].append(instance_labels)
            results['mask_boundaries'].append(mask_boundary)
        
        return results

def load_grounded_sam_masks_overlay(mask_dir, img_paths):
    """Load Grounded SAM masks"""
    masks_data = {}
    back_projector = Enhanced3DBBoxBackProjector()
    
    for i, img_path in enumerate(img_paths):
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_file = os.path.join(mask_dir, f"{frame_name}_results.json")
        # mask_file = os.path.join(mask_dir, f"{frame_name}.json")

        if os.path.exists(mask_file):
            try:
                with open(mask_file, 'r') as f:
                    data = json.load(f)
                
                print(f"Loading masks for {frame_name}: found {len(data['annotations'])} annotations")
                
                frame_masks = []
                for ann_idx, ann in enumerate(data['annotations']):
                    try:
                        mask = back_projector.decode_rle_mask(ann['segmentation'])
                        
                        if mask is not None and np.sum(mask) > 0:
                            frame_masks.append({
                                'mask': mask,
                                'class_name': ann['class_name'],
                                'score': ann['score'][0] if isinstance(ann['score'], list) else ann['score'],
                                'bbox': ann['bbox']
                            })
                            print(f"  ✓ {ann['class_name']}: {mask.shape}, {np.sum(mask)} pixels")
                        else:
                            print(f"  ✗ Failed to decode {ann['class_name']} mask")
                    except Exception as e:
                        print(f"  ✗ Error processing annotation {ann_idx}: {e}")
                
                masks_data[i] = frame_masks
                print(f"Successfully loaded {len(frame_masks)} valid masks for {frame_name}")
                
            except Exception as e:
                print(f"Error loading {mask_file}: {e}")
                masks_data[i] = []
        else:
            print(f"Mask file not found: {mask_file}")
            masks_data[i] = []
    
    return masks_data

def save_point_cloud_as_ply(points, colors, output_path, confidence=None, confidence_threshold=0.5):
    """
    Save point cloud as PLY file
    
    Args:
        points: numpy array of shape (H, W, 3) or (N, 3)
        colors: numpy array of shape (H, W, 3) or (N, 3), values in [0, 1]
        output_path: path to save PLY file
        confidence: optional confidence values
        confidence_threshold: minimum confidence to include point
    """
    # Handle different input shapes
    if points.ndim == 3:  # (H, W, 3)
        H, W = points.shape[:2]
        points = points.reshape(-1, 3)
        
        # Reshape colors to match
        if colors.ndim == 3:
            colors = colors.reshape(-1, 3)
        elif colors.ndim == 4:  # (1, H, W, 3)
            colors = colors[0].reshape(-1, 3)
        
        if confidence is not None:
            if confidence.ndim == 2:  # (H, W)
                confidence = confidence.reshape(-1)
            elif confidence.ndim == 3:  # (1, H, W)
                confidence = confidence[0].reshape(-1)
    
    # Ensure colors has the same number of points as points array
    if len(colors) != len(points):
        print(f"  Warning: Color/point mismatch: {len(colors)} colors, {len(points)} points")
        # Try to fix common cases
        if colors.ndim == 2 and colors.shape[0] == 1:
            colors = colors[0]
        if len(colors) < len(points):
            # Pad with white
            colors = np.vstack([colors, np.ones((len(points) - len(colors), 3))])
        elif len(colors) > len(points):
            # Truncate
            colors = colors[:len(points)]
    
    # Filter out invalid points (NaN or Inf)
    valid_mask = np.isfinite(points).all(axis=1)
    if confidence is not None:
        # Also filter by confidence if provided
        if confidence.shape[0] == valid_mask.shape[0]:
            valid_mask &= (confidence > confidence_threshold)
    
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    if len(points) == 0:
        print(f"  Warning: No valid points to save for {output_path}")
        return 0
    
    # Convert colors to uint8 (0-255)
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    # Create PLY header
    num_points = len(points)
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # Write PLY file
    with open(output_path, 'w') as f:
        f.write(header)
        for i in range(num_points):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n")
    
    return num_points

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference with mask overlays and toggle."
    )

    parser.add_argument(
    "--save_ply",
    action="store_true",
    default=True,
    help="Save point clouds as PLY files for each frame",
    )

    parser.add_argument(
        "--revisit",
        type=int,
        default=1,
        help="Number of revisit passes (1=online only, 2+=revisiting enabled)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=_resolve_model_path(),
        help="Path to the pretrained model checkpoint (auto-resolved from backends/cut3r/src/, checkpoints/, or ~/.cache/wildlift/).",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="Path to the directory containing Grounded SAM mask JSON files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="Output directory",
    )
    parser.add_argument(
        "--blend_mode",
        type=str,
        default="overlay",
        choices=["overlay", "highlight", "mask_only", "original"],
        help="Default visualization mode",
    )

    parser.add_argument(
        "--dji_log",
        type=str,
        default="",
        help="Path to DJI SRT log file for gimbal data extraction.",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="kalman",
        choices=["kalman", "simple"],
        help="Tracking method: 'kalman' (default, 3D Kalman filter + re-ID) or 'simple' (online tracker).",
    )

    parser.add_argument(
        "--gps_refine",
        action="store_true",
        help="Enable GPS-based pose refinement using DJI SRT GPS data.",
    )

    return parser.parse_args()

def prepare_input(
    img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True
):
    """Prepare input views for inference from a list of image paths."""
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
    else:
        # Handle raymaps case (keeping original logic)
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views

def parse_seq_path(p):
    """Parse sequence path (directory of images or video file)."""
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        img_paths = [p for p in img_paths if p.lower().endswith(img_extensions)]
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname

# def prepare_output_overlay_unified(outputs, outdir, revisit=1, use_pose=True, 
#                                    masks_data=None, original_images=None, model_size=512, 
#                                    blend_mode='overlay', gimbal_data=None, 
#                                    enable_3d_bboxes=True, enable_2d_projection=True,
#                                    frame_names=None):
#     """
#     Unified function to process inference outputs with optional features:
#     - Overlay mask backprojection (always enabled if masks_data provided)
#     - 3D bounding boxes with tracking (optional)
#     - 2D projections (optional)
    
#     Returns:
#         Base: (pts3ds_other, colors_to_use, conf_other, cam_dict, instance_labels, visualization_results)
#         + bounding_boxes (if enable_3d_bboxes=True)
#         + annotated_images (if enable_2d_projection=True and bounding_boxes exist)
#     """
#     from src.dust3r.utils.camera import pose_encoding_to_camera
#     from src.dust3r.post_process import estimate_focal_knowing_depth
#     from src.dust3r.utils.geometry import geotrf

#     print(f"\n🔄 === PROCESSING OUTPUTS ===")
#     print(f"Features enabled: 3D bboxes={enable_3d_bboxes}, 2D projection={enable_2d_projection}")
#     print(f"Blend mode: {blend_mode}")

#     # ==================================================================================
#     # STEP 1: Process raw inference outputs
#     # ==================================================================================
    
#     # Only keep outputs corresponding to one full pass
#     valid_length = len(outputs["pred"]) // revisit
#     outputs["pred"] = outputs["pred"][-valid_length:]
#     outputs["views"] = outputs["views"][-valid_length:]

#     # Extract 3D points and confidence
#     pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
#     pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
#     conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
#     conf_other = [output["conf"].cpu() for output in outputs["pred"]]
#     pts3ds_self = torch.cat(pts3ds_self_ls, 0)

#     print(f"✓ Extracted 3D points for {len(pts3ds_self_ls)} frames")

#     # ==================================================================================
#     # STEP 2: Recover camera poses and setup camera parameters
#     # ==================================================================================
    
#     # Recover camera poses
#     pr_poses = [
#         pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
#         for pred in outputs["pred"]
#     ]
#     R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
#     t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

#     # Apply pose transformation if enabled
#     if use_pose:
#         transformed_pts3ds_other = []
#         for pose, pself in zip(pr_poses, pts3ds_self):
#             transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
#         pts3ds_other = transformed_pts3ds_other
#         conf_other = conf_self

#     # Estimate focal length and camera parameters
#     B, H, W, _ = pts3ds_self.shape
#     pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
#     focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

#     # Get original colors
#     original_colors_list = [
#         0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
#     ]

#     # Create camera dictionary
#     cam_dict = {
#         "focal": focal.cpu().numpy(),
#         "pp": pp.cpu().numpy(),
#         "R": R_c2w.cpu().numpy(),
#         "t": t_c2w.cpu().numpy(),
#     }

#     print(f"✓ Configured camera parameters")

#     # ==================================================================================
#     # STEP 3: Process mask overlays and visualization
#     # ==================================================================================
    
#     visualization_results = None
#     instance_labels = None
    
#     if masks_data is not None:
#         print(f"\n🎨 === MASK OVERLAY PROCESSING ===")
#         back_projector = Enhanced3DBBoxBackProjector()
#         visualization_results = back_projector.backproject_masks_overlay(
#             pts3ds_self_ls, original_colors_list, masks_data, original_images, model_size
#         )
        
#         instance_labels = visualization_results['instance_labels']
        
#         # Count mask coverage
#         total_masked = sum(np.sum(labels > 0) for labels in instance_labels)
#         total_points = sum(labels.size for labels in instance_labels)
#         coverage_pct = (total_masked / total_points) * 100 if total_points > 0 else 0
        
#         print(f"✓ Processed mask overlays: {total_masked:,} points ({coverage_pct:.1f}%) covered")
#     else:
#         print("⚠️ No mask data provided - using original colors only")

#     # ==================================================================================
#     # STEP 4: Compute 3D bounding boxes with tracking (optional)
#     # ==================================================================================
    
#     bounding_boxes = None
#     tracking_stats = None
    
#     if enable_3d_bboxes and masks_data is not None:
#         print(f"\n📦 === 3D BOUNDING BOXES + TRACKING ===")
        
#         # Ensure we have the back_projector
#         if 'back_projector' not in locals():
#             back_projector = Enhanced3DBBoxBackProjector()
        
#         # Get camera poses for bbox computation
#         bbox_poses = pr_poses if use_pose else [np.eye(4) for _ in range(len(pts3ds_self_ls))]
#         coord_system = "world coordinates" if use_pose else "local coordinates"
#         print(f"Using {coord_system} for bounding box computation")
        
#         # Compute bounding boxes with tracking
#         bounding_boxes = back_projector.compute_instance_bboxes_unified(
#             pts3ds_list=pts3ds_self_ls, 
#             instance_labels_list=instance_labels, 
#             masks_data=masks_data,
#             camera_poses=bbox_poses,
#             gimbal_data=gimbal_data,
#             use_pose_transform=use_pose
#         )
        
#         # Calculate tracking statistics
#         tracking_stats = _calculate_tracking_statistics(bounding_boxes)
        
#         total_bboxes = sum(len(frame_bboxes) for frame_bboxes in bounding_boxes)
#         print(f"✓ Created {total_bboxes} tracked 3D bounding boxes!")
#         print(f"✓ Tracking: {tracking_stats['total_tracks']} tracks, "
#               f"avg length {tracking_stats['avg_tracklet_length']:.1f} frames")
        
#     elif enable_3d_bboxes:
#         print("⚠️ 3D bounding boxes requested but no mask data available")

#     # ==================================================================================
#     # STEP 5: Generate 2D projections (optional)
#     # ==================================================================================
    
#     annotated_images = None

#     if enable_2d_projection and bounding_boxes is not None:
#         total_bboxes = sum(len(frame_bboxes) for frame_bboxes in bounding_boxes)
#         if total_bboxes > 0:
#             print(f"\n📸 === 2D BBOX PROJECTION WITH TRACKS ===")
            
#             # Compute track information
#             track_trajectories, track_colors, track_info = compute_track_trajectories_and_colors(bounding_boxes)
            
#             # Convert original images for projection
#             projection_images = []
#             for view_output in outputs["views"]:
#                 img = 0.5 * (view_output["img"] + 1.0)
#                 projection_images.append(img)
            
#             # Use enhanced projection with track visualization
#             annotated_images = add_2d_projection_with_tracks(
#                 original_images=projection_images,
#                 bounding_boxes=bounding_boxes,
#                 cam_dict=cam_dict,
#                 output_dir=outdir,
#                 track_colors=track_colors,
#                 track_info=track_info,
#                 frame_names=frame_names
#             )
            
#             print(f"✅ Generated tracked 2D projections for {len(annotated_images)} images")
#         else:
#             print("⚠️ No bounding boxes found for 2D projection")
#     elif enable_2d_projection:
#         print("⚠️ 2D projection requested but no bounding boxes available")

#     # ==================================================================================
#     # STEP 6: Choose visualization colors based on blend mode
#     # ==================================================================================
    
#     if visualization_results is not None:
#         mode_mapping = {
#             'original': 'original_colors',
#             'overlay': 'overlay_colors', 
#             'highlight': 'highlight_colors',
#             'mask_only': 'mask_only_colors'
#         }
        
#         color_key = mode_mapping.get(blend_mode, 'overlay_colors')
#         colors_to_use = visualization_results[color_key]
        
#         print(f"✓ Using visualization mode: {blend_mode}")
#     else:
#         colors_to_use = original_colors_list
#         print("✓ Using original image colors")

#     # ==================================================================================
#     # STEP 7: Save all outputs to disk
#     # ==================================================================================
    
#     print(f"\n💾 === SAVING OUTPUTS ===")
    
#     # Prepare tensors for saving
#     pts3ds_self_tosave = pts3ds_self
#     depths_tosave = pts3ds_self_tosave[..., 2]
#     pts3ds_other_tosave = torch.cat(pts3ds_other)
#     conf_self_tosave = torch.cat(conf_self)
#     conf_other_tosave = torch.cat(conf_other)
#     colors_tosave = torch.cat([torch.tensor(c) for c in colors_to_use])
#     cam2world_tosave = torch.cat(pr_poses)
    
#     # Prepare intrinsics
#     intrinsics_tosave = (
#         torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
#     )
#     intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
#     intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
#     intrinsics_tosave[:, 0, 2] = pp[:, 0]
#     intrinsics_tosave[:, 1, 2] = pp[:, 1]

#     # Create output directories (MODIFIED: Remove visualization mode folders)
#     _create_output_directories(outdir, visualization_results=None, bounding_boxes=bounding_boxes)
    
#     # Save frame-by-frame data (MODIFIED: Use frame names)
#     # _save_frame_data(
#     #     outdir=outdir,
#     #     pts3ds_self=pts3ds_self,
#     #     depths_tosave=depths_tosave,
#     #     conf_self_tosave=conf_self_tosave,
#     #     colors_tosave=colors_tosave,
#     #     cam2world_tosave=cam2world_tosave,
#     #     intrinsics_tosave=intrinsics_tosave,
#     #     instance_labels=instance_labels,
#     #     visualization_results=None,  # MODIFIED: Don't save visualization folders
#     #     bounding_boxes=bounding_boxes,
#     #     frame_names=frame_names
#     # )
    
#     # Save frame-by-frame data (MODIFIED: Use frame names and add pts3ds_other)
#     _save_frame_data(
#         outdir=outdir,
#         pts3ds_self=pts3ds_self,
#         depths_tosave=depths_tosave,
#         conf_self_tosave=conf_self_tosave,
#         colors_tosave=colors_tosave,
#         cam2world_tosave=cam2world_tosave,
#         intrinsics_tosave=intrinsics_tosave,
#         instance_labels=instance_labels,
#         visualization_results=None,
#         bounding_boxes=bounding_boxes,
#         frame_names=frame_names,
#         pts3ds_other=pts3ds_other,  # Add this
#         save_ply=True  # Add this (or make it configurable via args)
#     )

#     # Save tracking summary
#     if tracking_stats is not None:
#         _save_tracking_summary(outdir, tracking_stats, bounding_boxes)
    
#     print(f"✓ Saved outputs to {outdir}")

#     # ==================================================================================
#     # STEP 8: Return results based on enabled features
#     # ==================================================================================
    
#     base_result = (pts3ds_other, colors_to_use, conf_other, cam_dict, instance_labels, visualization_results)
    
#     if enable_3d_bboxes and enable_2d_projection:
#         return base_result + (bounding_boxes, annotated_images)
#     elif enable_3d_bboxes:
#         return base_result + (bounding_boxes,)
#     else:
#         return base_result

def prepare_output_overlay_unified(outputs, outdir, revisit=1, use_pose=True, 
                                   masks_data=None, original_images=None, model_size=512, 
                                   blend_mode='overlay', gimbal_data=None, 
                                   enable_3d_bboxes=True, enable_2d_projection=True,
                                   frame_names=None):
    """
    Unified function to process inference outputs with optional features:
    - Overlay mask backprojection (always enabled if masks_data provided)
    - 3D bounding boxes with tracking (optional)
    - 2D projections (optional)
    
    Returns:
        Base: (pts3ds_other, colors_to_use, conf_other, cam_dict, instance_labels, visualization_results)
        + bounding_boxes (if enable_3d_bboxes=True)
        + annotated_images (if enable_2d_projection=True and bounding_boxes exist)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf

    print(f"\n🔄 === PROCESSING OUTPUTS ===")
    print(f"Features enabled: 3D bboxes={enable_3d_bboxes}, 2D projection={enable_2d_projection}")
    print(f"Blend mode: {blend_mode}")

    # ==================================================================================
    # STEP 1: Process raw inference outputs
    # ==================================================================================
    
    # Only keep outputs corresponding to one full pass
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # Extract 3D points and confidence
    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    print(f"✓ Extracted 3D points for {len(pts3ds_self_ls)} frames")

    # ==================================================================================
    # STEP 2: Recover camera poses and setup camera parameters
    # ==================================================================================
    
    # Recover camera poses
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    # Apply pose transformation if enabled
    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length and camera parameters
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    # Get original colors
    original_colors_list = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    # Create camera dictionary
    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
        "model_size": (H, W),  # Store actual model dimensions for reprojection
    }

    print(f"✓ Configured camera parameters")

    # ==================================================================================
    # STEP 3: Process mask overlays and visualization
    # ==================================================================================
    
    visualization_results = None
    instance_labels = None
    
    if masks_data is not None:
        print(f"\n🎨 === MASK OVERLAY PROCESSING ===")
        back_projector = Enhanced3DBBoxBackProjector()
        visualization_results = back_projector.backproject_masks_overlay(
            pts3ds_self_ls, original_colors_list, masks_data, original_images, model_size
        )
        
        instance_labels = visualization_results['instance_labels']
        
        # Count mask coverage
        total_masked = sum(np.sum(labels > 0) for labels in instance_labels)
        total_points = sum(labels.size for labels in instance_labels)
        coverage_pct = (total_masked / total_points) * 100 if total_points > 0 else 0
        
        print(f"✓ Processed mask overlays: {total_masked:,} points ({coverage_pct:.1f}%) covered")
    else:
        print("⚠️ No mask data provided - using original colors only")

    # ==================================================================================
    # STEP 4: Compute 3D bounding boxes with tracking (optional)
    # ==================================================================================
    
    bounding_boxes = None
    tracking_stats = None
    
    if enable_3d_bboxes and masks_data is not None:
        print(f"\n📦 === 3D BOUNDING BOXES + TRACKING ===")
        
        # Ensure we have the back_projector
        if 'back_projector' not in locals():
            back_projector = Enhanced3DBBoxBackProjector()
        
        # Get camera poses for bbox computation
        bbox_poses = pr_poses if use_pose else [np.eye(4) for _ in range(len(pts3ds_self_ls))]
        coord_system = "world coordinates" if use_pose else "local coordinates"
        print(f"Using {coord_system} for bounding box computation")
        
        # Compute bounding boxes with tracking
        bounding_boxes = back_projector.compute_instance_bboxes_unified(
            pts3ds_list=pts3ds_self_ls, 
            instance_labels_list=instance_labels, 
            masks_data=masks_data,
            camera_poses=bbox_poses,
            gimbal_data=gimbal_data,
            use_pose_transform=use_pose
        )
        
        # Calculate tracking statistics
        tracking_stats = _calculate_tracking_statistics(bounding_boxes)
        
        total_bboxes = sum(len(frame_bboxes) for frame_bboxes in bounding_boxes)
        print(f"✓ Created {total_bboxes} tracked 3D bounding boxes!")
        print(f"✓ Tracking: {tracking_stats['total_tracks']} tracks, "
              f"avg length {tracking_stats['avg_tracklet_length']:.1f} frames")
        
    elif enable_3d_bboxes:
        print("⚠️ 3D bounding boxes requested but no mask data available")

    # ==================================================================================
    # STEP 5: Generate 2D projections (optional)
    # ==================================================================================
    
    annotated_images = None

    if enable_2d_projection and bounding_boxes is not None:
        total_bboxes = sum(len(frame_bboxes) for frame_bboxes in bounding_boxes)
        if total_bboxes > 0:
            print(f"\n📸 === 2D BBOX PROJECTION WITH TRACKS ===")
            
            # Compute track information
            track_trajectories, track_colors, track_info = compute_track_trajectories_and_colors(bounding_boxes)
            
            # Convert original images for projection
            # projection_images = []
            # for view_output in outputs["views"]:
            #     img = 0.5 * (view_output["img"] + 1.0)
            #     projection_images.append(img)

            # Resize original images to model size for correct projection
            # (camera intrinsics are computed at model resolution)
            model_h, model_w = cam_dict["model_size"]
            projection_images = []
            for orig_img in original_images:
                resized = cv2.resize(orig_img, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
                projection_images.append(resized)
            print(f"Resized {len(projection_images)} images from original to model size ({model_w}x{model_h})")

            # Use enhanced projection with track visualization
            annotated_images = add_2d_projection_with_tracks(
                original_images=projection_images,
                bounding_boxes=bounding_boxes,
                cam_dict=cam_dict,
                output_dir=outdir,
                track_colors=track_colors,
                track_info=track_info,
                frame_names=frame_names
            )
            
            print(f"✅ Generated tracked 2D projections for {len(annotated_images)} images")
        else:
            print("⚠️ No bounding boxes found for 2D projection")
    elif enable_2d_projection:
        print("⚠️ 2D projection requested but no bounding boxes available")

    # ==================================================================================
    # STEP 6: Choose visualization colors based on blend mode
    # ==================================================================================
    
    if visualization_results is not None:
        mode_mapping = {
            'original': 'original_colors',
            'overlay': 'overlay_colors', 
            'highlight': 'highlight_colors',
            'mask_only': 'mask_only_colors'
        }
        
        color_key = mode_mapping.get(blend_mode, 'overlay_colors')
        colors_to_use = visualization_results[color_key]
        
        print(f"✓ Using visualization mode: {blend_mode}")
    else:
        colors_to_use = original_colors_list
        print("✓ Using original image colors")

    # ==================================================================================
    # STEP 7: Save all outputs to disk
    # ==================================================================================
    
    print(f"\n💾 === SAVING OUTPUTS ===")
    
    # Prepare tensors for saving
    pts3ds_self_tosave = pts3ds_self
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)
    conf_self_tosave = torch.cat(conf_self)
    conf_other_tosave = torch.cat(conf_other)
    # FIX: Don't concatenate colors - pass the list directly
    colors_tosave = colors_to_use  # Keep as list
    cam2world_tosave = torch.cat(pr_poses)
    
    # Prepare intrinsics
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # Create output directories
    _create_output_directories(outdir, visualization_results=None, bounding_boxes=bounding_boxes)
    
    # Save frame-by-frame data
    _save_frame_data(
        outdir=outdir,
        pts3ds_self=pts3ds_self,
        depths_tosave=depths_tosave,
        conf_self_tosave=conf_self_tosave,
        colors_tosave=colors_tosave,  # Pass list directly
        cam2world_tosave=cam2world_tosave,
        intrinsics_tosave=intrinsics_tosave,
        instance_labels=instance_labels,
        visualization_results=None,
        bounding_boxes=bounding_boxes,
        frame_names=frame_names,
        pts3ds_other=pts3ds_other,
        save_ply=True
    )

    # Save tracking summary
    if tracking_stats is not None:
        _save_tracking_summary(outdir, tracking_stats, bounding_boxes)
    
    print(f"✓ Saved outputs to {outdir}")

    # ==================================================================================
    # STEP 8: Return results based on enabled features
    # ==================================================================================
    
    base_result = (pts3ds_other, colors_to_use, conf_other, cam_dict, instance_labels, visualization_results)
    
    if enable_3d_bboxes and enable_2d_projection:
        return base_result + (bounding_boxes, annotated_images)
    elif enable_3d_bboxes:
        return base_result + (bounding_boxes,)
    else:
        return base_result
        
def _calculate_tracking_statistics(bounding_boxes):
    """Calculate tracking statistics from bounding boxes"""
    all_track_ids = set()
    track_detections = {}  # track_id -> list of frames
    
    for frame_idx, frame_bboxes in enumerate(bounding_boxes):
        for bbox in frame_bboxes:
            if hasattr(bbox, 'track_id') and bbox.track_id is not None:
                track_id = bbox.track_id
                all_track_ids.add(track_id)
                
                if track_id not in track_detections:
                    track_detections[track_id] = []
                track_detections[track_id].append({
                    'frame': frame_idx,
                    'class_name': bbox.class_name,
                    'confidence': bbox.confidence
                })
    
    # Calculate statistics
    tracklet_lengths = [len(detections) for detections in track_detections.values()]
    avg_length = np.mean(tracklet_lengths) if tracklet_lengths else 0
    
    # Per-class statistics
    class_stats = {}
    for track_id, detections in track_detections.items():
        if detections:
            class_name = detections[0]['class_name']
            if class_name not in class_stats:
                class_stats[class_name] = {'count': 0, 'lengths': []}
            class_stats[class_name]['count'] += 1
            class_stats[class_name]['lengths'].append(len(detections))
    
    return {
        'total_tracks': len(all_track_ids),
        'track_detections': track_detections,
        'avg_tracklet_length': avg_length,
        'max_tracklet_length': max(tracklet_lengths) if tracklet_lengths else 0,
        'class_stats': class_stats
    }

def _create_output_directories(outdir, visualization_results, bounding_boxes):
    """Create all necessary output directories (MODIFIED: No visualization folders)"""
    # MODIFIED: Only create essential directories, not visualization mode folders
    base_dirs = ["depth", "conf", "camera"]  # Removed "color" directory
    
    for dir_name in base_dirs:
        os.makedirs(os.path.join(outdir, dir_name), exist_ok=True)
    
    # MODIFIED: Don't create visualization mode directories
    if bounding_boxes is not None:
        os.makedirs(os.path.join(outdir, "bounding_boxes"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "instance_labels"), exist_ok=True)

# def _save_frame_data(outdir, pts3ds_self, depths_tosave, conf_self_tosave, colors_tosave, 
#                     cam2world_tosave, intrinsics_tosave, instance_labels, 
#                     visualization_results, bounding_boxes, frame_names=None):
#     """Save all frame-by-frame data (MODIFIED: Use frame names, don't save visualization modes)"""
#     import imageio.v2 as iio
    
#     for f_id in range(len(pts3ds_self)):
#         # Get frame name from original filename
#         if frame_names and f_id < len(frame_names):
#             frame_name = frame_names[f_id]
#         else:
#             frame_name = f"{f_id:06d}"
        
#         # Basic data
#         depth = depths_tosave[f_id].cpu().numpy()
#         conf = conf_self_tosave[f_id].cpu().numpy()
#         c2w = cam2world_tosave[f_id].cpu().numpy()
#         intrins = intrinsics_tosave[f_id].cpu().numpy()
        
#         # Save basic outputs with original frame names
#         np.save(os.path.join(outdir, "depth", f"{frame_name}.npy"), depth)
#         np.save(os.path.join(outdir, "conf", f"{frame_name}.npy"), conf)
#         np.savez(
#             os.path.join(outdir, "camera", f"{frame_name}.npz"),
#             pose=c2w,
#             intrinsics=intrins,
#         )
        
#         # MODIFIED: Don't save color and visualization mode images
        
#         # Save instance labels with original frame names
#         if instance_labels is not None:
#             np.save(os.path.join(outdir, "instance_labels", f"{frame_name}.npy"), 
#                    instance_labels[f_id])
        
#         # Save bounding boxes with tracking information and original frame names
#         if bounding_boxes is not None and f_id < len(bounding_boxes):
#             frame_bboxes = bounding_boxes[f_id]
#             bbox_data = []
#             for bbox in frame_bboxes:
#                 bbox_dict = {
#                     'center': [float(x) for x in bbox.center.tolist()],
#                     'dimensions': [float(x) for x in bbox.dimensions.tolist()],
#                     'rotation_matrix': [[float(x) for x in row] for row in bbox.rotation_matrix.tolist()],
#                     'class_name': str(bbox.class_name),
#                     'confidence': float(bbox.confidence),
#                     'instance_id': int(bbox.instance_id)
#                 }
                
#                 # Add tracking information
#                 if hasattr(bbox, 'track_id') and bbox.track_id is not None:
#                     bbox_dict['track_id'] = int(bbox.track_id)
#                     bbox_dict['persistent_instance_id'] = int(bbox.persistent_instance_id)
#                 else:
#                     bbox_dict['track_id'] = -1
#                     bbox_dict['persistent_instance_id'] = -1
                    
#                 bbox_data.append(bbox_dict)
            
#             with open(os.path.join(outdir, "bounding_boxes", f"{frame_name}.json"), 'w') as f:
#                 json.dump(bbox_data, f, indent=2)

def _save_frame_data(outdir, pts3ds_self, depths_tosave, conf_self_tosave, colors_tosave, 
                    cam2world_tosave, intrinsics_tosave, instance_labels, 
                    visualization_results, bounding_boxes, frame_names=None,
                    pts3ds_other=None, save_ply=True):
    """Save all frame-by-frame data including PLY files"""
    import imageio.v2 as iio
    
    # Create PLY directories if saving PLY files
    if save_ply:
        os.makedirs(os.path.join(outdir, "point_clouds"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "point_clouds_world"), exist_ok=True)
    
    print(f"\n💾 Saving frame data...")
    
    for f_id in range(len(pts3ds_self)):
        # Get frame name from original filename
        if frame_names and f_id < len(frame_names):
            frame_name = frame_names[f_id]
        else:
            frame_name = f"{f_id:06d}"
        
        # Extract data for this frame
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        
        # Save depth, confidence, and camera parameters
        np.save(os.path.join(outdir, "depth", f"{frame_name}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{frame_name}.npy"), conf)
        np.savez(
            os.path.join(outdir, "camera", f"{frame_name}.npz"),
            pose=c2w,
            intrinsics=intrins,
        )
        
        # Save instance labels if available
        if instance_labels is not None and f_id < len(instance_labels):
            os.makedirs(os.path.join(outdir, "instance_labels"), exist_ok=True)
            np.save(
                os.path.join(outdir, "instance_labels", f"{frame_name}.npy"), 
                instance_labels[f_id]
            )
        
        # Save bounding boxes
        if bounding_boxes is not None and f_id < len(bounding_boxes):
            frame_bboxes = bounding_boxes[f_id]
            if len(frame_bboxes) > 0:
                bbox_data = []
                for bbox in frame_bboxes:
                    bbox_dict = {
                        'center': [float(x) for x in bbox.center.tolist()],
                        'dimensions': [float(x) for x in bbox.dimensions.tolist()],
                        'rotation_matrix': [[float(x) for x in row] for row in bbox.rotation_matrix.tolist()],
                        'class_name': str(bbox.class_name),
                        'confidence': float(bbox.confidence),
                        'instance_id': int(bbox.instance_id),
                        'track_id': int(bbox.track_id) if hasattr(bbox, 'track_id') and bbox.track_id is not None else -1,
                        'persistent_instance_id': int(bbox.persistent_instance_id) if hasattr(bbox, 'persistent_instance_id') and bbox.persistent_instance_id is not None else -1
                    }
                    bbox_data.append(bbox_dict)
                
                os.makedirs(os.path.join(outdir, "bounding_boxes"), exist_ok=True)
                with open(os.path.join(outdir, "bounding_boxes", f"{frame_name}.json"), 'w') as f:
                    json.dump(bbox_data, f, indent=2)
        
        # Save PLY files
        if save_ply:
            # Get points for this frame
            pts3d = pts3ds_self[f_id]
            if torch.is_tensor(pts3d):
                pts3d = pts3d.cpu().numpy()
            
            # FIX: Get colors for this specific frame from the list
            if isinstance(colors_tosave, list):
                # colors_tosave is a list of frames, get the f_id-th frame
                colors = colors_tosave[f_id]
            else:
                # Fallback for tensor format (shouldn't happen with new code)
                colors = colors_tosave[f_id]
            
            if torch.is_tensor(colors):
                colors = colors.cpu().numpy()
            
            # Remove any extra batch dimensions
            if pts3d.ndim == 4:  # (1, H, W, 3)
                pts3d = pts3d[0]
            if colors.ndim == 4:  # (1, H, W, 3)
                colors = colors[0]
            
            # Ensure colors match points shape
            if colors.shape[:2] != pts3d.shape[:2]:
                print(f"  Warning: Shape mismatch - pts3d: {pts3d.shape}, colors: {colors.shape}")
                
                # Try to reshape colors to match
                H, W = pts3d.shape[:2]
                if colors.size == H * W * 3:
                    colors = colors.reshape(H, W, 3)
                    print(f"  ✓ Reshaped colors from {colors.shape} to ({H}, {W}, 3)")
                else:
                    print(f"  Error: Cannot match color dimensions to points")
                    print(f"     Expected: {H * W * 3} elements, got: {colors.size}")
                    colors = np.ones((H, W, 3)) * 0.5  # Gray fallback
                    print(f"  ✓ Using gray fallback colors: {colors.shape}")
            
            confidence = conf
            
            # Save camera-coordinate PLY
            ply_path_camera = os.path.join(outdir, "point_clouds", f"{frame_name}.ply")
            num_points_camera = save_point_cloud_as_ply(pts3d, colors, ply_path_camera, confidence)
            
            # Save world-coordinate PLY if available
            if pts3ds_other is not None and f_id < len(pts3ds_other):
                pts3d_world = pts3ds_other[f_id]
                if torch.is_tensor(pts3d_world):
                    pts3d_world = pts3d_world.cpu().numpy()
                if pts3d_world.ndim == 4:  # (1, H, W, 3)
                    pts3d_world = pts3d_world[0]
                
                ply_path_world = os.path.join(outdir, "point_clouds_world", f"{frame_name}.ply")
                num_points_world = save_point_cloud_as_ply(pts3d_world, colors, ply_path_world, confidence)
                
                print(f"  Frame {frame_name}: Saved {num_points_camera:,} points (camera) and {num_points_world:,} points (world)")
            else:
                print(f"  Frame {frame_name}: Saved {num_points_camera:,} points (camera coords)")
    
    print(f"✅ Saved all frame data to {outdir}")

    # Save mask-track mapping for annotator tool
    if bounding_boxes is not None:
        _save_mask_track_mapping(outdir, bounding_boxes, frame_names)


def _save_mask_track_mapping(outdir, bounding_boxes, frame_names):
    """Save mapping from track_id to mask annotation index for each frame.

    This enables the annotator to directly look up which mask belongs to which track,
    instead of using error-prone IoU heuristics when multiple animals are close together.

    The mapping file format:
    {
        "frame_name": {
            "track_id": mask_annotation_index,
            ...
        },
        ...
    }

    Args:
        outdir: Output directory path
        bounding_boxes: List of lists of BoundingBox3D objects per frame
        frame_names: List of frame name strings
    """
    mask_track_mapping = {}

    for f_id, frame_bboxes in enumerate(bounding_boxes):
        # Get frame key (name or index)
        if frame_names and f_id < len(frame_names):
            frame_key = str(frame_names[f_id])
        else:
            frame_key = str(f_id)

        mask_track_mapping[frame_key] = {}
        for bbox in frame_bboxes:
            if hasattr(bbox, 'track_id') and bbox.track_id is not None and bbox.track_id != -1:
                # instance_id is 1-indexed, mask annotation index is 0-indexed
                mask_index = int(bbox.instance_id) - 1
                mask_track_mapping[frame_key][str(bbox.track_id)] = mask_index

    mapping_path = os.path.join(outdir, "mask_track_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mask_track_mapping, f, indent=2)

    # Count stats
    total_mappings = sum(len(frame_map) for frame_map in mask_track_mapping.values())
    print(f"✅ Saved mask-track mapping ({total_mappings} mappings across {len(mask_track_mapping)} frames) to {mapping_path}")


def _save_tracking_summary(outdir, tracking_stats, bounding_boxes):
    """Save tracking summary and detailed track data"""
    # Create tracking summary
    summary = {
        'total_tracks': tracking_stats['total_tracks'],
        'frames_processed': len(bounding_boxes),
        'total_detections': sum(len(frame_bboxes) for frame_bboxes in bounding_boxes),
        'avg_tracklet_length': tracking_stats['avg_tracklet_length'],
        'max_tracklet_length': tracking_stats['max_tracklet_length'],
        'tracks': {}
    }
    
    # Add details for each track
    for track_id, detections in tracking_stats['track_detections'].items():
        summary['tracks'][track_id] = {
            'length': len(detections),
            'class_name': detections[0]['class_name'],
            'first_frame': min(d['frame'] for d in detections),
            'last_frame': max(d['frame'] for d in detections),
            'avg_confidence': sum(d['confidence'] for d in detections) / len(detections),
            'frames': [d['frame'] for d in detections]  # List of all frames this track appears in
        }
    
    # Add per-class statistics
    summary['class_statistics'] = {}
    for class_name, class_stats in tracking_stats['class_stats'].items():
        avg_length = np.mean(class_stats['lengths'])
        summary['class_statistics'][class_name] = {
            'track_count': class_stats['count'],
            'avg_track_length': avg_length,
            'min_track_length': min(class_stats['lengths']),
            'max_track_length': max(class_stats['lengths'])
        }
    
    # Save summary
    with open(os.path.join(outdir, "tracking_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Tracking summary saved: {tracking_stats['total_tracks']} tracks")
    
    # Print class statistics
    for class_name, class_info in summary['class_statistics'].items():
        print(f"  {class_name}: {class_info['track_count']} tracks, "
              f"avg {class_info['avg_track_length']:.1f} frames")

def extract_frame_names_from_paths(img_paths):
    """Extract frame names (without extension) from image paths"""
    frame_names = []
    for img_path in img_paths:
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        frame_names.append(frame_name)
    return frame_names
        
def run_inference_with_2d_projection(args):
    """Execute the full inference and visualization pipeline with 2D bbox projection."""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    add_path_to_dust3r(args.model_path)

    from src.dust3r.inference import inference, inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # NEW: Extract frame names from image paths
    frame_names = extract_frame_names_from_paths(img_paths)
    print(f"📝 Extracted frame names: {frame_names[:3]}..." if len(frame_names) > 3 else f"📝 Frame names: {frame_names}")

    # Load masks
    masks_data = None
    if args.mask_dir and os.path.exists(args.mask_dir):
        print(f"Loading masks from {args.mask_dir}...")
        masks_data = load_grounded_sam_masks_overlay(args.mask_dir, img_paths)
        
        total_masks = sum(len(frame_masks) for frame_masks in masks_data.values())
        if total_masks == 0:
            print("\n⚠️  WARNING: No valid masks found!")
            print("Install pycocotools: pip install pycocotools")
            masks_data = None
        else:
            print(f"✓ Successfully loaded {total_masks} masks across {len(masks_data)} frames")
    else:
        print("No mask directory provided. Proceeding without masks.")

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=args.revisit,
        update=True,
    )
    
    # Store original images for coordinate mapping
    # original_images = [view["img"] for view in views]

    print("Loading original images for 2D projection...")
    original_images = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is not None:
            original_images.append(img)
        else:
            print(f"Warning: Could not load {img_path}")
            # Fallback to model image if loading fails
            original_images.append(views[len(original_images)]["img"])

    # NEW: Parse DJI logs if provided
    gimbal_data = None
    if args.dji_log and os.path.exists(args.dji_log):
        print(f"🚁 Loading DJI gimbal data from {args.dji_log}...")
        
        # Extract frame indices we're actually processing
        frame_indices = []
        for img_path in img_paths:
            frame_name = os.path.splitext(os.path.basename(img_path))[0]
            try:
                frame_idx = int(frame_name)
                frame_indices.append(frame_idx)
            except ValueError:
                print(f"⚠️ Could not parse frame index from {frame_name}")
        
        gimbal_data = parse_dji_logs(args.dji_log, frame_indices)
        
        if gimbal_data:
            print(f"✅ Loaded gimbal data for {len(gimbal_data)} frames")
            # TEST: Print some gimbal data to verify it's working
            for frame_idx, data in list(gimbal_data.items())[:3]:  # Show first 3 frames
                print(f"  Frame {frame_idx}: pitch={data['pitch']:.1f}°, roll={data['roll']:.1f}°, alt={data['altitude']:.1f}m")
        else:
            print("⚠️ No gimbal data loaded")
    
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference(views, model, device)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization with overlay backprojection AND 2D projection
    print("Preparing output for visualization...")
    result = prepare_output_overlay_unified(
        outputs=outputs, 
        outdir=args.output_dir, 
        revisit=args.revisit, 
        use_pose=True, 
        masks_data=masks_data, 
        original_images=original_images, 
        model_size=args.size, 
        blend_mode=args.blend_mode, 
        gimbal_data=gimbal_data,
        enable_3d_bboxes=True,      # Enable 3D bounding boxes
        enable_2d_projection=True,   # Enable 2D projection
        frame_names=frame_names     # NEW: Pass frame names
    )
    
    # Handle variable return values
    if len(result) == 8:  # Full feature set: base + bboxes + 2D projection
        pts3ds_other, colors, conf, cam_dict, instance_labels, visualization_results, bounding_boxes, annotated_images = result
    elif len(result) == 7:  # Base + bboxes only
        pts3ds_other, colors, conf, cam_dict, instance_labels, visualization_results, bounding_boxes = result
        annotated_images = None
    else:  # Base only (6 values)
        pts3ds_other, colors, conf, cam_dict, instance_labels, visualization_results = result
        bounding_boxes, annotated_images = None, None

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c if isinstance(c, np.ndarray) else c.cpu().numpy() for c in colors]
    
    edge_colors = [None] * len(pts3ds_to_vis)

    # Create enhanced viewer with toggle functionality
    print("Launching enhanced point cloud viewer...")
    print(f"🎨 Mask visualization modes:")
    print(f"  📷 Original Colors - Natural image colors")
    print(f"  🎨 Mask Overlay - Highlighted instances on original")  
    print(f"  ✨ Mask Highlight - Brightened instances")
    print(f"  🎯 Masks Only - Instance colors with gray background")
    
    if annotated_images is not None:
        print(f"📸 2D Projection: {len(annotated_images)} images with projected 3D bboxes saved!")
    
    track_visualization_data = None
    if bounding_boxes is not None:
        print(f"\n🎨 === PREPARING TRACK VISUALIZATION ===")
        track_trajectories, track_colors, track_info = compute_track_trajectories_and_colors(bounding_boxes)
        
        total_tracks = len(track_trajectories)
        longest_track = max(len(traj) for traj in track_trajectories.values()) if track_trajectories else 0
        print(f"📊 Track Summary: {total_tracks} tracks, longest: {longest_track} detections")
        
        # NEW: Enhanced track analysis with plots
        enhanced_track_analysis(bounding_boxes, track_trajectories, track_colors, track_info, args.output_dir)

    # Post-processing: Apply Kalman re-tracker if selected
    if getattr(args, 'tracker', 'kalman') == 'kalman' and os.path.isdir(args.output_dir):
        print(f"\n🔄 === APPLYING KALMAN RE-TRACKER ===")
        try:
            from wildlift.rt.retracker import ImprovedTracker, KalmanTrack
            import subprocess
            # Run retracker as a post-processing step on the saved outputs
            retrack_cmd = [
                sys.executable, '-m', 'wildlift.rt.retracker',
                '--result_dir', args.output_dir,
                '--source_images', args.seq_path,
                '--output_subfolder', 'retracked'
            ]
            print(f"  Running: {' '.join(retrack_cmd)}")
            subprocess.run(retrack_cmd, check=True)
            print("  Kalman re-tracking complete. Results in: {}/retracked/".format(args.output_dir))
        except Exception as e:
            print(f"  WARNING: Kalman re-tracking failed: {e}")
            print("  Falling back to initial tracking results.")

    # GPS-based pose refinement (if enabled)
    if getattr(args, 'gps_refine', False) and gimbal_data is not None:
        print(f"\n🌍 === GPS-BASED POSE REFINEMENT ===")
        try:
            gps_data = None
            if args.dji_log:
                gps_data = parse_dji_logs_with_gps(args.dji_log, [])
            if gps_data:
                # Get poses from cam_dict
                pr_poses = [torch.tensor(cam_dict[k]['cam_c2w']).unsqueeze(0) for k in sorted(cam_dict.keys())]
                refined_poses = refine_poses_with_gps(pr_poses, gps_data)
                print(f"  GPS refinement applied to {len(refined_poses)} poses")
            else:
                print("  No GPS data available for refinement")
        except Exception as e:
            print(f"  WARNING: GPS refinement failed: {e}")

    viewer = PointCloudViewer(
        model, state_args, pts3ds_to_vis, colors_to_vis, conf, cam_dict,
        device=device, edge_color_list=edge_colors, show_camera=True,
        vis_threshold=args.vis_threshold, size=args.size,
        visualization_modes=visualization_results, bounding_boxes=bounding_boxes
    )

    # NEW: Add simplified track visualization
    # if bounding_boxes is not None:
    #     add_track_visualization_to_viewer(viewer, bounding_boxes, track_trajectories, track_colors, track_info)

    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print("No inputs found! Please provide --seq_path")
        return
    else:
        run_inference_with_2d_projection(args)


if __name__ == "__main__":
    main()