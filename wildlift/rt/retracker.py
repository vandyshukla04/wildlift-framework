#!/usr/bin/env python3
"""
retrack_3d.py — Post-processing re-tracker using 3D Kalman filtering + dormant track re-ID.

Reads existing CUT3R outputs (bbox JSONs, instance labels, camera params) and re-runs
tracking with improved occlusion handling. No CUT3R re-run needed.

Usage:
    python retrack_3d.py --result_dir results/paper_final/thursday/giraffes/gira-1_1 \
                         --source_images /path/to/source/frames \
                         --output_subfolder retracked
"""

import argparse
import json
import os
import glob
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Kalman Filter (constant velocity, 6-state)
# =============================================================================

class KalmanTrack:
    """Per-track 3D Kalman filter with constant-velocity model.
    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """

    def __init__(self, center, track_id, class_name, confidence, dt=1.0):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.dt = dt

        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        self.x[:3] = center

        # State covariance
        self.P = np.eye(6)
        self.P[3:, 3:] *= 10.0  # High initial velocity uncertainty

        # Transition matrix (constant velocity)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)

        # Process noise
        q = 0.05  # Process noise magnitude
        self.Q = np.eye(6) * q
        self.Q[3:, 3:] *= 2.0  # More noise on velocity

        # Measurement noise
        self.R = np.eye(3) * 0.1

        # Track metadata
        self.frames_missing = 0
        self.detection_count = 1
        self.first_frame = -1
        self.last_frame = -1
        self.last_mask = None
        self.last_confidence = confidence

    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    @property
    def predicted_center(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    def update(self, measurement):
        """Update state with measurement [x, y, z]."""
        y = measurement - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.frames_missing = 0
        self.detection_count += 1


# =============================================================================
# Improved Tracker
# =============================================================================

class ImprovedTracker:
    """3D tracker with Kalman prediction and dormant track re-identification."""

    def __init__(self, max_distance=8.0, mask_iou_threshold=0.15,
                 distance_weight=0.7, iou_weight=0.3,
                 max_missing_frames=20, dormant_timeout=100):
        self.max_distance = max_distance
        self.mask_iou_threshold = mask_iou_threshold
        self.distance_weight = distance_weight
        self.iou_weight = iou_weight
        self.max_missing_frames = max_missing_frames
        self.dormant_timeout = dormant_timeout

        self.active_tracks = {}      # track_id -> KalmanTrack
        self.dormant_tracks = {}     # track_id -> KalmanTrack
        self.next_track_id = 0

    def update(self, detections, frame_idx):
        """Process detections for one frame.

        detections: list of dicts with keys: center, class_name, confidence, mask, dimensions, rotation_matrix
        Returns: list of assigned track_ids (same order as detections)
        """
        # First frame — initialize all tracks
        if not self.active_tracks and not self.dormant_tracks:
            return self._initialize_tracks(detections, frame_idx)

        # Predict all active tracks
        for track in self.active_tracks.values():
            track.predict()
        # Also predict dormant tracks (keeps velocity model running)
        for track in self.dormant_tracks.values():
            track.predict()

        if len(detections) == 0:
            self._increment_missing(frame_idx)
            return []

        # Build cost matrix for active tracks
        track_ids = list(self.active_tracks.keys())
        n_det = len(detections)
        n_trk = len(track_ids)

        assignments = {}  # det_idx -> track_id
        unmatched_dets = set(range(n_det))
        matched_tracks = set()

        if n_trk > 0:
            cost_matrix = np.full((n_det, n_trk), 1e6)

            for i, det in enumerate(detections):
                for j, tid in enumerate(track_ids):
                    track = self.active_tracks[tid]

                    # Class filter
                    if det['class_name'].lower() != track.class_name.lower():
                        continue

                    # 3D distance (predicted vs measured)
                    dist = np.linalg.norm(det['center'] - track.predicted_center)
                    if dist > self.max_distance:
                        continue

                    # Mask IoU — soft gate: only reject if distance is also large
                    iou = self._compute_mask_iou(det.get('mask'), track.last_mask)
                    if iou < self.mask_iou_threshold and track.last_mask is not None and det.get('mask') is not None:
                        # Allow match if 3D distance is small (Kalman prediction is confident)
                        if dist > 1.0:
                            continue

                    dist_cost = dist / self.max_distance
                    iou_cost = 1.0 - iou
                    cost_matrix[i, j] = self.distance_weight * dist_cost + self.iou_weight * iou_cost

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e5:
                    tid = track_ids[c]
                    assignments[r] = tid
                    unmatched_dets.discard(r)
                    matched_tracks.add(tid)

                    # Update track
                    track = self.active_tracks[tid]
                    track.update(detections[r]['center'])
                    track.last_mask = detections[r].get('mask')
                    track.last_confidence = detections[r]['confidence']
                    track.last_frame = frame_idx

        # Try to re-identify unmatched detections from dormant tracks
        still_unmatched = set()
        for det_idx in unmatched_dets:
            reid_tid = self._try_reidentify(detections[det_idx], frame_idx)
            if reid_tid is not None:
                assignments[det_idx] = reid_tid
            else:
                still_unmatched.add(det_idx)

        # Create new tracks for remaining unmatched detections
        for det_idx in still_unmatched:
            det = detections[det_idx]
            tid = self.next_track_id
            self.next_track_id += 1
            track = KalmanTrack(det['center'], tid, det['class_name'], det['confidence'])
            track.first_frame = frame_idx
            track.last_frame = frame_idx
            track.last_mask = det.get('mask')
            self.active_tracks[tid] = track
            assignments[det_idx] = tid

        # Handle unmatched active tracks
        for tid in list(self.active_tracks.keys()):
            if tid not in matched_tracks:
                self.active_tracks[tid].frames_missing += 1
                if self.active_tracks[tid].frames_missing > self.max_missing_frames:
                    self._move_to_dormant(tid)

        # Expire old dormant tracks
        for tid in list(self.dormant_tracks.keys()):
            if self.dormant_tracks[tid].frames_missing > self.dormant_timeout:
                del self.dormant_tracks[tid]

        # Return track IDs in detection order
        return [assignments.get(i, -1) for i in range(n_det)]

    def _initialize_tracks(self, detections, frame_idx):
        track_ids = []
        for det in detections:
            tid = self.next_track_id
            self.next_track_id += 1
            track = KalmanTrack(det['center'], tid, det['class_name'], det['confidence'])
            track.first_frame = frame_idx
            track.last_frame = frame_idx
            track.last_mask = det.get('mask')
            self.active_tracks[tid] = track
            track_ids.append(tid)
        return track_ids

    def _move_to_dormant(self, track_id):
        track = self.active_tracks.pop(track_id)
        self.dormant_tracks[track_id] = track

    def _try_reidentify(self, detection, frame_idx):
        """Try to match a detection against dormant tracks."""
        best_tid = None
        best_cost = 1e6

        for tid, track in self.dormant_tracks.items():
            if detection['class_name'].lower() != track.class_name.lower():
                continue

            dist = np.linalg.norm(detection['center'] - track.predicted_center)
            # Use a larger distance threshold for re-ID (objects may have moved)
            if dist > self.max_distance * 2.0:
                continue

            cost = dist / self.max_distance
            if cost < best_cost:
                best_cost = cost
                best_tid = tid

        if best_tid is not None and best_cost < 1.5:  # Generous threshold for re-ID
            # Reactivate track
            track = self.dormant_tracks.pop(best_tid)
            track.update(detection['center'])
            track.last_mask = detection.get('mask')
            track.last_confidence = detection['confidence']
            track.last_frame = frame_idx
            self.active_tracks[best_tid] = track
            return best_tid

        return None

    def _increment_missing(self, frame_idx):
        for tid in list(self.active_tracks.keys()):
            self.active_tracks[tid].frames_missing += 1
            if self.active_tracks[tid].frames_missing > self.max_missing_frames:
                self._move_to_dormant(tid)
        for tid in list(self.dormant_tracks.keys()):
            if self.dormant_tracks[tid].frames_missing > self.dormant_timeout:
                del self.dormant_tracks[tid]

    @staticmethod
    def _compute_mask_iou(mask1, mask2):
        if mask1 is None or mask2 is None:
            return 0.0
        if mask1.shape != mask2.shape:
            return 0.0
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        if union == 0:
            return 0.0
        return intersection / union


# =============================================================================
# Data Loading
# =============================================================================

def load_results(result_dir):
    """Load per-frame bounding boxes, instance labels, and camera params."""
    bbox_dir = os.path.join(result_dir, "bounding_boxes")
    label_dir = os.path.join(result_dir, "instance_labels")
    camera_dir = os.path.join(result_dir, "camera")

    bbox_files = sorted(glob.glob(os.path.join(bbox_dir, "*.json")))
    if not bbox_files:
        raise ValueError(f"No bbox JSONs found in {bbox_dir}")

    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in bbox_files]

    # Load original mask_track_mapping to get correct bbox→instance mapping
    orig_mapping_path = os.path.join(result_dir, "mask_track_mapping.json")
    orig_mapping = None
    if os.path.exists(orig_mapping_path):
        with open(orig_mapping_path) as f:
            orig_mapping = json.load(f)

    frames = []
    for fname, bbox_file in zip(frame_names, bbox_files):
        with open(bbox_file) as f:
            bboxes = json.load(f)

        # Load instance labels
        label_path = os.path.join(label_dir, f"{fname}.npy")
        instance_labels = np.load(label_path) if os.path.exists(label_path) else None

        # Load camera
        cam_path = os.path.join(camera_dir, f"{fname}.npz")
        cam_data = np.load(cam_path) if os.path.exists(cam_path) else None

        # Build bbox→mask_index mapping from original mask_track_mapping
        # Original mapping: {track_id_str: mask_index} where mask_index is 0-based det index
        # Bbox JSON stores track_id per bbox, so we can reverse-lookup
        bbox_to_instance = {}
        if orig_mapping and fname in orig_mapping:
            frame_map = orig_mapping[fname]
            for tid_str, mask_idx in frame_map.items():
                orig_tid = int(tid_str)
                # Find which bbox has this track_id
                for bi, b in enumerate(bboxes):
                    if b.get('track_id') == orig_tid:
                        # mask_idx is the 0-based detection index in original
                        # instance label value = mask_idx + 1
                        bbox_to_instance[bi] = mask_idx + 1
                        break

        # Build detections
        detections = []
        for bi, bbox in enumerate(bboxes):
            center = np.array(bbox['center'])
            det = {
                'center': center,
                'dimensions': np.array(bbox['dimensions']),
                'rotation_matrix': np.array(bbox['rotation_matrix']),
                'class_name': bbox['class_name'],
                'confidence': bbox['confidence'],
                'original_track_id': bbox.get('track_id', -1),
                'mask': None,
            }

            # Assign mask using the correct mapping
            if instance_labels is not None:
                if bi in bbox_to_instance:
                    inst_val = bbox_to_instance[bi]
                    det['mask'] = (instance_labels == inst_val).astype(np.uint8)
                    det['instance_label_value'] = inst_val
                else:
                    # Fallback: assume bbox[i] → instance i+1
                    inst_val = bi + 1
                    if inst_val in np.unique(instance_labels):
                        det['mask'] = (instance_labels == inst_val).astype(np.uint8)
                        det['instance_label_value'] = inst_val

            detections.append(det)

        frames.append({
            'name': fname,
            'detections': detections,
            'instance_labels': instance_labels,
            'camera': cam_data,
        })

    return frames, frame_names


# =============================================================================
# Rendering
# =============================================================================

COLOR_PALETTE = [
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


def get_track_color(track_id, all_track_ids):
    """Get consistent color for a track ID."""
    sorted_ids = sorted(all_track_ids)
    idx = sorted_ids.index(track_id) if track_id in sorted_ids else track_id
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]


def get_bbox_corners(center, dimensions, rotation_matrix):
    """Get 8 corners of a 3D bounding box."""
    l, w, h = dimensions
    corners_local = np.array([
        [-l/2, -w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2],
        [-l/2,  w/2,  h/2],
    ])
    corners_world = (np.array(rotation_matrix) @ corners_local.T).T + np.array(center)
    return corners_world


def draw_3d_bbox_wireframe(img, corners_2d, color_bgr, thickness=2):
    """Draw 3D bounding box wireframe."""
    try:
        c = corners_2d.astype(int)
        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom
            (4,5),(5,6),(6,7),(7,4),  # top
            (0,4),(1,5),(2,6),(3,7),  # verticals
        ]
        for a, b in edges:
            cv2.line(img, tuple(c[a]), tuple(c[b]), color_bgr, thickness)
    except Exception:
        pass


def render_annotated_frame(source_img, detections, track_ids, all_track_ids, cam_data):
    """Render annotated frame with track-colored 3D bboxes."""
    img = source_img.copy()
    if cam_data is None:
        return img

    pose = cam_data['pose']
    intrinsics = cam_data['intrinsics']
    K = intrinsics

    for det, tid in zip(detections, track_ids):
        if tid < 0:
            continue

        color_rgb = get_track_color(tid, all_track_ids)
        color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])

        corners_3d = get_bbox_corners(det['center'], det['dimensions'], det['rotation_matrix'])

        try:
            # Transform to camera coords
            corners_h = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
            corners_cam = (np.linalg.inv(pose) @ corners_h.T)[:3].T

            if np.any(corners_cam[:, 2] <= 0):
                continue

            corners_2d_h = (K @ corners_cam.T).T
            corners_2d = corners_2d_h[:, :2] / corners_2d_h[:, 2:3]

            h, w = img.shape[:2]
            if np.any(corners_2d < -w) or np.any(corners_2d > 2*w):
                continue

            draw_3d_bbox_wireframe(img, corners_2d, color_bgr)

            # Label
            center_2d = np.mean(corners_2d, axis=0).astype(int)
            center_2d[0] = max(10, min(center_2d[0], w - 100))
            center_2d[1] = max(20, min(center_2d[1], h - 10))
            label = f"T{tid}: {det['class_name']}"
            cv2.putText(img, label, tuple(center_2d), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color_bgr, 1, cv2.LINE_AA)
        except Exception:
            continue

    return img


# =============================================================================
# Trajectory Plots
# =============================================================================

def save_trajectory_plots(trajectories, all_track_ids, output_dir):
    """Save 2D and 3D trajectory plots."""
    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for tid, traj in trajectories.items():
        color = get_track_color(tid, all_track_ids)
        centers = np.array([t['center'] for t in traj])
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                color=color, linewidth=2, label=f"T{tid}")
        ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2],
                  color=color, s=50, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Track Trajectories (Retracked)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'track_trajectories_3d.png'), dpi=150)
    plt.close()

    # 2D plot (top-down XZ)
    fig, ax = plt.subplots(figsize=(12, 8))
    for tid, traj in trajectories.items():
        color = get_track_color(tid, all_track_ids)
        centers = np.array([t['center'] for t in traj])
        ax.plot(centers[:, 0], centers[:, 2], color=color, linewidth=2, label=f"T{tid}")
        ax.scatter(centers[0, 0], centers[0, 2], color=color, s=50, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('2D Track Trajectories — Top Down (Retracked)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'track_trajectories_2d.png'), dpi=150)
    plt.close()


# =============================================================================
# Output
# =============================================================================

def save_outputs(output_dir, frames, frame_names, all_track_assignments,
                 all_track_ids, trajectories, source_images_dir=None):
    """Save all outputs to the retracked subfolder."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotated_2d'), exist_ok=True)

    # 1. mask_track_mapping.json
    mask_track_mapping = {}
    for frame_idx, frame in enumerate(frames):
        fname = frame_names[frame_idx]
        track_ids = all_track_assignments[frame_idx]
        mapping = {}
        for det_idx, tid in enumerate(track_ids):
            if tid >= 0:
                det = frame['detections'][det_idx]
                # Use the actual instance label value (0-based index for mapping)
                inst_val = det.get('instance_label_value')
                if inst_val is not None:
                    mapping[str(tid)] = inst_val - 1  # convert to 0-based mask index
                else:
                    mapping[str(tid)] = det_idx
        mask_track_mapping[fname] = mapping

    with open(os.path.join(output_dir, 'mask_track_mapping.json'), 'w') as f:
        json.dump(mask_track_mapping, f, indent=2)

    # 2. tracking_summary.json
    track_detections = {}
    for frame_idx, frame in enumerate(frames):
        track_ids = all_track_assignments[frame_idx]
        for det_idx, tid in enumerate(track_ids):
            if tid < 0:
                continue
            if tid not in track_detections:
                track_detections[tid] = {
                    'length': 0,
                    'class_name': frame['detections'][det_idx]['class_name'],
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'avg_confidence': 0.0,
                    'frames': [],
                }
            td = track_detections[tid]
            td['length'] += 1
            td['last_frame'] = frame_idx
            td['frames'].append(frame_idx)
            td['avg_confidence'] += frame['detections'][det_idx]['confidence']

    for tid, td in track_detections.items():
        if td['length'] > 0:
            td['avg_confidence'] /= td['length']

    lengths = [td['length'] for td in track_detections.values()]
    summary = {
        'total_tracks': len(track_detections),
        'frames_processed': len(frames),
        'total_detections': sum(lengths),
        'avg_tracklet_length': float(np.mean(lengths)) if lengths else 0,
        'max_tracklet_length': max(lengths) if lengths else 0,
        'tracks': {str(k): v for k, v in track_detections.items()},
    }

    with open(os.path.join(output_dir, 'tracking_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # 3. Trajectory plots
    save_trajectory_plots(trajectories, all_track_ids, output_dir)

    # 4. Annotated 2D frames
    if source_images_dir:
        src_exts = ['*.png', '*.jpg', '*.jpeg']
        src_files = []
        for ext in src_exts:
            src_files.extend(glob.glob(os.path.join(source_images_dir, ext)))
        src_files = sorted(src_files)

        # Try to match source images to frame names
        src_by_name = {}
        for sf in src_files:
            name = os.path.splitext(os.path.basename(sf))[0]
            src_by_name[name] = sf

        for frame_idx, frame in enumerate(frames):
            fname = frame_names[frame_idx]
            src_path = src_by_name.get(fname)
            if src_path is None:
                # Try matching by index
                if frame_idx < len(src_files):
                    src_path = src_files[frame_idx]

            if src_path is None:
                continue

            src_img = cv2.imread(src_path)
            if src_img is None:
                continue

            # Resize source to match instance label size if needed
            if frame['instance_labels'] is not None:
                lh, lw = frame['instance_labels'].shape[:2]
                sh, sw = src_img.shape[:2]
                if (sh, sw) != (lh, lw):
                    src_img = cv2.resize(src_img, (lw, lh))

            track_ids = all_track_assignments[frame_idx]
            annotated = render_annotated_frame(
                src_img, frame['detections'], track_ids, all_track_ids, frame['camera']
            )

            out_path = os.path.join(output_dir, 'annotated_2d', f'{fname}_tracked.png')
            cv2.imwrite(out_path, annotated)

    print(f"Outputs saved to {output_dir}")
    print(f"  Total tracks: {summary['total_tracks']}")
    print(f"  Avg tracklet length: {summary['avg_tracklet_length']:.1f}")
    print(f"  Max tracklet length: {summary['max_tracklet_length']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Re-track CUT3R outputs with 3D Kalman + dormant re-ID')
    parser.add_argument('--result_dir', required=True, help='Path to CUT3R result directory')
    parser.add_argument('--source_images', default=None, help='Path to source image frames')
    parser.add_argument('--output_subfolder', default='retracked', help='Subfolder name for outputs')
    parser.add_argument('--max_missing_frames', type=int, default=20)
    parser.add_argument('--dormant_timeout', type=int, default=100)
    parser.add_argument('--distance_weight', type=float, default=0.7)
    parser.add_argument('--iou_weight', type=float, default=0.3)
    parser.add_argument('--max_distance', type=float, default=8.0)
    args = parser.parse_args()

    output_dir = os.path.join(args.result_dir, args.output_subfolder)

    print(f"Loading results from {args.result_dir}...")
    frames, frame_names = load_results(args.result_dir)
    print(f"Loaded {len(frames)} frames")

    # Run improved tracker
    tracker = ImprovedTracker(
        max_distance=args.max_distance,
        mask_iou_threshold=0.15,
        distance_weight=args.distance_weight,
        iou_weight=args.iou_weight,
        max_missing_frames=args.max_missing_frames,
        dormant_timeout=args.dormant_timeout,
    )

    all_track_assignments = []
    all_track_ids = set()

    for frame_idx, frame in enumerate(frames):
        track_ids = tracker.update(frame['detections'], frame_idx)
        all_track_assignments.append(track_ids)
        for tid in track_ids:
            if tid >= 0:
                all_track_ids.add(tid)

    all_track_ids = sorted(all_track_ids)
    print(f"Tracking complete: {len(all_track_ids)} tracks")

    # Build trajectories
    trajectories = {}
    for frame_idx, frame in enumerate(frames):
        track_ids = all_track_assignments[frame_idx]
        for det_idx, tid in enumerate(track_ids):
            if tid < 0:
                continue
            if tid not in trajectories:
                trajectories[tid] = []
            trajectories[tid].append({
                'frame': frame_idx,
                'center': frame['detections'][det_idx]['center'],
                'confidence': frame['detections'][det_idx]['confidence'],
            })

    # Save outputs
    save_outputs(output_dir, frames, frame_names, all_track_assignments,
                 all_track_ids, trajectories, args.source_images)


if __name__ == '__main__':
    main()
