#!/usr/bin/env python3
"""
3D Bbox Editor (v6)

Features:
- Edit 3D bounding boxes (position, rotation, dimensions)
- Label semantic faces (front, top, left) for re-identification
- Optimize dimensions while maintaining species proportions
- Track-specific point cloud highlighting via mask backprojection
- Fixed interpolation to include keyframe 2
- Ground plane detection and bbox grounding to point clouds
- NEW: Interpolate Semantic Faces Only - propagate/interpolate semantic labels
        between keyframes without changing bbox positions/dimensions

Usage:
    python annotator_tool_v6.py --auto_bboxes results/DIR/bounding_boxes \
        --output corrected_bboxes \
        --mask_dir examples/wd_data/zebras/scene1/grounded-sam
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import viser

# Simple bbox class
class BBox3D:
    def __init__(self, center, dimensions, rotation_matrix, class_name, track_id, frame_idx,
                 confidence=None, instance_id=None, persistent_instance_id=None):
        self.center = np.array(center)
        self.dimensions = np.array(dimensions)
        self.rotation_matrix = np.array(rotation_matrix)
        self.class_name = class_name
        self.track_id = track_id
        self.frame_idx = frame_idx
        # Fields needed for demo_masks compatibility
        self.confidence = confidence if confidence is not None else 1.0
        self.instance_id = instance_id if instance_id is not None else -1
        self.persistent_instance_id = persistent_instance_id if persistent_instance_id is not None else -1

    def get_corners(self):
        """Get 8 corners"""
        l, w, h = self.dimensions
        corners_local = np.array([
            [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
            [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
            [l/2, w/2, h/2], [-l/2, w/2, h/2]
        ])
        return (self.rotation_matrix @ corners_local.T).T + self.center

    def get_edges(self):
        """Get edge indices for wireframe"""
        return [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
        ]

    def get_faces(self):
        """Get 6 faces of the bbox as lists of corner indices and their centers"""
        faces = {
            0: ([0, 1, 5, 4], 'face_0'),  # Front (along +X)
            1: ([2, 3, 7, 6], 'face_1'),  # Back (along -X)
            2: ([0, 3, 7, 4], 'face_2'),  # Left (along -Y)
            3: ([1, 2, 6, 5], 'face_3'),  # Right (along +Y)
            4: ([4, 5, 6, 7], 'face_4'),  # Top (along +Z)
            5: ([0, 1, 2, 3], 'face_5')   # Bottom (along -Z)
        }

        corners = self.get_corners()
        face_data = {}

        for face_id, (corner_ids, name) in faces.items():
            face_corners = corners[corner_ids]
            face_center = face_corners.mean(axis=0)
            face_normal = self._compute_face_normal(face_corners)

            face_data[face_id] = {
                'corners': face_corners,
                'center': face_center,
                'normal': face_normal,
                'name': name
            }

        return face_data

    def _compute_face_normal(self, face_corners):
        """Compute outward normal for a face"""
        # Use first 3 corners to compute normal
        v1 = face_corners[1] - face_corners[0]
        v2 = face_corners[2] - face_corners[0]
        normal = np.cross(v1, v2)
        return normal / (np.linalg.norm(normal) + 1e-8)


class SimpleBBoxEditor:
    """Minimal bbox editor - no callback loops!"""

    # Species-specific body proportions (length/height, width/height)
    SPECIES_PROPORTIONS = {
        'elephant': {'length': 1.72, 'width': 0.78},  # African elephant
        'rhino': {'length': 1.80, 'width': 0.85},     # Stockier build
        'zebra': {'length': 1.65, 'width': 0.55},     # Horse-like proportions
        'giraffe': {'length': 1.10, 'width': 0.50},   # Tall, narrow body
        'default': {'length': 1.72, 'width': 0.78},   # Fallback to elephant
    }

    # Ground detection configuration
    GROUND_CONFIG = {
        'search_radius': 3.0,          # Search radius around bbox center (meters)
        'ransac_threshold': 0.05,      # RANSAC inlier distance threshold (meters)
        'ransac_iterations': 500,      # Number of RANSAC iterations
        'min_inliers': 50,             # Minimum points to consider valid ground
        'ground_normal_tolerance': 0.4, # Max deviation from vertical (radians, ~23 degrees)
        'percentile_fallback': 10,     # Percentile for fallback ground estimation
        'height_axis': 'auto',         # 'auto', 'y', or 'z' - which axis is vertical
    }

    def __init__(self, auto_bboxes_dir, output_dir, point_clouds_dir=None, images_dir=None,
                 mask_dir=None, reload_annotations=True, port=8080):
        self.auto_bboxes_dir = Path(auto_bboxes_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store results dir for saving corrected bboxes
        self.results_dir = self.auto_bboxes_dir.parent

        # Try to auto-find point clouds
        if point_clouds_dir is None:
            pc_world = self.results_dir / "point_clouds_world"
            pc_local = self.results_dir / "point_clouds"
            if pc_world.exists():
                self.point_clouds_dir = pc_world
            elif pc_local.exists():
                self.point_clouds_dir = pc_local
            else:
                self.point_clouds_dir = None
        else:
            self.point_clouds_dir = Path(point_clouds_dir)

        # Auto-find images directory for 2D visualization
        if images_dir is None:
            images_path = self.results_dir / "images"
            self.images_dir = images_path if images_path.exists() else None
        else:
            self.images_dir = Path(images_dir)

        # Auto-find grounded-sam mask directory (for track highlighting)
        if mask_dir is None:
            self.mask_dir = self._auto_find_mask_dir()
        else:
            self.mask_dir = Path(mask_dir)

        # Print mask dir status
        if self.mask_dir and self.mask_dir.exists():
            print(f"✓ Mask directory: {self.mask_dir}")
        else:
            print(f"⚠️ Mask directory not found (track highlighting may be limited)")

        # Auto-find depths directory for generating point clouds if needed
        # Check both "depths" (plural) and "depth" (singular)
        depths_path = self.results_dir / "depths"
        depth_path = self.results_dir / "depth"
        if depths_path.exists():
            self.depths_dir = depths_path
        elif depth_path.exists():
            self.depths_dir = depth_path
        else:
            self.depths_dir = None

        # NEW v5: Load mask-track mapping for accurate track highlighting
        # Load this BEFORE bboxes so we can remap track IDs if retracked
        self.mask_track_mapping, self._track_remap = self._load_mask_track_mapping()

        # Load bbox data (needed for frame_indices), remap track IDs if retracked
        self.auto_bboxes = self._load_bboxes()
        self.frame_indices = sorted(self.auto_bboxes.keys())
        self.corrections = {}  # track_id -> {frame_idx -> bbox}

        # Semantic face labels: track_id -> {frame_idx -> {'front': face_id, 'top': face_id, 'left': face_id}}
        self.semantic_faces = {}

        # Load previous session data if available and requested
        self.reload_annotations = reload_annotations
        if reload_annotations:
            self._load_previous_session()
        else:
            print("ℹ️ Skipping reload of previous annotations (--no_reload specified)")

        # Face visualization handles
        self.face_handles = []
        # Auto-save timer
        self.auto_save_timer = None
        self.last_save_time = None

        # Try to load camera parameters (needs frame_indices)
        self.cam_dict = self._load_camera_params()

        # Point clouds (loaded on demand)
        self.point_clouds = {}  # frame_idx -> (points, colors)

        # NEW in v4: Track-specific highlighted point cloud handle
        self.track_pc_handle = None
        self.highlight_track_points = True  # Toggle for highlighting

        # Viser server (use same pattern as viser_utils.py)
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")  # CRITICAL!

        # State
        self.current_frame_idx = 0
        self.current_frame = self.frame_indices[0] if self.frame_indices else 0
        self.selected_track = None

        # Scene handles - organized by type
        self.pc_handle = None
        self.bbox_handles = {}  # bbox_id -> list of handles
        self.original_bbox_handles = []  # Original bboxes for comparison
        self.gizmo_handle = None

        # Flags to prevent update loops
        self.updating_sliders = False
        self.updating_dropdowns = False

        # Keyframe interpolation state
        self.keyframe_1 = None  # (frame_idx, track_id, BBox3D, semantic_labels_dict)
        self.keyframe_2 = None  # (frame_idx, track_id, BBox3D, semantic_labels_dict)
        self.pending_interpolation = None  # Stores params when confirmation needed

        # Setup UI
        self._setup_ui()

        # Initial render
        self._render_frame()

        print(f"\n{'='*60}")
        print(f"BBOX EDITOR v6 READY (with Semantic Face Interpolation)")
        print(f"{'='*60}")
        print(f"Open: http://localhost:{port}")
        print(f"Frames: {len(self.frame_indices)}")
        print(f"NEW: Use 'Interpolate Semantic Faces' to propagate labels only")
        print(f"{'='*60}\n")

    def _load_camera_params(self):
        """Load camera parameters from results dir"""
        try:
            # Check for single camera_parameters.npz file first
            cam_file = self.results_dir / "camera_parameters.npz"
            if cam_file.exists():
                data = np.load(cam_file)
                print(f"✓ Loaded camera parameters from {cam_file}")
                return {
                    'focal': data['focals'],
                    'pp': data['principal_points'],
                    'R': data['Rs'],
                    't': data['ts']
                }

            # Otherwise, load from individual camera/*.npz files (demo_masks.py format)
            camera_dir = self.results_dir / "camera"
            if camera_dir.exists():
                print(f"✓ Loading camera parameters from {camera_dir}")

                # Load camera params for each frame
                focals = []
                pps = []
                Rs = []
                ts = []

                for frame_idx in self.frame_indices:
                    cam_file = camera_dir / f"{frame_idx}.npz"
                    if not cam_file.exists():
                        # Try with 4-digit zero-padded name (common format)
                        cam_file = camera_dir / f"{frame_idx:04d}.npz"
                    if not cam_file.exists():
                        # Try with 6-digit zero-padded name
                        cam_file = camera_dir / f"{frame_idx:06d}.npz"

                    if cam_file.exists():
                        data = np.load(cam_file)
                        # Extract from pose and intrinsics
                        intrinsics = data['intrinsics']  # [3, 3]
                        pose = data['pose']  # [4, 4] c2w matrix

                        # Extract focal and principal point from intrinsics
                        focal = intrinsics[0, 0]  # fx (assuming square pixels)
                        pp = np.array([intrinsics[0, 2], intrinsics[1, 2]])

                        # Extract R and t from pose (c2w)
                        R = pose[:3, :3]
                        t = pose[:3, 3]

                        focals.append(focal)
                        pps.append(pp)
                        Rs.append(R)
                        ts.append(t)
                    else:
                        print(f"⚠️ Missing camera file for frame {frame_idx}")
                        return None

                if len(focals) == len(self.frame_indices):
                    return {
                        'focal': np.array(focals),
                        'pp': np.array(pps),
                        'R': np.array(Rs),
                        't': np.array(ts)
                    }
                else:
                    print(f"⚠️ Incomplete camera data: {len(focals)}/{len(self.frame_indices)} frames")
                    return None

        except Exception as e:
            print(f"⚠️ Could not load camera parameters: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _auto_find_mask_dir(self):
        """Auto-discover grounded-sam mask directory from results path

        Strategy: Parse results_dir to extract dataset/scene, then look for:
            examples/wd_data/{species}/{scene}/grounded-sam/

        Returns:
            Path object or None if not found
        """
        parts = self.results_dir.parts

        # Look for "results" directory in path
        if "results" in parts:
            idx = parts.index("results")
            if len(parts) > idx + 2:
                dataset = parts[idx + 1]
                scene = parts[idx + 2]

                # Map dataset name to species directory
                species_map = {
                    'zebra': 'zebras', 'zebras': 'zebras',
                    'rhino': 'rhinos', 'rhinos': 'rhinos',
                    'elephant': 'elephants', 'elephants': 'elephants',
                    'giraffe': 'giraffes', 'giraffes': 'giraffes',
                }

                species = species_map.get(dataset.lower())
                if species:
                    mask_path = Path(f"examples/wd_data/{species}/{scene}/grounded-sam")
                    if mask_path.exists():
                        print(f"✓ Auto-detected mask directory: {mask_path}")
                        return mask_path

        print("⚠️ Could not auto-detect mask directory. Use --mask_dir to specify.")
        return None

    def _load_mask_for_frame(self, frame_idx, class_name=None, current_bbox=None, track_id=None):
        """Load grounded-sam mask for a specific frame and track/class.

        NEW in v5: Uses mask_track_mapping for direct lookup when track_id is provided,
        eliminating IoU-based heuristics that fail with close animals.

        Args:
            frame_idx: Frame index to load
            class_name: Class name to match (e.g., 'zebra', 'elephant')
            current_bbox: Current BBox3D (for selecting best annotation if multiple exist)
            track_id: Track ID for direct mask lookup via mask_track_mapping

        Returns:
            Binary mask (H, W) numpy array, or None if not found
        """
        if self.mask_dir is None:
            return None

        # NEW v5: Direct lookup via mask_track_mapping (most reliable)
        if track_id is not None and self.mask_track_mapping:
            mask_idx = self.mask_track_mapping.get(str(frame_idx), {}).get(str(track_id))
            if mask_idx is not None:
                mask = self._decode_mask_by_index(frame_idx, int(mask_idx))
                if mask is not None:
                    return mask
                # Fall through to legacy method if decode fails

        # FALLBACK: Legacy IoU-based selection (for old results without mapping)
        json_file = self.mask_dir / f"{frame_idx}_results.json"
        if not json_file.exists():
            return None

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Filter annotations by class_name
            matching_annotations = [
                ann for ann in data['annotations']
                if class_name and ann['class_name'].lower() == class_name.lower()
            ]

            if not matching_annotations:
                return None

            # If multiple annotations, pick closest to current bbox projection
            if len(matching_annotations) > 1 and current_bbox is not None:
                selected_ann = self._select_best_mask_annotation(
                    matching_annotations, current_bbox, frame_idx
                )
            else:
                selected_ann = matching_annotations[0]

            # Decode RLE mask
            try:
                from pycocotools import mask as mask_utils
            except ImportError:
                print("⚠️ pycocotools not installed - track highlighting disabled")
                return None

            rle = selected_ann['segmentation']
            binary_mask = mask_utils.decode(rle)  # Returns (H, W) binary array
            return binary_mask

        except Exception as e:
            return None

    def _select_best_mask_annotation(self, annotations, current_bbox, frame_idx):
        """Select annotation that best matches current bbox 2D projection

        Strategy: Project 3D bbox to 2D, compute IoU with each mask's 2D bbox
        """
        if not self.cam_dict:
            return annotations[0]

        frame_to_idx = {frame: idx for idx, frame in enumerate(self.frame_indices)}
        if frame_idx not in frame_to_idx:
            return annotations[0]

        idx = frame_to_idx[frame_idx]
        focal = self.cam_dict['focal'][idx]
        pp = self.cam_dict['pp'][idx]
        R_mat = self.cam_dict['R'][idx]
        t = self.cam_dict['t'][idx]

        # Build camera matrices
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_mat
        camera_pose[:3, 3] = t

        # Get image dimensions
        img_h, img_w = annotations[0]['segmentation']['size']

        # Scale factors (dust3r: 288x512, masks: original size)
        model_h, model_w = 288, 512
        scale_x = img_w / model_w
        scale_y = img_h / model_h

        K = np.array([
            [focal * scale_x, 0, pp[0] * scale_x],
            [0, focal * scale_x, pp[1] * scale_y],
            [0, 0, 1]
        ])

        # Project 3D bbox to 2D
        corners_3d = current_bbox.get_corners()
        corners_cam = (np.linalg.inv(camera_pose) @
                       np.concatenate([corners_3d, np.ones((8, 1))], axis=1).T)[:3].T
        corners_2d = (K @ corners_cam.T).T
        corners_2d = corners_2d[:, :2] / corners_2d[:, 2:]

        # Get 2D bbox from projected corners
        x_min, y_min = corners_2d.min(axis=0)
        x_max, y_max = corners_2d.max(axis=0)
        projected_bbox = [x_min, y_min, x_max, y_max]

        # Compute IoU with each annotation
        best_iou = 0
        best_ann = annotations[0]

        for ann in annotations:
            mask_bbox = ann['bbox']  # [x1, y1, x2, y2] xyxy
            iou = self._compute_bbox_2d_iou(projected_bbox, mask_bbox)
            if iou > best_iou:
                best_iou = iou
                best_ann = ann

        return best_ann

    def _compute_bbox_2d_iou(self, bbox1, bbox2):
        """Compute IoU between two 2D bboxes in xyxy format"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-8)

    def _load_bboxes(self):
        """Load auto bboxes from JSON files.

        If retracked mapping is available, remaps bbox track_ids from the
        original (fragmented) track labels to the retracked (corrected) ones.
        """
        remap = self._track_remap  # per-frame {old_track: new_track} or None
        remapped_count = 0

        bboxes = {}
        for json_file in sorted(self.auto_bboxes_dir.glob("*.json")):
            # Skip non-numeric JSON files (e.g., mask_track_mapping.json)
            if not json_file.stem.isdigit():
                continue
            frame_idx = int(json_file.stem)
            with open(json_file) as f:
                data = json.load(f)

            # Get frame-level remap if available
            frame_remap = remap.get(str(frame_idx), {}) if remap else {}

            frame_bboxes = []
            for bbox_dict in data:
                track_id = bbox_dict['track_id']

                # Remap track_id if retracked mapping provides a new label
                if frame_remap:
                    new_track = frame_remap.get(str(track_id))
                    if new_track is not None:
                        track_id = int(new_track)
                        remapped_count += 1

                bbox = BBox3D(
                    center=bbox_dict['center'],
                    dimensions=bbox_dict['dimensions'],
                    rotation_matrix=bbox_dict['rotation_matrix'],
                    class_name=bbox_dict['class_name'],
                    track_id=track_id,
                    frame_idx=frame_idx,
                    confidence=bbox_dict.get('confidence', 1.0),
                    instance_id=bbox_dict.get('instance_id', -1),
                    persistent_instance_id=bbox_dict.get('persistent_instance_id', -1)
                )
                frame_bboxes.append(bbox)

            bboxes[frame_idx] = frame_bboxes

        print(f"Loaded {len(bboxes)} frames")
        if remapped_count > 0:
            print(f"  Remapped {remapped_count} bbox track IDs to retracked labels")
        return bboxes

    def _load_mask_track_mapping(self):
        """Load mask-track mapping from results dir if available.

        This mapping file (mask_track_mapping.json) is generated by demo_masks.py
        and provides a direct lookup from track_id to mask annotation index,
        eliminating the need for error-prone IoU-based mask selection.

        Prefers retracked/mask_track_mapping.json (3D Kalman filter results)
        over the original if available, as it has more accurate track labels.

        Returns:
            tuple: (mapping, track_remap) where:
                - mapping: {frame_idx: {track_id: mask_index}} or None
                - track_remap: {frame_idx: {old_track_id: new_track_id}} or None
                  (only set when using retracked mapping, to remap bbox track IDs)
        """
        retracked_file = self.results_dir / "retracked" / "mask_track_mapping.json"
        mapping_file = self.results_dir / "mask_track_mapping.json"

        if retracked_file.exists() and mapping_file.exists():
            try:
                with open(retracked_file) as f:
                    retracked_mapping = json.load(f)
                with open(mapping_file) as f:
                    original_mapping = json.load(f)

                # Build per-frame old_track_id -> new_track_id remap
                # by matching through shared mask indices
                track_remap = {}
                for frame_id in original_mapping:
                    if frame_id not in retracked_mapping:
                        continue
                    orig_map = original_mapping[frame_id]   # {old_track: mask_idx}
                    retk_map = retracked_mapping[frame_id]  # {new_track: mask_idx}
                    # Invert retracked: mask_idx -> new_track
                    mask_to_new = {v: k for k, v in retk_map.items()}
                    frame_remap = {}
                    for old_track, mask_idx in orig_map.items():
                        if mask_idx in mask_to_new:
                            frame_remap[old_track] = mask_to_new[mask_idx]
                    if frame_remap:
                        track_remap[frame_id] = frame_remap

                print(f"✓ Loaded RETRACKED mask-track mapping from {retracked_file}")
                # Summarize remap
                all_old = set()
                all_new = set()
                for fr in track_remap.values():
                    all_old.update(fr.keys())
                    all_new.update(fr.values())
                print(f"  Track remap: {len(all_old)} original tracks -> {len(all_new)} retracked tracks")
                return retracked_mapping, track_remap
            except Exception as e:
                print(f"⚠️ Failed to load retracked mask-track mapping: {e}, falling back to original")

        if mapping_file.exists():
            try:
                with open(mapping_file) as f:
                    mapping = json.load(f)
                print(f"✓ Loaded mask-track mapping from {mapping_file}")
                return mapping, None
            except Exception as e:
                print(f"⚠️ Failed to load mask-track mapping: {e}")
        else:
            print(f"ℹ️ No mask-track mapping found (run demo_masks.py to generate)")
        return None, None

    def _decode_mask_by_index(self, frame_idx, mask_index):
        """Decode mask at specific index in the annotations list.

        Args:
            frame_idx: Frame index/name
            mask_index: Index into the annotations array (0-indexed)

        Returns:
            Binary mask (H, W) numpy array, or None if not found
        """
        if self.mask_dir is None:
            return None

        json_file = self.mask_dir / f"{frame_idx}_results.json"
        if not json_file.exists():
            return None

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            annotations = data.get('annotations', [])
            if mask_index < 0 or mask_index >= len(annotations):
                print(f"⚠️ Mask index {mask_index} out of range (0-{len(annotations)-1}) for frame {frame_idx}")
                return None

            try:
                from pycocotools import mask as mask_utils
            except ImportError:
                print("⚠️ pycocotools not installed - mask decoding disabled")
                return None

            rle = annotations[mask_index]['segmentation']
            return mask_utils.decode(rle)

        except Exception as e:
            print(f"⚠️ Failed to decode mask at index {mask_index} for frame {frame_idx}: {e}")
            return None

    def _load_previous_session(self):
        """Load previously saved corrections and semantic faces from output_dir

        This ensures continuity between editing sessions - users don't lose work
        when restarting the annotator on the same dataset.

        Loads from:
            - output_dir/bounding_boxes/*.json -> updates auto_bboxes with corrected values
            - output_dir/corrected_labels/corrections.json -> restores corrections dict
            - output_dir/corrected_labels/semantic_faces/manual_labels.json -> restores semantic_faces
        """
        corrected_bboxes_dir = self.output_dir / "bounding_boxes"
        corrected_labels_dir = self.output_dir / "corrected_labels"

        loaded_anything = False

        # 1. Check if previously corrected bboxes exist - use them as base instead of auto_bboxes
        if corrected_bboxes_dir.exists():
            json_files = list(corrected_bboxes_dir.glob("*.json"))
            if json_files:
                print(f"\n📂 Found previous session data in {self.output_dir}")
                print(f"   Loading {len(json_files)} corrected bbox files...")

                # Replace auto_bboxes with the corrected versions
                for json_file in sorted(json_files):
                    try:
                        frame_idx = int(json_file.stem)
                        with open(json_file) as f:
                            data = json.load(f)

                        frame_bboxes = []
                        for bbox_dict in data:
                            bbox = BBox3D(
                                center=bbox_dict['center'],
                                dimensions=bbox_dict['dimensions'],
                                rotation_matrix=bbox_dict['rotation_matrix'],
                                class_name=bbox_dict['class_name'],
                                track_id=bbox_dict.get('track_id', -1),
                                frame_idx=frame_idx,
                                confidence=bbox_dict.get('confidence', 1.0),
                                instance_id=bbox_dict.get('instance_id', -1),
                                persistent_instance_id=bbox_dict.get('persistent_instance_id', -1)
                            )
                            frame_bboxes.append(bbox)

                        self.auto_bboxes[frame_idx] = frame_bboxes
                    except Exception as e:
                        print(f"   ⚠️ Error loading {json_file.name}: {e}")

                # Update frame indices in case new frames were added
                self.frame_indices = sorted(self.auto_bboxes.keys())
                print(f"   ✓ Loaded corrected bboxes for {len(json_files)} frames")
                loaded_anything = True

        # 2. Load corrections.json to restore the corrections dict
        corrections_file = corrected_labels_dir / "corrections.json"
        if corrections_file.exists():
            try:
                with open(corrections_file) as f:
                    data = json.load(f)

                for track_id_str, frames in data.items():
                    track_id = int(track_id_str)
                    self.corrections[track_id] = {}

                    for frame_idx_str, bbox_dict in frames.items():
                        frame_idx = int(frame_idx_str)
                        # Support both old format (with track_id/frame_idx) and new minimal format
                        bbox = BBox3D(
                            center=bbox_dict['center'],
                            dimensions=bbox_dict['dimensions'],
                            rotation_matrix=bbox_dict['rotation_matrix'],
                            class_name=bbox_dict['class_name'],
                            track_id=bbox_dict.get('track_id', track_id),  # Use key if not in dict
                            frame_idx=bbox_dict.get('frame_idx', frame_idx),  # Use key if not in dict
                            confidence=bbox_dict.get('confidence', 1.0),
                            instance_id=bbox_dict.get('instance_id', -1),
                            persistent_instance_id=bbox_dict.get('persistent_instance_id', -1)
                        )
                        self.corrections[track_id][frame_idx] = bbox

                total_corrections = sum(len(frames) for frames in self.corrections.values())
                print(f"   ✓ Restored {total_corrections} corrections for {len(self.corrections)} tracks")
                loaded_anything = True
            except Exception as e:
                print(f"   ⚠️ Error loading corrections.json: {e}")

        # 3. Load semantic face labels
        semantic_file = corrected_labels_dir / "semantic_faces" / "manual_labels.json"
        if semantic_file.exists():
            try:
                with open(semantic_file) as f:
                    data = json.load(f)

                for track_id_str, frames in data.items():
                    track_id = int(track_id_str)
                    self.semantic_faces[track_id] = {}

                    for frame_idx_str, labels in frames.items():
                        frame_idx = int(frame_idx_str)
                        self.semantic_faces[track_id][frame_idx] = labels

                total_labels = sum(len(frames) for frames in self.semantic_faces.values())
                print(f"   ✓ Restored {total_labels} semantic face labels for {len(self.semantic_faces)} tracks")
                loaded_anything = True
            except Exception as e:
                print(f"   ⚠️ Error loading semantic faces: {e}")

        if loaded_anything:
            print(f"   ✅ Previous session restored successfully!\n")
        else:
            print(f"ℹ️ No previous session found - starting fresh")

    def _load_point_cloud(self, frame_idx):
        """Load point cloud for a frame (PLY, or generate from depth+image)"""
        # Check if already loaded
        if frame_idx in self.point_clouds:
            return self.point_clouds[frame_idx]

        # Debug: Show what's available for adaptive loading
        if frame_idx == self.frame_indices[0]:  # Only print once for first frame
            print(f"\n🔍 Adaptive Point Cloud Loading:")
            print(f"   - PLY dir: {self.point_clouds_dir}")
            print(f"   - Depth dir: {self.depths_dir}")
            print(f"   - Images dir: {self.images_dir}")
            print(f"   - Camera dict: {'✓' if self.cam_dict is not None else '✗'}\n")

        # Try loading from PLY files first
        if self.point_clouds_dir is not None:
            # Try multiple filename formats: plain number, zero-padded (4 digits)
            ply_candidates = [
                self.point_clouds_dir / f"{frame_idx}.ply",
                self.point_clouds_dir / f"{frame_idx:04d}.ply",
            ]
            ply_file = None
            for candidate in ply_candidates:
                if candidate.exists():
                    ply_file = candidate
                    break

            if ply_file is not None:
                try:
                    import trimesh
                    mesh = trimesh.load(str(ply_file))
                    points = np.array(mesh.vertices)

                    # Get colors if available
                    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                        colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0
                    else:
                        colors = np.ones_like(points) * 0.5

                    # No subsampling - keep full resolution to match demo_masks_fixed.py
                    # Full PLY files have ~147k points (288x512)

                    self.point_clouds[frame_idx] = (points, colors)
                    return points, colors

                except Exception as e:
                    print(f"⚠️ Failed to load PLY {frame_idx}: {e}")

        # Fallback: Generate point cloud from depth + image if available
        if self.depths_dir is not None and self.images_dir is not None and self.cam_dict is not None:
            try:
                points, colors = self._generate_point_cloud_from_depth(frame_idx)
                if points is not None:
                    self.point_clouds[frame_idx] = (points, colors)
                    return points, colors
            except Exception as e:
                print(f"⚠️ Failed to generate point cloud from depth {frame_idx}: {e}")

        return None, None

    def _generate_point_cloud_from_depth(self, frame_idx):
        """Generate colored point cloud from depth map and image"""
        import cv2

        # Load depth map - try multiple filename formats
        depth_candidates = [
            self.depths_dir / f"{frame_idx}.npy",
            self.depths_dir / f"{frame_idx:04d}.npy",
        ]
        depth_file = None
        for candidate in depth_candidates:
            if candidate.exists():
                depth_file = candidate
                break
        if depth_file is None:
            return None, None

        depth = np.load(str(depth_file))

        # Load image for colors - try multiple filename formats
        img_candidates = [
            self.images_dir / f"{frame_idx}.jpg",
            self.images_dir / f"{frame_idx:04d}.jpg",
            self.images_dir / f"{frame_idx}.png",
            self.images_dir / f"{frame_idx:04d}.png",
        ]
        img_file = None
        for candidate in img_candidates:
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            return None, None

        img = cv2.imread(str(img_file))
        if img is None:
            return None, None

        # Get camera intrinsics
        frame_to_idx = {frame: idx for idx, frame in enumerate(self.frame_indices)}
        if frame_idx not in frame_to_idx:
            return None, None

        idx = frame_to_idx[frame_idx]
        focal = self.cam_dict['focal'][idx]
        pp = self.cam_dict['pp'][idx]

        # Resize depth and image to match camera parameters
        h, w = depth.shape
        model_h, model_w = 288, 512  # Default dust3r output size

        # Resize depth if needed
        if (h, w) != (model_h, model_w):
            depth = cv2.resize(depth, (model_w, model_h), interpolation=cv2.INTER_NEAREST)

        # Always resize image to match depth dimensions
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        if img.shape[:2] != (model_h, model_w):
            img_rgb = cv2.resize(img_rgb, (model_w, model_h), interpolation=cv2.INTER_LINEAR)

        # Generate 3D points from depth
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Back-project to 3D (camera coordinates)
        z = depth
        x = (u - pp[0]) * z / focal
        y = (v - pp[1]) * z / focal

        # Stack to (H, W, 3)
        points_cam = np.stack([x, y, z], axis=-1)

        # Transform to world coordinates using camera pose
        R_cam = self.cam_dict['R'][idx]
        t_cam = self.cam_dict['t'][idx]

        # c2w transform
        points_flat = points_cam.reshape(-1, 3)
        points_world = (R_cam @ points_flat.T).T + t_cam

        # Get colors
        colors_flat = img_rgb.reshape(-1, 3)

        # Filter out invalid points (zero depth)
        valid_mask = depth.flatten() > 0
        points_world = points_world[valid_mask]
        colors_flat = colors_flat[valid_mask]

        # NO SUBSAMPLING - keep all valid points to match demo_masks.py PLY density
        # Typical point count: ~147,456 points (288 × 512)

        print(f"✓ Generated point cloud from depth+image for frame {frame_idx}: {len(points_world)} points")
        return points_world, colors_flat

    def _setup_ui(self):
        """Setup GUI controls"""

        # Frame navigation buttons
        prev_frame_btn = self.server.add_gui_button("◀ Previous Frame")

        @prev_frame_btn.on_click
        def _(_):
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self.updating_sliders = True
                self.frame_slider.value = self.current_frame_idx
                self.updating_sliders = False
                self._render_frame()

        next_frame_btn = self.server.add_gui_button("Next Frame ▶")

        @next_frame_btn.on_click
        def _(_):
            if self.current_frame_idx < len(self.frame_indices) - 1:
                self.current_frame_idx += 1
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self.updating_sliders = True
                self.frame_slider.value = self.current_frame_idx
                self.updating_sliders = False
                self._render_frame()

        # Frame slider
        self.frame_slider = self.server.add_gui_slider(
            "Frame",
            min=0,
            max=len(self.frame_indices) - 1,
            step=1,
            initial_value=0
        )

        @self.frame_slider.on_update
        def _(_):
            if not self.updating_sliders:
                self.current_frame_idx = int(self.frame_slider.value)
                self.current_frame = self.frame_indices[self.current_frame_idx]
                self._render_frame()

        # Track dropdown
        all_tracks = set()
        for bboxes in self.auto_bboxes.values():
            for bbox in bboxes:
                all_tracks.add(bbox.track_id)

        track_options = ["(None)"] + [f"Track {tid}" for tid in sorted(all_tracks)]
        self.track_dropdown = self.server.add_gui_dropdown(
            "Select Track",
            options=track_options
        )

        @self.track_dropdown.on_update
        def _(_):
            if self.track_dropdown.value == "(None)":
                self.selected_track = None
            else:
                self.selected_track = int(self.track_dropdown.value.split()[-1])
            self._update_selection()
            self._render_frame()  # FIX: Re-render to show selected track immediately

        # Info display
        self.info_text = self.server.add_gui_text(
            "Info",
            initial_value="Select a track to edit",
            disabled=True
        )

        # Point cloud controls
        if self.point_clouds_dir:
            self.point_size_slider = self.server.add_gui_slider(
                "Point Size",
                min=0.001,
                max=0.02,
                step=0.001,
                initial_value=0.005
            )

            @self.point_size_slider.on_update
            def _(_):
                if self.pc_handle is not None:
                    self.pc_handle.point_size = self.point_size_slider.value

        # Dimension editing sliders (for selected bbox)
        # Use dim[0], dim[1], dim[2] labels so user can see which is which when dragging
        self.dim_0_slider = self.server.add_gui_slider(
            "dim[0]",
            min=0.1,
            max=5.0,
            step=0.01,
            initial_value=1.0
        )

        self.dim_1_slider = self.server.add_gui_slider(
            "dim[1]",
            min=0.1,
            max=5.0,
            step=0.01,
            initial_value=1.0
        )

        self.dim_2_slider = self.server.add_gui_slider(
            "dim[2]",
            min=0.1,
            max=5.0,
            step=0.01,
            initial_value=1.0
        )

        @self.dim_0_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        @self.dim_1_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        @self.dim_2_slider.on_update
        def _(_):
            if not self.updating_sliders and self.selected_track is not None:
                self._update_bbox_dimensions()

        # Elephant dimension mapping
        self.height_is = self.server.add_gui_dropdown(
            "Height is",
            options=["dim[0]", "dim[1]", "dim[2]"],
            initial_value="dim[2]"
        )

        # Snap to proportions button (label will be updated dynamically based on species)
        self.snap_btn = self.server.add_gui_button("Snap to Proportions")

        @self.snap_btn.on_click
        def _(_):
            self._snap_to_proportions()

        # NEW v5: Ground snapping buttons
        snap_ground_btn = self.server.add_gui_button("⬇️ Snap to Ground")

        @snap_ground_btn.on_click
        def _(_):
            self._snap_to_ground()

        snap_ground_all_btn = self.server.add_gui_button("⬇️ Snap All Frames to Ground")

        @snap_ground_all_btn.on_click
        def _(_):
            self._snap_to_ground_all_frames()

        # Ground detection method dropdown
        self.ground_method_dropdown = self.server.add_gui_dropdown(
            "Ground Method",
            options=["RANSAC (robust)", "Lowest Points", "Track Points"],
            initial_value="RANSAC (robust)"
        )

        # Copy from previous frame button
        copy_prev_btn = self.server.add_gui_button("Copy BBox from Previous Frame")

        @copy_prev_btn.on_click
        def _(_):
            self._copy_from_previous_frame()

        # Copy from next frame button
        copy_next_btn = self.server.add_gui_button("Copy BBox from Next Frame")

        @copy_next_btn.on_click
        def _(_):
            self._copy_from_next_frame()

        # Semantic face labeling (auto-apply)
        face_options = ["(None)"] + [f"Face {i}" for i in range(6)]

        self.front_face_dropdown = self.server.add_gui_dropdown(
            "Front Face",
            options=face_options,
            initial_value="(None)"
        )

        @self.front_face_dropdown.on_update
        def _(_):
            if not self.updating_dropdowns:
                self._auto_apply_and_save()

        self.top_face_dropdown = self.server.add_gui_dropdown(
            "Top Face",
            options=face_options,
            initial_value="(None)"
        )

        @self.top_face_dropdown.on_update
        def _(_):
            if not self.updating_dropdowns:
                self._auto_apply_and_save()

        self.left_face_dropdown = self.server.add_gui_dropdown(
            "Left Face",
            options=face_options,
            initial_value="(None)"
        )

        @self.left_face_dropdown.on_update
        def _(_):
            if not self.updating_dropdowns:
                self._auto_apply_and_save()

        # HYBRID++ mode buttons
        propagate_btn = self.server.add_gui_button("🔄 Propagate to All Frames")

        @propagate_btn.on_click
        def _(_):
            self._propagate_to_all_frames()

        # Keyframe interpolation workflow
        keyframe_1_btn = self.server.add_gui_button("📍 Mark as Keyframe 1")

        @keyframe_1_btn.on_click
        def _(_):
            self._mark_keyframe_1()

        keyframe_2_btn = self.server.add_gui_button("📍 Mark as Keyframe 2")

        @keyframe_2_btn.on_click
        def _(_):
            self._mark_keyframe_2()

        interpolate_btn = self.server.add_gui_button("🎬 Interpolate Between Keyframes")

        @interpolate_btn.on_click
        def _(_):
            self._interpolate_between_keyframes()

        # NEW v6: Interpolate semantic faces only (without changing bbox positions)
        interpolate_semantic_btn = self.server.add_gui_button("🏷️ Interpolate Semantic Faces Only")

        @interpolate_semantic_btn.on_click
        def _(_):
            self._interpolate_semantic_faces_only()

        # Confirmation button (hidden by default, shown when needed)
        self.confirm_interpolate_btn = self.server.add_gui_button(
            "⚠️ Confirm Overwrite & Interpolate",
            visible=False
        )

        @self.confirm_interpolate_btn.on_click
        def _(_):
            if self.pending_interpolation:
                self._perform_interpolation(*self.pending_interpolation)
                self.pending_interpolation = None
                self.confirm_interpolate_btn.visible = False

        # Info text to show keyframe status (separate for KF1 and KF2)
        self.keyframe_1_text = self.server.add_gui_text(
            "Keyframe 1",
            initial_value="Not set"
        )

        self.keyframe_2_text = self.server.add_gui_text(
            "Keyframe 2",
            initial_value="Not set"
        )

        next_unann_btn = self.server.add_gui_button("⏭️ Next Unannotated Frame")

        @next_unann_btn.on_click
        def _(_):
            self._next_unannotated_frame()

        # Manual save button (optional - auto-save still happens)
        save_btn = self.server.add_gui_button("💾 Save Now")

        @save_btn.on_click
        def _(_):
            self._manual_save()

        # Checkbox to show original bboxes for comparison
        self.show_original_checkbox = self.server.add_gui_checkbox(
            "Show Original Bboxes",
            initial_value=False,
            hint="Toggle to compare original vs corrected bboxes"
        )

        @self.show_original_checkbox.on_update
        def _(_):
            self._render_frame()

        # Track highlighting checkbox
        self.highlight_track_checkbox = self.server.add_gui_checkbox(
            "Highlight Track Points",
            initial_value=True,
            hint="Highlight points belonging to selected track using mask backprojection"
        )

        @self.highlight_track_checkbox.on_update
        def _(_):
            self._render_frame()

    def _render_frame(self):
        """Render current frame - full re-render"""
        # Clear all scene objects
        self._clear_scene()

        # Load and render point cloud
        points, colors = self._load_point_cloud(self.current_frame)
        if points is not None:
            point_size = self.point_size_slider.value if hasattr(self, 'point_size_slider') else 0.005

            # If a track is selected and highlighting is ON, dim the full point cloud
            if (self.selected_track is not None and
                hasattr(self, 'highlight_track_checkbox') and
                self.highlight_track_checkbox.value):
                # Dim the colors for non-track points
                dimmed_colors = colors * 0.3
                self.pc_handle = self.server.add_point_cloud(
                    name=f"/pc",
                    points=points,
                    colors=dimmed_colors,
                    point_size=point_size * 0.7  # Slightly smaller
                )
                # Render highlighted track points separately
                self._render_track_highlighted_points(points, colors, point_size)
            else:
                self.pc_handle = self.server.add_point_cloud(
                    name=f"/pc",
                    points=points,
                    colors=colors,
                    point_size=point_size
                )

        # Get bboxes for current frame
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])

        if not frame_bboxes:
            self._update_info_text()
            return

        # Render bboxes - only selected track when one is selected (reduces visual clutter)
        if self.selected_track is not None:
            # Only render the selected track's bbox
            for bbox in frame_bboxes:
                if bbox.track_id == self.selected_track:
                    self._render_bbox(bbox)
                    break
        else:
            # No track selected - show all bboxes so user can pick one
            for bbox in frame_bboxes:
                self._render_bbox(bbox)

        # If "Show Original" is ON and we have a selected track with corrections,
        # show the original bbox for comparison (only for selected track)
        if (hasattr(self, 'show_original_checkbox') and
            self.show_original_checkbox.value and
            self.selected_track is not None):
            # Find the selected track's original bbox
            for bbox in frame_bboxes:
                if bbox.track_id == self.selected_track:
                    if self._has_correction(bbox.track_id, self.current_frame):
                        # There IS a correction, so show original for comparison
                        self._render_bbox_as_original(bbox)
                    break

        # Update selection (gizmo + sliders)
        self._update_selection()

        # Update info
        self._update_info_text()

    def _render_bbox(self, bbox):
        """Render a single bbox wireframe"""
        # Use corrected version if exists
        if self._has_correction(bbox.track_id, self.current_frame):
            bbox = self.corrections[bbox.track_id][self.current_frame]
            color = (0.0, 1.0, 0.0)  # Green for corrected
        else:
            color = (1.0, 0.0, 0.0)  # Red for auto

        # Highlight selected
        if self.selected_track is not None and bbox.track_id == self.selected_track:
            color = (0.0, 1.0, 1.0)  # Cyan for selected
            line_width = 5.0
        else:
            line_width = 2.0

        # Create handle list for this bbox (use setdefault to avoid race conditions)
        bbox_id = f"bbox_{bbox.track_id}"
        if bbox_id not in self.bbox_handles:
            self.bbox_handles[bbox_id] = []

        # Render wireframe
        corners = bbox.get_corners()
        for i, (start_idx, end_idx) in enumerate(bbox.get_edges()):
            line_points = np.array([corners[start_idx], corners[end_idx]])

            handle = self.server.add_spline_catmull_rom(
                name=f"/{bbox_id}_edge_{i}",
                positions=line_points,
                color=color,
                line_width=line_width,
                segments=2
            )
            # Defensive append (handles race conditions)
            if bbox_id in self.bbox_handles:
                self.bbox_handles[bbox_id].append(handle)

        # Add center point
        handle = self.server.add_point_cloud(
            name=f"/{bbox_id}_center",
            points=bbox.center.reshape(1, 3),
            colors=np.array(color).reshape(1, 3),
            point_size=0.02
        )
        # Defensive append (handles race conditions)
        if bbox_id in self.bbox_handles:
            self.bbox_handles[bbox_id].append(handle)

        # Render face labels if this is the selected bbox (auto-show when dropdowns have selections)
        if self.selected_track is not None and bbox.track_id == self.selected_track:
            self._render_face_labels(bbox)

    def _render_face_labels(self, bbox):
        """Render semantic face labels as colored edges (clean, minimal)"""
        # Get current semantic selections
        selected_faces = {}
        if self.front_face_dropdown.value != "(None)":
            selected_faces['front'] = int(self.front_face_dropdown.value.split()[-1])
        if self.top_face_dropdown.value != "(None)":
            selected_faces['top'] = int(self.top_face_dropdown.value.split()[-1])
        if self.left_face_dropdown.value != "(None)":
            selected_faces['left'] = int(self.left_face_dropdown.value.split()[-1])

        # Semantic label colors
        semantic_colors = {
            'front': (1.0, 0.0, 0.0),   # Red
            'top': (0.0, 1.0, 0.0),     # Green
            'left': (0.0, 0.0, 1.0),    # Blue
        }

        # Face corner indices (which 4 corners form each face)
        face_corner_indices = {
            0: [0, 1, 5, 4],  # Front
            1: [2, 3, 7, 6],  # Back
            2: [0, 3, 7, 4],  # Left
            3: [1, 2, 6, 5],  # Right
            4: [4, 5, 6, 7],  # Top
            5: [0, 1, 2, 3],  # Bottom
        }

        # Get all 8 corners of the bbox
        corners = bbox.get_corners()

        # Render edges for each selected semantic face
        for semantic_name, face_id in selected_faces.items():
            if face_id not in face_corner_indices:
                continue

            face_color = semantic_colors[semantic_name]
            corner_idxs = face_corner_indices[face_id]

            # Draw the 4 edges of this face
            # Edge 0: corner[0] -> corner[1]
            # Edge 1: corner[1] -> corner[2]
            # Edge 2: corner[2] -> corner[3]
            # Edge 3: corner[3] -> corner[0]
            for i in range(4):
                start_idx = corner_idxs[i]
                end_idx = corner_idxs[(i + 1) % 4]

                line_points = np.array([corners[start_idx], corners[end_idx]])

                handle = self.server.add_spline_catmull_rom(
                    name=f"/face_{semantic_name}_edge_{i}",
                    positions=line_points,
                    color=face_color,
                    line_width=2.0,  # Same thickness as bbox
                    segments=2
                )
                self.face_handles.append(handle)

    def _render_bbox_as_original(self, bbox):
        """Render original bbox in gray for comparison (when corrections exist)"""
        # Always use gray color for original comparison view
        color = (0.8, 0.8, 0.8)  # Light gray
        line_width = 0.5  # Thinner than normal

        # Create unique ID for original bbox
        bbox_id = f"original_{bbox.track_id}"

        # Render wireframe
        corners = bbox.get_corners()
        for i, (start_idx, end_idx) in enumerate(bbox.get_edges()):
            line_points = np.array([corners[start_idx], corners[end_idx]])

            handle = self.server.add_spline_catmull_rom(
                name=f"/{bbox_id}_edge_{i}",
                positions=line_points,
                color=color,
                line_width=line_width,
                segments=2
            )
            self.original_bbox_handles.append(handle)

        # Add center point (smaller and transparent)
        handle = self.server.add_point_cloud(
            name=f"/{bbox_id}_center",
            points=bbox.center.reshape(1, 3),
            colors=np.array(color).reshape(1, 3),
            point_size=0.01  # Smaller than normal
        )
        self.original_bbox_handles.append(handle)

    def _update_selection(self):
        """Update gizmo and sliders for selected bbox"""
        # Remove old gizmo
        if self.gizmo_handle is not None:
            try:
                self.gizmo_handle.remove()
            except (KeyError, Exception):
                pass
            self.gizmo_handle = None

        if self.selected_track is None:
            return

        # Get selected bbox
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])
        selected_bbox = next((b for b in frame_bboxes if b.track_id == self.selected_track), None)

        if selected_bbox is None:
            self._update_info_text()
            return

        # Use corrected version if exists
        if self._has_correction(self.selected_track, self.current_frame):
            selected_bbox = self.corrections[self.selected_track][self.current_frame]

        # Update sliders WITHOUT triggering callbacks
        self.updating_sliders = True
        self.dim_0_slider.value = float(selected_bbox.dimensions[0])
        self.dim_1_slider.value = float(selected_bbox.dimensions[1])
        self.dim_2_slider.value = float(selected_bbox.dimensions[2])
        self.updating_sliders = False

        # Create transform gizmo
        self.gizmo_handle = self.server.add_transform_controls(
            name=f"/gizmo",
            position=tuple(selected_bbox.center),
            wxyz=tuple(R.from_matrix(selected_bbox.rotation_matrix).as_quat()[[3, 0, 1, 2]])
        )

        @self.gizmo_handle.on_update
        def _(transform):
            self._update_bbox_from_gizmo(transform)

        # Update semantic face dropdowns if labels exist for this track/frame
        self._update_semantic_dropdowns()

        # Update snap button label based on species
        self._update_snap_button_for_species(selected_bbox.class_name)

        # Update info
        self._update_info_text()

    def _update_semantic_dropdowns(self):
        """Update semantic face dropdowns to show existing labels"""
        # Prevent triggering auto-apply callbacks while loading
        self.updating_dropdowns = True

        # Reset to "(None)" first
        self.front_face_dropdown.value = "(None)"
        self.top_face_dropdown.value = "(None)"
        self.left_face_dropdown.value = "(None)"

        # If we have labels for this track/frame, load them
        if (self.selected_track in self.semantic_faces and
            self.current_frame in self.semantic_faces[self.selected_track]):

            labels = self.semantic_faces[self.selected_track][self.current_frame]

            if 'front' in labels:
                self.front_face_dropdown.value = f"Face {labels['front']}"
            if 'top' in labels:
                self.top_face_dropdown.value = f"Face {labels['top']}"
            if 'left' in labels:
                self.left_face_dropdown.value = f"Face {labels['left']}"

        self.updating_dropdowns = False

    def _update_snap_button_for_species(self, class_name):
        """Update snap button label based on current species"""
        if class_name and class_name.lower() in self.SPECIES_PROPORTIONS:
            species_cap = class_name.capitalize()
            # Update button label dynamically
            self.snap_btn.label = f"Snap to {species_cap} Proportions"
        else:
            self.snap_btn.label = "Snap to Default Proportions"

    def _clear_scene(self):
        """Clear all scene objects"""
        # Clear point cloud
        if self.pc_handle is not None:
            try:
                self.pc_handle.remove()
            except:
                pass
            self.pc_handle = None

        # NEW in v4: Clear track-highlighted point cloud
        if self.track_pc_handle is not None:
            try:
                self.track_pc_handle.remove()
            except:
                pass
            self.track_pc_handle = None

        # Clear all bbox handles (copy to avoid concurrent modification)
        for bbox_id, handles in list(self.bbox_handles.items()):
            for handle in handles:
                try:
                    handle.remove()
                except:
                    pass
        self.bbox_handles.clear()

        # Clear original bbox handles
        for handle in list(self.original_bbox_handles):
            try:
                handle.remove()
            except:
                pass
        self.original_bbox_handles.clear()

        # Clear face handles
        for handle in list(self.face_handles):
            try:
                handle.remove()
            except:
                pass
        self.face_handles.clear()

        # Clear gizmo
        if self.gizmo_handle is not None:
            try:
                self.gizmo_handle.remove()
            except:
                pass
            self.gizmo_handle = None

    def _render_track_highlighted_points(self, all_points, all_colors, point_size):
        """Render highlighted points for the selected track using mask backprojection

        Uses grounded-sam mask to identify which points belong to the selected track,
        then renders them with bright colors and larger point size.
        """
        if self.selected_track is None:
            return

        # Get current bbox for class name
        current_bbox = self._get_current_bbox()
        if current_bbox is None:
            return

        # Load mask for current frame and track
        # NEW v5: Pass track_id for direct lookup via mask_track_mapping
        mask = self._load_mask_for_frame(
            self.current_frame,
            class_name=current_bbox.class_name,
            current_bbox=current_bbox,
            track_id=self.selected_track  # Direct lookup when mapping exists
        )

        if mask is None:
            # Fallback: highlight points inside bbox
            self._render_bbox_interior_points(all_points, all_colors, current_bbox, point_size)
            return

        # Backproject mask to 3D points
        track_mask = self._backproject_mask_to_points(mask, all_points)

        if track_mask is None or not np.any(track_mask):
            # Fallback to bbox-based selection
            self._render_bbox_interior_points(all_points, all_colors, current_bbox, point_size)
            return

        # Extract highlighted points
        track_points = all_points[track_mask]
        track_colors = all_colors[track_mask]

        # Brighten colors for visibility
        bright_colors = np.clip(track_colors * 1.5 + 0.2, 0, 1)

        # Render highlighted points
        if len(track_points) > 0:
            self.track_pc_handle = self.server.add_point_cloud(
                name="/track_pc",
                points=track_points,
                colors=bright_colors,
                point_size=point_size * 1.5  # Larger for visibility
            )

    def _backproject_mask_to_points(self, mask, points):
        """Backproject 2D mask to identify which 3D points are inside the mask

        Uses camera parameters to project points to 2D and check mask coverage.

        Args:
            mask: (H, W) binary mask
            points: (N, 3) array of 3D points

        Returns:
            (N,) boolean array indicating which points are inside the mask
        """
        if self.cam_dict is None:
            return None

        frame_to_idx = {frame: idx for idx, frame in enumerate(self.frame_indices)}
        if self.current_frame not in frame_to_idx:
            return None

        idx = frame_to_idx[self.current_frame]

        # Get camera parameters
        focal = self.cam_dict['focal'][idx]
        pp = self.cam_dict['pp'][idx]
        R_mat = self.cam_dict['R'][idx]
        t = self.cam_dict['t'][idx]

        # Build camera pose (c2w)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_mat
        camera_pose[:3, 3] = t

        # Get mask dimensions
        mask_h, mask_w = mask.shape

        # Scale camera params to mask size
        model_h, model_w = 288, 512  # dust3r output size
        scale_x = mask_w / model_w
        scale_y = mask_h / model_h

        focal_scaled = focal * scale_x
        pp_scaled = pp * np.array([scale_x, scale_y])

        K = np.array([
            [focal_scaled, 0, pp_scaled[0]],
            [0, focal_scaled, pp_scaled[1]],
            [0, 0, 1]
        ])

        # Transform points to camera space
        points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        points_cam = (np.linalg.inv(camera_pose) @ points_h.T)[:3].T

        # Filter points behind camera
        valid_depth = points_cam[:, 2] > 0.01
        if not np.any(valid_depth):
            return None

        # Project to 2D
        points_2d = np.zeros((len(points), 2))
        points_2d[valid_depth] = (K @ points_cam[valid_depth].T).T[:, :2] / points_cam[valid_depth, 2:3]

        # Check which points fall inside mask
        u = points_2d[:, 0].astype(int)
        v = points_2d[:, 1].astype(int)

        # Bounds check
        in_bounds = (u >= 0) & (u < mask_w) & (v >= 0) & (v < mask_h) & valid_depth

        # Check mask values
        inside_mask = np.zeros(len(points), dtype=bool)
        inside_mask[in_bounds] = mask[v[in_bounds], u[in_bounds]] > 0

        return inside_mask

    def _render_bbox_interior_points(self, all_points, all_colors, bbox, point_size):
        """Fallback: highlight points inside the bbox volume

        Used when mask backprojection is not available.
        """
        # Get local coordinates by inverse-transforming
        points_local = (np.linalg.inv(bbox.rotation_matrix) @ (all_points - bbox.center).T).T

        # Check if inside bbox bounds
        half_dims = bbox.dimensions / 2
        inside = (
            (np.abs(points_local[:, 0]) <= half_dims[0]) &
            (np.abs(points_local[:, 1]) <= half_dims[1]) &
            (np.abs(points_local[:, 2]) <= half_dims[2])
        )

        if not np.any(inside):
            return

        # Extract and render
        track_points = all_points[inside]
        track_colors = all_colors[inside]

        # Brighten colors
        bright_colors = np.clip(track_colors * 1.5 + 0.2, 0, 1)

        self.track_pc_handle = self.server.add_point_cloud(
            name="/track_pc",
            points=track_points,
            colors=bright_colors,
            point_size=point_size * 1.5
        )

    def _has_correction(self, track_id, frame_idx):
        """Check if correction exists"""
        return track_id in self.corrections and frame_idx in self.corrections[track_id]

    def _get_or_create_correction(self, track_id, frame_idx):
        """Get existing correction or create from original bbox"""
        if not self._has_correction(track_id, frame_idx):
            # Get original bbox
            frame_bboxes = self.auto_bboxes.get(frame_idx, [])
            original_bbox = next((b for b in frame_bboxes if b.track_id == track_id), None)
            if original_bbox is None:
                return None

            # Create correction entry
            if track_id not in self.corrections:
                self.corrections[track_id] = {}

            # Deep copy
            self.corrections[track_id][frame_idx] = BBox3D(
                center=original_bbox.center.copy(),
                dimensions=original_bbox.dimensions.copy(),
                rotation_matrix=original_bbox.rotation_matrix.copy(),
                class_name=original_bbox.class_name,
                track_id=original_bbox.track_id,
                frame_idx=original_bbox.frame_idx
            )

        return self.corrections[track_id][frame_idx]

    def _update_bbox_from_gizmo(self, transform):
        """Update bbox position/rotation from gizmo"""
        if self.selected_track is None:
            return

        # Get or create correction
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return

        # Update center
        bbox.center = np.array(transform.position)

        # Update rotation
        quat_wxyz = np.array(transform.wxyz)
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        bbox.rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

        # Re-render just the bboxes
        self._rerender_bboxes()

        # Schedule auto-save
        self._schedule_auto_save()

    def _update_bbox_dimensions(self):
        """Update bbox dimensions from sliders"""
        if self.selected_track is None:
            return

        # Get or create correction
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return

        # Update dimensions
        bbox.dimensions[0] = self.dim_0_slider.value
        bbox.dimensions[1] = self.dim_1_slider.value
        bbox.dimensions[2] = self.dim_2_slider.value

        # Re-render just the bboxes
        self._rerender_bboxes()

        # Schedule auto-save
        self._schedule_auto_save()

    def _snap_to_proportions(self):
        """Snap bbox to species-specific proportions

        User tells us which dimension is height, we calculate the other two based on
        the species of the current track (elephant, rhino, zebra, giraffe).
        """
        if self.selected_track is None:
            return

        # Get or create correction
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            return

        # Which dimension index is height? (0, 1, or 2)
        height_idx = int(self.height_is.value.split('[')[1].split(']')[0])

        # Get the other two indices
        all_indices = {0, 1, 2}
        other_indices = list(all_indices - {height_idx})

        # Get current height value
        height = bbox.dimensions[height_idx]

        # Get proportions for this species
        bbox_class = bbox.class_name.lower() if bbox.class_name else 'default'
        proportions = self.SPECIES_PROPORTIONS.get(
            bbox_class,
            self.SPECIES_PROPORTIONS['default']
        )
        target_length = height * proportions['length']
        target_width = height * proportions['width']

        # Assign to the other two dimensions
        # The larger dimension becomes length, smaller becomes width
        idx1, idx2 = other_indices[0], other_indices[1]
        dim1, dim2 = bbox.dimensions[idx1], bbox.dimensions[idx2]

        if dim1 > dim2:
            # idx1 is length, idx2 is width
            bbox.dimensions[idx1] = target_length
            bbox.dimensions[idx2] = target_width
        else:
            # idx2 is length, idx1 is width
            bbox.dimensions[idx2] = target_length
            bbox.dimensions[idx1] = target_width

        # Update sliders
        self.updating_sliders = True
        self.dim_0_slider.value = float(bbox.dimensions[0])
        self.dim_1_slider.value = float(bbox.dimensions[1])
        self.dim_2_slider.value = float(bbox.dimensions[2])
        self.updating_sliders = False

        # Re-render bboxes
        self._rerender_bboxes()

        species_name = bbox.class_name.capitalize() if bbox.class_name else 'default'
        print(f"✓ Snapped to {species_name} proportions:")
        print(f"  dim[0] = {bbox.dimensions[0]:.2f}m")
        print(f"  dim[1] = {bbox.dimensions[1]:.2f}m")
        print(f"  dim[2] = {bbox.dimensions[2]:.2f}m")
        print(f"  (Height was dim[{height_idx}] = {height:.2f}m)")

    # =========================================================================
    # NEW v5: Ground Detection and Snapping Methods
    # =========================================================================

    def _detect_vertical_axis(self, points):
        """Detect which axis is the vertical (height) axis based on point cloud spread.

        In viser with set_up_direction("-y"), the Y axis is typically vertical (inverted).
        But we verify by checking point distribution.

        Returns:
            int: 0 for X, 1 for Y, 2 for Z
            int: direction multiplier (1 if up is positive, -1 if up is negative)
        """
        # Check the spread of points along each axis
        ranges = np.ptp(points, axis=0)  # Range (max - min) for each axis

        # The vertical axis typically has the smallest horizontal spread relative to scene
        # But more reliable: check which axis aligns with "up" direction set in viser
        # Since we use set_up_direction("-y"), Y is vertical with negative being up

        # For robustness, check point distribution
        # Ground points should cluster at one end of the vertical axis

        # Default assumption based on viser setup: Y-axis, negative is up
        # So ground is at maximum Y values (most positive Y = lowest point)
        return 1, -1  # Y-axis, up is negative direction

    def _get_points_near_bbox(self, points, colors, bbox, radius=None):
        """Get points within a radius of the bbox center (XZ plane for ground detection).

        Args:
            points: (N, 3) array of 3D points
            colors: (N, 3) array of RGB colors
            bbox: BBox3D object
            radius: Search radius (uses config default if None)

        Returns:
            local_points: Points near the bbox
            local_colors: Colors of those points
            indices: Original indices of the selected points
        """
        if radius is None:
            radius = self.GROUND_CONFIG['search_radius']

        # Get the vertical axis
        vert_axis, _ = self._detect_vertical_axis(points)

        # Compute horizontal distance (ignoring vertical axis)
        horiz_axes = [i for i in range(3) if i != vert_axis]

        center_horiz = bbox.center[horiz_axes]
        points_horiz = points[:, horiz_axes]

        distances = np.linalg.norm(points_horiz - center_horiz, axis=1)
        mask = distances < radius

        return points[mask], colors[mask] if colors is not None else None, np.where(mask)[0]

    def _fit_ground_plane_ransac(self, points):
        """Fit a ground plane using RANSAC.

        Args:
            points: (N, 3) array of 3D points

        Returns:
            plane_params: (a, b, c, d) where ax + by + cz + d = 0, or None if failed
            inliers: Boolean mask of inlier points
        """
        config = self.GROUND_CONFIG
        n_points = len(points)

        if n_points < 3:
            return None, None

        best_inliers = None
        best_n_inliers = 0
        best_plane = None

        for _ in range(config['ransac_iterations']):
            # Randomly sample 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)

            if norm_len < 1e-8:
                continue

            normal = normal / norm_len
            d = -np.dot(normal, p1)

            # Check if this could be a ground plane (normal should be mostly vertical)
            vert_axis, up_dir = self._detect_vertical_axis(points)
            vert_component = abs(normal[vert_axis])

            if vert_component < np.cos(config['ground_normal_tolerance']):
                continue  # Not vertical enough to be ground

            # Count inliers
            distances = np.abs(np.dot(points, normal) + d)
            inliers = distances < config['ransac_threshold']
            n_inliers = np.sum(inliers)

            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inliers = inliers
                best_plane = (normal[0], normal[1], normal[2], d)

        if best_n_inliers < config['min_inliers']:
            return None, None

        return best_plane, best_inliers

    def _estimate_ground_level_percentile(self, points):
        """Estimate ground level using percentile of lowest points.

        Args:
            points: (N, 3) array of 3D points

        Returns:
            ground_height: Estimated ground level on the vertical axis
        """
        vert_axis, up_dir = self._detect_vertical_axis(points)

        # Get vertical coordinates
        vert_coords = points[:, vert_axis]

        # Ground is at the "bottom" - depends on up direction
        if up_dir > 0:
            # Up is positive, so ground is at minimum
            ground_height = np.percentile(vert_coords, self.GROUND_CONFIG['percentile_fallback'])
        else:
            # Up is negative (like viser -y), so ground is at maximum
            ground_height = np.percentile(vert_coords, 100 - self.GROUND_CONFIG['percentile_fallback'])

        return ground_height, vert_axis

    def _detect_ground_from_track_points(self, points, colors, bbox):
        """Detect ground using points belonging to the tracked object.

        Uses mask backprojection to find object points, then estimates
        ground as the lowest points of the object (feet/hooves).

        Args:
            points: Full point cloud
            colors: Point colors
            bbox: Current bbox

        Returns:
            ground_height: Estimated ground level
            vert_axis: Which axis is vertical
        """
        # Try to use mask to get track-specific points
        mask = self._load_mask_for_frame(
            self.current_frame,
            class_name=bbox.class_name,
            current_bbox=bbox
        )

        if mask is not None:
            track_mask = self._backproject_mask_to_points(mask, points)
            if track_mask is not None and np.sum(track_mask) > 10:
                track_points = points[track_mask]
                # Ground is at the lowest points of the tracked object
                return self._estimate_ground_level_percentile(track_points)

        # Fallback: use points inside bbox
        local_points, _, _ = self._get_points_near_bbox(points, colors, bbox, radius=max(bbox.dimensions))
        if len(local_points) > 10:
            return self._estimate_ground_level_percentile(local_points)

        return None, None

    def _compute_ground_level(self, points, colors, bbox, method='ransac'):
        """Compute ground level using the specified method.

        Args:
            points: (N, 3) point cloud
            colors: (N, 3) colors
            bbox: Current BBox3D
            method: 'ransac', 'percentile', or 'track'

        Returns:
            ground_height: Height of ground on vertical axis
            vert_axis: Which axis is vertical (0, 1, or 2)
            success: Whether detection succeeded
        """
        if len(points) == 0:
            return None, None, False

        # Get points near the bbox
        local_points, local_colors, _ = self._get_points_near_bbox(points, colors, bbox)

        if len(local_points) < 10:
            print(f"⚠️ Not enough points near bbox ({len(local_points)} found)")
            return None, None, False

        vert_axis, up_dir = self._detect_vertical_axis(local_points)

        if method == 'ransac':
            plane, inliers = self._fit_ground_plane_ransac(local_points)

            if plane is not None:
                # Extract ground height from plane
                # Plane: ax + by + cz + d = 0
                # For a point on the plane with x=cx, z=cz (horizontal coords of bbox center)
                a, b, c, d = plane
                horiz_axes = [i for i in range(3) if i != vert_axis]

                # Solve for vertical coordinate at bbox center horizontal position
                if abs(plane[vert_axis]) > 1e-8:
                    # ground_height = -(a*cx + c*cz + d) / b  (if vert_axis=1)
                    horiz_contrib = sum(plane[ax] * bbox.center[ax] for ax in horiz_axes)
                    ground_height = -(horiz_contrib + d) / plane[vert_axis]
                    return ground_height, vert_axis, True

            # Fallback to percentile if RANSAC fails
            print("  RANSAC failed, falling back to percentile method")
            method = 'percentile'

        if method == 'percentile':
            ground_height, vert_axis = self._estimate_ground_level_percentile(local_points)
            return ground_height, vert_axis, True

        if method == 'track':
            ground_height, vert_axis = self._detect_ground_from_track_points(points, colors, bbox)
            if ground_height is not None:
                return ground_height, vert_axis, True
            # Fallback
            ground_height, vert_axis = self._estimate_ground_level_percentile(local_points)
            return ground_height, vert_axis, True

        return None, None, False

    def _snap_bbox_to_ground(self, bbox, ground_height, vert_axis):
        """Adjust bbox center so its bottom face aligns with the ground.

        Args:
            bbox: BBox3D to modify (modified in place)
            ground_height: Ground level on vertical axis
            vert_axis: Which axis is vertical

        Returns:
            offset: How much the bbox was moved
        """
        # Get the height dimension based on user selection
        height_idx = int(self.height_is.value.split('[')[1].split(']')[0])
        half_height = bbox.dimensions[height_idx] / 2

        # Current bottom of bbox
        # The bottom face is at center - half_height along the LOCAL vertical
        # But we need to account for rotation...

        # For simplicity, assume the bbox is roughly aligned with world axes
        # The "bottom" in local coords is -half_height on the height dimension

        # Get current bbox corners to find actual bottom
        corners = bbox.get_corners()
        # Bottom face corners are indices 0, 1, 2, 3 (see get_corners: -h/2 in local z)
        # But the height dimension might not be z in local coords...

        # More robust: find the lowest corners based on world vertical axis
        vert_coords = corners[:, vert_axis]
        _, up_dir = self._detect_vertical_axis(corners)

        if up_dir > 0:
            # Up is positive, bottom is minimum
            current_bottom = np.min(vert_coords)
        else:
            # Up is negative, bottom is maximum
            current_bottom = np.max(vert_coords)

        # Compute offset needed
        offset = ground_height - current_bottom

        # Apply offset to center
        old_center = bbox.center.copy()
        bbox.center[vert_axis] += offset

        return offset

    def _snap_to_ground(self):
        """Snap current bbox to ground level (single frame)."""
        if self.selected_track is None:
            print("⚠️ No track selected!")
            return

        # Get or create correction
        bbox = self._get_or_create_correction(self.selected_track, self.current_frame)
        if bbox is None:
            print("⚠️ No bbox found for current frame!")
            return

        # Load point cloud for current frame
        points, colors = self._load_point_cloud(self.current_frame)
        if points is None:
            print("⚠️ No point cloud available for current frame!")
            return

        # Get selected method
        method_map = {
            "RANSAC (robust)": "ransac",
            "Lowest Points": "percentile",
            "Track Points": "track"
        }
        method = method_map.get(self.ground_method_dropdown.value, "ransac")

        print(f"\n⬇️ Snapping bbox to ground (method: {method})...")
        print(f"  Bbox center before: {bbox.center}")

        # Detect ground
        ground_height, vert_axis, success = self._compute_ground_level(
            points, colors, bbox, method=method
        )

        if not success:
            print("⚠️ Ground detection failed!")
            return

        print(f"  Detected ground at {['X', 'Y', 'Z'][vert_axis]}={ground_height:.3f}")

        # Snap bbox
        offset = self._snap_bbox_to_ground(bbox, ground_height, vert_axis)

        print(f"  Bbox center after: {bbox.center}")
        print(f"  Moved by: {offset:.3f}m on {['X', 'Y', 'Z'][vert_axis]}-axis")
        print(f"✓ Bbox grounded successfully!")

        # Update gizmo and re-render
        self._update_selection()
        self._rerender_bboxes()

        # Schedule auto-save
        self._schedule_auto_save()

    def _snap_to_ground_all_frames(self):
        """Snap bbox to ground for all frames where this track exists."""
        if self.selected_track is None:
            print("⚠️ No track selected!")
            return

        # Get selected method
        method_map = {
            "RANSAC (robust)": "ransac",
            "Lowest Points": "percentile",
            "Track Points": "track"
        }
        method = method_map.get(self.ground_method_dropdown.value, "ransac")

        print(f"\n⬇️ Snapping track {self.selected_track} to ground in all frames...")
        print(f"  Method: {method}")

        success_count = 0
        fail_count = 0

        for frame_idx in self.frame_indices:
            # Check if track exists in this frame
            track_exists = False
            for bbox in self.auto_bboxes.get(frame_idx, []):
                if bbox.track_id == self.selected_track:
                    track_exists = True
                    break

            if not track_exists:
                continue

            # Get or create correction for this frame
            bbox = self._get_or_create_correction(self.selected_track, frame_idx)
            if bbox is None:
                fail_count += 1
                continue

            # Load point cloud
            points, colors = self._load_point_cloud(frame_idx)
            if points is None:
                fail_count += 1
                continue

            # Detect ground
            ground_height, vert_axis, success = self._compute_ground_level(
                points, colors, bbox, method=method
            )

            if not success:
                fail_count += 1
                continue

            # Snap bbox
            self._snap_bbox_to_ground(bbox, ground_height, vert_axis)
            success_count += 1

        print(f"\n✓ Ground snapping complete:")
        print(f"  Succeeded: {success_count} frames")
        print(f"  Failed: {fail_count} frames")

        # Re-render current frame
        self._render_frame()

        # Schedule auto-save
        self._schedule_auto_save()

    # =========================================================================
    # End of Ground Detection Methods
    # =========================================================================

    def _apply_semantic_labels(self):
        """Apply semantic face labels for current track and frame"""
        if self.selected_track is None:
            print("⚠️ No track selected!")
            return

        # Get face selections
        front_val = self.front_face_dropdown.value
        top_val = self.top_face_dropdown.value
        left_val = self.left_face_dropdown.value

        # Parse selections
        labels = {}
        if front_val != "(None)":
            labels['front'] = int(front_val.split()[-1])
        if top_val != "(None)":
            labels['top'] = int(top_val.split()[-1])
        if left_val != "(None)":
            labels['left'] = int(left_val.split()[-1])

        if not labels:
            print("⚠️ No faces selected!")
            return

        # Validate: all three must be different
        if len(labels) == 3:
            face_ids = set(labels.values())
            if len(face_ids) != 3:
                print("⚠️ Error: Front, Top, and Left must be different faces!")
                return

        # Store semantic labels
        if self.selected_track not in self.semantic_faces:
            self.semantic_faces[self.selected_track] = {}

        self.semantic_faces[self.selected_track][self.current_frame] = labels

        print(f"✓ Applied semantic face labels for track {self.selected_track} frame {self.current_frame}:")
        for semantic, face_id in labels.items():
            print(f"  {semantic.capitalize()}: Face {face_id}")

        # Re-render to show updated labels
        self._rerender_bboxes()

        # Trigger auto-save for semantic face labels
        self._schedule_auto_save()

    def _copy_from_previous_frame(self):
        """Copy bbox from previous frame to current frame"""
        if self.selected_track is None:
            print("No track selected!")
            return

        if self.current_frame_idx == 0:
            print("Already at first frame!")
            return

        # Get previous frame
        prev_frame = self.frame_indices[self.current_frame_idx - 1]

        # Check if previous frame has this track (either corrected or original)
        prev_bbox = None
        if self._has_correction(self.selected_track, prev_frame):
            prev_bbox = self.corrections[self.selected_track][prev_frame]
        else:
            # Try to find in original bboxes
            prev_frame_bboxes = self.auto_bboxes.get(prev_frame, [])
            prev_bbox = next((b for b in prev_frame_bboxes if b.track_id == self.selected_track), None)

        if prev_bbox is None:
            print(f"Track {self.selected_track} not found in previous frame {prev_frame}!")
            return

        # Create correction for current frame by copying from previous
        if self.selected_track not in self.corrections:
            self.corrections[self.selected_track] = {}

        self.corrections[self.selected_track][self.current_frame] = BBox3D(
            center=prev_bbox.center.copy(),
            dimensions=prev_bbox.dimensions.copy(),
            rotation_matrix=prev_bbox.rotation_matrix.copy(),
            class_name=prev_bbox.class_name,
            track_id=prev_bbox.track_id,
            frame_idx=self.current_frame,
            # Preserve metadata from previous frame
            confidence=prev_bbox.confidence,
            instance_id=prev_bbox.instance_id,
            persistent_instance_id=prev_bbox.persistent_instance_id
        )

        # Also copy semantic face labels if they exist
        if self.selected_track in self.semantic_faces and prev_frame in self.semantic_faces[self.selected_track]:
            if self.selected_track not in self.semantic_faces:
                self.semantic_faces[self.selected_track] = {}

            self.semantic_faces[self.selected_track][self.current_frame] = self.semantic_faces[self.selected_track][prev_frame].copy()
            print(f"✓ Also copied semantic face labels from frame {prev_frame}")

        print(f"✓ Copied bbox for track {self.selected_track} from frame {prev_frame} to {self.current_frame}")

        # Update sliders and re-render
        self._update_selection()
        self._rerender_bboxes()

        # Schedule auto-save
        self._schedule_auto_save()

    def _copy_from_next_frame(self):
        """Copy bbox from next frame to current frame"""
        if self.selected_track is None:
            print("No track selected!")
            return

        if self.current_frame_idx >= len(self.frame_indices) - 1:
            print("Already at last frame!")
            return

        # Get next frame
        next_frame = self.frame_indices[self.current_frame_idx + 1]

        # Check if next frame has this track (either corrected or original)
        next_bbox = None
        if self._has_correction(self.selected_track, next_frame):
            next_bbox = self.corrections[self.selected_track][next_frame]
        else:
            # Try to find in original bboxes
            next_frame_bboxes = self.auto_bboxes.get(next_frame, [])
            next_bbox = next((b for b in next_frame_bboxes if b.track_id == self.selected_track), None)

        if next_bbox is None:
            print(f"Track {self.selected_track} not found in next frame {next_frame}!")
            return

        # Create correction for current frame by copying from next
        if self.selected_track not in self.corrections:
            self.corrections[self.selected_track] = {}

        self.corrections[self.selected_track][self.current_frame] = BBox3D(
            center=next_bbox.center.copy(),
            dimensions=next_bbox.dimensions.copy(),
            rotation_matrix=next_bbox.rotation_matrix.copy(),
            class_name=next_bbox.class_name,
            track_id=next_bbox.track_id,
            frame_idx=self.current_frame,
            # Preserve metadata from next frame
            confidence=next_bbox.confidence,
            instance_id=next_bbox.instance_id,
            persistent_instance_id=next_bbox.persistent_instance_id
        )

        # Also copy semantic face labels if they exist
        if self.selected_track in self.semantic_faces and next_frame in self.semantic_faces[self.selected_track]:
            if self.selected_track not in self.semantic_faces:
                self.semantic_faces[self.selected_track] = {}

            self.semantic_faces[self.selected_track][self.current_frame] = self.semantic_faces[self.selected_track][next_frame].copy()
            print(f"✓ Also copied semantic face labels from frame {next_frame}")

        print(f"✓ Copied bbox for track {self.selected_track} from frame {next_frame} to {self.current_frame}")

        # Update sliders and re-render
        self._update_selection()
        self._rerender_bboxes()

        # Schedule auto-save
        self._schedule_auto_save()

    def _rerender_bboxes(self):
        """Re-render only the bboxes (faster than full re-render)

        FIX: Respects selected_track - only renders selected track's bbox when one is selected.
        """
        # Clear bbox handles (copy to avoid concurrent modification)
        for bbox_id, handles in list(self.bbox_handles.items()):
            for handle in handles:
                try:
                    handle.remove()
                except:
                    pass
        self.bbox_handles.clear()

        # Clear face handles
        for handle in list(self.face_handles):
            try:
                handle.remove()
            except:
                pass
        self.face_handles.clear()

        # Re-render bboxes - respect selected_track (same logic as _render_frame)
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])

        if self.selected_track is not None:
            # Only render selected track's bbox
            for bbox in frame_bboxes:
                if bbox.track_id == self.selected_track:
                    self._render_bbox(bbox)
                    break
        else:
            # No track selected - show all bboxes
            for bbox in frame_bboxes:
                self._render_bbox(bbox)

    def _save_corrections(self):
        """Save corrections to JSON and generate 2D projections"""
        # 1. Save to output_dir/corrected_labels/ (user edits only - for reference)
        corrected_labels_dir = self.output_dir / "corrected_labels"
        corrected_labels_dir.mkdir(exist_ok=True)

        output_file = corrected_labels_dir / "corrections.json"
        data = {}
        for track_id, frames in self.corrections.items():
            data[str(track_id)] = {}
            for frame_idx, bbox in frames.items():
                # Minimal format - track_id and frame_idx are already keys
                data[str(track_id)][str(frame_idx)] = {
                    'center': bbox.center.tolist(),
                    'dimensions': bbox.dimensions.tolist(),
                    'rotation_matrix': bbox.rotation_matrix.tolist(),
                    'class_name': bbox.class_name
                }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(self.corrections)} track corrections to {output_file}")

        # 2. Save corrected bboxes to output_dir/bounding_boxes/ (demo_masks compatible!)
        corrected_dir = self.output_dir / "bounding_boxes"
        corrected_dir.mkdir(exist_ok=True)

        # Build full bbox list per frame (corrected + original)
        all_frames = {}
        for frame_idx in self.frame_indices:
            # Start with original bboxes
            frame_bboxes = []
            for bbox in self.auto_bboxes.get(frame_idx, []):
                # Use corrected version if it exists
                if self._has_correction(bbox.track_id, frame_idx):
                    frame_bboxes.append(self.corrections[bbox.track_id][frame_idx])
                else:
                    frame_bboxes.append(bbox)
            all_frames[frame_idx] = frame_bboxes

        # Save each frame as JSON (streamlined format - only fields used by downstream tools)
        for frame_idx, frame_bboxes in all_frames.items():
            frame_file = corrected_dir / f"{frame_idx}.json"
            frame_data = []
            for bbox in frame_bboxes:
                frame_data.append({
                    'center': [float(x) for x in bbox.center.tolist()],
                    'dimensions': [float(x) for x in bbox.dimensions.tolist()],
                    'rotation_matrix': [[float(x) for x in row] for row in bbox.rotation_matrix.tolist()],
                    'class_name': str(bbox.class_name),
                    'confidence': float(bbox.confidence),
                    'track_id': int(bbox.track_id) if bbox.track_id is not None else -1
                })
            with open(frame_file, 'w') as f:
                json.dump(frame_data, f, indent=2)

        print(f"✓ Saved corrected bboxes to {corrected_dir}")

        # 3. Save semantic face labels (for semantic_face_propagator.py)
        if self.semantic_faces:
            semantic_dir = corrected_labels_dir / "semantic_faces"
            semantic_dir.mkdir(exist_ok=True)

            semantic_file = semantic_dir / "manual_labels.json"
            semantic_data = {}

            for track_id, frames in self.semantic_faces.items():
                semantic_data[str(track_id)] = {}
                for frame_idx, labels in frames.items():
                    semantic_data[str(track_id)][str(frame_idx)] = labels

            with open(semantic_file, 'w') as f:
                json.dump(semantic_data, f, indent=2)

            print(f"✓ Saved semantic face labels for {len(self.semantic_faces)} tracks to {semantic_file}")

        # 4. Generate 2D projections if images and camera params available
        if self.images_dir and self.cam_dict:
            self._generate_2d_projections(all_frames)
            # Also generate semantic face visualizations
            self._generate_semantic_face_visualizations(all_frames)
        else:
            print("⚠️ Skipping 2D projections (missing images or camera parameters)")

    def _generate_2d_projections(self, all_frames):
        """Generate 2D bbox projections on images"""
        import cv2

        print("\n📸 Generating 2D projections...")
        vis_dir = self.output_dir / "annotated_2d"
        vis_dir.mkdir(exist_ok=True)

        # Map frame indices to array indices
        frame_to_idx = {frame: idx for idx, frame in enumerate(self.frame_indices)}

        for frame_idx, frame_bboxes in all_frames.items():
            if frame_idx not in frame_to_idx:
                continue

            idx = frame_to_idx[frame_idx]

            # Load image - try multiple filename formats
            img_file = None
            for fmt in [f"{frame_idx}.jpg", f"{frame_idx}.png",
                       f"{frame_idx:04d}.jpg", f"{frame_idx:04d}.png"]:
                candidate = self.images_dir / fmt
                if candidate.exists():
                    img_file = candidate
                    break
            if img_file is None:
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                continue

            # Get camera params
            focal = self.cam_dict['focal'][idx]
            pp = self.cam_dict['pp'][idx]
            R = self.cam_dict['R'][idx]
            t = self.cam_dict['t'][idx]

            # Build full camera-to-world pose matrix
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R
            camera_pose[:3, 3] = t

            # Scale camera params to image size
            img_h, img_w = img.shape[:2]
            model_h, model_w = 288, 512
            scale_x = img_w / model_w
            scale_y = img_h / model_h

            focal_scaled = focal * scale_x
            pp_scaled = pp * np.array([scale_x, scale_y])

            K = np.array([
                [focal_scaled, 0, pp_scaled[0]],
                [0, focal_scaled, pp_scaled[1]],
                [0, 0, 1]
            ])

            # Project each bbox
            for bbox in frame_bboxes:
                # Check if this bbox was corrected
                if self._has_correction(bbox.track_id, frame_idx):
                    color = (0, 255, 0)  # Green for corrected
                else:
                    color = (255, 0, 0)  # Blue for original

                # Get 3D corners
                corners_3d = bbox.get_corners()

                # Transform to camera space (FIXED: invert c2w pose to get w2c)
                corners_cam = (np.linalg.inv(camera_pose) @
                             np.concatenate([corners_3d, np.ones((8, 1))], axis=1).T)[:3].T

                # Project to 2D
                corners_2d = (K @ corners_cam.T).T
                corners_2d = corners_2d[:, :2] / corners_2d[:, 2:]

                # Draw wireframe
                corners_2d = corners_2d.astype(int)
                edges = bbox.get_edges()
                for start_idx, end_idx in edges:
                    pt1 = tuple(corners_2d[start_idx])
                    pt2 = tuple(corners_2d[end_idx])
                    cv2.line(img, pt1, pt2, color, 2)

                # Draw track ID
                center_2d = corners_2d.mean(axis=0).astype(int)
                cv2.putText(img, f"T{bbox.track_id}", tuple(center_2d),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save
            output_file = vis_dir / f"{frame_idx}_corrected.png"
            cv2.imwrite(str(output_file), img)

        print(f"✓ Saved 2D projections to {vis_dir}")

    def _generate_semantic_face_visualizations(self, all_frames):
        """Generate 2D visualizations with only the 3 semantic faces (front/top/left)"""
        import cv2

        print("\n🎨 Generating semantic face visualizations...")
        vis_dir = self.output_dir / "annotated_2d_semantic_faces"
        vis_dir.mkdir(exist_ok=True)

        # Map frame indices to array indices
        frame_to_idx = {frame: idx for idx, frame in enumerate(self.frame_indices)}

        for frame_idx, frame_bboxes in all_frames.items():
            if frame_idx not in frame_to_idx:
                continue

            idx = frame_to_idx[frame_idx]

            # Load image - try multiple filename formats
            img_file = None
            for fmt in [f"{frame_idx}.jpg", f"{frame_idx}.png",
                       f"{frame_idx:04d}.jpg", f"{frame_idx:04d}.png"]:
                candidate = self.images_dir / fmt
                if candidate.exists():
                    img_file = candidate
                    break
            if img_file is None:
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                continue

            # Get camera params
            focal = self.cam_dict['focal'][idx]
            pp = self.cam_dict['pp'][idx]
            R = self.cam_dict['R'][idx]
            t = self.cam_dict['t'][idx]

            # Build full camera-to-world pose matrix
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R
            camera_pose[:3, 3] = t

            # Scale camera params to image size
            img_h, img_w = img.shape[:2]
            model_h, model_w = 288, 512
            scale_x = img_w / model_w
            scale_y = img_h / model_h

            focal_scaled = focal * scale_x
            pp_scaled = pp * np.array([scale_x, scale_y])

            K = np.array([
                [focal_scaled, 0, pp_scaled[0]],
                [0, focal_scaled, pp_scaled[1]],
                [0, 0, 1]
            ])

            # Project each bbox
            for bbox in frame_bboxes:
                # Get 3D corners
                corners_3d = bbox.get_corners()

                # Transform to camera space (FIXED: invert c2w pose to get w2c)
                corners_cam = (np.linalg.inv(camera_pose) @
                             np.concatenate([corners_3d, np.ones((8, 1))], axis=1).T)[:3].T

                # Project to 2D
                corners_2d = (K @ corners_cam.T).T
                corners_2d = corners_2d[:, :2] / corners_2d[:, 2:]
                corners_2d = corners_2d.astype(int)

                # Get face data
                faces_3d = bbox.get_faces()

                # Check if we have semantic labels for this track/frame
                has_semantic = (bbox.track_id in self.semantic_faces and
                               frame_idx in self.semantic_faces[bbox.track_id])

                # Only draw if we have semantic labels
                if has_semantic:
                    semantic_labels = self.semantic_faces[bbox.track_id][frame_idx]

                    # Define face quad corner indices
                    face_corner_indices = {
                        0: [0, 1, 5, 4],  # Front
                        1: [2, 3, 7, 6],  # Back
                        2: [0, 3, 7, 4],  # Left
                        3: [1, 2, 6, 5],  # Right
                        4: [4, 5, 6, 7],  # Top
                        5: [0, 1, 2, 3],  # Bottom
                    }

                    # Semantic face colors (BGR for OpenCV)
                    semantic_colors_bgr = {
                        'front': (0, 0, 255),    # Red
                        'top': (0, 255, 0),      # Green
                        'left': (255, 0, 0),     # Blue
                    }

                    # Only draw the 3 semantic faces
                    for sem_name, face_id in semantic_labels.items():
                        if face_id not in face_corner_indices:
                            continue

                        corner_idxs = face_corner_indices[face_id]
                        face_corners_2d = corners_2d[corner_idxs]

                        # Check if face is visible (all corners have positive z)
                        face_corners_cam = corners_cam[corner_idxs]
                        if np.all(face_corners_cam[:, 2] > 0):
                            color = semantic_colors_bgr[sem_name]

                            # Fill polygon with semi-transparent color
                            overlay = img.copy()
                            cv2.fillPoly(overlay, [face_corners_2d], color)

                            # Blend with original
                            alpha = 0.4
                            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                            # Draw face edges
                            for i in range(4):
                                pt1 = tuple(face_corners_2d[i])
                                pt2 = tuple(face_corners_2d[(i + 1) % 4])
                                cv2.line(img, pt1, pt2, color, 2)

                            # Add simple semantic label (no face numbers!)
                            face_center_2d = face_corners_2d.mean(axis=0).astype(int)
                            label_text = sem_name.upper()

                            cv2.putText(img, label_text, tuple(face_center_2d),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(img, label_text, tuple(face_center_2d),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw track ID
                center_2d = corners_2d.mean(axis=0).astype(int)
                cv2.putText(img, f"Track {bbox.track_id}", tuple(center_2d),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
                cv2.putText(img, f"Track {bbox.track_id}", tuple(center_2d),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save
            output_file = vis_dir / f"{frame_idx}_semantic_faces.png"
            cv2.imwrite(str(output_file), img)

        print(f"✓ Saved semantic face visualizations to {vis_dir}")

    def _get_current_bbox(self):
        """Get current bbox (corrected version if exists, otherwise original)"""
        if self.selected_track is None:
            return None

        # Get original bbox
        frame_bboxes = self.auto_bboxes.get(self.current_frame, [])
        bbox = next((b for b in frame_bboxes if b.track_id == self.selected_track), None)

        if bbox is None:
            return None

        # Use corrected version if exists
        if self._has_correction(self.selected_track, self.current_frame):
            bbox = self.corrections[self.selected_track][self.current_frame]

        return bbox

    def _update_info_text(self):
        """Update info text with track/frame info and last saved time"""
        import datetime

        # Build base info text
        if self.selected_track is None:
            base_text = f"Frame {self.current_frame}"
        else:
            selected_bbox = self._get_current_bbox()
            if selected_bbox is None:
                base_text = f"Track {self.selected_track} not in frame {self.current_frame}"
            else:
                base_text = f"Track {self.selected_track} | {selected_bbox.class_name} | Frame {self.current_frame}"

        # Add last saved indicator
        if self.last_save_time is not None:
            elapsed = (datetime.datetime.now() - self.last_save_time).total_seconds()
            if elapsed < 60:
                time_str = f"{int(elapsed)}s ago"
            elif elapsed < 3600:
                time_str = f"{int(elapsed / 60)}m ago"
            else:
                time_str = f"{int(elapsed / 3600)}h ago"

            full_text = f"{base_text} | 💾 Last saved: {time_str}"
        else:
            full_text = base_text

        self.info_text.value = full_text

    def _auto_apply_and_save(self):
        """Auto-apply semantic labels and schedule auto-save (called by dropdown callbacks)"""
        # Silently apply semantic labels
        front_val = self.front_face_dropdown.value
        top_val = self.top_face_dropdown.value
        left_val = self.left_face_dropdown.value

        # Parse selections
        labels = {}
        if front_val != "(None)":
            labels['front'] = int(front_val.split()[-1])
        if top_val != "(None)":
            labels['top'] = int(top_val.split()[-1])
        if left_val != "(None)":
            labels['left'] = int(left_val.split()[-1])

        # Store semantic labels (even if incomplete - user is working on it)
        if labels:
            if self.selected_track not in self.semantic_faces:
                self.semantic_faces[self.selected_track] = {}

            self.semantic_faces[self.selected_track][self.current_frame] = labels

            # Re-render to show colored face edges immediately
            self._rerender_bboxes()

        # Schedule auto-save
        self._schedule_auto_save()

    def _schedule_auto_save(self):
        """Schedule auto-save with 2-second debounce"""
        import threading

        # Cancel previous timer if exists
        if self.auto_save_timer is not None:
            self.auto_save_timer.cancel()

        # Schedule new save in 2 seconds
        self.auto_save_timer = threading.Timer(2.0, self._perform_auto_save)
        self.auto_save_timer.start()

    def _perform_auto_save(self):
        """Actually perform the auto-save"""
        import datetime

        self._save_corrections()
        self.last_save_time = datetime.datetime.now()
        self.auto_save_timer = None

        # Update info text to show last saved time
        self._update_info_text()

    def _manual_save(self):
        """Manual save triggered by user clicking Save Now button"""
        import datetime

        # Cancel pending auto-save timer
        if self.auto_save_timer is not None:
            self.auto_save_timer.cancel()
            self.auto_save_timer = None

        self._save_corrections()
        self.last_save_time = datetime.datetime.now()

        # Update info text
        self._update_info_text()

        print("💾 Manual save completed!")

    def _propagate_to_all_frames(self):
        """Propagate current bbox size/shape/rotation + face labels to all frames

        What this does:
        - Copies DIMENSIONS (width, height, length) from current frame to all frames
        - Copies ROTATION (orientation) from current frame to all frames
        - Copies FACE LABELS (front/top/left) from current frame to all frames
        - KEEPS ORIGINAL POSITION (center) for each frame (doesn't move bboxes)

        Use case: "This bbox in frame 100 is perfect - copy its size/shape to all other
        frames (but let them stay where they are)" + face labels so you don't have to
        relabel, just rotate if needed.
        """
        if self.selected_track is None:
            print("⚠️ No track selected!")
            return

        current_bbox = self._get_current_bbox()
        if current_bbox is None:
            print("⚠️ No bbox for current frame!")
            return

        # Get semantic face labels for current frame (if any)
        current_semantic_faces = None
        if self.selected_track in self.semantic_faces:
            if self.current_frame in self.semantic_faces[self.selected_track]:
                current_semantic_faces = self.semantic_faces[self.selected_track][self.current_frame]

        print(f"\n🔄 Propagating from frame {self.current_frame}:")
        print(f"  Dimensions: {current_bbox.dimensions}")
        print(f"  Rotation: {'Yes' if current_bbox.rotation_matrix is not None else 'No'}")
        print(f"  Face labels: {current_semantic_faces if current_semantic_faces else 'None'}")

        # Count how many frames will be affected
        affected_frames = 0

        # Propagate to all frames where this track exists
        for frame_idx in self.frame_indices:
            # Check if track exists in this frame (either in original or corrected)
            track_exists = False
            for bbox in self.auto_bboxes.get(frame_idx, []):
                if bbox.track_id == self.selected_track:
                    track_exists = True
                    break

            if not track_exists:
                continue

            # Create corrected bbox for this frame (copy dimensions, rotation from current)
            # But keep the original center position (don't move the bbox across frames)
            original_bbox = None
            for bbox in self.auto_bboxes.get(frame_idx, []):
                if bbox.track_id == self.selected_track:
                    original_bbox = bbox
                    break

            if original_bbox is None:
                continue

            # Create new bbox with current dimensions/rotation but original center
            new_bbox = BBox3D(
                center=original_bbox.center.copy(),  # Keep original position
                dimensions=current_bbox.dimensions.copy(),  # Use current dimensions
                rotation_matrix=current_bbox.rotation_matrix.copy(),  # Use current rotation
                class_name=current_bbox.class_name,
                track_id=self.selected_track,
                frame_idx=frame_idx,
                # Preserve original metadata
                confidence=original_bbox.confidence,
                instance_id=original_bbox.instance_id,
                persistent_instance_id=original_bbox.persistent_instance_id
            )

            # Store as correction
            if self.selected_track not in self.corrections:
                self.corrections[self.selected_track] = {}
            self.corrections[self.selected_track][frame_idx] = new_bbox

            # Propagate semantic face labels too
            if current_semantic_faces:
                if self.selected_track not in self.semantic_faces:
                    self.semantic_faces[self.selected_track] = {}
                self.semantic_faces[self.selected_track][frame_idx] = current_semantic_faces.copy()

            affected_frames += 1

        print(f"✓ Propagated to {affected_frames} frames for track {self.selected_track}")
        print(f"  All frames now have same dimensions, rotation, and face labels")
        print(f"  (but each frame kept its original position)")

        # Re-render current view
        self._rerender_bboxes()

        # Schedule auto-save
        self._schedule_auto_save()

    def _mark_keyframe_1(self):
        """Mark current frame and track as keyframe 1"""
        if self.selected_track is None:
            print("⚠️ No track selected! Select a track first.")
            return

        # Get current bbox (corrected if exists, else original)
        current_bbox = self._get_current_bbox()
        if current_bbox is None:
            print(f"⚠️ Track {self.selected_track} not found in frame {self.current_frame}!")
            return

        # Get semantic labels if they exist
        semantic_labels = None
        if (self.selected_track in self.semantic_faces and
            self.current_frame in self.semantic_faces[self.selected_track]):
            semantic_labels = self.semantic_faces[self.selected_track][self.current_frame].copy()

        # Store keyframe 1
        self.keyframe_1 = (
            self.current_frame,
            self.selected_track,
            current_bbox,  # BBox3D object
            semantic_labels  # Dict or None
        )

        # Update info text
        self.keyframe_1_text.value = f"Frame {self.current_frame}, Track {self.selected_track}"

        print(f"✓ Marked Keyframe 1: Frame {self.current_frame}, Track {self.selected_track}")
        if semantic_labels:
            print(f"  With semantic labels: {semantic_labels}")

    def _mark_keyframe_2(self):
        """Mark current frame and track as keyframe 2"""
        if self.selected_track is None:
            print("⚠️ No track selected! Select a track first.")
            return

        # Get current bbox (corrected if exists, else original)
        current_bbox = self._get_current_bbox()
        if current_bbox is None:
            print(f"⚠️ Track {self.selected_track} not found in frame {self.current_frame}!")
            return

        # Get semantic labels if they exist
        semantic_labels = None
        if (self.selected_track in self.semantic_faces and
            self.current_frame in self.semantic_faces[self.selected_track]):
            semantic_labels = self.semantic_faces[self.selected_track][self.current_frame].copy()

        # Store keyframe 2
        self.keyframe_2 = (
            self.current_frame,
            self.selected_track,
            current_bbox,  # BBox3D object
            semantic_labels  # Dict or None
        )

        # Update info text
        self.keyframe_2_text.value = f"Frame {self.current_frame}, Track {self.selected_track}"

        print(f"✓ Marked Keyframe 2: Frame {self.current_frame}, Track {self.selected_track}")
        if semantic_labels:
            print(f"  With semantic labels: {semantic_labels}")

    def _interpolate_between_keyframes(self):
        """Interpolate position and rotation between keyframe 1 and keyframe 2"""
        # Validation
        if self.keyframe_1 is None or self.keyframe_2 is None:
            print("⚠️ Both keyframes must be set! Mark Keyframe 1 and Keyframe 2 first.")
            return

        kf1_frame, kf1_track, kf1_bbox, kf1_labels = self.keyframe_1
        kf2_frame, kf2_track, kf2_bbox, kf2_labels = self.keyframe_2

        # Check same track
        if kf1_track != kf2_track:
            print(f"⚠️ Keyframes are for different tracks! KF1: {kf1_track}, KF2: {kf2_track}")
            return

        # Check frame order
        if kf1_frame >= kf2_frame:
            print(f"⚠️ Keyframe 1 must be before Keyframe 2! KF1: {kf1_frame}, KF2: {kf2_frame}")
            return

        # Get frame range
        start_frame = kf1_frame
        end_frame = kf2_frame
        track_id = kf1_track

        # Find intermediate frames (include end_frame so kf2 is also saved as correction)
        intermediate_frames = [f for f in self.frame_indices if start_frame < f <= end_frame]

        if not intermediate_frames:
            print(f"⚠️ No frames to interpolate between {start_frame} and {end_frame}!")
            return

        # Check for existing corrections and ask for confirmation
        existing_corrections = []
        for frame_idx in intermediate_frames:
            if self._has_correction(track_id, frame_idx):
                existing_corrections.append(frame_idx)

        if existing_corrections:
            print(f"\n⚠️ WARNING: {len(existing_corrections)} frames already have corrections:")
            print(f"   Frames: {existing_corrections[:5]}{'...' if len(existing_corrections) > 5 else ''}")
            print(f"   Interpolation will OVERWRITE these corrections.")
            print(f"   Click 'Confirm Overwrite & Interpolate' to proceed")

            # Store pending interpolation
            self.pending_interpolation = (
                track_id, start_frame, end_frame,
                kf1_bbox, kf2_bbox, kf1_labels, kf2_labels,
                intermediate_frames
            )

            # Show confirmation button
            self.confirm_interpolate_btn.visible = True
            return

        # No conflicts, proceed directly
        self._perform_interpolation(
            track_id, start_frame, end_frame,
            kf1_bbox, kf2_bbox, kf1_labels, kf2_labels,
            intermediate_frames
        )

    def _perform_interpolation(self, track_id, start_frame, end_frame, kf1_bbox, kf2_bbox,
                               kf1_labels, kf2_labels, intermediate_frames):
        """Actually perform the interpolation (separated for clarity)

        FIX in v4: Correctly saves kf2 frame with exact kf2 bbox values
        """
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        print(f"\n🎬 Interpolating {len(intermediate_frames)} frames between {start_frame} and {end_frame}...")

        # Extract keyframe data
        pos1 = kf1_bbox.center.copy()
        pos2 = kf2_bbox.center.copy()
        rot1_matrix = kf1_bbox.rotation_matrix.copy()
        rot2_matrix = kf2_bbox.rotation_matrix.copy()
        dims1 = kf1_bbox.dimensions.copy()
        dims2 = kf2_bbox.dimensions.copy()

        # Convert rotation matrices to scipy Rotation objects for SLERP
        rot1 = R.from_matrix(rot1_matrix)
        rot2 = R.from_matrix(rot2_matrix)

        # Create SLERP interpolator for rotations
        key_times = np.array([0.0, 1.0])  # Normalized time: start=0, end=1
        key_rots = R.from_matrix(np.array([rot1_matrix, rot2_matrix]))
        slerp = Slerp(key_times, key_rots)

        # Determine semantic label propagation strategy
        # If both keyframes have labels, we can intelligently propagate
        use_intelligent_labels = (kf1_labels is not None and kf2_labels is not None)

        # Initialize corrections dict for track if needed
        if track_id not in self.corrections:
            self.corrections[track_id] = {}
        if track_id not in self.semantic_faces:
            self.semantic_faces[track_id] = {}

        # Interpolate each intermediate frame
        for i, frame_idx in enumerate(intermediate_frames):
            # Calculate interpolation factor (0 to 1)
            # Frame positions in the sequence
            frame_position = self.frame_indices.index(frame_idx)
            start_position = self.frame_indices.index(start_frame)
            end_position = self.frame_indices.index(end_frame)

            # Linear interpolation factor
            t = (frame_position - start_position) / (end_position - start_position)

            # FIX in v4: For kf2 (t=1.0), use exact kf2 values to ensure consistency
            if frame_idx == end_frame:
                # Use exact kf2 values
                interpolated_pos = pos2.copy()
                interpolated_rot_matrix = rot2_matrix.copy()
                interpolated_dims = dims2.copy()
                labels_to_use = kf2_labels
            else:
                # Interpolate position (LERP)
                interpolated_pos = pos1 + t * (pos2 - pos1)

                # Interpolate rotation (SLERP)
                interpolated_rot = slerp([t])[0]  # slerp returns Rotation object
                interpolated_rot_matrix = interpolated_rot.as_matrix()

                # Interpolate dimensions (LERP) - NEW in v4
                interpolated_dims = dims1 + t * (dims2 - dims1)

                # Determine labels
                if use_intelligent_labels:
                    labels_to_use = kf1_labels if t < 0.5 else kf2_labels
                else:
                    labels_to_use = kf1_labels

            # Create correction entry
            self.corrections[track_id][frame_idx] = BBox3D(
                center=interpolated_pos,
                dimensions=interpolated_dims,
                rotation_matrix=interpolated_rot_matrix,
                class_name=kf1_bbox.class_name,
                track_id=track_id,
                frame_idx=frame_idx,
                # Preserve metadata
                confidence=kf1_bbox.confidence,
                instance_id=kf1_bbox.instance_id,
                persistent_instance_id=kf1_bbox.persistent_instance_id
            )

            # Save semantic labels
            if labels_to_use is not None:
                self.semantic_faces[track_id][frame_idx] = labels_to_use.copy()

        print(f"✓ Interpolated {len(intermediate_frames)} frames (including kf2):")
        print(f"  - Position: Smoothly transitioned from {pos1} to {pos2}")
        print(f"  - Rotation: SLERP between start and end orientations")
        print(f"  - Dimensions: Interpolated from {dims1} to {dims2}")
        if use_intelligent_labels:
            print(f"  - Semantic labels: Start labels for first half, end labels for second half")
        elif kf1_labels:
            print(f"  - Semantic labels: Propagated from keyframe 1")
        print(f"  - kf2 (frame {end_frame}): Saved with exact kf2 values")

        # Re-render current frame
        self._render_frame()

        # Schedule auto-save
        self._schedule_auto_save()

        # Clear keyframes for next interpolation
        self.keyframe_1 = None
        self.keyframe_2 = None
        self.keyframe_1_text.value = "Not set (cleared)"
        self.keyframe_2_text.value = "Not set (cleared)"

    def _interpolate_semantic_faces_only(self):
        """NEW v6: Interpolate/propagate semantic face labels between keyframes
        WITHOUT modifying bbox positions, rotations, or dimensions.

        Use case: All boxes are already correctly positioned (e.g., after running
        full interpolation), but semantic face labels need to be propagated or
        updated between two keyframes.

        Workflow:
        1. Mark Keyframe 1 (with correct semantic face labels)
        2. Mark Keyframe 2 (with correct semantic face labels)
        3. Click "Interpolate Semantic Faces Only"
        -> Labels from KF1 applied to first half, labels from KF2 to second half
        """
        # Validation
        if self.keyframe_1 is None or self.keyframe_2 is None:
            print("⚠️ Both keyframes must be set! Mark Keyframe 1 and Keyframe 2 first.")
            return

        kf1_frame, kf1_track, kf1_bbox, kf1_labels = self.keyframe_1
        kf2_frame, kf2_track, kf2_bbox, kf2_labels = self.keyframe_2

        # Check same track
        if kf1_track != kf2_track:
            print(f"⚠️ Keyframes are for different tracks! KF1: {kf1_track}, KF2: {kf2_track}")
            return

        # Check frame order
        if kf1_frame >= kf2_frame:
            print(f"⚠️ Keyframe 1 must be before Keyframe 2! KF1: {kf1_frame}, KF2: {kf2_frame}")
            return

        # Check that at least one keyframe has semantic labels
        if kf1_labels is None and kf2_labels is None:
            print("⚠️ Neither keyframe has semantic face labels! Set labels on at least one keyframe.")
            return

        # Get frame range
        start_frame = kf1_frame
        end_frame = kf2_frame
        track_id = kf1_track

        # Find all frames for this track between keyframes (inclusive of both)
        frames_to_update = [f for f in self.frame_indices if start_frame <= f <= end_frame]

        # Filter to only frames where the track exists
        valid_frames = []
        for frame_idx in frames_to_update:
            track_exists = any(
                bbox.track_id == track_id
                for bbox in self.auto_bboxes.get(frame_idx, [])
            )
            if track_exists:
                valid_frames.append(frame_idx)

        if not valid_frames:
            print(f"⚠️ No frames found for track {track_id} between {start_frame} and {end_frame}!")
            return

        print(f"\n🏷️ Interpolating semantic faces for {len(valid_frames)} frames...")
        print(f"   Track {track_id}: Frame {start_frame} -> Frame {end_frame}")
        print(f"   KF1 labels: {kf1_labels}")
        print(f"   KF2 labels: {kf2_labels}")

        # Initialize semantic_faces dict for track if needed
        if track_id not in self.semantic_faces:
            self.semantic_faces[track_id] = {}

        # Determine propagation strategy
        if kf1_labels is not None and kf2_labels is not None:
            # Both keyframes have labels - use intelligent split
            strategy = "split"
        elif kf1_labels is not None:
            # Only KF1 has labels - propagate forward
            strategy = "forward"
        else:
            # Only KF2 has labels - propagate backward
            strategy = "backward"

        # Calculate midpoint for split strategy
        start_position = self.frame_indices.index(start_frame)
        end_position = self.frame_indices.index(end_frame)
        mid_position = (start_position + end_position) // 2

        # Apply labels to each frame
        updated_count = 0
        for frame_idx in valid_frames:
            frame_position = self.frame_indices.index(frame_idx)

            if strategy == "split":
                # Use KF1 labels for first half, KF2 labels for second half
                if frame_position <= mid_position:
                    labels_to_apply = kf1_labels.copy()
                else:
                    labels_to_apply = kf2_labels.copy()
            elif strategy == "forward":
                labels_to_apply = kf1_labels.copy()
            else:  # backward
                labels_to_apply = kf2_labels.copy()

            self.semantic_faces[track_id][frame_idx] = labels_to_apply
            updated_count += 1

        print(f"✓ Applied semantic face labels to {updated_count} frames")
        if strategy == "split":
            print(f"  - Frames {start_frame} to {self.frame_indices[mid_position]}: KF1 labels")
            print(f"  - Frames {self.frame_indices[mid_position + 1]} to {end_frame}: KF2 labels")
        elif strategy == "forward":
            print(f"  - All frames: KF1 labels (propagated forward)")
        else:
            print(f"  - All frames: KF2 labels (propagated backward)")

        # Re-render current frame to show updated labels
        self._render_frame()

        # Update semantic dropdowns if on current frame
        self._update_semantic_dropdowns()

        # Schedule auto-save
        self._schedule_auto_save()

        # Clear keyframes for next operation
        self.keyframe_1 = None
        self.keyframe_2 = None
        self.keyframe_1_text.value = "Not set (cleared)"
        self.keyframe_2_text.value = "Not set (cleared)"

    def _next_unannotated_frame(self):
        """Jump to next frame that doesn't have corrections for current track"""
        if self.selected_track is None:
            print("⚠️ No track selected!")
            return

        # Find next frame without corrections
        found_frame = None
        for i in range(self.current_frame_idx + 1, len(self.frame_indices)):
            frame_idx = self.frame_indices[i]

            # Check if this frame has this track
            track_exists = False
            for bbox in self.auto_bboxes.get(frame_idx, []):
                if bbox.track_id == self.selected_track:
                    track_exists = True
                    break

            if not track_exists:
                continue

            # Check if already corrected
            if not self._has_correction(self.selected_track, frame_idx):
                found_frame = i
                break

        if found_frame is None:
            # Wrap around to beginning
            for i in range(0, self.current_frame_idx):
                frame_idx = self.frame_indices[i]

                track_exists = False
                for bbox in self.auto_bboxes.get(frame_idx, []):
                    if bbox.track_id == self.selected_track:
                        track_exists = True
                        break

                if not track_exists:
                    continue

                if not self._has_correction(self.selected_track, frame_idx):
                    found_frame = i
                    break

        if found_frame is None:
            print(f"✓ All frames annotated for track {self.selected_track}!")
            return

        # Jump to that frame
        self.frame_slider.value = found_frame
        print(f"→ Jumped to frame {self.frame_indices[found_frame]} (unannotated)")

    def run(self):
        """Main loop"""
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nSaving...")
            # Cancel pending auto-save timer
            if self.auto_save_timer is not None:
                self.auto_save_timer.cancel()
            self._save_corrections()
            print("Done!")


def main():
    parser = argparse.ArgumentParser(description="3D Bbox Editor v6 - with Semantic Face Interpolation")
    parser.add_argument("--auto_bboxes", required=True, help="Dir with auto bbox JSONs")
    parser.add_argument("--output", required=True, help="Output dir for corrections")
    parser.add_argument("--images", help="Dir with images for 2D visualization (optional)")
    parser.add_argument("--mask_dir", help="Dir with grounded-sam masks for track highlighting (auto-detected if not provided)")
    parser.add_argument("--no_reload", action="store_true",
                        help="Don't reload previous annotations from output dir (default: reload)")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")

    args = parser.parse_args()

    editor = SimpleBBoxEditor(
        auto_bboxes_dir=args.auto_bboxes,
        output_dir=args.output,
        images_dir=args.images,
        mask_dir=args.mask_dir,
        reload_annotations=not args.no_reload,  # Default is True (reload), --no_reload sets to False
        port=args.port
    )

    editor.run()


if __name__ == "__main__":
    main()


# - (a bit more complicated) we can have an additional option of select which face is the front, right and top for the current frame here instead of semantic propagation. Just an option