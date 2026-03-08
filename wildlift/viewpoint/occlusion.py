#!/usr/bin/env python3
"""
Occlusion Analyzer for WildLIFT Pipeline

Answers practical ecologist questions about animal visibility and occlusion
in drone footage using 3D ray-OBB intersection and 2D mask overlap.

Key features:
    - Ray-OBB intersection: cast rays from camera through animal faces,
      check if nearer animals block the view (geometrically exact in 3D)
    - 2D mask overlap: pixel-level occlusion validation using segmentation masks
    - Per-face breakdown: which body parts are occluded and by whom
    - Best frame selection: least-occluded view of each animal/body part
    - Never-seen detection: body parts that were never clearly visible
    - PDF report, JSON summary, and annotated frame overlays

Usage:
    python occlusion_analyzer.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ \\
        --pdf --json --annotated_frames
"""

import os
import json
import numpy as np
import cv2
import argparse
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False

from wildlift.viewpoint.analyzer import ViewpointAnalyzer, MaskCropExtractor, CONFIG


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FaceOcclusionDetail:
    """Per-face occlusion measurement for one frame."""
    face_name: str
    self_visible: bool          # camera can see this face (dot product > 0)
    visibility_score: float     # dot product magnitude
    ray_occlusion_pct: float    # % of face area blocked by other animals (ray-OBB)
    occluding_tracks: List[int] # which animals block this face
    effective_score: float      # visibility_score * (1 - ray_occlusion_pct/100)


@dataclass
class OcclusionRecord:
    """Per-frame, per-animal occlusion measurement."""
    track_id: int
    frame_name: str
    face_details: Dict[str, FaceOcclusionDetail]
    self_visibility_pct: float      # % of faces visible from viewing angle
    ray_occlusion_pct: float        # % of visible area blocked by other animals
    mask_overlap_pct: float         # 2D mask overlap (if available, else -1)
    total_visibility_pct: float     # combined effective visibility
    occluding_tracks: List[int]     # all tracks that block this animal


@dataclass
class TrackOcclusionSummary:
    """Aggregated occlusion stats across all frames for one animal."""
    track_id: int
    class_name: str
    total_frames: int
    mean_visibility_pct: float
    best_frame: str
    worst_frame: str
    best_visibility: float
    worst_visibility: float
    never_seen_faces: List[str]
    consistently_occluded_by: List[int]
    best_frames_per_face: Dict[str, List[str]]


# =============================================================================
# RAY-OBB INTERSECTION (SLAB METHOD)
# =============================================================================

def ray_obb_intersect(ray_origin, ray_dir, obb_center, obb_half_dims, obb_rotation):
    """
    Test ray against oriented bounding box using the slab method.

    Args:
        ray_origin: (3,) ray start point (camera position)
        ray_dir: (3,) normalized ray direction
        obb_center: (3,) center of the OBB
        obb_half_dims: (3,) half-dimensions [l/2, w/2, h/2]
        obb_rotation: (3,3) rotation matrix (columns = OBB local axes in world frame)

    Returns:
        float: distance along ray to first intersection, or None if no hit
    """
    delta = ray_origin - obb_center
    local_origin = obb_rotation.T @ delta
    local_dir = obb_rotation.T @ ray_dir

    t_min = -np.inf
    t_max = np.inf

    for axis in range(3):
        if abs(local_dir[axis]) < 1e-8:
            # Ray parallel to this slab
            if abs(local_origin[axis]) > obb_half_dims[axis]:
                return None  # outside slab
        else:
            t1 = (-obb_half_dims[axis] - local_origin[axis]) / local_dir[axis]
            t2 = (+obb_half_dims[axis] - local_origin[axis]) / local_dir[axis]
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)
            if t_min > t_max:
                return None

    if t_min > 0:
        return t_min
    elif t_max > 0:
        return t_max
    return None


def sample_face_points(face_corners, n_samples=8):
    """
    Generate a uniform grid of sample points on a quadrilateral face.

    Args:
        face_corners: (4, 3) array of face corner points (ordered around quad)
        n_samples: number of samples along each edge

    Returns:
        (n_samples*n_samples, 3) array of sample points
    """
    c0, c1, c2, c3 = face_corners
    points = []
    for u_i in range(n_samples):
        u = (u_i + 0.5) / n_samples
        for v_i in range(n_samples):
            v = (v_i + 0.5) / n_samples
            # Bilinear interpolation on quad
            p = (1 - u) * (1 - v) * c0 + u * (1 - v) * c1 + u * v * c2 + (1 - u) * v * c3
            points.append(p)
    return np.array(points)


# =============================================================================
# OCCLUSION ANALYZER
# =============================================================================

class OcclusionAnalyzer:
    """
    Occlusion analysis using 3D ray-OBB intersection and 2D mask overlap.
    Composes ViewpointAnalyzer for geometry, camera loading, and face visibility.
    """

    def __init__(self, annotator_output_dir, images_dir=None, ray_samples=8):
        self.annotator_output_dir = Path(annotator_output_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.ray_samples = ray_samples

        # Compose ViewpointAnalyzer for all geometry and data loading
        self.vp = ViewpointAnalyzer(annotator_output_dir, images_dir)

        # Set up mask extractor
        self.mask_extractor = None
        if images_dir:
            results_dir = self.annotator_output_dir.parent
            self.mask_extractor = MaskCropExtractor(
                images_dir=Path(images_dir),
                results_dir=results_dir,
            )

        # Mask directory for raw RLE loading
        self.mask_dir = None
        if self.mask_extractor and self.mask_extractor.mask_dir:
            self.mask_dir = self.mask_extractor.mask_dir

        # Cache frame qualities from viewpoint analyzer
        self._frame_qualities_cache = {}

        print(f"OcclusionAnalyzer initialized:")
        print(f"  Ray samples per face edge: {ray_samples} ({ray_samples**2} per face)")
        print(f"  Tracks: {self.vp.labeled_tracks}")
        print(f"  Frames: {len(self.vp.frame_order)}")

    # ------------------------------------------------------------------
    # Core 3D occlusion
    # ------------------------------------------------------------------

    def _get_bbox_as_obb(self, bbox_data):
        """Convert stored bbox data to OBB parameters (center, half_dims, rotation)."""
        center = np.array(bbox_data['center'])
        dims = np.array(bbox_data['dimensions'])
        rotation = np.array(bbox_data['rotation_matrix'])
        half_dims = dims / 2.0
        return center, half_dims, rotation

    def compute_face_ray_occlusion(self, cam_pos, face_data, other_obbs, own_track_id):
        """
        Cast rays from camera through sample points on a face and test
        against all other animals' OBBs.

        Args:
            cam_pos: (3,) camera position in world coords
            face_data: dict with 'corners', 'normal', 'center' for the face
            other_obbs: list of (track_id, center, half_dims, rotation) for other animals
            own_track_id: track id of the animal whose face we're testing

        Returns:
            (ray_occlusion_pct, occluding_tracks)
        """
        face_corners = face_data['corners']
        sample_pts = sample_face_points(face_corners, self.ray_samples)
        total_samples = len(sample_pts)

        if total_samples == 0:
            return 0.0, []

        occluded_count = 0
        occluder_set = set()

        for pt in sample_pts:
            ray_dir = pt - cam_pos
            dist_to_face = np.linalg.norm(ray_dir)
            if dist_to_face < 1e-8:
                continue
            ray_dir = ray_dir / dist_to_face

            for other_tid, obb_center, obb_half, obb_rot in other_obbs:
                if other_tid == own_track_id:
                    continue
                hit_t = ray_obb_intersect(cam_pos, ray_dir, obb_center, obb_half, obb_rot)
                if hit_t is not None and hit_t < dist_to_face:
                    occluded_count += 1
                    occluder_set.add(other_tid)
                    break  # this sample is occluded, no need to check more OBBs

        pct = (occluded_count / total_samples) * 100.0
        return pct, list(occluder_set)

    def compute_frame_occlusion(self, frame_name):
        """
        Compute occlusion for all tracked animals in a single frame.

        Returns:
            Dict[int, OcclusionRecord] keyed by track_id
        """
        # Load camera params
        camera_params = self.vp._load_camera_params(frame_name)
        if camera_params is None:
            return {}

        cam_pos = camera_params['t']

        # Collect all OBBs for animals present in this frame
        obbs = []  # (track_id, center, half_dims, rotation)
        track_bbox_data = {}  # track_id -> raw bbox_data
        for track_id in self.vp.labeled_tracks:
            if frame_name in self.vp.all_bbox_data.get(track_id, {}):
                bbox_data = self.vp.all_bbox_data[track_id][frame_name]
                center, half_dims, rotation = self._get_bbox_as_obb(bbox_data)
                obbs.append((track_id, center, half_dims, rotation))
                track_bbox_data[track_id] = bbox_data

        if len(obbs) == 0:
            return {}

        # Get image shape for quality computation
        img_shape = (480, 640, 3)
        if self.images_dir:
            for ext in ['.jpg', '.png']:
                img_path = self.images_dir / f"{frame_name}{ext}"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_shape = img.shape
                    break

        records = {}

        for track_id in track_bbox_data:
            if track_id not in self.vp.semantic_faces:
                continue
            if frame_name not in self.vp.semantic_faces[track_id]:
                continue

            semantic_faces = self.vp.semantic_faces[track_id][frame_name]
            face_details = {}
            all_occluders = set()
            visible_face_count = 0
            total_face_count = 0
            weighted_occlusion_sum = 0.0
            weighted_visibility_sum = 0.0

            for face_name in CONFIG['VISIBLE_FACES']:
                total_face_count += 1

                if face_name not in semantic_faces:
                    face_details[face_name] = FaceOcclusionDetail(
                        face_name=face_name,
                        self_visible=False,
                        visibility_score=0.0,
                        ray_occlusion_pct=0.0,
                        occluding_tracks=[],
                        effective_score=0.0,
                    )
                    continue

                face_data = semantic_faces[face_name]

                # Self-visibility (viewing angle)
                is_visible, vis_score = self.vp._calculate_face_visibility(
                    face_data, camera_params
                )

                if not is_visible:
                    face_details[face_name] = FaceOcclusionDetail(
                        face_name=face_name,
                        self_visible=False,
                        visibility_score=0.0,
                        ray_occlusion_pct=0.0,
                        occluding_tracks=[],
                        effective_score=0.0,
                    )
                    continue

                visible_face_count += 1

                # Ray-OBB inter-animal occlusion
                ray_occ_pct, occluders = self.compute_face_ray_occlusion(
                    cam_pos, face_data, obbs, track_id
                )
                all_occluders.update(occluders)

                effective = abs(vis_score) * (1.0 - ray_occ_pct / 100.0)

                face_details[face_name] = FaceOcclusionDetail(
                    face_name=face_name,
                    self_visible=True,
                    visibility_score=abs(vis_score),
                    ray_occlusion_pct=ray_occ_pct,
                    occluding_tracks=occluders,
                    effective_score=effective,
                )

                weighted_visibility_sum += abs(vis_score)
                weighted_occlusion_sum += abs(vis_score) * ray_occ_pct

            # Aggregate
            self_vis_pct = (visible_face_count / max(total_face_count, 1)) * 100.0

            if weighted_visibility_sum > 0:
                avg_ray_occ = weighted_occlusion_sum / weighted_visibility_sum
            else:
                avg_ray_occ = 0.0

            total_vis = self_vis_pct * (1.0 - avg_ray_occ / 100.0)

            # 2D mask overlap (if available)
            mask_overlap = -1.0  # sentinel: not computed

            records[track_id] = OcclusionRecord(
                track_id=track_id,
                frame_name=frame_name,
                face_details=face_details,
                self_visibility_pct=self_vis_pct,
                ray_occlusion_pct=avg_ray_occ,
                mask_overlap_pct=mask_overlap,
                total_visibility_pct=total_vis,
                occluding_tracks=list(all_occluders),
            )

        return records

    # ------------------------------------------------------------------
    # 2D mask overlap (validation)
    # ------------------------------------------------------------------

    def compute_mask_overlap(self, frame_name, camera_params=None):
        """
        Compute 2D mask overlap between animals in a frame.
        Returns Dict[track_id, overlap_pct].
        """
        if not PYCOCOTOOLS_AVAILABLE or self.mask_dir is None:
            return {}

        if camera_params is None:
            camera_params = self.vp._load_camera_params(frame_name)
        if camera_params is None:
            return {}

        cam_pos = camera_params['t']

        # Extract frame number for mask file lookup
        frame_num = re.findall(r'\d+', str(frame_name))
        frame_key = frame_num[0] if frame_num else frame_name

        # Load mask file
        json_file = None
        for name in [frame_key, frame_name]:
            candidate = self.mask_dir / f"{name}_results.json"
            if candidate.exists():
                json_file = candidate
                break
        if json_file is None:
            return {}

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
            annotations = results.get('annotations', [])
        except Exception:
            return {}

        # Load mask-track mapping
        mapping = self.mask_extractor.mask_track_mapping if self.mask_extractor else {}
        frame_mapping = mapping.get(str(frame_key), {})

        # Build track_id -> decoded mask
        track_masks = {}
        for track_id_str, mask_idx in frame_mapping.items():
            track_id = int(track_id_str)
            if mask_idx < len(annotations):
                rle = annotations[mask_idx].get('segmentation')
                if rle:
                    track_masks[track_id] = mask_utils.decode(rle).astype(bool)

        if len(track_masks) < 2:
            return {tid: 0.0 for tid in track_masks}

        # Depth ordering
        track_depths = {}
        for tid in track_masks:
            if tid in self.vp.all_bbox_data and frame_name in self.vp.all_bbox_data[tid]:
                center = np.array(self.vp.all_bbox_data[tid][frame_name]['center'])
                track_depths[tid] = np.linalg.norm(cam_pos - center)
            else:
                track_depths[tid] = 0.0

        # Compute overlap for each track (farther animal occluded by nearer)
        overlaps = {}
        for tid_a, mask_a in track_masks.items():
            mask_a_area = np.sum(mask_a)
            if mask_a_area == 0:
                overlaps[tid_a] = 0.0
                continue

            overlap_pixels = 0
            for tid_b, mask_b in track_masks.items():
                if tid_b == tid_a:
                    continue
                # Only count if B is nearer (smaller depth)
                if track_depths.get(tid_b, 0) < track_depths.get(tid_a, 0):
                    overlap_pixels += np.sum(mask_a & mask_b)

            overlaps[tid_a] = (overlap_pixels / mask_a_area) * 100.0

        return overlaps

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze_all_frames(self):
        """
        Run occlusion analysis on all frames.

        Returns:
            Dict[int, List[OcclusionRecord]] keyed by track_id
        """
        all_records = defaultdict(list)
        n_frames = len(self.vp.frame_order)

        print(f"\nAnalyzing occlusion across {n_frames} frames...")

        for i, frame_name in enumerate(self.vp.frame_order):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Frame {i+1}/{n_frames}: {frame_name}")

            frame_records = self.compute_frame_occlusion(frame_name)

            # Optionally add 2D mask overlap
            mask_overlaps = self.compute_mask_overlap(frame_name)
            for tid, record in frame_records.items():
                if tid in mask_overlaps:
                    record.mask_overlap_pct = mask_overlaps[tid]
                all_records[tid].append(record)

        print(f"  Done. {sum(len(v) for v in all_records.values())} records across {len(all_records)} tracks.")
        return dict(all_records)

    def summarize_track(self, track_id, records):
        """Aggregate occlusion records into a per-track summary."""
        if not records:
            return None

        # Class name
        class_name = 'unknown'
        if track_id in self.vp.all_bbox_data:
            first_frame = next(iter(self.vp.all_bbox_data[track_id].values()))
            class_name = first_frame.get('class_name', 'unknown')

        visibilities = [r.total_visibility_pct for r in records]
        best_idx = int(np.argmax(visibilities))
        worst_idx = int(np.argmin(visibilities))

        # Never-seen faces: faces where effective_score never exceeds threshold
        face_max_scores = defaultdict(float)
        face_best_frames = defaultdict(list)
        for r in records:
            for fname, fd in r.face_details.items():
                if fd.effective_score > face_max_scores[fname]:
                    face_max_scores[fname] = fd.effective_score
                face_best_frames[fname].append((fd.effective_score, r.frame_name))

        never_seen = [f for f in CONFIG['VISIBLE_FACES']
                      if face_max_scores.get(f, 0) < 0.05]

        # Best frames per face (top 5)
        best_per_face = {}
        for fname in CONFIG['VISIBLE_FACES']:
            sorted_frames = sorted(face_best_frames.get(fname, []),
                                   key=lambda x: x[0], reverse=True)
            best_per_face[fname] = [f for _, f in sorted_frames[:5]]

        # Consistently occluded by
        occluder_counts = Counter()
        for r in records:
            for tid in r.occluding_tracks:
                occluder_counts[tid] += 1
        # Tracks that occlude in >25% of frames
        threshold = len(records) * 0.25
        consistent_occluders = [tid for tid, cnt in occluder_counts.items()
                                if cnt >= threshold]

        return TrackOcclusionSummary(
            track_id=track_id,
            class_name=class_name,
            total_frames=len(records),
            mean_visibility_pct=float(np.mean(visibilities)),
            best_frame=records[best_idx].frame_name,
            worst_frame=records[worst_idx].frame_name,
            best_visibility=visibilities[best_idx],
            worst_visibility=visibilities[worst_idx],
            never_seen_faces=never_seen,
            consistently_occluded_by=consistent_occluders,
            best_frames_per_face=best_per_face,
        )

    # ------------------------------------------------------------------
    # Ecologist query methods
    # ------------------------------------------------------------------

    def get_best_frames(self, track_id, records, face=None, top_k=5):
        """Get least-occluded frames for a track (optionally for a specific face)."""
        if face is None:
            scored = [(r.total_visibility_pct, r.frame_name) for r in records]
        else:
            scored = []
            for r in records:
                fd = r.face_details.get(face)
                if fd:
                    scored.append((fd.effective_score, r.frame_name))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    def get_never_seen_faces(self, track_id, records):
        """Get faces that were never clearly visible."""
        face_max = defaultdict(float)
        for r in records:
            for fname, fd in r.face_details.items():
                face_max[fname] = max(face_max[fname], fd.effective_score)
        return [f for f in CONFIG['VISIBLE_FACES'] if face_max.get(f, 0) < 0.05]

    def get_hard_to_photograph(self, all_records, threshold=50.0):
        """Get track IDs with mean visibility below threshold."""
        hard = []
        for tid, records in all_records.items():
            mean_vis = np.mean([r.total_visibility_pct for r in records])
            if mean_vis < threshold:
                hard.append((tid, mean_vis))
        hard.sort(key=lambda x: x[1])
        return hard

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def generate_json_summary(self, all_records, output_path):
        """Export full occlusion analysis as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summaries = {}
        for tid, records in all_records.items():
            summary = self.summarize_track(tid, records)
            if summary is None:
                continue
            summaries[tid] = {
                'track_id': tid,
                'class_name': summary.class_name,
                'total_frames': summary.total_frames,
                'mean_visibility_pct': round(summary.mean_visibility_pct, 2),
                'best_frame': summary.best_frame,
                'worst_frame': summary.worst_frame,
                'best_visibility': round(summary.best_visibility, 2),
                'worst_visibility': round(summary.worst_visibility, 2),
                'never_seen_faces': summary.never_seen_faces,
                'consistently_occluded_by': summary.consistently_occluded_by,
                'best_frames_per_face': summary.best_frames_per_face,
                'per_frame': [],
            }
            for r in records:
                frame_entry = {
                    'frame': r.frame_name,
                    'self_visibility_pct': round(r.self_visibility_pct, 2),
                    'ray_occlusion_pct': round(r.ray_occlusion_pct, 2),
                    'mask_overlap_pct': round(r.mask_overlap_pct, 2),
                    'total_visibility_pct': round(r.total_visibility_pct, 2),
                    'occluding_tracks': r.occluding_tracks,
                    'faces': {},
                }
                for fname, fd in r.face_details.items():
                    frame_entry['faces'][fname] = {
                        'visible': fd.self_visible,
                        'visibility_score': round(fd.visibility_score, 3),
                        'ray_occlusion_pct': round(fd.ray_occlusion_pct, 2),
                        'occluding_tracks': fd.occluding_tracks,
                        'effective_score': round(fd.effective_score, 3),
                    }
                summaries[tid]['per_frame'].append(frame_entry)

        report = {
            'analysis_date': datetime.now().isoformat(),
            'scene': self.annotator_output_dir.parent.name,
            'ray_samples': self.ray_samples,
            'total_tracks': len(summaries),
            'tracks': {str(k): v for k, v in summaries.items()},
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Saved JSON summary to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Annotated frames
    # ------------------------------------------------------------------

    def generate_annotated_frames(self, all_records, output_dir, occlusion_threshold=30.0, skip_text=False):
        """
        Save annotated images with occlusion overlays for frames with
        significant occlusion.
        """
        if self.images_dir is None:
            print("No images_dir provided, skipping annotated frames.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect frames with significant occlusion
        flagged_frames = set()
        frame_records_map = defaultdict(dict)
        for tid, records in all_records.items():
            for r in records:
                frame_records_map[r.frame_name][tid] = r
                if r.ray_occlusion_pct > occlusion_threshold:
                    flagged_frames.add(r.frame_name)

        print(f"Generating annotated frames for {len(flagged_frames)} frames with >{occlusion_threshold}% occlusion...")

        count = 0
        for frame_name in sorted(flagged_frames):
            # Load image
            img = None
            for ext in ['.jpg', '.png', '.jpeg']:
                img_path = self.images_dir / f"{frame_name}{ext}"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    break
            if img is None:
                continue

            overlay = img.copy()
            records_in_frame = frame_records_map[frame_name]

            for tid, record in records_in_frame.items():
                if record.ray_occlusion_pct < occlusion_threshold:
                    continue

                # Try to load mask for this track to show occluded region
                mask = self._load_track_mask(frame_name, tid)

                if mask is not None:
                    # Red overlay on mask region proportional to occlusion
                    red_overlay = np.zeros_like(img)
                    red_overlay[:, :, 2] = 255  # Red channel
                    alpha = min(record.ray_occlusion_pct / 100.0, 0.7)
                    mask_bool = mask.astype(bool)
                    overlay[mask_bool] = cv2.addWeighted(
                        overlay[mask_bool], 1.0 - alpha,
                        red_overlay[mask_bool], alpha, 0
                    )

                # Add text label (unless skip_text is set)
                if not skip_text:
                    bbox_data = self.vp.all_bbox_data.get(tid, {}).get(frame_name)
                    if bbox_data is not None:
                        camera_params = self.vp._load_camera_params(frame_name)
                        if camera_params is not None:
                            center_3d = np.array(bbox_data['center'])
                            K = camera_params['K']
                            R = camera_params['R']
                            t = camera_params['t']
                            pose = np.eye(4)
                            pose[:3, :3] = R
                            pose[:3, 3] = t
                            center_cam = (np.linalg.inv(pose) @ np.append(center_3d, 1))[:3]
                            if center_cam[2] > 0:
                                center_2d = (K @ center_cam)
                                center_2d = (center_2d[:2] / center_2d[2]).astype(int)
                                h, w = img.shape[:2]
                                cx = max(10, min(center_2d[0], w - 200))
                                cy = max(30, min(center_2d[1], h - 10))

                                occluders_str = ','.join(str(t) for t in record.occluding_tracks)
                                label = f"T{tid}: {record.total_visibility_pct:.0f}% vis"
                                if occluders_str:
                                    label += f" (occ by T{occluders_str})"

                                cv2.putText(overlay, label, (cx, cy),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 2, cv2.LINE_AA)

            out_path = output_dir / f"{frame_name}_occlusion.png"
            cv2.imwrite(str(out_path), overlay)
            count += 1

        print(f"Saved {count} annotated frames to: {output_dir}")

    def _load_track_mask(self, frame_name, track_id):
        """Load decoded mask for a specific track in a frame."""
        if not PYCOCOTOOLS_AVAILABLE or self.mask_dir is None:
            return None

        frame_num = re.findall(r'\d+', str(frame_name))
        frame_key = frame_num[0] if frame_num else frame_name

        mapping = self.mask_extractor.mask_track_mapping if self.mask_extractor else {}
        mask_idx = mapping.get(str(frame_key), {}).get(str(track_id))
        if mask_idx is None:
            return None

        json_file = None
        for name in [frame_key, frame_name]:
            candidate = self.mask_dir / f"{name}_results.json"
            if candidate.exists():
                json_file = candidate
                break
        if json_file is None:
            return None

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
            annotations = results.get('annotations', [])
            if mask_idx < len(annotations):
                rle = annotations[mask_idx].get('segmentation')
                if rle:
                    return mask_utils.decode(rle)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------

    def generate_pdf_report(self, all_records, output_path):
        """Generate publication-quality occlusion analysis PDF."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        typo = CONFIG['TYPOGRAPHY']
        summaries = {}
        for tid, records in all_records.items():
            s = self.summarize_track(tid, records)
            if s:
                summaries[tid] = s

        with PdfPages(str(output_path)) as pdf:
            self._pdf_visibility_heatmap(pdf, all_records, summaries, typo)
            self._pdf_per_animal_summary(pdf, all_records, summaries, typo)
            self._pdf_hard_to_photograph(pdf, all_records, summaries, typo)

        print(f"Saved PDF report to: {output_path}")
        return output_path

    def _pdf_visibility_heatmap(self, pdf, all_records, summaries, typo):
        """Page 1: Visibility heatmap (tracks x selected frames)."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('OCCLUSION ANALYSIS - Visibility Heatmap',
                      fontsize=typo['title'], fontweight='bold', y=0.96)

        track_ids = sorted(all_records.keys())
        if not track_ids:
            pdf.savefig(fig)
            plt.close(fig)
            return

        # Collect all frames across tracks (sorted)
        all_frames = sorted(set(
            r.frame_name for records in all_records.values() for r in records
        ), key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

        # Subsample frames if too many
        max_cols = 50
        if len(all_frames) > max_cols:
            step = len(all_frames) // max_cols
            display_frames = all_frames[::step]
        else:
            display_frames = all_frames

        # Build heatmap matrix
        heat_data = np.full((len(track_ids), len(display_frames)), np.nan)
        frame_to_col = {f: i for i, f in enumerate(display_frames)}

        for row, tid in enumerate(track_ids):
            frame_map = {r.frame_name: r for r in all_records[tid]}
            for frame_name, col in frame_to_col.items():
                if frame_name in frame_map:
                    heat_data[row, col] = frame_map[frame_name].total_visibility_pct

        ax = fig.add_axes([0.12, 0.2, 0.78, 0.65])
        cmap = LinearSegmentedColormap.from_list('vis',
            ['#D32F2F', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50'])
        im = ax.imshow(heat_data, cmap=cmap, aspect='auto', vmin=0, vmax=100)

        ax.set_yticks(range(len(track_ids)))
        ax.set_yticklabels([f'Track {t}' for t in track_ids], fontsize=typo['body'])

        # X labels: show every Nth frame number
        n_labels = min(15, len(display_frames))
        label_step = max(1, len(display_frames) // n_labels)
        xtick_pos = list(range(0, len(display_frames), label_step))
        xtick_labels = []
        for pos in xtick_pos:
            fn = display_frames[pos]
            nums = re.findall(r'\d+', fn)
            xtick_labels.append(nums[0] if nums else fn)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=typo['caption'], rotation=45)
        ax.set_xlabel('Frame', fontsize=typo['body'])

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Total Visibility %', fontsize=typo['body'])

        # Stats text
        stats_ax = fig.add_axes([0.1, 0.04, 0.8, 0.1])
        stats_ax.axis('off')
        hard = self.get_hard_to_photograph(all_records, 50.0)
        stats_text = (
            f"Tracks: {len(track_ids)}  |  "
            f"Frames: {len(all_frames)}  |  "
            f"Hard to photograph (<50% vis): {len(hard)} tracks"
        )
        if hard:
            stats_text += f"\n  Difficult tracks: " + ", ".join(
                f"T{tid} ({vis:.0f}%)" for tid, vis in hard[:5]
            )
        stats_ax.text(0.5, 0.5, stats_text, ha='center', va='center',
                      fontsize=typo['body'],
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                                edgecolor='#E0E0E0'))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _pdf_per_animal_summary(self, pdf, all_records, summaries, typo):
        """Page 2+: Per-animal summary cards with face breakdown."""
        track_ids = sorted(summaries.keys())
        tracks_per_page = 3

        for page_start in range(0, len(track_ids), tracks_per_page):
            page_tracks = track_ids[page_start:page_start + tracks_per_page]

            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('Per-Animal Occlusion Summary',
                          fontsize=typo['subtitle'], fontweight='bold', y=0.97)

            n_tracks = len(page_tracks)
            track_height = 0.85 / n_tracks

            for idx, tid in enumerate(page_tracks):
                summary = summaries[tid]
                records = all_records[tid]

                top = 0.90 - idx * track_height
                bottom = top - track_height + 0.02

                # Track header
                header_ax = fig.add_axes([0.05, top - 0.03, 0.9, 0.03])
                header_ax.axis('off')
                header_ax.text(0.0, 0.5,
                               f'TRACK {tid} ({summary.class_name}) - '
                               f'Mean vis: {summary.mean_visibility_pct:.1f}% | '
                               f'{summary.total_frames} frames',
                               ha='left', va='center',
                               fontsize=typo['heading'], fontweight='bold')

                if summary.never_seen_faces:
                    header_ax.text(1.0, 0.5,
                                   f'Never seen: {", ".join(summary.never_seen_faces)}',
                                   ha='right', va='center',
                                   fontsize=typo['body'], color='#D32F2F')

                # Bar chart: per-face visibility vs occlusion
                bar_ax = fig.add_axes([0.05, bottom, 0.4, top - bottom - 0.05])
                faces = CONFIG['VISIBLE_FACES']
                x = np.arange(len(faces))
                width = 0.35

                # Compute mean per-face scores
                face_vis_means = []
                face_occ_means = []
                for fname in faces:
                    vis_scores = []
                    occ_scores = []
                    for r in records:
                        fd = r.face_details.get(fname)
                        if fd:
                            vis_scores.append(fd.visibility_score * 100 if fd.self_visible else 0)
                            occ_scores.append(fd.ray_occlusion_pct if fd.self_visible else 0)
                    face_vis_means.append(np.mean(vis_scores) if vis_scores else 0)
                    face_occ_means.append(np.mean(occ_scores) if occ_scores else 0)

                bars1 = bar_ax.bar(x - width/2, face_vis_means, width,
                                   label='Visibility', color='#4CAF50', alpha=0.8)
                bars2 = bar_ax.bar(x + width/2, face_occ_means, width,
                                   label='Occlusion', color='#D32F2F', alpha=0.8)

                bar_ax.set_xticks(x)
                bar_ax.set_xticklabels([f.upper()[:3] for f in faces],
                                       fontsize=typo['caption'])
                bar_ax.set_ylabel('%', fontsize=typo['caption'])
                bar_ax.legend(fontsize=typo['caption'], loc='upper right')
                bar_ax.set_ylim(0, 105)
                bar_ax.set_title('Mean Face Visibility vs Occlusion',
                                 fontsize=typo['body'])

                # Best/worst frames text
                text_ax = fig.add_axes([0.5, bottom, 0.45, top - bottom - 0.05])
                text_ax.axis('off')

                info_lines = [
                    f"Best frame: {summary.best_frame} ({summary.best_visibility:.1f}%)",
                    f"Worst frame: {summary.worst_frame} ({summary.worst_visibility:.1f}%)",
                ]
                if summary.consistently_occluded_by:
                    info_lines.append(
                        f"Often occluded by: T{', T'.join(str(t) for t in summary.consistently_occluded_by)}"
                    )
                info_lines.append("")
                info_lines.append("Best frames per face:")
                for fname in faces:
                    best = summary.best_frames_per_face.get(fname, [])[:3]
                    if best:
                        nums = [re.findall(r'\d+', f)[0] if re.findall(r'\d+', f) else f
                                for f in best]
                        info_lines.append(f"  {fname.upper()[:5]:5s}: {', '.join(nums)}")
                    else:
                        info_lines.append(f"  {fname.upper()[:5]:5s}: (none)")

                text_ax.text(0.0, 0.95, '\n'.join(info_lines),
                             ha='left', va='top',
                             fontsize=typo['body'], family='monospace',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='#FAFAFA', edgecolor='#E0E0E0'))

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _pdf_hard_to_photograph(self, pdf, all_records, summaries, typo):
        """Page: highlight animals that are consistently hard to see."""
        hard = self.get_hard_to_photograph(all_records, 50.0)
        if not hard:
            return  # no page needed

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Hard-to-Photograph Animals (<50% mean visibility)',
                      fontsize=typo['subtitle'], fontweight='bold', y=0.96)

        ax = fig.add_axes([0.1, 0.15, 0.8, 0.7])
        tids = [t for t, _ in hard]
        vis_vals = [v for _, v in hard]

        colors = ['#D32F2F' if v < 25 else '#FF9800' if v < 40 else '#FFC107'
                  for v in vis_vals]
        bars = ax.barh(range(len(tids)), vis_vals, color=colors, edgecolor='white')
        ax.set_yticks(range(len(tids)))
        labels = []
        for tid in tids:
            s = summaries.get(tid)
            label = f'Track {tid}'
            if s:
                label += f' ({s.class_name})'
            labels.append(label)
        ax.set_yticklabels(labels, fontsize=typo['body'])
        ax.set_xlabel('Mean Visibility %', fontsize=typo['body'])
        ax.set_xlim(0, 100)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax.text(51, -0.5, '50% threshold', fontsize=typo['caption'], color='gray')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, vis_vals)):
            ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=typo['caption'])

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Occlusion Analyzer - 3D ray-based occlusion analysis for wildlife ecology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full analysis with all outputs
    python occlusion_analyzer.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ \\
        --pdf --json --annotated_frames

    # JSON only, custom threshold
    python occlusion_analyzer.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --json --occlusion_threshold 20.0
        """
    )

    parser.add_argument("--annotator_output", required=True,
                        help="Path to annotator tool output directory")
    parser.add_argument("--images_dir", default=None,
                        help="Path to original images")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: {annotator_output}/occlusion_analysis)")
    parser.add_argument("--pdf", action="store_true",
                        help="Generate PDF report")
    parser.add_argument("--json", action="store_true",
                        help="Generate JSON summary")
    parser.add_argument("--annotated_frames", action="store_true",
                        help="Generate annotated frame images")
    parser.add_argument("--occlusion_threshold", type=float, default=30.0,
                        help="Occlusion threshold for annotated frames (default: 30%%)")
    parser.add_argument("--ray_samples", type=int, default=8,
                        help="Ray samples per face edge (default: 8, total=64 per face)")
    parser.add_argument("--top_k_frames", type=int, default=5,
                        help="Number of best frames to report per face")

    args = parser.parse_args()

    print("=" * 70)
    print("OCCLUSION ANALYZER")
    print("3D Ray-OBB + 2D Mask Overlap")
    print("=" * 70)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.annotator_output) / "occlusion_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    try:
        analyzer = OcclusionAnalyzer(
            annotator_output_dir=args.annotator_output,
            images_dir=args.images_dir,
            ray_samples=args.ray_samples,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Run analysis
    all_records = analyzer.analyze_all_frames()

    if not all_records:
        print("\nNo occlusion records generated. Check that bbox and semantic face data exist.")
        return 1

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for tid, records in sorted(all_records.items()):
        summary = analyzer.summarize_track(tid, records)
        if summary:
            print(f"  Track {tid} ({summary.class_name}): "
                  f"mean vis={summary.mean_visibility_pct:.1f}%, "
                  f"best={summary.best_frame} ({summary.best_visibility:.1f}%), "
                  f"never seen: {summary.never_seen_faces or 'none'}")
            if summary.consistently_occluded_by:
                print(f"    Consistently occluded by: {summary.consistently_occluded_by}")

    hard = analyzer.get_hard_to_photograph(all_records, 50.0)
    if hard:
        print(f"\n  Hard-to-photograph animals (<50% vis):")
        for tid, vis in hard:
            print(f"    Track {tid}: {vis:.1f}%")

    # Generate outputs
    if args.json:
        analyzer.generate_json_summary(
            all_records, output_dir / "occlusion_summary.json"
        )

    if args.annotated_frames:
        analyzer.generate_annotated_frames(
            all_records, output_dir / "occlusion_overlays",
            occlusion_threshold=args.occlusion_threshold,
        )

    if args.pdf:
        analyzer.generate_pdf_report(
            all_records, output_dir / "occlusion_report.pdf"
        )

    print(f"\nDone. Outputs in: {output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
