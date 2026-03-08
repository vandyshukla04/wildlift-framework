#!/usr/bin/env python3
"""
Viewpoint Analyzer v7 for WildLIFT Pipeline

Publication-quality aggregate PDF with enhanced visualization.

Key Changes from v6:
    - Fixed mask/crop loading bug in aggregate filmstrip
    - Typographic Scale: Consistent font sizing across all elements
    - Heat-encoded Matrix: Shows mean quality scores instead of presence count
    - Timeline Context: Frame numbers show temporal distribution
    - Rejected Frames: Can mark and display rejected frames in visualizations
    - Removed letter grades: Uses numeric coverage metrics instead

Usage:
    # Generate aggregate PDF from saved selections
    python viewpoint_analyzer_v7.py --annotator_output results/zebra/scene1/corrected/ \
        --images_dir data/zebra/scene1/images/ --load_saved --aggregate

    # Interactive selection + aggregate PDF with rejected frames
    python viewpoint_analyzer_v7.py --annotator_output results/zebra/scene1/corrected/ \
        --images_dir data/zebra/scene1/images/ --aggregate --select_rejected
"""

import os
import json
import numpy as np
import cv2
import glob
import argparse
from pathlib import Path
import re
import colorsys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, to_rgba, Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict, Counter
from scipy.stats import entropy

# Optional dependencies
try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Mask extraction will be limited.")


# =============================================================================
# GLOBAL CONFIGURATION (V7)
# =============================================================================

CONFIG = {
    # Reporting Settings
    'CROPS_PER_FACE': 5,
    'PDF_FIGURE_SIZE': (8.5, 11),
    'CROP_PADDING': 30,
    'CROP_BG_COLOR': 'white',

    # V7: Quality Filtering
    'MIN_QUALITY_THRESHOLD': 0.15,
    'MAX_CANDIDATES_TO_SHOW': 20,

    # V7: Typographic Scale (base 11pt, ratio 1.25)
    'TYPOGRAPHY': {
        'title': 16,       # Main titles
        'subtitle': 13,    # Section headers
        'heading': 11,     # Track/column headers
        'body': 9,         # Frame labels, stats
        'caption': 7,      # Small annotations
        'micro': 6,        # Tiny labels
    },

    # V7: Modern/Colorful Style - Orientation Colors
    'ORIENTATION_COLORS': {
        'front': '#E53935',   # Red 600
        'back': '#43A047',    # Green 600
        'left': '#1E88E5',    # Blue 600
        'right': '#FB8C00',   # Orange 600
        'top': '#8E24AA',     # Purple 600
        'bottom': '#00ACC1',  # Cyan 600
    },

    # Missing view indicator
    'MISSING_COLOR': '#BDBDBD',

    # Rejected frame indicator
    'REJECTED_COLOR': '#D32F2F',
    'REJECTED_ALPHA': 0.3,

    # Heat map colors (for quality matrix)
    'HEATMAP_COLORS': ['#FFFFFF', '#FFF9C4', '#FFEE58', '#FFC107', '#FF9800', '#FF5722'],

    # Semantic face labels
    'VISIBLE_FACES': ['front', 'back', 'left', 'right', 'top'],
    'ALL_FACES': ['front', 'back', 'left', 'right', 'top', 'bottom'],

    # Nature Methods publication style
    'NATURE_STYLE': {
        'ORIENTATION_COLORS': {  # Okabe-Ito colorblind-safe palette
            'front':  '#E69F00',  # Orange
            'back':   '#009E73',  # Bluish green
            'left':   '#0072B2',  # Blue
            'right':  '#D55E00',  # Vermillion
            'top':    '#CC79A7',  # Reddish purple
            'bottom': '#F0E442',  # Yellow
        },
        'TYPOGRAPHY': {  # Nature Methods: 5-7pt labels, sans-serif
            'title': 8,
            'subtitle': 7,
            'heading': 7,
            'body': 6,
            'caption': 5.5,
            'micro': 5,
        },
        'FIGURE_WIDTH_SINGLE': 3.504,   # 89mm
        'FIGURE_WIDTH_DOUBLE': 7.205,   # 183mm
        'DPI': 300,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CoverageResult:
    """Numeric coverage metrics (replaces letter grades)."""
    orientations_covered: int
    total_frames: int
    mean_quality: float
    coverage_percent: float
    missing_orientations: List[str]


@dataclass
class CandidateFrame:
    """A candidate frame for user selection."""
    frame: str
    quality_score: float
    frame_idx: int
    crop_image: Optional[np.ndarray] = None
    approved: bool = False
    rejected: bool = False


# =============================================================================
# VIEWPOINT ANALYZER CLASS (V7)
# =============================================================================

class ViewpointAnalyzer:
    """Viewpoint analysis with interactive selection support."""

    def __init__(self, annotator_output_dir, images_dir=None):
        self.annotator_output_dir = Path(annotator_output_dir)
        self.images_dir = Path(images_dir) if images_dir else None

        self._validate_input_paths()

        self.all_bbox_data = self._load_annotated_bboxes()
        self.semantic_faces = self._load_semantic_faces()
        self.labeled_tracks = list(self.semantic_faces.keys())
        self.frame_order = self._determine_frame_order()

        self.semantic_face_colors = CONFIG['ORIENTATION_COLORS']

        print(f"Viewpoint Analyzer v7 initialized:")
        print(f"  Annotator output: {self.annotator_output_dir}")
        print(f"  Tracks with semantic labels: {self.labeled_tracks}")
        print(f"  Total frames: {len(self.frame_order)}")

    def _validate_input_paths(self):
        bbox_dir = self.annotator_output_dir / "bounding_boxes"
        semantic_file = (self.annotator_output_dir /
                        "corrected_labels" / "semantic_faces" / "manual_labels.json")

        errors = []
        if not bbox_dir.exists():
            errors.append(f"Bounding boxes directory not found: {bbox_dir}")
        elif not list(bbox_dir.glob("*.json")):
            errors.append(f"No bounding box JSON files in: {bbox_dir}")

        if not semantic_file.exists():
            errors.append(f"Semantic face labels not found: {semantic_file}")

        if errors:
            raise FileNotFoundError("\n".join(errors))

    def _load_annotated_bboxes(self):
        bbox_dir = self.annotator_output_dir / "bounding_boxes"
        all_bbox_data = {}

        for bbox_file in sorted(bbox_dir.glob("*.json")):
            frame_name = bbox_file.stem
            with open(bbox_file, 'r') as f:
                frame_bboxes = json.load(f)

            for bbox in frame_bboxes:
                track_id = bbox.get('track_id')
                if track_id is None or track_id == -1:
                    continue

                if track_id not in all_bbox_data:
                    all_bbox_data[track_id] = {}

                all_bbox_data[track_id][frame_name] = {
                    'center': np.array(bbox['center']),
                    'dimensions': np.array(bbox['dimensions']),
                    'rotation_matrix': np.array(bbox['rotation_matrix']),
                    'track_id': track_id,
                    'class_name': bbox.get('class_name', 'animal'),
                    'confidence': bbox.get('confidence', 1.0)
                }

        total_frames = sum(len(frames) for frames in all_bbox_data.values())
        print(f"  Loaded bboxes: {len(all_bbox_data)} tracks, {total_frames} frame-instances")
        return all_bbox_data

    def _load_semantic_faces(self):
        semantic_file = (self.annotator_output_dir /
                        "corrected_labels" / "semantic_faces" / "manual_labels.json")

        with open(semantic_file, 'r') as f:
            raw_labels = json.load(f)

        semantic_faces = {}

        for track_id_str, frames_data in raw_labels.items():
            track_id = int(track_id_str)
            if track_id not in self.all_bbox_data:
                continue

            semantic_faces[track_id] = {}

            for frame_name, label_to_index in frames_data.items():
                frame_key = str(frame_name)
                if frame_key not in self.all_bbox_data[track_id]:
                    continue

                bbox_data = self.all_bbox_data[track_id][frame_key]
                all_faces = self.get_all_faces_from_bbox(bbox_data)

                semantic_faces[track_id][frame_key] = {}

                for semantic_label, face_index in label_to_index.items():
                    face_key = f'f{face_index}'
                    if face_key in all_faces:
                        semantic_faces[track_id][frame_key][semantic_label] = all_faces[face_key]

                self._infer_opposite_faces(semantic_faces[track_id][frame_key], all_faces)

        total_labels = sum(len(frames) for frames in semantic_faces.values())
        print(f"  Loaded semantic labels: {len(semantic_faces)} tracks, {total_labels} frame-label sets")
        return semantic_faces

    def _determine_frame_order(self):
        all_frame_names = set()
        for track_data in self.all_bbox_data.values():
            all_frame_names.update(track_data.keys())

        def extract_numeric_part(frame_name):
            numbers = re.findall(r'\d+', str(frame_name))
            return int(numbers[0]) if numbers else float('inf')

        return sorted(list(all_frame_names), key=extract_numeric_part)

    def _load_camera_params(self, frame_name):
        # Try multiple name variants: original, zero-padded (4-digit), unpadded
        frame_num = re.findall(r'\d+', str(frame_name))
        name_variants = [frame_name]
        if frame_num:
            padded = frame_num[0].zfill(4)
            unpadded = str(int(frame_num[0]))
            if padded != frame_name:
                name_variants.append(padded)
            if unpadded != frame_name:
                name_variants.append(unpadded)

        search_dirs = [
            self.annotator_output_dir / "camera",
            self.annotator_output_dir.parent / "camera",
            self.annotator_output_dir.parent.parent / "camera",
        ]

        camera_file = None
        for search_dir in search_dirs:
            for name in name_variants:
                candidate = search_dir / f"{name}.npz"
                if candidate.exists():
                    camera_file = candidate
                    break
            if camera_file is not None:
                break

        if camera_file is None:
            return None

        try:
            camera_data = np.load(camera_file)
            return {
                'K': camera_data['intrinsics'],
                'R': camera_data['pose'][:3, :3],
                't': camera_data['pose'][:3, 3]
            }
        except Exception as e:
            return None

    # Geometry methods
    def get_bbox_corners(self, center, dimensions, rotation_matrix):
        l, w, h = dimensions
        corners_local = np.array([
            [-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2],
            [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
            [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2],
            [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2],
        ])
        corners_world = (rotation_matrix @ corners_local.T).T + center
        return corners_world

    def compute_face_from_corners(self, corners, indices, box_center):
        face_corners = corners[indices]
        face_center = np.mean(face_corners, axis=0)
        edge1 = face_corners[1] - face_corners[0]
        edge2 = face_corners[3] - face_corners[0]
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        outward_vec = face_center - box_center
        if np.dot(normal, outward_vec) < 0:
            normal = -normal
        area = 0.5 * (np.linalg.norm(np.cross(edge1, edge2)) +
                     np.linalg.norm(np.cross(face_corners[2] - face_corners[0],
                                           face_corners[3] - face_corners[0])))
        return {'center': face_center, 'normal': normal, 'corners': face_corners, 'area': area}

    def get_all_faces_from_bbox(self, bbox_data):
        corners = self.get_bbox_corners(
            bbox_data['center'], bbox_data['dimensions'], bbox_data['rotation_matrix']
        )
        box_center = np.mean(corners, axis=0)
        face_indices = {
            'f0': [0, 1, 5, 4], 'f1': [2, 3, 7, 6], 'f2': [0, 3, 7, 4],
            'f3': [1, 2, 6, 5], 'f4': [4, 5, 6, 7], 'f5': [0, 1, 2, 3],
        }
        faces = {}
        for face_id, indices in face_indices.items():
            faces[face_id] = self.compute_face_from_corners(corners, indices, box_center)
        return faces

    def _infer_opposite_faces(self, semantic_assignments, all_faces):
        opposites = {'front': 'back', 'left': 'right', 'top': 'bottom'}
        assigned_faces = set()
        for semantic_label, face_data in semantic_assignments.items():
            for face_id, test_face in all_faces.items():
                if np.allclose(face_data['center'], test_face['center']):
                    assigned_faces.add(face_id)
                    break

        unassigned_faces = {fid: fdata for fid, fdata in all_faces.items()
                           if fid not in assigned_faces}

        for semantic_label, face_data in list(semantic_assignments.items()):
            if semantic_label in opposites:
                opposite_label = opposites[semantic_label]
                if opposite_label in semantic_assignments:
                    continue
                best_match = None
                best_score = 999
                best_match_id = None
                for face_id, test_face in unassigned_faces.items():
                    dot = np.dot(face_data['normal'], test_face['normal'])
                    if dot < best_score:
                        best_score = dot
                        best_match = test_face
                        best_match_id = face_id

                if best_match is not None and best_score < -0.8:
                    semantic_assignments[opposite_label] = best_match
                    if best_match_id in unassigned_faces:
                        del unassigned_faces[best_match_id]

    # Visibility and quality methods
    def _calculate_face_visibility(self, face_data, camera_params):
        try:
            t = camera_params['t']
            camera_pos = t
            face_to_camera = camera_pos - face_data['center']
            face_to_camera = face_to_camera / np.linalg.norm(face_to_camera)
            visibility_score = np.dot(face_data['normal'], face_to_camera)
            return visibility_score > 0, visibility_score
        except:
            return False, 0.0

    def _project_face_to_2d(self, face_data, camera_params, img_shape):
        try:
            face_corners_3d = face_data['corners']
            K = camera_params['K']
            R = camera_params['R']
            t = camera_params['t']

            camera_pose = np.eye(4)
            camera_pose[:3, :3] = R
            camera_pose[:3, 3] = t

            face_corners_h = np.concatenate([face_corners_3d, np.ones((4, 1))], axis=1)
            corners_cam = (np.linalg.inv(camera_pose) @ face_corners_h.T).T[:, :3]

            if np.any(corners_cam[:, 2] <= 0):
                return None

            corners_2d_hom = (K @ corners_cam.T).T
            corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]

            img_h, img_w = img_shape[:2]
            if (np.any(corners_2d[:, 0] < -img_w) or np.any(corners_2d[:, 0] > 2*img_w) or
                np.any(corners_2d[:, 1] < -img_h) or np.any(corners_2d[:, 1] > 2*img_h)):
                return None

            return corners_2d.astype(int)
        except:
            return None

    def _calculate_face_quality(self, face_data, camera_params, img_shape):
        try:
            is_visible, visibility_score = self._calculate_face_visibility(face_data, camera_params)
            if not is_visible:
                return 0.0

            corners_2d = self._project_face_to_2d(face_data, camera_params, img_shape)
            if corners_2d is None:
                return 0.0

            face_area_2d = cv2.contourArea(corners_2d)
            max_possible_area = img_shape[0] * img_shape[1]
            area_score = min(face_area_2d / max_possible_area, 1.0)

            img_center = np.array([img_shape[1]/2, img_shape[0]/2])
            face_center_2d = np.mean(corners_2d, axis=0)
            max_distance = np.sqrt(img_shape[0]**2 + img_shape[1]**2) / 2
            distance_score = 1.0 - np.linalg.norm(face_center_2d - img_center) / max_distance

            x_span = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
            y_span = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])
            if min(x_span, y_span) > 0:
                aspect_ratio = min(x_span, y_span) / max(x_span, y_span)
            else:
                aspect_ratio = 0.0

            quality_score = (
                0.25 * abs(visibility_score) +
                0.25 * area_score +
                0.25 * distance_score +
                0.25 * aspect_ratio
            )
            return quality_score
        except:
            return 0.0

    def compute_frame_qualities(self, track_id: int) -> Dict[str, Dict[str, float]]:
        """Compute quality scores for all frames and orientations."""
        if track_id not in self.semantic_faces:
            return {}

        semantic_labels = CONFIG['ALL_FACES']
        frame_quality_scores = {}

        sorted_frames = sorted(
            self.semantic_faces[track_id].keys(),
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
        )

        for frame_name in sorted_frames:
            camera_params = self._load_camera_params(frame_name)
            if not camera_params:
                continue

            img_shape = (480, 640, 3)
            if self.images_dir:
                img_path = self.images_dir / f"{frame_name}.jpg"
                if not img_path.exists():
                    img_path = self.images_dir / f"{frame_name}.png"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_shape = img.shape

            semantic_faces = self.semantic_faces[track_id][frame_name]
            frame_qualities = {}

            for label in semantic_labels:
                if label == 'bottom':
                    frame_qualities[label] = 0.0
                elif label in semantic_faces:
                    quality_score = self._calculate_face_quality(
                        semantic_faces[label], camera_params, img_shape
                    )
                    frame_qualities[label] = quality_score
                else:
                    frame_qualities[label] = 0.0

            frame_quality_scores[frame_name] = frame_qualities

        return frame_quality_scores

    def get_candidates_for_orientation(self, track_id: int, orientation: str,
                                       frame_qualities: Dict,
                                       min_quality: float = None,
                                       max_candidates: int = None) -> List[CandidateFrame]:
        if min_quality is None:
            min_quality = CONFIG['MIN_QUALITY_THRESHOLD']
        if max_candidates is None:
            max_candidates = CONFIG['MAX_CANDIDATES_TO_SHOW']

        candidates = []

        sorted_frames = sorted(
            frame_qualities.keys(),
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
        )

        for frame_idx, frame_name in enumerate(sorted_frames):
            quality = frame_qualities[frame_name].get(orientation, 0.0)
            if quality >= min_quality:
                candidates.append(CandidateFrame(
                    frame=frame_name,
                    quality_score=quality,
                    frame_idx=frame_idx
                ))

        candidates.sort(key=lambda x: x.quality_score, reverse=True)
        return candidates[:max_candidates]

    def get_quality_statistics(self, track_id: int, frame_qualities: Dict) -> Dict:
        stats = {}
        for orientation in CONFIG['VISIBLE_FACES']:
            qualities = [
                frame_qualities[f].get(orientation, 0.0)
                for f in frame_qualities
            ]
            non_zero = [q for q in qualities if q > 0]

            stats[orientation] = {
                'total_frames': len(qualities),
                'frames_with_visibility': len(non_zero),
                'max_quality': max(qualities) if qualities else 0,
                'mean_quality': np.mean(non_zero) if non_zero else 0,
                'frames_above_0.1': sum(1 for q in qualities if q >= 0.1),
                'frames_above_0.2': sum(1 for q in qualities if q >= 0.2),
                'frames_above_0.3': sum(1 for q in qualities if q >= 0.3),
            }
        return stats


# =============================================================================
# MASK CROP EXTRACTOR
# =============================================================================

class MaskCropExtractor:
    """Extracts masked animal crops from images."""

    def __init__(self, images_dir: Path, mask_dir: Path = None, results_dir: Path = None):
        self.images_dir = Path(images_dir) if images_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.results_dir = Path(results_dir) if results_dir else None
        self.mask_track_mapping = self._load_mask_track_mapping()

        # V7: Auto-discover mask_dir if not provided
        if self.mask_dir is None and self.images_dir is not None:
            self._auto_discover_mask_dir()

    def _auto_discover_mask_dir(self):
        """Auto-discover grounded-sam mask directory."""
        possible_dirs = [
            self.images_dir / "grounded-sam",
            self.images_dir.parent / "grounded-sam",
        ]
        if self.results_dir:
            possible_dirs.extend([
                self.results_dir.parent.parent / "grounded-sam",
                self.results_dir.parent / "grounded-sam",
            ])

        for candidate in possible_dirs:
            if candidate.exists() and list(candidate.glob("*_results.json")):
                self.mask_dir = candidate
                print(f"  Auto-discovered mask directory: {candidate}")
                break

    def _load_mask_track_mapping(self) -> Dict:
        if self.results_dir is None:
            return {}

        possible_paths = [
            self.results_dir / "mask_track_mapping.json",
            self.results_dir / "corrected_labels" / "mask_track_mapping.json",
            self.results_dir.parent / "mask_track_mapping.json",
        ]

        for mapping_file in possible_paths:
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    print(f"  Loaded mask-track mapping from: {mapping_file}")
                    return json.load(f)
        return {}

    def _load_image(self, frame_name: str) -> Optional[np.ndarray]:
        if self.images_dir is None:
            return None

        frame_num = re.findall(r'\d+', str(frame_name))
        frame_key = frame_num[0] if frame_num else frame_name

        # Build candidate names: original, digits-only, and zero-padded variants
        candidates = [frame_name, frame_key]
        if frame_key.isdigit():
            for width in [4, 5, 6, 8]:
                padded = frame_key.zfill(width)
                if padded not in candidates:
                    candidates.append(padded)

        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            for name in candidates:
                img_path = self.images_dir / f"{name}{ext}"
                if img_path.exists():
                    return cv2.imread(str(img_path))
        return None

    def _load_mask(self, frame_name: str, track_id: int) -> Optional[np.ndarray]:
        if self.mask_dir is None or not PYCOCOTOOLS_AVAILABLE:
            return None

        frame_num = re.findall(r'\d+', str(frame_name))
        frame_key = frame_num[0] if frame_num else frame_name

        # Try unpadded and zero-padded keys for mask_track_mapping lookup
        mask_idx = self.mask_track_mapping.get(str(frame_key), {}).get(str(track_id))
        if mask_idx is None and frame_key.isdigit():
            for width in [4, 5, 6, 8]:
                padded = frame_key.zfill(width)
                mask_idx = self.mask_track_mapping.get(padded, {}).get(str(track_id))
                if mask_idx is not None:
                    break

        # Build candidate names with zero-padded variants
        mask_candidates = [frame_key, frame_name]
        if frame_key.isdigit():
            for width in [4, 5, 6, 8]:
                padded = frame_key.zfill(width)
                if padded not in mask_candidates:
                    mask_candidates.append(padded)

        json_file = None
        for name in mask_candidates:
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

            if mask_idx is not None and mask_idx < len(annotations):
                ann = annotations[mask_idx]
                rle = ann.get('segmentation')
                if rle:
                    return mask_utils.decode(rle)

            if len(annotations) == 1:
                rle = annotations[0].get('segmentation')
                if rle:
                    return mask_utils.decode(rle)

        except Exception as e:
            pass

        return None

    def _create_error_image(self, size: int = 200) -> np.ndarray:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :, 2] = 80
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "IMG ERR"
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        text_x = (size - text_size[0]) // 2
        text_y = (size + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.6, (255, 255, 255), 2)
        return img

    def extract_crop(self, frame_name: str, track_id: int,
                    padding: int = None, background: str = None) -> Optional[np.ndarray]:
        if padding is None:
            padding = CONFIG['CROP_PADDING']
        if background is None:
            background = CONFIG['CROP_BG_COLOR']

        try:
            image = self._load_image(frame_name)
            if image is None:
                return self._create_error_image()

            mask = self._load_mask(frame_name, track_id)
            if mask is None:
                # Return full image cropped to center if no mask
                h, w = image.shape[:2]
                crop_size = min(h, w) // 2
                cx, cy = w // 2, h // 2
                x1 = max(0, cx - crop_size)
                x2 = min(w, cx + crop_size)
                y1 = max(0, cy - crop_size)
                y2 = min(h, cy + crop_size)
                return image[y1:y2, x1:x2].copy()

            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                return self._create_error_image()

            mask_x1, mask_x2 = xs.min(), xs.max()
            mask_y1, mask_y2 = ys.min(), ys.max()

            mask_w = mask_x2 - mask_x1
            mask_h = mask_y2 - mask_y1
            mask_cx = (mask_x1 + mask_x2) // 2
            mask_cy = (mask_y1 + mask_y2) // 2

            max_dim = max(mask_w, mask_h) + 2 * padding
            half_dim = max_dim // 2

            img_h, img_w = image.shape[:2]
            x1 = max(0, mask_cx - half_dim)
            x2 = min(img_w, mask_cx + half_dim)
            y1 = max(0, mask_cy - half_dim)
            y2 = min(img_h, mask_cy + half_dim)

            cropped_image = image[y1:y2, x1:x2].copy()
            cropped_mask = mask[y1:y2, x1:x2]

            if background == 'white':
                result = np.ones_like(cropped_image) * 255
                result[cropped_mask > 0] = cropped_image[cropped_mask > 0]
            else:
                result = cropped_image

            return result

        except Exception as e:
            return self._create_error_image()


# =============================================================================
# INTERACTIVE FRAME SELECTOR (V7 - with rejected frames)
# =============================================================================

class InteractiveFrameSelector:
    """Interactive GUI for selecting best frames per orientation."""

    def __init__(self, analyzer: ViewpointAnalyzer,
                 crop_extractor: MaskCropExtractor,
                 track_id: int,
                 select_rejected: bool = False):
        self.analyzer = analyzer
        self.crop_extractor = crop_extractor
        self.track_id = track_id
        self.selections = {}
        self.rejected = {}
        self.select_rejected = select_rejected

    def run_selection(self, min_quality: float = None) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict]:
        """Returns (selections, rejected, frame_qualities)"""
        if min_quality is None:
            min_quality = CONFIG['MIN_QUALITY_THRESHOLD']

        typo = CONFIG['TYPOGRAPHY']
        print(f"\n{'='*70}")
        print(f"INTERACTIVE FRAME SELECTION - Track {self.track_id}")
        print(f"{'='*70}")
        print(f"  Min quality threshold: {min_quality:.2f}")
        print(f"  Max candidates to show: {CONFIG['MAX_CANDIDATES_TO_SHOW']}")
        if self.select_rejected:
            print(f"  Rejected frame selection: ENABLED")

        frame_qualities = self.analyzer.compute_frame_qualities(self.track_id)

        stats = self.analyzer.get_quality_statistics(self.track_id, frame_qualities)
        print(f"\n  Quality Statistics:")
        for orient, s in stats.items():
            print(f"    {orient:6s}: {s['frames_with_visibility']:3d} visible, "
                  f"max={s['max_quality']:.2f}, "
                  f">=0.2: {s['frames_above_0.2']}, >=0.3: {s['frames_above_0.3']}")

        visible_labels = CONFIG['VISIBLE_FACES']
        self.selections = {label: [] for label in visible_labels}
        self.rejected = {label: [] for label in visible_labels}

        for orientation in visible_labels:
            candidates = self.analyzer.get_candidates_for_orientation(
                self.track_id, orientation, frame_qualities, min_quality
            )

            if not candidates:
                print(f"\n  [{orientation.upper()}] No candidates above threshold {min_quality:.2f}")
                print(f"    (max quality for {orientation}: {stats[orientation]['max_quality']:.3f})")
                continue

            print(f"\n  [{orientation.upper()}] Showing {len(candidates)} candidates (sorted by quality)")

            for candidate in candidates:
                if self.crop_extractor:
                    candidate.crop_image = self.crop_extractor.extract_crop(
                        candidate.frame, self.track_id
                    )

            selected, rejected = self._show_selection_gui(orientation, candidates)
            self.selections[orientation] = selected
            self.rejected[orientation] = rejected

            print(f"    Selected {len(selected)} frames: {selected}")
            if rejected:
                print(f"    Rejected {len(rejected)} frames: {rejected}")

        return self.selections, self.rejected, frame_qualities

    def _show_selection_gui(self, orientation: str,
                           candidates: List[CandidateFrame]) -> Tuple[List[str], List[str]]:
        """Show matplotlib GUI for selecting frames. Returns (selected, rejected)."""
        n_candidates = len(candidates)
        if n_candidates == 0:
            return [], []

        typo = CONFIG['TYPOGRAPHY']
        n_cols = min(5, n_candidates)
        n_rows = (n_candidates + n_cols - 1) // n_cols

        fig_width = 3 * n_cols
        fig_height = 3.5 * n_rows + 1.5

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

        title = f'Select frames for {orientation.upper()} view (Track {self.track_id})'
        if self.select_rejected:
            title += '\nLeft-click = APPROVE (green), Right-click = REJECT (red)'
        else:
            title += '\nClick to toggle selection (green = selected)'
        title += '\nClose window when done'

        fig.suptitle(title, fontsize=typo['subtitle'], fontweight='bold')

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # State: 0 = unselected, 1 = approved, -1 = rejected
        selection_state = [0] * n_candidates
        border_rects = []

        def update_borders():
            for i, (rect, state) in enumerate(zip(border_rects, selection_state)):
                if state == 1:  # Approved
                    rect.set_edgecolor('lime')
                    rect.set_linewidth(6)
                elif state == -1:  # Rejected
                    rect.set_edgecolor(CONFIG['REJECTED_COLOR'])
                    rect.set_linewidth(6)
                else:
                    rect.set_edgecolor('gray')
                    rect.set_linewidth(1)
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is None:
                return
            for i in range(n_candidates):
                row = i // n_cols
                col = i % n_cols
                if event.inaxes == axes[row, col]:
                    if self.select_rejected and event.button == 3:  # Right click
                        # Toggle reject
                        if selection_state[i] == -1:
                            selection_state[i] = 0
                        else:
                            selection_state[i] = -1
                    else:  # Left click
                        # Toggle approve
                        if selection_state[i] == 1:
                            selection_state[i] = 0
                        else:
                            selection_state[i] = 1
                    update_borders()
                    break

        for i, candidate in enumerate(candidates):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            if candidate.crop_image is not None:
                img = candidate.crop_image
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.set_facecolor('#E0E0E0')
                ax.text(0.5, 0.5, 'No image', ha='center', va='center',
                       transform=ax.transAxes, fontsize=typo['body'])

            frame_num = re.findall(r'\d+', str(candidate.frame))
            frame_str = frame_num[0] if frame_num else candidate.frame
            ax.set_title(f'F:{frame_str} Q:{candidate.quality_score:.3f}', fontsize=typo['body'])
            ax.set_xticks([])
            ax.set_yticks([])

            rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                 fill=False, edgecolor='gray', linewidth=1)
            ax.add_patch(rect)
            border_rects.append(rect)

        for i in range(n_candidates, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        fig.canvas.mpl_connect('button_press_event', on_click)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

        selected_frames = [candidates[i].frame for i in range(n_candidates) if selection_state[i] == 1]
        rejected_frames = [candidates[i].frame for i in range(n_candidates) if selection_state[i] == -1]

        return selected_frames, rejected_frames


# =============================================================================
# COVERAGE CALCULATOR (replaces QualityGrader)
# =============================================================================

class CoverageCalculator:
    """Calculate numeric coverage metrics (no letter grades)."""

    def calculate_coverage(self, selections: Dict[str, List[str]],
                          frame_qualities: Dict[str, Dict[str, float]] = None) -> CoverageResult:
        visible_labels = CONFIG['VISIBLE_FACES']

        orientations_with_data = sum(1 for label in visible_labels
                                     if len(selections.get(label, [])) > 0)
        total_frames = sum(len(selections.get(label, [])) for label in visible_labels)
        missing = [label for label in visible_labels if len(selections.get(label, [])) == 0]

        # Calculate mean quality if frame_qualities provided
        mean_quality = 0.0
        if frame_qualities:
            qualities = []
            for label in visible_labels:
                for frame in selections.get(label, []):
                    if frame in frame_qualities:
                        q = frame_qualities[frame].get(label, 0)
                        if q > 0:
                            qualities.append(q)
            if qualities:
                mean_quality = np.mean(qualities)

        coverage_percent = (orientations_with_data / 5.0) * 100

        return CoverageResult(
            orientations_covered=orientations_with_data,
            total_frames=total_frames,
            mean_quality=mean_quality,
            coverage_percent=coverage_percent,
            missing_orientations=missing
        )


# =============================================================================
# AGGREGATE REPORT GENERATOR (V7 - FIXED & ENHANCED)
# =============================================================================

class AggregateReportGenerator:
    """Generates publication-quality aggregate PDF with heat-encoded matrix."""

    def __init__(self, crop_extractor: MaskCropExtractor = None,
                 video_name: str = "",
                 frame_qualities_cache: Dict[int, Dict] = None):
        self.crop_extractor = crop_extractor
        self.video_name = video_name
        self.calculator = CoverageCalculator()
        self.orientation_colors = CONFIG['ORIENTATION_COLORS']
        self.missing_color = CONFIG['MISSING_COLOR']
        self.typo = CONFIG['TYPOGRAPHY']
        self.frame_qualities_cache = frame_qualities_cache or {}

    def load_all_saved_selections(self, viewpoint_dir: Path) -> Tuple[Dict, Dict]:
        """Load all approved_selections_track*.json files.
        Returns (selections, rejected)
        """
        all_selections = {}
        all_rejected = {}

        for f in sorted(viewpoint_dir.glob("approved_selections_track*.json")):
            with open(f, 'r') as fp:
                data = json.load(fp)
                track_id = data.get('track_id')
                if track_id is not None:
                    if 'selections' in data:
                        all_selections[track_id] = data['selections']
                    else:
                        all_selections[track_id] = {k: v for k, v in data.items()
                                                     if k in CONFIG['VISIBLE_FACES']}
                    # Load rejected if present
                    all_rejected[track_id] = data.get('rejected', {})

        return all_selections, all_rejected

    def generate_aggregate_report(self, all_selections: Dict[int, Dict[str, List[str]]],
                                  output_path: Path,
                                  all_rejected: Dict[int, Dict[str, List[str]]] = None,
                                  show_rejected: bool = False) -> Path:
        """Generate aggregate PDF with heat-encoded matrix and improved filmstrips."""
        output_path.parent.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("GENERATING AGGREGATE PDF (V7 - Heat Matrix + Timeline)")
        print(f"{'='*70}")
        print(f"  Tracks: {list(all_selections.keys())}")
        print(f"  Output: {output_path}")

        if all_rejected is None:
            all_rejected = {k: {} for k in all_selections.keys()}

        # Compute coverage for all tracks
        coverages = {}
        for track_id, selections in all_selections.items():
            fq = self.frame_qualities_cache.get(track_id, {})
            coverages[track_id] = self.calculator.calculate_coverage(selections, fq)

        with PdfPages(str(output_path)) as pdf:
            # Page 1: Heat-encoded quality matrix
            self._create_heat_matrix_page(pdf, all_selections, coverages)

            # Page 2+: Improved filmstrips with timeline
            self._create_filmstrip_pages(pdf, all_selections, coverages,
                                        all_rejected if show_rejected else None)

        print(f"\nSaved aggregate PDF to: {output_path}")
        return output_path

    def _create_heat_matrix_page(self, pdf: PdfPages,
                                 all_selections: Dict[int, Dict[str, List[str]]],
                                 coverages: Dict[int, CoverageResult]):
        """Create heat-encoded quality matrix page."""
        fig = plt.figure(figsize=(8.5, 11))

        # Title
        fig.suptitle(f'VIEWPOINT QUALITY MATRIX\n{self.video_name}',
                    fontsize=self.typo['title'], fontweight='bold', y=0.96)

        # Create main axes for the matrix
        ax_matrix = fig.add_axes([0.12, 0.35, 0.76, 0.52])

        visible_labels = CONFIG['VISIBLE_FACES']
        track_ids = sorted(all_selections.keys())
        n_tracks = len(track_ids)

        # Create heat map data
        heat_data = np.zeros((n_tracks, len(visible_labels)))

        for row_idx, track_id in enumerate(track_ids):
            selections = all_selections[track_id]
            fq = self.frame_qualities_cache.get(track_id, {})

            for col_idx, label in enumerate(visible_labels):
                frames = selections.get(label, [])
                if len(frames) > 0 and fq:
                    # Calculate mean quality for selected frames
                    qualities = [fq.get(f, {}).get(label, 0) for f in frames]
                    qualities = [q for q in qualities if q > 0]
                    heat_data[row_idx, col_idx] = np.mean(qualities) if qualities else 0.0
                elif len(frames) > 0:
                    # Has frames but no quality data - show as present (0.5)
                    heat_data[row_idx, col_idx] = 0.5
                else:
                    heat_data[row_idx, col_idx] = 0.0

        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('quality',
            ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0'])

        # Plot heatmap
        im = ax_matrix.imshow(heat_data, cmap=cmap, aspect='auto', vmin=0, vmax=0.5)

        # Set axis labels
        ax_matrix.set_xticks(range(len(visible_labels)))
        ax_matrix.set_xticklabels([l.upper() for l in visible_labels],
                                  fontsize=self.typo['heading'], fontweight='bold')
        ax_matrix.set_yticks(range(n_tracks))
        ax_matrix.set_yticklabels([f'Track {t}' for t in track_ids],
                                  fontsize=self.typo['body'])

        # Add grid
        ax_matrix.set_xticks(np.arange(-0.5, len(visible_labels), 1), minor=True)
        ax_matrix.set_yticks(np.arange(-0.5, n_tracks, 1), minor=True)
        ax_matrix.grid(which='minor', color='white', linewidth=2)

        # Add text annotations (frame count and quality)
        for row_idx, track_id in enumerate(track_ids):
            selections = all_selections[track_id]
            for col_idx, label in enumerate(visible_labels):
                frames = selections.get(label, [])
                n_frames = len(frames)
                quality = heat_data[row_idx, col_idx]

                if n_frames > 0:
                    text = f'{n_frames}\n({quality:.2f})'
                    color = 'white' if quality > 0.25 else 'black'
                else:
                    text = 'X'
                    color = '#999999'

                ax_matrix.text(col_idx, row_idx, text,
                              ha='center', va='center',
                              fontsize=self.typo['caption'], fontweight='bold',
                              color=color)

        # Colorbar
        cbar_ax = fig.add_axes([0.90, 0.35, 0.02, 0.52])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Mean Quality Score', fontsize=self.typo['body'])
        cbar.ax.tick_params(labelsize=self.typo['caption'])

        # Orientation color legend at top
        legend_ax = fig.add_axes([0.12, 0.88, 0.76, 0.03])
        legend_ax.axis('off')

        for i, label in enumerate(visible_labels):
            x = i / len(visible_labels) + 0.1 / len(visible_labels)
            color = self.orientation_colors[label]
            legend_ax.add_patch(FancyBboxPatch(
                (x, 0.1), 0.12, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=color, alpha=0.4,
                edgecolor=color, linewidth=2,
                transform=legend_ax.transAxes
            ))
            legend_ax.text(x + 0.06, 0.5, label.upper(),
                          ha='center', va='center',
                          fontsize=self.typo['caption'], fontweight='bold',
                          transform=legend_ax.transAxes)

        # Statistics section
        ax_stats = fig.add_axes([0.08, 0.06, 0.84, 0.22])
        ax_stats.axis('off')

        # Calculate statistics
        total_frames = sum(sum(len(v) for v in sel.values()) for sel in all_selections.values())
        orientations_used = set()
        for sel in all_selections.values():
            for label, frames in sel.items():
                if len(frames) > 0:
                    orientations_used.add(label)

        # Coverage distribution
        coverage_counts = Counter(c.orientations_covered for c in coverages.values())

        # Mean quality across all tracks
        all_qualities = [c.mean_quality for c in coverages.values() if c.mean_quality > 0]
        overall_mean_q = np.mean(all_qualities) if all_qualities else 0

        stats_text = (
            f"Total Tracks: {n_tracks}    |    "
            f"Total Frames: {total_frames}    |    "
            f"Orientations Used: {len(orientations_used)}/5    |    "
            f"Overall Mean Quality: {overall_mean_q:.3f}\n\n"
            f"Coverage Distribution:  "
        )
        for cov in sorted(coverage_counts.keys(), reverse=True):
            stats_text += f"{cov}/5: {coverage_counts[cov]} tracks  |  "

        ax_stats.text(0.5, 0.7, stats_text.rstrip('  |  '),
                     ha='center', va='center', fontsize=self.typo['body'],
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                              edgecolor='#E0E0E0', linewidth=1))

        # Frame range (temporal context)
        all_frame_nums = []
        for sel in all_selections.values():
            for frames in sel.values():
                for f in frames:
                    nums = re.findall(r'\d+', f)
                    if nums:
                        all_frame_nums.append(int(nums[0]))

        if all_frame_nums:
            min_frame = min(all_frame_nums)
            max_frame = max(all_frame_nums)
            ax_stats.text(0.5, 0.25, f'Frame Range: {min_frame} - {max_frame}',
                         ha='center', va='center', fontsize=self.typo['caption'],
                         color='#666666')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_filmstrip_pages(self, pdf: PdfPages,
                                all_selections: Dict[int, Dict[str, List[str]]],
                                coverages: Dict[int, CoverageResult],
                                all_rejected: Dict = None):
        """Create filmstrip pages with timeline context."""
        track_ids = sorted(all_selections.keys())
        visible_labels = CONFIG['VISIBLE_FACES']

        # Calculate tracks per page based on content
        tracks_per_page = 3

        for page_start in range(0, len(track_ids), tracks_per_page):
            page_tracks = track_ids[page_start:page_start + tracks_per_page]

            fig = plt.figure(figsize=(11, 8.5))

            page_num = page_start // tracks_per_page + 1
            total_pages = (len(track_ids) + tracks_per_page - 1) // tracks_per_page
            fig.suptitle(f'Viewpoint Filmstrips - Page {page_num}/{total_pages}',
                        fontsize=self.typo['subtitle'], fontweight='bold', y=0.97)

            n_page_tracks = len(page_tracks)
            track_height = 0.88 / n_page_tracks

            for track_idx, track_id in enumerate(page_tracks):
                selections = all_selections[track_id]
                coverage = coverages[track_id]
                rejected = all_rejected.get(track_id, {}) if all_rejected else {}

                track_top = 0.92 - track_idx * track_height
                track_bottom = track_top - track_height + 0.03

                self._draw_track_filmstrip(fig, track_id, selections, coverage,
                                          track_top, track_bottom, rejected)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _draw_track_filmstrip(self, fig, track_id: int,
                              selections: Dict[str, List[str]],
                              coverage: CoverageResult,
                              top: float, bottom: float,
                              rejected: Dict[str, List[str]] = None):
        """Draw a single track's filmstrip with proper positioning."""
        visible_labels = CONFIG['VISIBLE_FACES']

        # Collect all frames with their orientations
        all_frames = []
        for label in visible_labels:
            for frame in selections.get(label, []):
                all_frames.append((label, frame, False))  # (label, frame, is_rejected)
            if rejected:
                for frame in rejected.get(label, []):
                    all_frames.append((label, frame, True))

        # Sort by frame number for temporal order
        def get_frame_num(item):
            nums = re.findall(r'\d+', item[1])
            return int(nums[0]) if nums else 0
        all_frames.sort(key=get_frame_num)

        # Track header
        header_height = 0.03
        header_ax = fig.add_axes([0.03, top - header_height, 0.94, header_height])
        header_ax.axis('off')

        # Header background gradient
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        header_ax.imshow(gradient, aspect='auto', cmap='Blues', alpha=0.2,
                        extent=[0, 1, 0, 1])

        header_ax.text(0.01, 0.5, f'TRACK {track_id}',
                      ha='left', va='center',
                      fontsize=self.typo['heading'], fontweight='bold')

        # Coverage badge (no letter grade)
        coverage_text = f'{coverage.orientations_covered}/5 views | {coverage.total_frames} frames'
        if coverage.mean_quality > 0:
            coverage_text += f' | Q={coverage.mean_quality:.2f}'

        header_ax.text(0.99, 0.5, coverage_text,
                      ha='right', va='center',
                      fontsize=self.typo['body'],
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                               alpha=0.8, edgecolor='#90CAF9'))

        # Filmstrip area - use GridSpec for better layout
        strip_top = top - header_height - 0.01
        strip_height = (strip_top - bottom) * 0.75

        if len(all_frames) == 0:
            strip_ax = fig.add_axes([0.03, strip_top - strip_height, 0.94, strip_height])
            strip_ax.axis('off')
            strip_ax.text(0.5, 0.5, 'No frames selected',
                         ha='center', va='center',
                         fontsize=self.typo['body'], color='gray', style='italic')
        else:
            # Calculate frame dimensions
            n_frames = len(all_frames)
            max_frames_per_row = 10
            n_rows = (n_frames + max_frames_per_row - 1) // max_frames_per_row
            frames_in_row = min(n_frames, max_frames_per_row)

            frame_width = 0.88 / frames_in_row
            frame_height = strip_height / n_rows

            for i, (label, frame_name, is_rejected) in enumerate(all_frames):
                row = i // max_frames_per_row
                col = i % max_frames_per_row

                # Calculate position in figure coordinates
                x_pos = 0.05 + col * frame_width
                y_pos = strip_top - (row + 1) * frame_height

                # Create axes for this frame
                frame_ax = fig.add_axes([x_pos, y_pos, frame_width * 0.92, frame_height * 0.85])

                # Load and display image
                crop_img = None
                if self.crop_extractor:
                    crop_img = self.crop_extractor.extract_crop(frame_name, track_id)

                if crop_img is not None:
                    if len(crop_img.shape) == 3 and crop_img.shape[2] == 3:
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    frame_ax.imshow(crop_img)

                    # Add rejected overlay
                    if is_rejected:
                        frame_ax.imshow(np.ones_like(crop_img) * 255,
                                       alpha=CONFIG['REJECTED_ALPHA'])
                        frame_ax.plot([0, 1], [0, 1], 'r-', linewidth=2, transform=frame_ax.transAxes)
                        frame_ax.plot([0, 1], [1, 0], 'r-', linewidth=2, transform=frame_ax.transAxes)
                else:
                    frame_ax.set_facecolor('#F5F5F5')
                    frame_ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                 fontsize=self.typo['caption'], color='gray')

                frame_ax.axis('off')

                # Color-coded border
                color = CONFIG['REJECTED_COLOR'] if is_rejected else self.orientation_colors[label]
                for spine in frame_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2 if not is_rejected else 3)

                # Frame number (temporal context)
                frame_num = re.findall(r'\d+', str(frame_name))
                frame_str = frame_num[0] if frame_num else frame_name
                frame_ax.text(0.5, 1.08, frame_str,
                             ha='center', va='bottom',
                             fontsize=self.typo['micro'],
                             transform=frame_ax.transAxes)

                # Orientation badge
                badge_text = label.upper()[:3]
                if is_rejected:
                    badge_text = 'REJ'
                frame_ax.text(0.5, -0.12, badge_text,
                             ha='center', va='top',
                             fontsize=self.typo['micro'], fontweight='bold',
                             transform=frame_ax.transAxes,
                             bbox=dict(boxstyle='round,pad=0.1',
                                      facecolor=color, alpha=0.6, edgecolor='none'))

        # Timeline indicator
        timeline_y = bottom + 0.01
        timeline_ax = fig.add_axes([0.03, timeline_y, 0.94, 0.015])
        timeline_ax.axis('off')

        # Get frame range for this track
        frame_nums = []
        for label in visible_labels:
            for f in selections.get(label, []):
                nums = re.findall(r'\d+', f)
                if nums:
                    frame_nums.append(int(nums[0]))

        if frame_nums:
            min_f, max_f = min(frame_nums), max(frame_nums)
            span = max_f - min_f if max_f > min_f else 1

            # Draw timeline bar
            timeline_ax.axhline(y=0.5, xmin=0.05, xmax=0.95, color='#BDBDBD', linewidth=2)

            # Draw frame markers
            for label in visible_labels:
                color = self.orientation_colors[label]
                for f in selections.get(label, []):
                    nums = re.findall(r'\d+', f)
                    if nums:
                        fnum = int(nums[0])
                        x_pos = 0.05 + 0.9 * (fnum - min_f) / span
                        timeline_ax.plot(x_pos, 0.5, 'o', color=color, markersize=4,
                                        transform=timeline_ax.transAxes)

            # Timeline labels
            timeline_ax.text(0.02, 0.5, str(min_f), ha='right', va='center',
                            fontsize=self.typo['micro'], transform=timeline_ax.transAxes)
            timeline_ax.text(0.98, 0.5, str(max_f), ha='left', va='center',
                            fontsize=self.typo['micro'], transform=timeline_ax.transAxes)

        # Missing views indicator
        if coverage.missing_orientations:
            missing_y = bottom - 0.01
            fig.text(0.03, missing_y, 'Missing: ' + ', '.join(coverage.missing_orientations),
                    fontsize=self.typo['caption'], color='#666666', style='italic')


# =============================================================================
# PDF REPORT GENERATOR (V7 - Per-track, kept for compatibility)
# =============================================================================

class PDFReportGenerator:
    """Generates filmstrip PDF with vertical strip layout (per-track)."""

    def __init__(self, analyzer: ViewpointAnalyzer,
                 crop_extractor: MaskCropExtractor = None):
        self.analyzer = analyzer
        self.crop_extractor = crop_extractor
        self.calculator = CoverageCalculator()
        self.typo = CONFIG['TYPOGRAPHY']

    def generate_report(self, track_id: int, selections: Dict[str, List[str]],
                       frame_qualities: Dict,
                       rejected: Dict[str, List[str]] = None,
                       output_path: Path = None) -> Path:
        if output_path is None:
            output_path = (self.analyzer.annotator_output_dir /
                          "viewpoint_analysis" / f"filmstrip_v7_track{track_id}.pdf")

        output_path.parent.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print("GENERATING FILMSTRIP PDF (V7)")
        print(f"{'='*70}")

        coverage = self.calculator.calculate_coverage(selections, frame_qualities)

        with PdfPages(str(output_path)) as pdf:
            self._create_vertical_filmstrip_page(pdf, track_id, selections,
                                                frame_qualities, coverage, rejected)

        print(f"\nSaved PDF report to: {output_path}")
        return output_path

    def _create_vertical_filmstrip_page(self, pdf: PdfPages, track_id: int,
                                        selections: Dict[str, List[str]],
                                        frame_qualities: Dict,
                                        coverage: CoverageResult,
                                        rejected: Dict[str, List[str]] = None):
        visible_labels = CONFIG['VISIBLE_FACES']
        n_cols = len(visible_labels)

        max_frames = max(len(selections.get(l, [])) for l in visible_labels)
        if rejected:
            max_frames = max(max_frames, max(len(rejected.get(l, [])) for l in visible_labels))
        max_frames = max(max_frames, 1)

        fig = plt.figure(figsize=(11, 8.5))

        n_rows = max_frames + 1
        gs = GridSpec(n_rows, n_cols, figure=fig,
                     height_ratios=[0.15] + [0.85 / max_frames] * max_frames,
                     left=0.02, right=0.98, top=0.92, bottom=0.02,
                     hspace=0.08, wspace=0.05)

        total_selected = sum(len(selections.get(l, [])) for l in visible_labels)
        fig.suptitle(f'Track {track_id} - Viewpoint Filmstrip (V7)\n'
                    f'{coverage.orientations_covered}/5 views | {total_selected} frames | '
                    f'Mean Q={coverage.mean_quality:.2f}',
                    fontsize=self.typo['subtitle'], fontweight='bold')

        for col_idx, label in enumerate(visible_labels):
            ax_header = fig.add_subplot(gs[0, col_idx])
            ax_header.axis('off')

            n_selected = len(selections.get(label, []))
            color = self.analyzer.semantic_face_colors[label] if n_selected > 0 else CONFIG['MISSING_COLOR']

            ax_header.text(0.5, 0.5, f'{label.upper()}\n({n_selected})',
                          ha='center', va='center',
                          fontsize=self.typo['heading'], fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor=color, alpha=0.4))

        for col_idx, label in enumerate(visible_labels):
            selected_frames = selections.get(label, [])
            orientation_color = self.analyzer.semantic_face_colors[label]

            for row_idx in range(max_frames):
                ax = fig.add_subplot(gs[row_idx + 1, col_idx])

                if row_idx < len(selected_frames):
                    frame_name = selected_frames[row_idx]
                    quality = frame_qualities.get(frame_name, {}).get(label, 0)

                    crop_image = None
                    if self.crop_extractor:
                        crop_image = self.crop_extractor.extract_crop(frame_name, track_id)

                    if crop_image is not None:
                        if crop_image.shape[-1] == 3:
                            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                        ax.imshow(crop_image)

                        frame_num = re.findall(r'\d+', str(frame_name))
                        frame_str = frame_num[0] if frame_num else frame_name
                        ax.text(0.02, 0.98, f'F:{frame_str}',
                               transform=ax.transAxes, fontsize=self.typo['caption'],
                               verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.1',
                                        facecolor='white', alpha=0.8))
                        ax.text(0.02, 0.02, f'Q:{quality:.2f}',
                               transform=ax.transAxes, fontsize=self.typo['caption'],
                               verticalalignment='bottom',
                               bbox=dict(boxstyle='round,pad=0.1',
                                        facecolor='white', alpha=0.8))

                        for spine in ax.spines.values():
                            spine.set_edgecolor(orientation_color)
                            spine.set_linewidth(3)
                    else:
                        ax.set_facecolor('#EEEEEE')
                        ax.text(0.5, 0.5, 'No image', ha='center', va='center',
                               fontsize=self.typo['caption'], color='gray')
                else:
                    ax.set_facecolor('#F8F8F8')

                ax.set_xticks([])
                ax.set_yticks([])

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def save_json_report(self, track_id: int, selections: Dict[str, List[str]],
                        frame_qualities: Dict,
                        rejected: Dict[str, List[str]] = None,
                        output_path: Path = None) -> Path:
        if output_path is None:
            output_path = (self.analyzer.annotator_output_dir /
                          "viewpoint_analysis" / f"selections_v7_track{track_id}.json")

        output_path.parent.mkdir(exist_ok=True)

        coverage = self.calculator.calculate_coverage(selections, frame_qualities)

        report = {
            'track_id': track_id,
            'timestamp': datetime.now().isoformat(),
            'version': 'v7',
            'coverage': {
                'orientations_covered': coverage.orientations_covered,
                'total_frames': coverage.total_frames,
                'mean_quality': float(coverage.mean_quality),
                'coverage_percent': float(coverage.coverage_percent),
                'missing_orientations': coverage.missing_orientations
            },
            'selections': selections,
            'rejected': rejected or {},
            'frame_qualities': {
                frame: {k: float(v) for k, v in quals.items()}
                for frame, quals in frame_qualities.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Saved JSON report to: {output_path}")
        return output_path


# =============================================================================
# MAIN ENTRY POINT (V7)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Viewpoint Analyzer v7 - Heat Matrix + Timeline + Rejected Frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate aggregate PDF from saved selections (recommended)
    python viewpoint_analyzer_v7.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --load_saved --aggregate

    # Interactive selection with rejected frames + aggregate PDF
    python viewpoint_analyzer_v7.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --aggregate --select_rejected

    # Process specific tracks with aggregate and show rejected
    python viewpoint_analyzer_v7.py --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ --track_id 0 1 5 --aggregate --show_rejected
        """
    )

    parser.add_argument("--annotator_output", required=True,
                       help="Path to annotator tool output directory")
    parser.add_argument("--images_dir", default=None,
                       help="Path to original images")
    parser.add_argument("--mask_dir", default=None,
                       help="Path to grounded-SAM mask directory")
    parser.add_argument("--results_dir", default=None,
                       help="Path to results directory")
    parser.add_argument("--min_quality", type=float, default=None,
                       help=f"Minimum quality threshold (default: {CONFIG['MIN_QUALITY_THRESHOLD']})")
    parser.add_argument("--max_candidates", type=int, default=None,
                       help=f"Max candidates per orientation (default: {CONFIG['MAX_CANDIDATES_TO_SHOW']})")
    parser.add_argument("--use_saved_selections", action="store_true",
                       help="Use previously saved selections instead of interactive mode")
    parser.add_argument("--load_saved", action="store_true",
                       help="Load from existing selection files (alias for --use_saved_selections)")
    parser.add_argument("--aggregate", action="store_true",
                       help="Generate aggregate PDF combining all tracks")
    parser.add_argument("--track_id", type=int, nargs='*', default=None,
                       help="Specific track ID(s) to process")
    parser.add_argument("--select_tracks", action="store_true",
                       help="Interactively select which tracks to process")
    parser.add_argument("--select_rejected", action="store_true",
                       help="Enable rejected frame selection in interactive mode")
    parser.add_argument("--show_rejected", action="store_true",
                       help="Show rejected frames in output PDFs")
    parser.add_argument("--video_name", default=None,
                       help="Video name for PDF title (auto-detected if not provided)")
    parser.add_argument("--semantic", action="store_true",
                       help="Run semantic face propagation after viewpoint analysis "
                            "(requires wildlift.viewpoint.semantic_propagator)")

    args = parser.parse_args()

    # Handle alias
    if args.load_saved:
        args.use_saved_selections = True

    # Update CONFIG
    if args.min_quality is not None:
        CONFIG['MIN_QUALITY_THRESHOLD'] = args.min_quality
    if args.max_candidates is not None:
        CONFIG['MAX_CANDIDATES_TO_SHOW'] = args.max_candidates

    print("=" * 70)
    print("VIEWPOINT ANALYZER v7")
    print("Heat Matrix + Timeline + Rejected Frames")
    print("=" * 70)
    print(f"  Min quality threshold: {CONFIG['MIN_QUALITY_THRESHOLD']}")
    print(f"  Aggregate mode: {args.aggregate}")
    print(f"  Select rejected: {args.select_rejected}")

    # Auto-detect video name
    video_name = args.video_name
    if video_name is None:
        output_path = Path(args.annotator_output)
        if output_path.name == "corrected_bboxes":
            video_name = output_path.parent.name
        else:
            video_name = output_path.name
    print(f"  Video name: {video_name}")

    try:
        analyzer = ViewpointAnalyzer(
            annotator_output_dir=args.annotator_output,
            images_dir=args.images_dir
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Set up mask crop extractor with auto-discovery
    crop_extractor = None
    if args.images_dir:
        mask_dir = args.mask_dir
        results_dir = args.results_dir

        # Auto-detect results_dir
        if results_dir is None:
            for candidate in [Path(args.annotator_output).parent, Path(args.annotator_output)]:
                if (candidate / "mask_track_mapping.json").exists():
                    results_dir = str(candidate)
                    break

        # Create extractor (it will auto-discover mask_dir if needed)
        crop_extractor = MaskCropExtractor(
            images_dir=Path(args.images_dir),
            mask_dir=Path(mask_dir) if mask_dir else None,
            results_dir=Path(results_dir) if results_dir else Path(args.annotator_output).parent
        )
        print(f"  Crop extraction enabled")
        if crop_extractor.mask_dir:
            print(f"  Mask directory: {crop_extractor.mask_dir}")

    # Determine which tracks to process
    tracks_to_process = []
    viewpoint_dir = Path(args.annotator_output) / "viewpoint_analysis"

    all_selections = {}
    all_rejected = {}
    frame_qualities_cache = {}

    if args.use_saved_selections and viewpoint_dir.exists():
        # Load from saved selections
        agg_gen = AggregateReportGenerator(crop_extractor, video_name)
        all_selections, all_rejected = agg_gen.load_all_saved_selections(viewpoint_dir)

        if args.track_id:
            all_selections = {k: v for k, v in all_selections.items() if k in args.track_id}
            all_rejected = {k: v for k, v in all_rejected.items() if k in args.track_id}

        tracks_to_process = list(all_selections.keys())
        print(f"\n  Loaded saved selections for {len(tracks_to_process)} track(s): {tracks_to_process}")

        # Compute frame qualities for loaded tracks
        for track_id in tracks_to_process:
            frame_qualities_cache[track_id] = analyzer.compute_frame_qualities(track_id)
    else:
        # Interactive or specified tracks
        if args.select_tracks or (args.track_id is not None and len(args.track_id) == 0):
            print(f"\n  Available tracks with semantic labels: {analyzer.labeled_tracks}")
            print("  Select tracks to process (comma-separated, or 'all' for all tracks):")
            user_input = input("  > ").strip()

            if user_input.lower() == 'all' or user_input == '':
                tracks_to_process = analyzer.labeled_tracks
            else:
                try:
                    tracks_to_process = [int(t.strip()) for t in user_input.split(',')]
                    invalid = [t for t in tracks_to_process if t not in analyzer.labeled_tracks]
                    if invalid:
                        print(f"  WARNING: Tracks {invalid} don't have semantic labels, skipping them")
                        tracks_to_process = [t for t in tracks_to_process if t in analyzer.labeled_tracks]
                except ValueError:
                    print("  Invalid input. Processing all tracks.")
                    tracks_to_process = analyzer.labeled_tracks
        elif args.track_id is not None and len(args.track_id) > 0:
            tracks_to_process = args.track_id
            invalid = [t for t in tracks_to_process if t not in analyzer.labeled_tracks]
            if invalid:
                print(f"  WARNING: Tracks {invalid} don't have semantic labels, skipping them")
                tracks_to_process = [t for t in tracks_to_process if t in analyzer.labeled_tracks]
        else:
            tracks_to_process = analyzer.labeled_tracks

    if not tracks_to_process:
        print("\nERROR: No valid tracks to process")
        return 1

    print(f"\n  Will process {len(tracks_to_process)} track(s): {tracks_to_process}")

    # Process each track (if not loaded from saved)
    if not args.use_saved_selections or not all_selections:
        for track_idx, track_id in enumerate(tracks_to_process):
            print(f"\n{'#'*70}")
            print(f"# PROCESSING TRACK {track_id} ({track_idx + 1}/{len(tracks_to_process)})")
            print(f"{'#'*70}")

            selection_file = viewpoint_dir / f"approved_selections_track{track_id}.json"

            if args.use_saved_selections and selection_file.exists():
                print(f"\n  Loading saved selections from: {selection_file}")
                with open(selection_file, 'r') as f:
                    saved_data = json.load(f)
                selections = saved_data.get('selections', saved_data)
                rejected = saved_data.get('rejected', {})
                frame_qualities = analyzer.compute_frame_qualities(track_id)
            else:
                selector = InteractiveFrameSelector(analyzer, crop_extractor, track_id,
                                                   select_rejected=args.select_rejected)
                selections, rejected, frame_qualities = selector.run_selection(CONFIG['MIN_QUALITY_THRESHOLD'])

                selection_file.parent.mkdir(exist_ok=True)
                save_data = {
                    'track_id': track_id,
                    'timestamp': datetime.now().isoformat(),
                    'min_quality_threshold': CONFIG['MIN_QUALITY_THRESHOLD'],
                    'selections': selections,
                    'rejected': rejected
                }
                with open(selection_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                print(f"\n  Saved selections to: {selection_file}")

            print(f"\n{'='*70}")
            print(f"SELECTION SUMMARY - Track {track_id}")
            print(f"{'='*70}")
            for label in CONFIG['VISIBLE_FACES']:
                frames = selections.get(label, [])
                rej = rejected.get(label, [])
                print(f"  {label.upper():6s}: {len(frames)} selected, {len(rej)} rejected - {frames}")

            if crop_extractor and not args.aggregate:
                report_gen = PDFReportGenerator(analyzer, crop_extractor)
                report_gen.generate_report(track_id, selections, frame_qualities, rejected)
                report_gen.save_json_report(track_id, selections, frame_qualities, rejected)

            all_selections[track_id] = selections
            all_rejected[track_id] = rejected
            frame_qualities_cache[track_id] = frame_qualities

    # Generate aggregate PDF if requested
    if args.aggregate:
        agg_gen = AggregateReportGenerator(crop_extractor, video_name, frame_qualities_cache)
        output_path = viewpoint_dir / f"aggregate_filmstrip_v7_{video_name}.pdf"
        agg_gen.generate_aggregate_report(all_selections, output_path,
                                         all_rejected, show_rejected=args.show_rejected)

    # Final summary
    print(f"\n{'='*70}")
    print(f"ALL TRACKS COMPLETE - Processed {len(tracks_to_process)} tracks")
    print(f"{'='*70}")
    for track_id, selections in all_selections.items():
        total_frames = sum(len(v) for v in selections.values())
        orientations_covered = sum(1 for v in selections.values() if len(v) > 0)
        rejected_count = sum(len(v) for v in all_rejected.get(track_id, {}).values())
        print(f"  Track {track_id}: {total_frames} selected, {rejected_count} rejected, "
              f"{orientations_covered}/5 orientations")

    # Optional semantic face propagation
    if args.semantic:
        print("\n" + "=" * 70)
        print("SEMANTIC FACE PROPAGATION")
        print("=" * 70)
        try:
            from wildlift.viewpoint.semantic_propagator import TrackletViewpointAnalyzer
            sem_analyzer = TrackletViewpointAnalyzer(args.annotator_output, args.images_dir)
            sem_analyzer.run()
            print("Semantic face propagation complete.")
        except ImportError:
            print("ERROR: Could not import semantic_propagator. "
                  "Run separately: python -m wildlift.viewpoint.semantic_propagator")
        except Exception as e:
            print(f"WARNING: Semantic propagation failed: {e}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
