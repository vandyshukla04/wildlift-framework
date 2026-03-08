#!/usr/bin/env python3
"""
Viewpoint-Conditioned Re-Identification Proof of Concept

Demonstrates that viewpoint-aware feature matching outperforms viewpoint-agnostic
matching for discriminating between animals in a single scene.

Core idea: CUT3R gives us 3D geometry, so we KNOW which side of an animal we're
looking at. By comparing left-to-left and front-to-front (instead of arbitrary views),
we remove viewpoint as a nuisance variable.

Usage:
    # Single sequence
    python viewpoint_reid_poc.py \\
        --results_dir results/paper_final/thursday/zebras/zebr-14_2 \\
        --images_dir examples/wd_data/zebras/zebr-14_2 \\
        --output_dir results/reid_poc/zebr-14_2

    # All multi-track sequences
    python viewpoint_reid_poc.py --run_all --output_dir results/reid_poc

    # With full-resolution video crops
    python viewpoint_reid_poc.py --run_all --output_dir results/reid_poc \\
        --full_res_video /mnt/d/.../DJI_20250802125512_0006_V.MP4
"""

import os
import sys
import json
import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add CUT3R root to path
sys.path.insert(0, str(Path(__file__).parent))

from wildlift.viewpoint.analyzer import ViewpointAnalyzer, MaskCropExtractor, CONFIG

ORIENTATIONS = ['front', 'back', 'left', 'right', 'top']
DINO_MODEL = 'dinov2_vitb14'
CROP_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Orientation fallback chains for conditioned matching
ORIENTATION_FALLBACK = {
    'front': ['front', 'left', 'right', 'back', 'top'],
    'back':  ['back', 'left', 'right', 'front', 'top'],
    'left':  ['left', 'front', 'back', 'right', 'top'],
    'right': ['right', 'front', 'back', 'left', 'top'],
    'top':   ['top', 'front', 'back', 'left', 'right'],
}

# Sequence configurations for --run_all
SEQUENCES = [
    {
        'name': 'zebr-14_2',
        'animal': 'zebras',
        'results_dir': 'results/paper_final/thursday/zebras/zebr-14_2',
        'images_dir': 'examples/wd_data/zebras/zebr-14_2',
        'retrack_dir': 'retracked',  # use retracked mask_track_mapping
    },
    {
        'name': 'zebr-3',
        'animal': 'zebras',
        'results_dir': 'results/paper_final/thursday/zebras/zebr-3',
        'images_dir': 'examples/wd_data/zebras/zebr-3',
    },
    {
        'name': 'rhin-57_2',
        'animal': 'rhinos_cami',
        'results_dir': 'results/paper_final/thursday/rhinos_cami/rhin-57_2',
        'images_dir': 'examples/wd_data/rhinos_cami/rhin-57_2',
    },
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FrameCrop:
    """A single extracted crop with metadata."""
    track_id: int
    frame_name: str
    orientation: str
    quality_score: float
    crop_image: np.ndarray  # BGR, resized to CROP_SIZE x CROP_SIZE
    all_orientations: Dict[str, float] = field(default_factory=dict)


@dataclass
class GalleryEntry:
    """Gallery representation for one identity."""
    track_id: int
    avg_feature: np.ndarray                         # averaged across all crops
    orientation_features: Dict[str, np.ndarray]     # per-orientation averaged
    orientation_qualities: Dict[str, float]         # mean quality per orientation
    num_crops: int = 0


@dataclass
class ReIDMetrics:
    """Evaluation metrics for a re-ID method."""
    method_name: str
    rank1: float
    rank3: float
    mAP: float
    per_orientation_rank1: Dict[str, float]
    confusion_matrix: np.ndarray
    num_queries: int
    id_list: List[int] = field(default_factory=list)


# =============================================================================
# Data Loading
# =============================================================================

def _frame_sort_key(frame_name: str) -> int:
    nums = re.findall(r'\d+', frame_name)
    return int(nums[0]) if nums else 0


def load_scene_data(results_dir: str, images_dir: str, retrack_dir: str = None):
    """
    Load all tracklet data for a scene using existing ViewpointAnalyzer and MaskCropExtractor.

    Returns:
        analyzer: ViewpointAnalyzer instance
        crop_extractor: MaskCropExtractor instance
        track_crops: {track_id: {frame_name: FrameCrop}}
        track_class_names: {track_id: class_name}
    """
    results_path = Path(results_dir)
    images_path = Path(images_dir)

    # Determine annotator output directory (where bboxes + semantic faces live)
    annotator_output = results_path / 'corrected_bboxes'
    if not annotator_output.exists():
        # rhin-57_2 style: labels at root level
        annotator_output = results_path

    # Initialize ViewpointAnalyzer
    analyzer = ViewpointAnalyzer(str(annotator_output), str(images_path))

    # Initialize MaskCropExtractor
    # For retracked sequences, use the retrack dir for mask_track_mapping
    if retrack_dir:
        retrack_path = results_path / retrack_dir
        crop_extractor = MaskCropExtractor(
            images_dir=images_path,
            results_dir=retrack_path,
        )
    else:
        crop_extractor = MaskCropExtractor(
            images_dir=images_path,
            results_dir=results_path,
        )

    # Extract crops for all labeled tracks
    track_crops = {}
    track_class_names = {}

    for track_id in analyzer.labeled_tracks:
        print(f"  Processing track {track_id}...")
        frame_qualities = analyzer.compute_frame_qualities(track_id)
        if not frame_qualities:
            print(f"    Skipped: no frame qualities computed")
            continue

        # Get class name from bbox data
        first_frame = next(iter(analyzer.all_bbox_data.get(track_id, {})), None)
        if first_frame:
            track_class_names[track_id] = analyzer.all_bbox_data[track_id][first_frame].get('class_name', 'animal')

        crops = {}
        sorted_frames = sorted(frame_qualities.keys(), key=_frame_sort_key)

        for frame_name in sorted_frames:
            qualities = frame_qualities[frame_name]

            # Find dominant orientation (highest quality, excluding bottom)
            visible = {o: q for o, q in qualities.items()
                       if q > 0.05 and o != 'bottom'}
            if not visible:
                continue

            dominant_ori = max(visible, key=visible.get)

            # Extract crop
            crop = crop_extractor.extract_crop(frame_name, track_id,
                                               padding=20, background='white')
            if crop is None:
                continue

            # Check it's not an error image (solid dark red)
            if crop.shape[0] < 50 or crop.shape[1] < 50:
                continue

            # Resize for DINOv2
            crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                      interpolation=cv2.INTER_LANCZOS4)

            crops[frame_name] = FrameCrop(
                track_id=track_id,
                frame_name=frame_name,
                orientation=dominant_ori,
                quality_score=visible[dominant_ori],
                crop_image=crop_resized,
                all_orientations=qualities,
            )

        if crops:
            track_crops[track_id] = crops
            ori_counts = Counter(c.orientation for c in crops.values())
            print(f"    {len(crops)} crops: {dict(ori_counts)}")
        else:
            print(f"    Skipped: no valid crops extracted")

    return analyzer, crop_extractor, track_crops, track_class_names


def upgrade_crops_to_full_res(track_crops: Dict[int, Dict[str, FrameCrop]],
                               full_res_dir: str,
                               crop_extractor: MaskCropExtractor):
    """
    Re-extract crops from full-resolution images.
    Loads full-res image, upscales the mask to match, extracts crop at high res,
    then resizes back to CROP_SIZE for the feature extractor.
    """
    full_res_path = Path(full_res_dir)
    if not full_res_path.exists():
        print(f"  Full-res dir not found: {full_res_path}")
        return track_crops

    upgraded = 0
    for track_id, crops in track_crops.items():
        for frame_name, crop_data in crops.items():
            # Find full-res image
            hr_img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = full_res_path / f"{frame_name}{ext}"
                if candidate.exists():
                    hr_img_path = candidate
                    break
            if hr_img_path is None:
                continue

            hr_img = cv2.imread(str(hr_img_path))
            if hr_img is None:
                continue

            # Load mask at original resolution and upscale
            mask = crop_extractor._load_mask(frame_name, track_id)
            if mask is None:
                continue

            hr_h, hr_w = hr_img.shape[:2]
            mask_h, mask_w = mask.shape[:2]

            if mask_h != hr_h or mask_w != hr_w:
                mask_upscaled = cv2.resize(mask, (hr_w, hr_h),
                                           interpolation=cv2.INTER_NEAREST)
            else:
                mask_upscaled = mask

            # Extract crop from full-res image with mask
            ys, xs = np.where(mask_upscaled > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            mask_x1, mask_x2 = xs.min(), xs.max()
            mask_y1, mask_y2 = ys.min(), ys.max()
            mask_cx = (mask_x1 + mask_x2) // 2
            mask_cy = (mask_y1 + mask_y2) // 2
            mask_w_box = mask_x2 - mask_x1
            mask_h_box = mask_y2 - mask_y1

            padding = int(max(mask_w_box, mask_h_box) * 0.1)
            max_dim = max(mask_w_box, mask_h_box) + 2 * padding
            half_dim = max_dim // 2

            x1 = max(0, mask_cx - half_dim)
            x2 = min(hr_w, mask_cx + half_dim)
            y1 = max(0, mask_cy - half_dim)
            y2 = min(hr_h, mask_cy + half_dim)

            cropped = hr_img[y1:y2, x1:x2].copy()
            cropped_mask = mask_upscaled[y1:y2, x1:x2]

            # White background
            result = np.ones_like(cropped) * 255
            result[cropped_mask > 0] = cropped[cropped_mask > 0]

            # Resize to CROP_SIZE
            result_resized = cv2.resize(result, (CROP_SIZE, CROP_SIZE),
                                        interpolation=cv2.INTER_LANCZOS4)
            crop_data.crop_image = result_resized
            upgraded += 1

    print(f"  Upgraded {upgraded} crops to full resolution ({full_res_path})")
    return track_crops


# =============================================================================
# DINOv2 Feature Extraction
# =============================================================================

def _prepare_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR crop to ImageNet-normalized RGB tensor."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


class DINOv2Extractor:
    """Zero-shot feature extraction using DINOv2."""

    def __init__(self, model_name: str = DINO_MODEL, device: str = DEVICE):
        self.device = device
        print(f"\nLoading DINOv2: {model_name} on {device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model.to(device)
        self.feat_dim = self.model.embed_dim
        self.name = f"DINOv2-{model_name.split('_')[-1]}"
        print(f"  Feature dim: {self.feat_dim}")

    @torch.no_grad()
    def extract_batch(self, crops: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Extract L2-normalized [CLS] features for a list of BGR crops."""
        all_feats = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            tensors = torch.stack([_prepare_tensor(c) for c in batch])
            tensors = tensors.to(self.device)
            feats = self.model(tensors)
            feats = F.normalize(feats, p=2, dim=1)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)


class MegaDescriptorExtractor:
    """Wildlife-specific feature extraction using MegaDescriptor (via timm)."""

    def __init__(self, model_name: str = 'hf-hub:BVRA/MegaDescriptor-T-224',
                 device: str = DEVICE):
        import timm
        self.device = device
        print(f"\nLoading MegaDescriptor: {model_name} on {device}...")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        self.model.to(device)
        # Get feature dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            self.feat_dim = self.model(dummy).shape[1]
        self.name = "MegaDescriptor-T"
        print(f"  Feature dim: {self.feat_dim}")

    @torch.no_grad()
    def extract_batch(self, crops: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Extract L2-normalized features for a list of BGR crops."""
        all_feats = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            tensors = torch.stack([_prepare_tensor(c) for c in batch])
            tensors = tensors.to(self.device)
            feats = self.model(tensors)
            feats = F.normalize(feats, p=2, dim=1)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)


# =============================================================================
# Gallery Building
# =============================================================================

def build_gallery(track_crops: Dict[int, Dict[str, FrameCrop]],
                  frame_sets: Dict[int, List[str]],
                  extractor: DINOv2Extractor) -> List[GalleryEntry]:
    """
    Build gallery entries from specified frame subsets per track.
    """
    entries = []

    for track_id, frames in frame_sets.items():
        if track_id not in track_crops:
            continue

        crops_dict = track_crops[track_id]
        valid_frames = [f for f in frames if f in crops_dict]
        if not valid_frames:
            continue

        crop_list = [crops_dict[f].crop_image for f in valid_frames]
        orientations = [crops_dict[f].orientation for f in valid_frames]
        qualities = [crops_dict[f].quality_score for f in valid_frames]

        features = extractor.extract_batch(crop_list)

        # Viewpoint-agnostic: simple average
        avg_feat = np.mean(features, axis=0)
        avg_feat /= np.linalg.norm(avg_feat)

        # Viewpoint-conditioned: quality-weighted per-orientation average
        ori_features = {}
        ori_qualities = {}
        for ori in ORIENTATIONS:
            ori_idx = [i for i, o in enumerate(orientations) if o == ori]
            if not ori_idx:
                continue
            ori_feats = features[ori_idx]
            ori_quals = np.array([qualities[i] for i in ori_idx])
            weights = ori_quals / (ori_quals.sum() + 1e-8)
            weighted_feat = np.average(ori_feats, axis=0, weights=weights)
            weighted_feat /= np.linalg.norm(weighted_feat)
            ori_features[ori] = weighted_feat
            ori_qualities[ori] = float(np.mean(ori_quals))

        entries.append(GalleryEntry(
            track_id=track_id,
            avg_feature=avg_feat,
            orientation_features=ori_features,
            orientation_qualities=ori_qualities,
            num_crops=len(valid_frames),
        ))

    return entries


# =============================================================================
# Matching
# =============================================================================

def match_agnostic(query_feat: np.ndarray,
                   gallery: List[GalleryEntry]) -> List[Tuple[int, float]]:
    """Cosine similarity against average gallery features."""
    scores = [(g.track_id, float(np.dot(query_feat, g.avg_feature)))
              for g in gallery]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def match_conditioned(query_feat: np.ndarray, query_ori: str,
                      query_quality: float,
                      gallery: List[GalleryEntry]) -> List[Tuple[int, float]]:
    """
    Compare same-orientation features, with fallback chain.
    Boost exact-orientation matches, penalize fallbacks.
    """
    scores = []
    for g in gallery:
        gallery_feat = None
        matched_ori = None

        # Try orientation fallback chain
        for fallback_ori in ORIENTATION_FALLBACK.get(query_ori, [query_ori]):
            if fallback_ori in g.orientation_features:
                gallery_feat = g.orientation_features[fallback_ori]
                matched_ori = fallback_ori
                break

        if gallery_feat is None:
            gallery_feat = g.avg_feature
            matched_ori = 'avg'

        sim = float(np.dot(query_feat, gallery_feat))

        # Weight by match quality
        if matched_ori == query_ori:
            weight = 1.0
        elif matched_ori == 'avg':
            weight = 0.5
        else:
            weight = 0.7

        scores.append((g.track_id, sim * weight))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(query_results: List[Dict], method_name: str) -> ReIDMetrics:
    """
    Compute re-ID metrics from query results.

    Each result: {true_id, orientation, ranked_ids: [(track_id, score), ...]}
    """
    if not query_results:
        return ReIDMetrics(method_name=method_name, rank1=0, rank3=0, mAP=0,
                           per_orientation_rank1={}, confusion_matrix=np.zeros((1, 1)),
                           num_queries=0)

    id_list = sorted(set(r['true_id'] for r in query_results))
    id_to_idx = {tid: i for i, tid in enumerate(id_list)}
    n_ids = len(id_list)
    confusion = np.zeros((n_ids, n_ids), dtype=int)

    rank1_correct = 0
    rank3_correct = 0
    aps = []
    per_ori = defaultdict(lambda: [0, 0])  # [correct, total]

    for r in query_results:
        true_id = r['true_id']
        ranked = [x[0] for x in r['ranked_ids']]
        ori = r['orientation']

        # Rank accuracy
        if ranked and ranked[0] == true_id:
            rank1_correct += 1
            per_ori[ori][0] += 1
        if true_id in ranked[:3]:
            rank3_correct += 1
        per_ori[ori][1] += 1

        # Confusion matrix
        if ranked:
            pred_idx = id_to_idx.get(ranked[0], 0)
            true_idx = id_to_idx.get(true_id, 0)
            confusion[true_idx, pred_idx] += 1

        # Average precision
        relevant = 0
        ap_sum = 0.0
        for rank, rid in enumerate(ranked, 1):
            if rid == true_id:
                relevant += 1
                ap_sum += relevant / rank
        aps.append(ap_sum / max(relevant, 1))

    n = len(query_results)
    per_ori_rank1 = {ori: c[0] / c[1] if c[1] > 0 else 0.0
                     for ori, c in per_ori.items()}

    return ReIDMetrics(
        method_name=method_name,
        rank1=rank1_correct / n if n else 0,
        rank3=rank3_correct / n if n else 0,
        mAP=float(np.mean(aps)) if aps else 0,
        per_orientation_rank1=per_ori_rank1,
        confusion_matrix=confusion,
        num_queries=n,
        id_list=id_list,
    )


# =============================================================================
# Experiments
# =============================================================================

def run_temporal_split(track_crops: Dict[int, Dict[str, FrameCrop]],
                       extractor: DINOv2Extractor,
                       split_ratio: float = 0.5) -> Optional[Dict]:
    """
    Experiment 1: Temporal Split Re-ID.

    First split_ratio of frames → gallery, rest → query.
    """
    gallery_frame_sets = {}
    query_crops_list = []

    for track_id, crops in track_crops.items():
        frames = sorted(crops.keys(), key=_frame_sort_key)
        if len(frames) < 4:
            continue

        split_idx = max(2, int(len(frames) * split_ratio))
        gallery_frames = frames[:split_idx]
        query_frames = frames[split_idx:]

        gallery_frame_sets[track_id] = gallery_frames

        for f in query_frames:
            query_crops_list.append(crops[f])

    if len(gallery_frame_sets) < 2 or not query_crops_list:
        print("  Not enough data for temporal split experiment")
        return None

    # Build gallery
    gallery = build_gallery(track_crops, gallery_frame_sets, extractor)
    if len(gallery) < 2:
        return None

    # Extract query features
    query_features = extractor.extract_batch(
        [q.crop_image for q in query_crops_list])

    # Run both matchers
    agnostic_results = []
    conditioned_results = []

    for i, qcrop in enumerate(query_crops_list):
        feat = query_features[i]

        ag_ranked = match_agnostic(feat, gallery)
        vc_ranked = match_conditioned(feat, qcrop.orientation,
                                      qcrop.quality_score, gallery)

        result_base = {'true_id': qcrop.track_id, 'orientation': qcrop.orientation}
        agnostic_results.append({**result_base, 'ranked_ids': ag_ranked})
        conditioned_results.append({**result_base, 'ranked_ids': vc_ranked})

    return {
        'agnostic': compute_metrics(agnostic_results, 'Agnostic'),
        'conditioned': compute_metrics(conditioned_results, 'VP-Conditioned'),
        'num_queries': len(query_crops_list),
        'gallery_sizes': {g.track_id: g.num_crops for g in gallery},
        'query_crops': query_crops_list,
        'query_features': query_features,
        'gallery': gallery,
        'agnostic_results': agnostic_results,
        'conditioned_results': conditioned_results,
    }


def run_fragmentation(track_crops: Dict[int, Dict[str, FrameCrop]],
                      extractor: DINOv2Extractor,
                      num_fragments: int = 3,
                      gap_frames: int = 3) -> Optional[Dict]:
    """
    Experiment 2: Track Fragmentation Re-ID.

    Split tracks into fragments. Fragment 0 → gallery, rest → queries.
    """
    fragments = []  # (track_id, frag_idx, [frame_names])

    for track_id, crops in track_crops.items():
        frames = sorted(crops.keys(), key=_frame_sort_key)
        if len(frames) < num_fragments * 3:
            continue

        frag_size = len(frames) // num_fragments
        for fi in range(num_fragments):
            start = fi * frag_size
            end = start + frag_size if fi < num_fragments - 1 else len(frames)
            # Gap at boundaries
            if fi > 0:
                start += gap_frames // 2
            if fi < num_fragments - 1:
                end -= gap_frames // 2
            frag_frames = frames[start:end]
            if frag_frames:
                fragments.append((track_id, fi, frag_frames))

    gallery_frags = [(tid, fi, ff) for tid, fi, ff in fragments if fi == 0]
    query_frags = [(tid, fi, ff) for tid, fi, ff in fragments if fi > 0]

    if len(gallery_frags) < 2 or not query_frags:
        print("  Not enough data for fragmentation experiment")
        return None

    # Build gallery from fragment 0
    gallery_frame_sets = {tid: ff for tid, _, ff in gallery_frags}
    gallery = build_gallery(track_crops, gallery_frame_sets, extractor)
    if len(gallery) < 2:
        return None

    # Match query fragments
    agnostic_results = []
    conditioned_results = []

    for tid, fi, frag_frames in query_frags:
        if tid not in track_crops:
            continue
        valid_frames = [f for f in frag_frames if f in track_crops[tid]]
        if not valid_frames:
            continue

        crop_list = [track_crops[tid][f].crop_image for f in valid_frames]
        features = extractor.extract_batch(crop_list)

        # Average feature for the fragment
        avg_feat = np.mean(features, axis=0)
        avg_feat /= np.linalg.norm(avg_feat)

        # Dominant orientation of fragment
        orientations = [track_crops[tid][f].orientation for f in valid_frames]
        dominant_ori = Counter(orientations).most_common(1)[0][0]
        avg_quality = np.mean([track_crops[tid][f].quality_score for f in valid_frames])

        ag_ranked = match_agnostic(avg_feat, gallery)
        vc_ranked = match_conditioned(avg_feat, dominant_ori, avg_quality, gallery)

        result_base = {'true_id': tid, 'orientation': dominant_ori}
        agnostic_results.append({**result_base, 'ranked_ids': ag_ranked})
        conditioned_results.append({**result_base, 'ranked_ids': vc_ranked})

    if not agnostic_results:
        return None

    return {
        'agnostic': compute_metrics(agnostic_results, 'Agnostic'),
        'conditioned': compute_metrics(conditioned_results, 'VP-Conditioned'),
        'num_fragments': len(fragments),
        'num_query_fragments': len(query_frags),
    }


# =============================================================================
# Report Generation
# =============================================================================

def _metrics_to_dict(m: ReIDMetrics) -> Dict:
    return {
        'method': m.method_name,
        'rank1': m.rank1,
        'rank3': m.rank3,
        'mAP': m.mAP,
        'per_orientation_rank1': m.per_orientation_rank1,
        'num_queries': m.num_queries,
    }


def generate_report(exp1: Optional[Dict], exp2: Optional[Dict],
                    track_crops: Dict[int, Dict[str, FrameCrop]],
                    track_class_names: Dict[int, str],
                    scene_name: str, output_dir: Path):
    """Generate PDF report and JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_dir / 'reid_poc_report.pdf') as pdf:
        # Page 1: Metrics comparison
        _plot_metrics_comparison(pdf, exp1, exp2, scene_name)

        # Page 2: Per-orientation breakdown (exp1)
        if exp1:
            _plot_orientation_breakdown(pdf, exp1)

        # Page 3: Confusion matrices (exp1)
        if exp1:
            _plot_confusion_matrices(pdf, exp1, track_class_names)

        # Page 4: Gallery exemplars
        _plot_gallery_exemplars(pdf, track_crops, track_class_names)

        # Page 5: Match examples
        if exp1 and 'query_crops' in exp1:
            _plot_match_examples(pdf, exp1, track_crops, track_class_names)

    # JSON summary
    summary = {
        'scene': scene_name,
        'timestamp': datetime.now().isoformat(),
        'num_tracks': len(track_crops),
        'total_crops': sum(len(c) for c in track_crops.values()),
        'experiment_1_temporal_split': {
            'agnostic': _metrics_to_dict(exp1['agnostic']),
            'conditioned': _metrics_to_dict(exp1['conditioned']),
        } if exp1 else None,
        'experiment_2_fragmentation': {
            'agnostic': _metrics_to_dict(exp2['agnostic']),
            'conditioned': _metrics_to_dict(exp2['conditioned']),
        } if exp2 else None,
    }
    with open(output_dir / 'reid_poc_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Report: {output_dir / 'reid_poc_report.pdf'}")
    print(f"  Results: {output_dir / 'reid_poc_results.json'}")


def _plot_metrics_comparison(pdf, exp1, exp2, scene_name):
    """Bar chart comparing Rank-1, Rank-3, mAP across methods."""
    n_exps = sum(1 for e in [exp1, exp2] if e is not None)
    fig, axes = plt.subplots(1, max(n_exps, 1), figsize=(6 * max(n_exps, 1), 5))
    if n_exps == 1:
        axes = [axes]

    plot_data = []
    if exp1:
        plot_data.append((exp1, 'Exp 1: Temporal Split'))
    if exp2:
        plot_data.append((exp2, 'Exp 2: Fragmentation'))

    for ax, (exp, title) in zip(axes, plot_data):
        metrics = ['Rank-1', 'Rank-3', 'mAP']
        x = np.arange(len(metrics))
        width = 0.35

        ag = exp['agnostic']
        vc = exp['conditioned']

        ag_vals = [ag.rank1, ag.rank3, ag.mAP]
        vc_vals = [vc.rank1, vc.rank3, vc.mAP]

        bars1 = ax.bar(x - width / 2, ag_vals, width, label='Agnostic',
                        color='#5B9BD5', alpha=0.85)
        bars2 = ax.bar(x + width / 2, vc_vals, width, label='VP-Conditioned',
                        color='#ED7D31', alpha=0.85)

        # Value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.1%}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Score')
        ax.set_title(f'{title}\n({ag.num_queries} queries)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Viewpoint-Conditioned vs Agnostic Re-ID: {scene_name}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_orientation_breakdown(pdf, exp1):
    """Per-orientation rank-1 accuracy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ag = exp1['agnostic']
    vc = exp1['conditioned']
    all_oris = sorted(set(list(ag.per_orientation_rank1.keys()) +
                          list(vc.per_orientation_rank1.keys())))

    if not all_oris:
        plt.close(fig)
        return

    x = np.arange(len(all_oris))
    width = 0.35

    ag_vals = [ag.per_orientation_rank1.get(o, 0) for o in all_oris]
    vc_vals = [vc.per_orientation_rank1.get(o, 0) for o in all_oris]

    ax.bar(x - width / 2, ag_vals, width, label='Agnostic', color='#5B9BD5')
    ax.bar(x + width / 2, vc_vals, width, label='VP-Conditioned', color='#ED7D31')

    ax.set_xlabel('Query Orientation')
    ax.set_ylabel('Rank-1 Accuracy')
    ax.set_title('Per-Orientation Re-ID Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(all_oris)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_confusion_matrices(pdf, exp1, class_names):
    """Side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in [(axes[0], 'agnostic', 'Agnostic'),
                            (axes[1], 'conditioned', 'VP-Conditioned')]:
        m = exp1[key]
        cm = m.confusion_matrix
        labels = [f'T{tid}' for tid in m.id_list]

        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(f'{title}\nRank-1={m.rank1:.1%}  mAP={m.mAP:.3f}')
        ax.set_xlabel('Predicted ID')
        ax.set_ylabel('True ID')

        if len(labels) <= 8:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)

            # Annotate cells
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if cm[i, j] > 0:
                        ax.text(j, i, str(int(cm[i, j])), ha='center', va='center',
                                fontsize=9, color='white' if cm[i, j] > cm.max() / 2 else 'black')

        fig.colorbar(im, ax=ax)

    fig.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_gallery_exemplars(pdf, track_crops, class_names):
    """Show exemplar crops organized by orientation per track."""
    n_tracks = len(track_crops)
    if n_tracks == 0:
        return

    fig, axes = plt.subplots(n_tracks, len(ORIENTATIONS),
                              figsize=(3 * len(ORIENTATIONS), 3 * n_tracks))
    if n_tracks == 1:
        axes = [axes]

    for row, (track_id, crops) in enumerate(sorted(track_crops.items())):
        for col, ori in enumerate(ORIENTATIONS):
            ax = axes[row][col]

            # Find best crop for this orientation
            ori_crops = [(f, c) for f, c in crops.items() if c.orientation == ori]
            ori_crops.sort(key=lambda x: x[1].quality_score, reverse=True)

            if ori_crops:
                best = ori_crops[0][1]
                img_rgb = cv2.cvtColor(best.crop_image, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(f'{ori}\nq={best.quality_score:.2f}', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')

            ax.axis('off')
            if col == 0:
                cname = class_names.get(track_id, 'animal')
                ax.set_ylabel(f'Track {track_id}\n({cname})', fontsize=9)

    fig.suptitle('Gallery: Best Exemplar per Orientation per Track',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_match_examples(pdf, exp1, track_crops, class_names):
    """Show correct and incorrect match examples."""
    agnostic_results = exp1.get('agnostic_results', [])
    conditioned_results = exp1.get('conditioned_results', [])
    query_crops = exp1.get('query_crops', [])

    if not agnostic_results or not query_crops:
        return

    # Find cases where conditioned correct but agnostic wrong (and vice versa)
    conditioned_wins = []
    agnostic_wins = []
    both_correct = []

    for i, (ag, vc, qc) in enumerate(zip(agnostic_results, conditioned_results, query_crops)):
        ag_correct = ag['ranked_ids'][0][0] == ag['true_id'] if ag['ranked_ids'] else False
        vc_correct = vc['ranked_ids'][0][0] == vc['true_id'] if vc['ranked_ids'] else False

        if vc_correct and not ag_correct:
            conditioned_wins.append(i)
        elif ag_correct and not vc_correct:
            agnostic_wins.append(i)
        elif ag_correct and vc_correct:
            both_correct.append(i)

    # Plot up to 6 examples: 3 conditioned wins, 3 agnostic wins
    examples = []
    for idx in conditioned_wins[:3]:
        examples.append((idx, 'VP-Conditioned wins'))
    for idx in agnostic_wins[:3]:
        examples.append((idx, 'Agnostic wins'))
    if len(examples) < 6:
        for idx in both_correct[:6 - len(examples)]:
            examples.append((idx, 'Both correct'))

    if not examples:
        return

    n_examples = min(len(examples), 6)
    fig, axes = plt.subplots(n_examples, 3, figsize=(9, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]

    for row, (idx, label) in enumerate(examples[:n_examples]):
        qc = query_crops[idx]
        ag = agnostic_results[idx]
        vc = conditioned_results[idx]

        # Query crop
        ax = axes[row][0]
        img_rgb = cv2.cvtColor(qc.crop_image, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f'Query: T{qc.track_id} ({qc.orientation})\n{label}',
                     fontsize=8, fontweight='bold')
        ax.axis('off')

        # Agnostic match
        ax = axes[row][1]
        ag_pred = ag['ranked_ids'][0][0] if ag['ranked_ids'] else -1
        ag_score = ag['ranked_ids'][0][1] if ag['ranked_ids'] else 0
        correct = ag_pred == qc.track_id
        ax.set_title(f'Agnostic → T{ag_pred} (sim={ag_score:.3f})\n'
                     f'{"CORRECT" if correct else "WRONG"}',
                     fontsize=8, color='green' if correct else 'red')
        ax.axis('off')

        # Conditioned match
        ax = axes[row][2]
        vc_pred = vc['ranked_ids'][0][0] if vc['ranked_ids'] else -1
        vc_score = vc['ranked_ids'][0][1] if vc['ranked_ids'] else 0
        correct = vc_pred == qc.track_id
        ax.set_title(f'VP-Cond → T{vc_pred} (sim={vc_score:.3f})\n'
                     f'{"CORRECT" if correct else "WRONG"}',
                     fontsize=8, color='green' if correct else 'red')
        ax.axis('off')

    fig.suptitle('Match Examples: Query → Predicted Identity',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Full-Resolution Video Frame Extraction (optional)
# =============================================================================

def extract_full_res_frames(video_path: str, frame_numbers: List[int],
                            output_dir: Path) -> Dict[int, Path]:
    """
    Extract specific frames from a video at full resolution using ffmpeg.
    Returns {frame_number: path_to_extracted_frame}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}

    # Get video fps
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
        capture_output=True, text=True
    )
    fps_str = result.stdout.strip()
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    print(f"  Video FPS: {fps:.2f}")

    for frame_num in frame_numbers:
        out_path = output_dir / f'{frame_num}.jpg'
        if out_path.exists():
            extracted[frame_num] = out_path
            continue

        timestamp = frame_num / fps
        subprocess.run(
            ['ffmpeg', '-ss', f'{timestamp:.4f}', '-i', video_path,
             '-vframes', '1', '-q:v', '2', str(out_path),
             '-y', '-loglevel', 'error'],
            check=True
        )
        if out_path.exists():
            extracted[frame_num] = out_path

    print(f"  Extracted {len(extracted)}/{len(frame_numbers)} full-res frames")
    return extracted


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Viewpoint-Conditioned Re-ID Proof of Concept')
    parser.add_argument('--results_dir', type=str,
                        help='Path to CUT3R results directory for a sequence')
    parser.add_argument('--images_dir', type=str,
                        help='Path to source images directory')
    parser.add_argument('--output_dir', type=str, default='results/reid_poc',
                        help='Output directory')
    parser.add_argument('--run_all', action='store_true',
                        help='Run on all configured multi-track sequences')
    parser.add_argument('--retrack_dir', type=str, default=None,
                        help='Subdirectory with retracked mask_track_mapping.json')
    parser.add_argument('--dino_model', type=str, default='dinov2_vitb14',
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'],
                        help='DINOv2 model variant')
    parser.add_argument('--extractor', type=str, default='dino',
                        choices=['dino', 'mega', 'both'],
                        help='Feature extractor: dino, mega (MegaDescriptor), or both')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help='Gallery/query temporal split ratio')
    parser.add_argument('--num_fragments', type=int, default=3,
                        help='Number of fragments for experiment 2')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='DINOv2 inference batch size')
    parser.add_argument('--skip_exp2', action='store_true',
                        help='Skip fragmentation experiment')
    parser.add_argument('--full_res_video', type=str, default=None,
                        help='Path to full-resolution source video for better crops')
    parser.add_argument('--full_res_dir', type=str, default=None,
                        help='Path to pre-extracted full-resolution frames directory')
    args = parser.parse_args()

    root = Path(__file__).parent

    if args.run_all:
        sequences = SEQUENCES
    else:
        if not args.results_dir or not args.images_dir:
            parser.error('--results_dir and --images_dir required unless --run_all')
        sequences = [{
            'name': Path(args.results_dir).name,
            'animal': Path(args.results_dir).parent.name,
            'results_dir': args.results_dir,
            'images_dir': args.images_dir,
            'retrack_dir': args.retrack_dir,
        }]

    # Initialize feature extractors
    extractors = []
    if args.extractor in ('dino', 'both'):
        extractors.append(DINOv2Extractor(args.dino_model))
    if args.extractor in ('mega', 'both'):
        extractors.append(MegaDescriptorExtractor())

    all_results = {}

    for seq in sequences:
        name = seq['name']
        animal = seq.get('animal', '')
        retrack = seq.get('retrack_dir')

        print(f"\n{'=' * 60}")
        print(f"  {animal}/{name}")
        print(f"{'=' * 60}")

        results_dir = str(root / seq['results_dir']) if not os.path.isabs(seq['results_dir']) else seq['results_dir']
        images_dir = str(root / seq['images_dir']) if not os.path.isabs(seq['images_dir']) else seq['images_dir']
        output_path = Path(args.output_dir)
        output_dir = (root / output_path / name) if not output_path.is_absolute() else (output_path / name)

        # Load data
        analyzer, crop_extractor, track_crops, track_class_names = load_scene_data(
            results_dir, images_dir, retrack)

        if len(track_crops) < 2:
            print(f"  Skipping: only {len(track_crops)} tracks with crops (need >= 2)")
            continue

        # Upgrade to full-res crops if available
        full_res = args.full_res_dir or seq.get('full_res_dir')
        if full_res:
            track_crops = upgrade_crops_to_full_res(track_crops, full_res, crop_extractor)

        total_crops = sum(len(c) for c in track_crops.values())
        print(f"\n  Loaded {len(track_crops)} tracks, {total_crops} total crops")

        # Run experiments with each extractor
        seq_results = {}
        for extractor in extractors:
            ext_name = extractor.name
            print(f"\n  === Extractor: {ext_name} ===")

            # Experiment 1: Temporal Split
            print(f"\n  --- Experiment 1: Temporal Split ---")
            exp1 = run_temporal_split(track_crops, extractor, args.split_ratio)
            if exp1:
                ag = exp1['agnostic']
                vc = exp1['conditioned']
                print(f"    Agnostic:      Rank-1={ag.rank1:.1%}  mAP={ag.mAP:.3f}")
                print(f"    VP-Conditioned: Rank-1={vc.rank1:.1%}  mAP={vc.mAP:.3f}")
                delta = vc.rank1 - ag.rank1
                print(f"    Delta:          {delta:+.1%}")

            # Experiment 2: Track Fragmentation
            exp2 = None
            if not args.skip_exp2:
                print(f"\n  --- Experiment 2: Track Fragmentation ---")
                exp2 = run_fragmentation(track_crops, extractor, args.num_fragments)
                if exp2:
                    ag = exp2['agnostic']
                    vc = exp2['conditioned']
                    print(f"    Agnostic:      Rank-1={ag.rank1:.1%}  mAP={ag.mAP:.3f}")
                    print(f"    VP-Conditioned: Rank-1={vc.rank1:.1%}  mAP={vc.mAP:.3f}")
                    delta = vc.rank1 - ag.rank1
                    print(f"    Delta:          {delta:+.1%}")

            # Generate report per extractor
            ext_output_dir = output_dir / ext_name if len(extractors) > 1 else output_dir
            generate_report(exp1, exp2, track_crops, track_class_names,
                            f"{animal}/{name} ({ext_name})", ext_output_dir)
            seq_results[ext_name] = {'exp1': exp1, 'exp2': exp2}

        all_results[name] = seq_results

    # Final summary
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")

    for name, ext_results in all_results.items():
        print(f"\n  {name}:")
        for ext_name, results in ext_results.items():
            print(f"    [{ext_name}]")
            if results['exp1']:
                e1 = results['exp1']
                d = e1['conditioned'].rank1 - e1['agnostic'].rank1
                print(f"      Temporal Split:   Agnostic={e1['agnostic'].rank1:.1%}  "
                      f"VP-Cond={e1['conditioned'].rank1:.1%}  ({d:+.1%})")
            if results['exp2']:
                e2 = results['exp2']
                d = e2['conditioned'].rank1 - e2['agnostic'].rank1
                print(f"      Fragmentation:    Agnostic={e2['agnostic'].rank1:.1%}  "
                      f"VP-Cond={e2['conditioned'].rank1:.1%}  ({d:+.1%})")


if __name__ == '__main__':
    main()
