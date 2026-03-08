#!/usr/bin/env python3
"""
Tracklet Quality Report Generator

Generates intuitive, actionable PDF reports for wildlife tracklet analysis.
Consumes output from viewpoint_analyzer_v2.py and produces:
- Letter grades (A/B/C/F) with plain-English explanations
- Detailed metrics with thresholds and recommendations
- Masked animal crops from best exemplar frames
- Use-case fitness assessments (Data Collection QC, Re-ID Training, Publication)

Usage:
    python tracklet_report_generator.py --annotator_output results/zebra/scene1/corrected/
    python tracklet_report_generator.py --annotator_output results/zebra/scene1/corrected/ \
        --images_dir data/zebra/scene1/images/ --mask_dir data/zebra/scene1/grounded-sam/
"""

import os
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Import ViewpointAnalyzer for analysis
from wildlift.viewpoint.analyzer import ViewpointAnalyzer

# Optional: pycocotools for mask decoding
try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Mask extraction will be limited.")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GradeResult:
    """Result of grading a metric."""
    letter: str  # A, B, C, F
    score: float  # 0-1 normalized score
    label: str  # "Excellent", "Good", "Fair", "Poor"
    description: str  # Plain-English explanation
    color: str  # For visualization


@dataclass
class UseCaseFitness:
    """Fitness assessment for a specific use case."""
    verdict: str  # "KEEP", "FLAG", "RECAPTURE" / "USE", "AUGMENT", "EXCLUDE" / "USABLE", "LIMITED"
    confidence: float  # 0-1
    reasons: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


@dataclass
class TrackletGrades:
    """All grades for a single tracklet."""
    overall: GradeResult
    coverage: GradeResult
    diversity: GradeResult
    quality: GradeResult
    completeness: GradeResult
    reid_readiness: GradeResult


@dataclass
class TrackletInterpretation:
    """Human-readable interpretation of tracklet metrics."""
    summary: str
    issues: List[str]
    recommendations: List[str]
    strengths: List[str]


# =============================================================================
# QUALITY GRADER CLASS
# =============================================================================

class QualityGrader:
    """
    Converts raw metrics to intuitive letter grades with explanations.

    Grade thresholds are configurable but defaults are:
        A (Excellent): >= 0.8
        B (Good):      >= 0.6
        C (Fair):      >= 0.4
        F (Poor):      < 0.4
    """

    # Default thresholds
    THRESHOLDS = {
        'coverage_per_view': {'A': 0.25, 'B': 0.15, 'C': 0.08},  # Fraction of frames
        'coverage_views': {'A': 5, 'B': 4, 'C': 3},  # Number of views with >10%
        'diversity': {'A': 0.80, 'B': 0.60, 'C': 0.40},
        'quality': {'A': 0.70, 'B': 0.50, 'C': 0.30},
        'completeness': {'A': 0.80, 'B': 0.60, 'C': 0.40},
        'reid_readiness': {'A': 0.75, 'B': 0.55, 'C': 0.35},
        'occlusion_rate': {'A': 0.10, 'B': 0.20, 'C': 0.35},  # Inverted: lower is better
    }

    GRADE_COLORS = {
        'A': '#2ECC71',  # Green
        'B': '#3498DB',  # Blue
        'C': '#F39C12',  # Orange
        'F': '#E74C3C',  # Red
    }

    GRADE_LABELS = {
        'A': 'Excellent',
        'B': 'Good',
        'C': 'Fair',
        'F': 'Poor',
    }

    def __init__(self, thresholds: Dict = None):
        """Initialize grader with optional custom thresholds."""
        if thresholds:
            self.thresholds = {**self.THRESHOLDS, **thresholds}
        else:
            self.thresholds = self.THRESHOLDS.copy()

    def _score_to_grade(self, score: float, metric: str, inverted: bool = False) -> str:
        """Convert a 0-1 score to a letter grade."""
        thresholds = self.thresholds.get(metric, {'A': 0.8, 'B': 0.6, 'C': 0.4})

        if inverted:
            # Lower is better (e.g., occlusion rate)
            if score <= thresholds['A']:
                return 'A'
            elif score <= thresholds['B']:
                return 'B'
            elif score <= thresholds['C']:
                return 'C'
            else:
                return 'F'
        else:
            # Higher is better
            if score >= thresholds['A']:
                return 'A'
            elif score >= thresholds['B']:
                return 'B'
            elif score >= thresholds['C']:
                return 'C'
            else:
                return 'F'

    def grade_coverage(self, coverage_vector: Dict[str, float]) -> GradeResult:
        """Grade viewpoint coverage."""
        visible_labels = ['front', 'back', 'left', 'right', 'top']

        # Count views with sufficient coverage (>10%)
        good_views = sum(1 for label in visible_labels
                        if coverage_vector.get(label, 0) >= 0.10)

        # Get thresholds
        thresholds = self.thresholds['coverage_views']

        if good_views >= thresholds['A']:
            letter = 'A'
            description = f"All {good_views}/5 orientations have good coverage"
        elif good_views >= thresholds['B']:
            letter = 'B'
            missing = 5 - good_views
            description = f"{good_views}/5 orientations covered, {missing} underrepresented"
        elif good_views >= thresholds['C']:
            letter = 'C'
            missing = 5 - good_views
            description = f"Only {good_views}/5 orientations covered, {missing} missing"
        else:
            letter = 'F'
            description = f"Poor coverage: only {good_views}/5 orientations visible"

        # Identify specific gaps
        gaps = [label for label in visible_labels
                if coverage_vector.get(label, 0) < 0.10]
        if gaps:
            description += f" (missing: {', '.join(gaps)})"

        return GradeResult(
            letter=letter,
            score=good_views / 5.0,
            label=self.GRADE_LABELS[letter],
            description=description,
            color=self.GRADE_COLORS[letter]
        )

    def grade_diversity(self, normalized_diversity: float) -> GradeResult:
        """Grade viewpoint diversity (Shannon entropy normalized)."""
        letter = self._score_to_grade(normalized_diversity, 'diversity')

        if letter == 'A':
            description = "Views evenly distributed across orientations"
        elif letter == 'B':
            description = "Good viewpoint spread with minor imbalance"
        elif letter == 'C':
            description = "Viewpoints concentrated in few orientations"
        else:
            description = "Very uneven viewpoint distribution"

        return GradeResult(
            letter=letter,
            score=normalized_diversity,
            label=self.GRADE_LABELS[letter],
            description=description,
            color=self.GRADE_COLORS[letter]
        )

    def grade_quality(self, avg_quality: float,
                      quality_per_orientation: Dict[str, float] = None) -> GradeResult:
        """Grade overall image quality."""
        letter = self._score_to_grade(avg_quality, 'quality')

        if letter == 'A':
            description = "High quality views with good visibility"
        elif letter == 'B':
            description = "Acceptable quality for most applications"
        elif letter == 'C':
            description = "Quality issues may affect analysis"
        else:
            description = "Poor quality - distant or blurry captures"

        # Add specific quality issues
        if quality_per_orientation:
            low_quality = [label for label, q in quality_per_orientation.items()
                          if q < 0.3 and not np.isnan(q) and q > 0]
            if low_quality:
                description += f" (low quality: {', '.join(low_quality)})"

        return GradeResult(
            letter=letter,
            score=avg_quality,
            label=self.GRADE_LABELS[letter],
            description=description,
            color=self.GRADE_COLORS[letter]
        )

    def grade_completeness(self, completeness_score: float) -> GradeResult:
        """Grade coverage completeness (min coverage across views)."""
        letter = self._score_to_grade(completeness_score, 'completeness')

        if letter == 'A':
            description = "All orientations well represented"
        elif letter == 'B':
            description = "Most orientations adequately covered"
        elif letter == 'C':
            description = "Some orientations significantly underrepresented"
        else:
            description = "Major gaps in viewpoint coverage"

        return GradeResult(
            letter=letter,
            score=completeness_score,
            label=self.GRADE_LABELS[letter],
            description=description,
            color=self.GRADE_COLORS[letter]
        )

    def grade_reid_readiness(self, reid_score: float) -> GradeResult:
        """Grade re-identification readiness."""
        letter = self._score_to_grade(reid_score, 'reid_readiness')

        if letter == 'A':
            description = "Excellent for re-ID training"
        elif letter == 'B':
            description = "Good for re-ID with minor limitations"
        elif letter == 'C':
            description = "Marginal for re-ID, may need augmentation"
        else:
            description = "Not recommended for re-ID training"

        return GradeResult(
            letter=letter,
            score=reid_score,
            label=self.GRADE_LABELS[letter],
            description=description,
            color=self.GRADE_COLORS[letter]
        )

    def grade_occlusion(self, occlusion_rate: float) -> GradeResult:
        """Grade occlusion rate (lower is better)."""
        letter = self._score_to_grade(occlusion_rate, 'occlusion_rate', inverted=True)

        if letter == 'A':
            description = "Very low occlusion - animals well separated"
        elif letter == 'B':
            description = "Low occlusion - mostly clear views"
        elif letter == 'C':
            description = "Moderate occlusion - some blocking"
        else:
            description = "High occlusion - dense grouping, frequent blocking"

        return GradeResult(
            letter=letter,
            score=1.0 - occlusion_rate,  # Invert for consistency (higher = better)
            label=self.GRADE_LABELS[letter],
            description=description,
            color=self.GRADE_COLORS[letter]
        )

    def compute_overall_grade(self, grades: TrackletGrades) -> GradeResult:
        """Compute overall grade from component grades."""
        # Weighted average: coverage and diversity matter most
        weights = {
            'coverage': 0.25,
            'diversity': 0.25,
            'quality': 0.20,
            'completeness': 0.15,
            'reid_readiness': 0.15,
        }

        weighted_score = (
            weights['coverage'] * grades.coverage.score +
            weights['diversity'] * grades.diversity.score +
            weights['quality'] * grades.quality.score +
            weights['completeness'] * grades.completeness.score +
            weights['reid_readiness'] * grades.reid_readiness.score
        )

        # Determine letter grade
        if weighted_score >= 0.75:
            letter = 'A'
        elif weighted_score >= 0.55:
            letter = 'B'
        elif weighted_score >= 0.35:
            letter = 'C'
        else:
            letter = 'F'

        # Build description
        descriptions = {
            'A': "Excellent tracklet quality for all use cases",
            'B': "Good tracklet suitable for most applications",
            'C': "Fair quality with notable limitations",
            'F': "Poor quality - consider recapture or exclude",
        }

        return GradeResult(
            letter=letter,
            score=weighted_score,
            label=self.GRADE_LABELS[letter],
            description=descriptions[letter],
            color=self.GRADE_COLORS[letter]
        )

    def grade_tracklet(self, profile: Dict) -> TrackletGrades:
        """Grade all aspects of a tracklet."""
        # Extract metrics from profile
        coverage_vector = profile.get('coverage_vector', {})
        normalized_diversity = profile.get('normalized_diversity', 0.0)
        completeness_score = profile.get('completeness_score', 0.0)
        reid_score = profile.get('reid_readiness_score', 0.0)
        quality_per_orientation = profile.get('average_quality_per_orientation', {})

        # Calculate average quality (excluding NaN and bottom)
        valid_qualities = [q for label, q in quality_per_orientation.items()
                         if label != 'bottom' and not np.isnan(q) and q > 0]
        avg_quality = np.mean(valid_qualities) if valid_qualities else 0.0

        # Grade each component
        coverage_grade = self.grade_coverage(coverage_vector)
        diversity_grade = self.grade_diversity(normalized_diversity)
        quality_grade = self.grade_quality(avg_quality, quality_per_orientation)
        completeness_grade = self.grade_completeness(completeness_score)
        reid_grade = self.grade_reid_readiness(reid_score)

        # Create grades object
        grades = TrackletGrades(
            overall=None,  # Will be computed below
            coverage=coverage_grade,
            diversity=diversity_grade,
            quality=quality_grade,
            completeness=completeness_grade,
            reid_readiness=reid_grade,
        )

        # Compute overall grade
        grades.overall = self.compute_overall_grade(grades)

        return grades


# =============================================================================
# USE CASE FITNESS EVALUATORS
# =============================================================================

class UseCaseEvaluator:
    """Evaluates tracklet fitness for different use cases."""

    def __init__(self, grader: QualityGrader = None):
        self.grader = grader or QualityGrader()

    def evaluate_for_data_collection_qc(self, profile: Dict,
                                        grades: TrackletGrades) -> UseCaseFitness:
        """Evaluate if tracklet should be kept, flagged, or recaptured."""
        reasons = []
        caveats = []

        # Decision logic
        overall_letter = grades.overall.letter
        coverage_letter = grades.coverage.letter

        if overall_letter in ['A', 'B'] and coverage_letter in ['A', 'B']:
            verdict = "KEEP"
            confidence = 0.90 if overall_letter == 'A' else 0.75
            reasons.append("Meets quality standards for data collection")
        elif overall_letter == 'C' or coverage_letter == 'C':
            verdict = "FLAG"
            confidence = 0.60
            reasons.append("Quality concerns require manual review")

            # Identify specific issues
            if grades.coverage.letter in ['C', 'F']:
                caveats.append(f"Coverage issue: {grades.coverage.description}")
            if grades.quality.letter in ['C', 'F']:
                caveats.append(f"Quality issue: {grades.quality.description}")
        else:
            verdict = "RECAPTURE"
            confidence = 0.80
            reasons.append("Below minimum quality threshold")
            caveats.append("Consider recapturing this individual")

        return UseCaseFitness(
            verdict=verdict,
            confidence=confidence,
            reasons=reasons,
            caveats=caveats
        )

    def evaluate_for_reid_training(self, profile: Dict,
                                   grades: TrackletGrades) -> UseCaseFitness:
        """Evaluate suitability for re-ID model training."""
        reasons = []
        caveats = []

        reid_letter = grades.reid_readiness.letter
        diversity_letter = grades.diversity.letter

        if reid_letter == 'A' and diversity_letter in ['A', 'B']:
            verdict = "USE"
            confidence = grades.reid_readiness.score
            reasons.append("High-quality training data")
        elif reid_letter in ['A', 'B']:
            verdict = "USE"
            confidence = grades.reid_readiness.score
            reasons.append("Suitable for training")

            if diversity_letter in ['C', 'F']:
                caveats.append("Limited viewpoint diversity may reduce model generalization")
        elif reid_letter == 'C':
            verdict = "AUGMENT"
            confidence = 0.50
            reasons.append("Usable with data augmentation")
            caveats.append("Apply viewpoint augmentation techniques")

            gaps = profile.get('coverage_gaps', [])
            if gaps:
                caveats.append(f"Missing views: {', '.join(gaps)}")
        else:
            verdict = "EXCLUDE"
            confidence = 0.85
            reasons.append("Not recommended for training")
            caveats.append("Would likely introduce noise into model")

        return UseCaseFitness(
            verdict=verdict,
            confidence=confidence,
            reasons=reasons,
            caveats=caveats
        )

    def evaluate_for_publication(self, profile: Dict,
                                 grades: TrackletGrades) -> UseCaseFitness:
        """Evaluate suitability for publication/analysis."""
        reasons = []
        caveats = []

        overall_letter = grades.overall.letter

        if overall_letter in ['A', 'B']:
            verdict = "USABLE"
            confidence = grades.overall.score
            reasons.append("Suitable for publication")

            # Note any minor issues for transparency
            gaps = profile.get('coverage_gaps', [])
            if gaps:
                caveats.append(f"Note: {', '.join(gaps)} views underrepresented")
        else:
            verdict = "LIMITED"
            confidence = 0.50
            reasons.append("Usable with documented limitations")

            # Document specific issues
            if grades.coverage.letter in ['C', 'F']:
                caveats.append(f"Coverage limitation: {grades.coverage.description}")
            if grades.diversity.letter in ['C', 'F']:
                caveats.append("Viewpoint bias should be noted in methods")

        return UseCaseFitness(
            verdict=verdict,
            confidence=confidence,
            reasons=reasons,
            caveats=caveats
        )


# =============================================================================
# INTERPRETATION GENERATOR
# =============================================================================

class InterpretationGenerator:
    """Generates human-readable interpretations and recommendations."""

    def generate_interpretation(self, profile: Dict,
                               grades: TrackletGrades) -> TrackletInterpretation:
        """Generate comprehensive interpretation for a tracklet."""
        issues = []
        recommendations = []
        strengths = []

        # Identify strengths
        if grades.diversity.letter in ['A', 'B']:
            strengths.append("Good viewpoint diversity")
        if grades.coverage.letter == 'A':
            strengths.append("Complete coverage of all orientations")
        if grades.quality.letter in ['A', 'B']:
            strengths.append("High image quality")
        if grades.reid_readiness.letter == 'A':
            strengths.append("Excellent for re-identification")

        # Identify issues and generate recommendations
        coverage_vector = profile.get('coverage_vector', {})
        quality_per_orientation = profile.get('average_quality_per_orientation', {})
        visible_labels = ['front', 'back', 'left', 'right', 'top']

        # Coverage issues
        for label in visible_labels:
            coverage = coverage_vector.get(label, 0)
            if coverage < 0.08:
                issues.append(f"'{label}' view severely underrepresented ({coverage:.0%} of frames)")
                recommendations.append(f"Capture additional frames showing the {label} of the animal")
            elif coverage < 0.15:
                issues.append(f"'{label}' view underrepresented ({coverage:.0%} of frames)")

        # Quality issues
        for label in visible_labels:
            quality = quality_per_orientation.get(label, 0)
            if not np.isnan(quality) and 0 < quality < 0.3:
                issues.append(f"'{label}' view has low quality ({quality:.2f})")
                recommendations.append(f"For {label} views, capture at closer range or higher resolution")

        # Diversity issues
        if grades.diversity.letter in ['C', 'F']:
            issues.append("Viewpoints concentrated in few orientations")
            recommendations.append("Vary capture angles to improve viewpoint distribution")

        # Generate summary
        overall = grades.overall
        if overall.letter == 'A':
            summary = "Excellent tracklet with comprehensive viewpoint coverage and high quality."
        elif overall.letter == 'B':
            summary = "Good tracklet suitable for most applications with minor limitations."
        elif overall.letter == 'C':
            summary = "Fair tracklet with notable gaps that may affect some analyses."
        else:
            summary = "Poor quality tracklet - consider recapture or exclude from analysis."

        return TrackletInterpretation(
            summary=summary,
            issues=issues,
            recommendations=recommendations,
            strengths=strengths
        )


# =============================================================================
# MASK CROP EXTRACTOR
# =============================================================================

class MaskCropExtractor:
    """Extracts masked animal crops from images."""

    def __init__(self, images_dir: Path, mask_dir: Path,
                 annotator_output_dir: Path):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.annotator_output_dir = annotator_output_dir

        # Load mask-track mapping if available
        self.mask_track_mapping = self._load_mask_track_mapping()

    def _load_mask_track_mapping(self) -> Dict:
        """Load mask-to-track mapping from annotator output."""
        mapping_file = self.annotator_output_dir / "corrected_labels" / "mask_track_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)
        return {}

    def _load_image(self, frame_name: str) -> Optional[np.ndarray]:
        """Load image for a frame."""
        if self.images_dir is None:
            return None

        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            img_path = self.images_dir / f"{frame_name}{ext}"
            if img_path.exists():
                return cv2.imread(str(img_path))

        return None

    def _load_mask(self, frame_name: str, track_id: int) -> Optional[np.ndarray]:
        """Load mask for a specific track in a frame."""
        if self.mask_dir is None or not PYCOCOTOOLS_AVAILABLE:
            return None

        # Try direct lookup via mapping
        frame_key = frame_name
        # Try extracting frame index
        import re
        frame_match = re.findall(r'\d+', frame_name)
        if frame_match:
            frame_key = frame_match[0]

        mask_idx = self.mask_track_mapping.get(str(frame_key), {}).get(str(track_id))

        # Load from grounded-SAM JSON
        json_file = self.mask_dir / f"{frame_key}_results.json"
        if not json_file.exists():
            # Try with frame_name
            json_file = self.mask_dir / f"{frame_name}_results.json"

        if not json_file.exists():
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

            # Fallback: use first annotation (if only one animal)
            if len(annotations) == 1:
                rle = annotations[0].get('segmentation')
                if rle:
                    return mask_utils.decode(rle)

        except Exception as e:
            print(f"Warning: Could not load mask for {frame_name}, track {track_id}: {e}")

        return None

    def extract_crop(self, frame_name: str, track_id: int,
                    bbox_data: Dict = None, padding: int = 30,
                    background: str = 'white') -> Optional[np.ndarray]:
        """
        Extract a masked crop of an animal.

        Args:
            frame_name: Frame identifier
            track_id: Track ID
            bbox_data: Optional bounding box data (for crop bounds)
            padding: Pixels to add around the crop
            background: 'white' or 'transparent'

        Returns:
            Cropped image with mask applied, or None if unavailable
        """
        # Load image
        image = self._load_image(frame_name)
        if image is None:
            return None

        # Load mask
        mask = self._load_mask(frame_name, track_id)
        if mask is None:
            # Return uncropped image region if no mask available
            if bbox_data is not None:
                return self._crop_without_mask(image, bbox_data, padding)
            return None

        # Find bounding box of mask
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        x1, x2 = max(0, xs.min() - padding), min(image.shape[1], xs.max() + padding)
        y1, y2 = max(0, ys.min() - padding), min(image.shape[0], ys.max() + padding)

        # Crop image and mask
        cropped_image = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2]

        # Apply mask
        if background == 'white':
            # White background
            result = np.ones_like(cropped_image) * 255
            result[cropped_mask > 0] = cropped_image[cropped_mask > 0]
        else:
            # Transparent background (RGBA)
            result = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = (cropped_mask * 255).astype(np.uint8)

        return result

    def _crop_without_mask(self, image: np.ndarray, bbox_data: Dict,
                          padding: int) -> np.ndarray:
        """Crop image using bounding box when mask is unavailable."""
        # This is a fallback - ideally we have masks
        # For 3D bbox, project center and use dimensions
        center = bbox_data.get('center', None)
        if center is None:
            return None

        # Simple crop around center (approximate)
        h, w = image.shape[:2]
        cx, cy = int(w / 2), int(h / 2)  # Default to image center

        crop_size = 200  # Default crop size
        x1 = max(0, cx - crop_size - padding)
        x2 = min(w, cx + crop_size + padding)
        y1 = max(0, cy - crop_size - padding)
        y2 = min(h, cy + crop_size + padding)

        return image[y1:y2, x1:x2].copy()


# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

class PDFReportGenerator:
    """Generates comprehensive PDF reports with metrics, grades, and crops."""

    def __init__(self, analyzer: ViewpointAnalyzer,
                 grader: QualityGrader = None,
                 evaluator: UseCaseEvaluator = None,
                 interpreter: InterpretationGenerator = None,
                 crop_extractor: MaskCropExtractor = None):
        self.analyzer = analyzer
        self.grader = grader or QualityGrader()
        self.evaluator = evaluator or UseCaseEvaluator(self.grader)
        self.interpreter = interpreter or InterpretationGenerator()
        self.crop_extractor = crop_extractor

    def generate_report(self, tracklet_profiles: Dict,
                       occlusion_stats: Dict = None,
                       output_path: Path = None) -> Path:
        """
        Generate complete PDF report.

        Args:
            tracklet_profiles: Output from ViewpointAnalyzer.compute_tracklet_viewpoint_profiles()
            occlusion_stats: Optional occlusion statistics
            output_path: Output PDF path (default: viewpoint_analysis/tracklet_quality_report.pdf)

        Returns:
            Path to generated PDF
        """
        if output_path is None:
            output_path = self.analyzer.annotator_output_dir / "viewpoint_analysis" / "tracklet_quality_report.pdf"

        output_path.parent.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("GENERATING PDF QUALITY REPORT")
        print("="*70)

        # Grade all tracklets
        all_grades = {}
        all_interpretations = {}
        all_fitness = {}

        for track_id, profile in tracklet_profiles.items():
            grades = self.grader.grade_tracklet(profile)
            interpretation = self.interpreter.generate_interpretation(profile, grades)
            fitness = {
                'qc': self.evaluator.evaluate_for_data_collection_qc(profile, grades),
                'reid': self.evaluator.evaluate_for_reid_training(profile, grades),
                'publication': self.evaluator.evaluate_for_publication(profile, grades),
            }

            all_grades[track_id] = grades
            all_interpretations[track_id] = interpretation
            all_fitness[track_id] = fitness

        # Generate PDF
        with PdfPages(str(output_path)) as pdf:
            # Page 1: Executive Summary
            self._create_summary_page(pdf, tracklet_profiles, all_grades, all_fitness)

            # Pages 2-N: Individual Tracklet Details
            for track_id in sorted(tracklet_profiles.keys()):
                self._create_tracklet_page(
                    pdf, track_id,
                    tracklet_profiles[track_id],
                    all_grades[track_id],
                    all_interpretations[track_id],
                    all_fitness[track_id]
                )

            # Occlusion Analysis Page (if available)
            if occlusion_stats:
                self._create_occlusion_page(pdf, occlusion_stats)

            # Final Page: Recommendations
            self._create_recommendations_page(
                pdf, tracklet_profiles, all_grades,
                all_interpretations, all_fitness
            )

        print(f"\nSaved PDF report to: {output_path}")
        return output_path

    def _create_summary_page(self, pdf: PdfPages, tracklet_profiles: Dict,
                            all_grades: Dict, all_fitness: Dict):
        """Create executive summary page."""
        fig = plt.figure(figsize=(11, 8.5))

        # Title
        fig.suptitle('TRACKLET QUALITY REPORT', fontsize=20, fontweight='bold', y=0.98)

        # Subtitle with metadata
        scene_name = self.analyzer.annotator_output_dir.name
        date_str = datetime.now().strftime('%Y-%m-%d')
        fig.text(0.5, 0.93, f'Scene: {scene_name}  |  Date: {date_str}',
                ha='center', fontsize=12, style='italic')

        gs = GridSpec(3, 2, figure=fig, height_ratios=[0.3, 0.5, 0.2],
                     left=0.08, right=0.92, top=0.88, bottom=0.08,
                     hspace=0.3, wspace=0.3)

        # Summary statistics
        ax_stats = fig.add_subplot(gs[0, :])
        ax_stats.axis('off')

        total_tracks = len(tracklet_profiles)
        total_frames = sum(p.get('total_frames', 0) for p in tracklet_profiles.values())

        # Calculate overall dataset grade
        avg_score = np.mean([g.overall.score for g in all_grades.values()])
        if avg_score >= 0.75:
            dataset_grade = 'A'
        elif avg_score >= 0.55:
            dataset_grade = 'B'
        elif avg_score >= 0.35:
            dataset_grade = 'C'
        else:
            dataset_grade = 'F'

        grade_color = self.grader.GRADE_COLORS[dataset_grade]

        stats_text = (
            f"Tracks Analyzed: {total_tracks}          "
            f"Total Frames: {total_frames}          "
            f"Overall Grade: {dataset_grade} ({self.grader.GRADE_LABELS[dataset_grade]})"
        )
        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=grade_color, alpha=0.3))

        # Tracklet summary table
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

        # Create table data
        table_data = [['Track', 'Grade', 'Coverage', 'Diversity', 'Quality', 'Re-ID Ready']]
        cell_colors = [['#E8E8E8'] * 6]

        for track_id in sorted(tracklet_profiles.keys()):
            grades = all_grades[track_id]
            fitness = all_fitness[track_id]

            coverage_str = f"{int(grades.coverage.score * 5)}/5 views"
            reid_ready = "Yes" if fitness['reid'].verdict == "USE" else (
                "Marginal" if fitness['reid'].verdict == "AUGMENT" else "No"
            )

            row = [
                f"Track {track_id}",
                grades.overall.letter,
                coverage_str,
                f"{grades.diversity.score:.2f}",
                f"{grades.quality.score:.2f}",
                reid_ready
            ]
            table_data.append(row)

            # Color code by grade
            row_color = self.grader.GRADE_COLORS[grades.overall.letter]
            cell_colors.append([row_color + '40'] * 6)  # Add alpha

        table = ax_table.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            cellColours=cell_colors
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Key issues section
        ax_issues = fig.add_subplot(gs[2, :])
        ax_issues.axis('off')

        # Collect key issues
        key_issues = []
        for track_id, grades in all_grades.items():
            if grades.coverage.letter in ['C', 'F']:
                key_issues.append(f"Track {track_id}: {grades.coverage.description}")
            if grades.quality.letter == 'F':
                key_issues.append(f"Track {track_id}: {grades.quality.description}")

        if key_issues:
            issues_text = "KEY ISSUES:\n" + "\n".join(f"  - {issue}" for issue in key_issues[:4])
        else:
            issues_text = "KEY ISSUES:\n  No critical issues detected"

        ax_issues.text(0.05, 0.9, issues_text, va='top', fontsize=10,
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_tracklet_page(self, pdf: PdfPages, track_id: int,
                             profile: Dict, grades: TrackletGrades,
                             interpretation: TrackletInterpretation,
                             fitness: Dict):
        """Create detailed page for a single tracklet."""
        fig = plt.figure(figsize=(11, 8.5))

        # Title
        grade_color = grades.overall.color
        fig.suptitle(f'TRACK {track_id} DETAIL', fontsize=18, fontweight='bold', y=0.98)

        # Grade badge
        fig.text(0.92, 0.96, grades.overall.letter, fontsize=24, fontweight='bold',
                ha='center', va='center', color='white',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=grade_color))

        gs = GridSpec(3, 3, figure=fig,
                     left=0.08, right=0.92, top=0.90, bottom=0.08,
                     hspace=0.35, wspace=0.3)

        # Top row: Metrics summary
        ax_metrics = fig.add_subplot(gs[0, :2])
        ax_metrics.axis('off')

        metrics_text = (
            f"Grade: {grades.overall.letter} ({grades.overall.label})     "
            f"Re-ID Readiness: {grades.reid_readiness.score:.0%}\n"
            f"Diversity: {grades.diversity.score:.2f}     "
            f"Completeness: {grades.completeness.score:.2f}     "
            f"Total Frames: {profile.get('total_frames', 0)}"
        )
        ax_metrics.text(0.0, 0.5, metrics_text, fontsize=11, va='center',
                       fontfamily='monospace')

        # Grade breakdown
        ax_breakdown = fig.add_subplot(gs[0, 2])
        ax_breakdown.axis('off')

        breakdown_lines = [
            f"Coverage:    {grades.coverage.letter}",
            f"Diversity:   {grades.diversity.letter}",
            f"Quality:     {grades.quality.letter}",
            f"Completeness:{grades.completeness.letter}",
        ]
        for i, line in enumerate(breakdown_lines):
            letter = line.split()[-1]
            color = self.grader.GRADE_COLORS.get(letter, 'black')
            ax_breakdown.text(0.0, 0.8 - i*0.25, line, fontsize=10,
                            fontfamily='monospace', color=color, fontweight='bold')

        # Middle row: Exemplar crops
        ax_crops = fig.add_subplot(gs[1, :])
        ax_crops.set_title('BEST VIEW EXEMPLARS', fontsize=12, fontweight='bold')
        ax_crops.axis('off')

        # Extract and display crops
        visible_labels = ['front', 'back', 'left', 'right', 'top']
        optimal_frames = profile.get('optimal_exemplar_frames', {})

        # Create sub-axes for crops
        crop_width = 0.18
        crop_gap = 0.02
        start_x = 0.05

        for i, label in enumerate(visible_labels):
            exemplar = optimal_frames.get(label, {})
            frame_name = exemplar.get('frame')
            quality = exemplar.get('quality_score', 0)

            # Position for this crop
            ax_pos = [start_x + i * (crop_width + crop_gap), 0.1, crop_width, 0.7]
            ax_crop = ax_crops.inset_axes(ax_pos)

            # Try to extract crop
            crop_image = None
            if frame_name and self.crop_extractor:
                crop_image = self.crop_extractor.extract_crop(frame_name, track_id)

            if crop_image is not None:
                # Convert BGR to RGB for display
                if crop_image.shape[-1] == 3:
                    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                elif crop_image.shape[-1] == 4:
                    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGRA2RGBA)
                ax_crop.imshow(crop_image)
            else:
                # Placeholder
                ax_crop.text(0.5, 0.5, 'N/A', ha='center', va='center',
                           fontsize=14, color='gray')
                ax_crop.set_facecolor('#E0E0E0')

            ax_crop.axis('off')
            ax_crop.set_title(f'{label.upper()}\nQ: {quality:.2f}', fontsize=9)

        # Bottom row: Assessment and recommendations
        ax_assessment = fig.add_subplot(gs[2, :2])
        ax_assessment.axis('off')

        assessment_lines = ["ASSESSMENT:"]
        for strength in interpretation.strengths[:3]:
            assessment_lines.append(f"  + {strength}")
        for issue in interpretation.issues[:3]:
            assessment_lines.append(f"  - {issue}")

        ax_assessment.text(0.0, 0.95, '\n'.join(assessment_lines), va='top',
                          fontsize=9, fontfamily='monospace')

        # Use-case fitness
        ax_fitness = fig.add_subplot(gs[2, 2])
        ax_fitness.axis('off')

        qc = fitness['qc']
        reid = fitness['reid']
        pub = fitness['publication']

        fitness_lines = [
            "USE-CASE FITNESS:",
            f"  Data QC:     {qc.verdict}",
            f"  Re-ID:       {reid.verdict}",
            f"  Publication: {pub.verdict}",
        ]
        ax_fitness.text(0.0, 0.95, '\n'.join(fitness_lines), va='top',
                       fontsize=9, fontfamily='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_occlusion_page(self, pdf: PdfPages, occlusion_stats: Dict):
        """Create occlusion analysis page."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('INTER-ANIMAL OCCLUSION ANALYSIS', fontsize=18, fontweight='bold', y=0.98)

        gs = GridSpec(2, 2, figure=fig,
                     left=0.08, right=0.92, top=0.90, bottom=0.08,
                     hspace=0.3, wspace=0.3)

        # Grade the occlusion
        occlusion_rate = occlusion_stats.get('occlusion_rate', 0)
        occlusion_grade = self.grader.grade_occlusion(occlusion_rate)

        # Summary panel
        ax_summary = fig.add_subplot(gs[0, 0])
        ax_summary.axis('off')

        summary_text = (
            f"Occlusion Rate: {occlusion_rate:.1%}\n"
            f"Grade: {occlusion_grade.letter} ({occlusion_grade.label})\n\n"
            f"Frames Analyzed: {occlusion_stats.get('total_frames_analyzed', 0)}\n"
            f"Frames with 2+ Animals: {occlusion_stats.get('frames_with_multiple_animals', 0)}\n"
            f"Frames with Occlusion: {occlusion_stats.get('frames_with_occlusion', 0)}\n\n"
            f"Mean Severity: {occlusion_stats.get('mean_severity', 0):.3f}\n"
            f"Max Severity: {occlusion_stats.get('max_severity', 0):.3f}"
        )
        ax_summary.text(0.05, 0.95, summary_text, va='top', fontsize=11,
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5',
                                facecolor=occlusion_grade.color, alpha=0.3))

        # Per-track exposure
        ax_exposure = fig.add_subplot(gs[0, 1])
        per_track = occlusion_stats.get('per_track_exposure', {})

        if per_track:
            track_ids = sorted(per_track.keys())
            exposures = [per_track[tid] for tid in track_ids]

            colors = ['#E74C3C' if e > 0.2 else '#F39C12' if e > 0.1 else '#2ECC71'
                     for e in exposures]
            ax_exposure.bar(range(len(track_ids)), exposures, color=colors)
            ax_exposure.set_xticks(range(len(track_ids)))
            ax_exposure.set_xticklabels([f'T{tid}' for tid in track_ids])
            ax_exposure.set_ylabel('Mean Occlusion Exposure')
            ax_exposure.set_title('Per-Track Occlusion Exposure')
        else:
            ax_exposure.text(0.5, 0.5, 'No multi-animal frames', ha='center', va='center')
            ax_exposure.set_title('Per-Track Occlusion Exposure')

        # Temporal profile
        ax_temporal = fig.add_subplot(gs[1, :])
        temporal = occlusion_stats.get('temporal_profile', [])

        if temporal:
            multi_animal_frames = [t for t in temporal if t.get('num_animals', 0) >= 2]
            if multi_animal_frames:
                max_occs = [t.get('max_occlusion', 0) for t in multi_animal_frames]
                ax_temporal.fill_between(range(len(max_occs)), max_occs, alpha=0.3, color='red')
                ax_temporal.plot(max_occs, color='red', linewidth=1)
                ax_temporal.set_xlabel('Multi-Animal Frame Index')
                ax_temporal.set_ylabel('Max Occlusion')
                ax_temporal.set_ylim(0, 1)

        ax_temporal.set_title('Occlusion Over Time')
        ax_temporal.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_recommendations_page(self, pdf: PdfPages, tracklet_profiles: Dict,
                                    all_grades: Dict, all_interpretations: Dict,
                                    all_fitness: Dict):
        """Create final recommendations page."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('RECOMMENDATIONS', fontsize=18, fontweight='bold', y=0.98)

        gs = GridSpec(3, 1, figure=fig,
                     left=0.08, right=0.92, top=0.90, bottom=0.08,
                     hspace=0.3)

        # Data Collection QC
        ax_qc = fig.add_subplot(gs[0])
        ax_qc.axis('off')

        keep_tracks = [tid for tid, f in all_fitness.items() if f['qc'].verdict == "KEEP"]
        flag_tracks = [tid for tid, f in all_fitness.items() if f['qc'].verdict == "FLAG"]
        recapture_tracks = [tid for tid, f in all_fitness.items() if f['qc'].verdict == "RECAPTURE"]

        qc_text = "DATA COLLECTION QC:\n"
        qc_text += f"  Keep: {', '.join(f'Track {t}' for t in keep_tracks) or 'None'}\n"
        qc_text += f"  Flag for review: {', '.join(f'Track {t}' for t in flag_tracks) or 'None'}\n"
        qc_text += f"  Consider recapture: {', '.join(f'Track {t}' for t in recapture_tracks) or 'None'}"

        ax_qc.text(0.0, 0.9, qc_text, va='top', fontsize=11, fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5', alpha=0.8))

        # Re-ID Training
        ax_reid = fig.add_subplot(gs[1])
        ax_reid.axis('off')

        use_tracks = [tid for tid, f in all_fitness.items() if f['reid'].verdict == "USE"]
        augment_tracks = [tid for tid, f in all_fitness.items() if f['reid'].verdict == "AUGMENT"]
        exclude_tracks = [tid for tid, f in all_fitness.items() if f['reid'].verdict == "EXCLUDE"]

        reid_text = "RE-ID TRAINING PREP:\n"
        reid_text += f"  Ready for training: {', '.join(f'Track {t}' for t in use_tracks) or 'None'}\n"
        reid_text += f"  Use with augmentation: {', '.join(f'Track {t}' for t in augment_tracks) or 'None'}\n"
        reid_text += f"  Exclude: {', '.join(f'Track {t}' for t in exclude_tracks) or 'None'}"

        ax_reid.text(0.0, 0.9, reid_text, va='top', fontsize=11, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB', alpha=0.8))

        # Publication Notes
        ax_pub = fig.add_subplot(gs[2])
        ax_pub.axis('off')

        pub_text = "PUBLICATION NOTES:\n"

        # Collect all caveats
        all_caveats = []
        for tid, f in all_fitness.items():
            for caveat in f['publication'].caveats:
                all_caveats.append(f"Track {tid}: {caveat}")

        if all_caveats:
            pub_text += "  Caveats to document:\n"
            for caveat in all_caveats[:5]:
                pub_text += f"    - {caveat}\n"
        else:
            pub_text += "  No significant caveats - data suitable for publication"

        # General recommendations
        all_recommendations = []
        for interp in all_interpretations.values():
            all_recommendations.extend(interp.recommendations)

        if all_recommendations:
            # Count most common recommendations
            rec_counts = Counter(all_recommendations)
            common_recs = rec_counts.most_common(3)

            pub_text += "\n  General recommendations:\n"
            for rec, count in common_recs:
                pub_text += f"    - {rec}\n"

        ax_pub.text(0.0, 0.9, pub_text, va='top', fontsize=11, fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def save_enhanced_json(self, tracklet_profiles: Dict,
                          all_grades: Dict, all_interpretations: Dict,
                          all_fitness: Dict, output_path: Path = None) -> Path:
        """Save enhanced JSON with grades, interpretations, and fitness."""
        if output_path is None:
            output_path = self.analyzer.annotator_output_dir / "viewpoint_analysis" / "tracklet_quality_report.json"

        output_path.parent.mkdir(exist_ok=True)

        # Build enhanced data
        enhanced_data = {}

        for track_id, profile in tracklet_profiles.items():
            grades = all_grades[track_id]
            interpretation = all_interpretations[track_id]
            fitness = all_fitness[track_id]

            enhanced_data[str(track_id)] = {
                'grades': {
                    'overall': {'letter': grades.overall.letter, 'score': grades.overall.score,
                               'label': grades.overall.label, 'description': grades.overall.description},
                    'coverage': {'letter': grades.coverage.letter, 'score': grades.coverage.score,
                                'description': grades.coverage.description},
                    'diversity': {'letter': grades.diversity.letter, 'score': grades.diversity.score,
                                 'description': grades.diversity.description},
                    'quality': {'letter': grades.quality.letter, 'score': grades.quality.score,
                               'description': grades.quality.description},
                    'completeness': {'letter': grades.completeness.letter, 'score': grades.completeness.score},
                    'reid_readiness': {'letter': grades.reid_readiness.letter, 'score': grades.reid_readiness.score,
                                      'description': grades.reid_readiness.description},
                },
                'interpretation': {
                    'summary': interpretation.summary,
                    'issues': interpretation.issues,
                    'recommendations': interpretation.recommendations,
                    'strengths': interpretation.strengths,
                },
                'use_case_fitness': {
                    'data_collection_qc': {
                        'verdict': fitness['qc'].verdict,
                        'confidence': fitness['qc'].confidence,
                        'reasons': fitness['qc'].reasons,
                        'caveats': fitness['qc'].caveats,
                    },
                    'reid_training': {
                        'verdict': fitness['reid'].verdict,
                        'confidence': fitness['reid'].confidence,
                        'reasons': fitness['reid'].reasons,
                        'caveats': fitness['reid'].caveats,
                    },
                    'publication': {
                        'verdict': fitness['publication'].verdict,
                        'confidence': fitness['publication'].confidence,
                        'reasons': fitness['publication'].reasons,
                        'caveats': fitness['publication'].caveats,
                    },
                },
                'raw_metrics': self._convert_numpy_types(profile),
            }

        with open(output_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)

        print(f"Saved enhanced JSON to: {output_path}")
        return output_path

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        else:
            return obj


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate intuitive quality reports for wildlife tracklet analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tracklet_report_generator.py --annotator_output results/zebra/scene1/corrected/

  python tracklet_report_generator.py --annotator_output results/zebra/scene1/corrected/ \\
      --images_dir data/zebra/scene1/images/ \\
      --mask_dir data/zebra/scene1/grounded-sam/
        """
    )

    parser.add_argument('--annotator_output', type=str, required=True,
                       help='Path to annotator tool output directory')
    parser.add_argument('--images_dir', type=str, default=None,
                       help='Path to original images for crop extraction')
    parser.add_argument('--mask_dir', type=str, default=None,
                       help='Path to grounded-SAM mask directory')
    parser.add_argument('--compute_occlusion', action='store_true',
                       help='Include inter-animal occlusion analysis')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PDF path (default: annotator_output/viewpoint_analysis/tracklet_quality_report.pdf)')

    args = parser.parse_args()

    # Initialize ViewpointAnalyzer
    print("Initializing ViewpointAnalyzer...")
    analyzer = ViewpointAnalyzer(
        annotator_output_dir=args.annotator_output,
        images_dir=args.images_dir
    )

    # Compute viewpoint profiles
    print("\nComputing tracklet viewpoint profiles...")
    tracklet_profiles = analyzer.compute_tracklet_viewpoint_profiles()

    # Compute occlusion if requested
    occlusion_stats = None
    if args.compute_occlusion:
        print("\nComputing occlusion statistics...")
        occlusion_stats = analyzer.compute_occlusion_statistics()

    # Initialize crop extractor if images and masks available
    crop_extractor = None
    if args.images_dir and args.mask_dir:
        crop_extractor = MaskCropExtractor(
            images_dir=Path(args.images_dir),
            mask_dir=Path(args.mask_dir),
            annotator_output_dir=Path(args.annotator_output)
        )
        print(f"\nMask crop extraction enabled")
    elif args.images_dir:
        # Try to auto-find mask directory
        mask_dir = Path(args.annotator_output).parent.parent / "grounded-sam"
        if mask_dir.exists():
            crop_extractor = MaskCropExtractor(
                images_dir=Path(args.images_dir),
                mask_dir=mask_dir,
                annotator_output_dir=Path(args.annotator_output)
            )
            print(f"\nAuto-detected mask directory: {mask_dir}")

    # Initialize report generator
    grader = QualityGrader()
    evaluator = UseCaseEvaluator(grader)
    interpreter = InterpretationGenerator()

    report_generator = PDFReportGenerator(
        analyzer=analyzer,
        grader=grader,
        evaluator=evaluator,
        interpreter=interpreter,
        crop_extractor=crop_extractor
    )

    # Grade all tracklets
    all_grades = {}
    all_interpretations = {}
    all_fitness = {}

    for track_id, profile in tracklet_profiles.items():
        grades = grader.grade_tracklet(profile)
        interpretation = interpreter.generate_interpretation(profile, grades)
        fitness = {
            'qc': evaluator.evaluate_for_data_collection_qc(profile, grades),
            'reid': evaluator.evaluate_for_reid_training(profile, grades),
            'publication': evaluator.evaluate_for_publication(profile, grades),
        }

        all_grades[track_id] = grades
        all_interpretations[track_id] = interpretation
        all_fitness[track_id] = fitness

    # Generate PDF report
    output_path = Path(args.output) if args.output else None
    pdf_path = report_generator.generate_report(
        tracklet_profiles=tracklet_profiles,
        occlusion_stats=occlusion_stats,
        output_path=output_path
    )

    # Save enhanced JSON
    report_generator.save_enhanced_json(
        tracklet_profiles=tracklet_profiles,
        all_grades=all_grades,
        all_interpretations=all_interpretations,
        all_fitness=all_fitness
    )

    # Print console summary
    print("\n" + "="*70)
    print("TRACKLET QUALITY SUMMARY")
    print("="*70)

    for track_id in sorted(tracklet_profiles.keys()):
        grades = all_grades[track_id]
        fitness = all_fitness[track_id]
        interpretation = all_interpretations[track_id]

        print(f"\nTRACK {track_id}: Grade {grades.overall.letter} ({grades.overall.label})")
        print(f"  Coverage: {grades.coverage.letter}  Diversity: {grades.diversity.letter}  "
              f"Quality: {grades.quality.letter}  Re-ID: {grades.reid_readiness.letter}")
        print(f"  {interpretation.summary}")

        if interpretation.issues:
            print(f"  Issues: {interpretation.issues[0]}")

        print(f"  QC: {fitness['qc'].verdict}  |  Re-ID: {fitness['reid'].verdict}  |  "
              f"Publication: {fitness['publication'].verdict}")

    print(f"\n{'='*70}")
    print(f"PDF Report: {pdf_path}")
    print("="*70)


if __name__ == "__main__":
    main()
