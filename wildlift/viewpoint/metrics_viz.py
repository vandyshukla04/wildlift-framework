#!/usr/bin/env python3
"""
Viewpoint Metrics Visualizer for WildLIFT Pipeline

Nature Methods publication-quality temporal metric plots: per-face quality
Q_t^(f), effective visibility E_f, coverage vector C_f, diversity index H,
dominant viewpoint ribbon, quality heatmap, and coverage dot chart.

Usage:
    # Quality-only plots (fast)
    python viewpoint_metrics_viz.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/

    # With pre-computed occlusion data
    python viewpoint_metrics_viz.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ \\
        --occlusion_json results/zebra/scene1/corrected/occlusion_analysis/occlusion_summary.json

    # With live occlusion analysis (slow)
    python viewpoint_metrics_viz.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ \\
        --with_occlusion
"""

import os
import json
import argparse
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm
from scipy.stats import entropy

from wildlift.viewpoint.analyzer import ViewpointAnalyzer, MaskCropExtractor, CONFIG

# Optional: occlusion analyzer (only needed with --with_occlusion)
try:
    from wildlift.viewpoint.occlusion import OcclusionAnalyzer
    OCCLUSION_AVAILABLE = True
except ImportError:
    OCCLUSION_AVAILABLE = False


def _apply_nature_rcparams():
    """Set matplotlib rcParams for Nature Methods publication style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 6,
        'axes.labelsize': 6,
        'axes.titlesize': 7,
        'axes.titleweight': 'bold',
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'legend.fontsize': 5,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'lines.linewidth': 0.8,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.25,
        'image.cmap': 'inferno',
    })


class MetricsVisualizer:
    """
    Nature Methods publication-quality temporal metrics visualization.

    Per-track pages:
      Page 1: Temporal quality lines + dominant viewpoint ribbon + coverage bars + summary
      Page 2: Quality heatmap (faces x frames) + stacked visibility area + coverage dots
      Page 3 (if occlusion): Effective visibility lines + occlusion impact comparison

    Standalone exports:
      Per-track PNG: vertical heatmap + coverage dot chart + summary table
      Aggregate PNG: viewpoint timeline ribbons + mean quality matrix
    """

    # Perceptually uniform colormap for quality heatmaps
    QUALITY_CMAP = 'inferno'
    QUALITY_VMAX = 0.6

    # No-data fill for ribbons
    NONE_COLOR = '#D9D9D9'

    def __init__(self, annotator_output_dir, images_dir=None,
                 with_occlusion=False, ray_samples=8,
                 nature_style=True, show_all_faces=False):
        self.vp = ViewpointAnalyzer(annotator_output_dir, images_dir)
        self.occlusion_analyzer = None
        self.occlusion_records = None
        self._effective_vis_cache = {}
        self.show_all_faces = show_all_faces

        if with_occlusion:
            if not OCCLUSION_AVAILABLE:
                print("WARNING: occlusion_analyzer not importable, skipping occlusion")
            else:
                self.occlusion_analyzer = OcclusionAnalyzer(
                    annotator_output_dir, images_dir, ray_samples=ray_samples
                )

        self._quality_cache = {}
        self._coverage_cache = {}
        self._diversity_cache = {}

        # Style selection
        if nature_style and 'NATURE_STYLE' in CONFIG:
            _apply_nature_rcparams()
            ns = CONFIG['NATURE_STYLE']
            self.typo = ns['TYPOGRAPHY']
            self.face_colors = ns['ORIENTATION_COLORS']
            self.fig_w_single = ns['FIGURE_WIDTH_SINGLE']
            self.fig_w_double = ns['FIGURE_WIDTH_DOUBLE']
            self.target_dpi = ns['DPI']
        else:
            self.typo = CONFIG['TYPOGRAPHY']
            self.face_colors = CONFIG['ORIENTATION_COLORS']
            self.fig_w_single = 3.504
            self.fig_w_double = 7.205
            self.target_dpi = 300

        self.visible_faces = CONFIG['VISIBLE_FACES']

    # ------------------------------------------------------------------
    # Data computation
    # ------------------------------------------------------------------

    def compute_track_qualities(self, track_id):
        if track_id not in self._quality_cache:
            self._quality_cache[track_id] = self.vp.compute_frame_qualities(track_id)
        return self._quality_cache[track_id]

    def compute_coverage_vector(self, track_id):
        if track_id in self._coverage_cache:
            return self._coverage_cache[track_id]
        fq = self.compute_track_qualities(track_id)
        total = len(fq)
        if total == 0:
            self._coverage_cache[track_id] = {f: 0.0 for f in self.visible_faces}
            return self._coverage_cache[track_id]
        coverage = {}
        for face in self.visible_faces:
            visible_count = sum(1 for frame_q in fq.values() if frame_q.get(face, 0.0) > 0)
            coverage[face] = visible_count / total
        self._coverage_cache[track_id] = coverage
        return coverage

    def compute_diversity_index(self, track_id):
        if track_id in self._diversity_cache:
            return self._diversity_cache[track_id]
        cv = self.compute_coverage_vector(track_id)
        values = [max(cv.get(f, 0.0), 1e-10) for f in self.visible_faces]
        total = sum(values)
        if total <= 0:
            self._diversity_cache[track_id] = 0.0
            return 0.0
        p = [v / total for v in values]
        H = entropy(p) / np.log(len(self.visible_faces))
        self._diversity_cache[track_id] = float(H)
        return float(H)

    def compute_effective_visibility(self, track_id):
        if track_id in self._effective_vis_cache:
            return self._effective_vis_cache[track_id]
        if self.occlusion_records is None:
            return None
        records = self.occlusion_records.get(track_id, [])
        if not records:
            return None
        result = {}
        for r in records:
            face_scores = {}
            for face_name, fd in r.face_details.items():
                face_scores[face_name] = fd.effective_score
            result[r.frame_name] = face_scores
        self._effective_vis_cache[track_id] = result
        return result

    def run_occlusion_analysis(self):
        if self.occlusion_analyzer is None:
            return
        print("\nRunning occlusion analysis (this may take a while)...")
        self.occlusion_records = self.occlusion_analyzer.analyze_all_frames()
        print(f"  Occlusion analysis complete for {len(self.occlusion_records)} tracks")

    def load_occlusion_from_json(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self._effective_vis_cache = {}
        for tid_str, track_data in data.get('tracks', {}).items():
            track_id = int(tid_str)
            frames = {}
            for entry in track_data.get('per_frame', []):
                face_scores = {}
                for face_name, face_data in entry.get('faces', {}).items():
                    face_scores[face_name] = face_data.get('effective_score', 0.0)
                frames[entry['frame']] = face_scores
            self._effective_vis_cache[track_id] = frames
        print(f"  Loaded occlusion data for {len(self._effective_vis_cache)} tracks from {json_path}")

    # ------------------------------------------------------------------
    # Smart face filtering
    # ------------------------------------------------------------------

    def _get_active_faces(self, track_id):
        """Return only faces with >0 coverage for this track."""
        if self.show_all_faces:
            return self.visible_faces
        cv = self.compute_coverage_vector(track_id)
        active = [f for f in self.visible_faces if cv.get(f, 0.0) > 0]
        return active if active else self.visible_faces

    def _get_active_faces_multi(self, track_ids):
        """Return faces with >0 coverage in ANY of the given tracks."""
        if self.show_all_faces:
            return self.visible_faces
        active = set()
        for tid in track_ids:
            cv = self.compute_coverage_vector(tid)
            active.update(f for f in self.visible_faces if cv.get(f, 0.0) > 0)
        result = [f for f in self.visible_faces if f in active]
        return result if result else self.visible_faces

    # ------------------------------------------------------------------
    # Variable-width face layout
    # ------------------------------------------------------------------

    def _compute_face_widths(self, active_set):
        """Compute variable column widths: active faces wide, inactive faces narrow.

        Returns (all_faces, widths, x_edges, x_centers) where:
          - all_faces: list of all visible faces in order
          - widths: per-face widths (active=4, inactive=1)
          - x_edges: cumulative edge positions for pcolormesh
          - x_centers: center positions for tick labels
        """
        all_faces = list(self.visible_faces)
        widths = [1.0 for f in all_faces]  # uniform width for all faces
        x_edges = [0.0]
        for w in widths:
            x_edges.append(x_edges[-1] + w)
        x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(len(all_faces))]
        return all_faces, widths, np.array(x_edges), x_centers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sorted_frames(self, frame_dict):
        return sorted(
            frame_dict.keys(),
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
        )

    def _frame_number(self, frame_name):
        nums = re.findall(r'\d+', str(frame_name))
        return nums[0] if nums else frame_name

    def _set_sparse_xticks(self, ax, sorted_frames, n):
        if n > 20:
            step = max(1, n // 10)
            positions = list(range(0, n, step))
            labels = [self._frame_number(sorted_frames[p]) for p in positions]
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=self.typo['micro'])

    def _clean_axes(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=self.typo['caption'])
        ax.grid(True, alpha=0.25, linewidth=0.3)

    def _panel_label(self, ax, label, x=-0.08, y=1.06):
        """Add bold panel label (a, b, c, ...) in axes coordinates."""
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=self.typo['title'], fontweight='bold',
                va='top', ha='left')

    def _get_class_name(self, track_id):
        if track_id in self.vp.all_bbox_data:
            first = next(iter(self.vp.all_bbox_data[track_id].values()))
            return first.get('class_name', 'unknown')
        return 'unknown'

    # ------------------------------------------------------------------
    # Plot methods - Page 1
    # ------------------------------------------------------------------

    def _plot_temporal_quality(self, ax, track_id, frame_qualities):
        """Temporal quality line plot: Q_t^(f) per face over frames."""
        sorted_frames = self._sorted_frames(frame_qualities)
        n = len(sorted_frames)
        indices = list(range(n))
        active = self._get_active_faces(track_id)

        for face in active:
            values = [frame_qualities[f].get(face, 0.0) for f in sorted_frames]
            ax.plot(indices, values, color=self.face_colors[face],
                    linewidth=0.8, alpha=0.85, label=face.capitalize())

        ax.set_xlabel('Frame')
        ax.set_ylabel('Quality $Q$')
        ax.set_title('Temporal quality  $Q_t^{(f)}$')
        ax.set_ylim(-0.02, 0.72)
        ax.legend(loc='upper right', ncol=len(active),
                  columnspacing=0.6, handlelength=1.0)
        self._clean_axes(ax)
        self._set_sparse_xticks(ax, sorted_frames, n)

    def _plot_dominant_viewpoint_ribbon(self, ax, track_id, frame_qualities):
        """Dominant viewpoint ribbon: colored bar per frame showing best visible face."""
        sorted_frames = self._sorted_frames(frame_qualities)
        n = len(sorted_frames)

        dominant_views = []
        dominant_colors = []
        for frame_name in sorted_frames:
            fq = frame_qualities[frame_name]
            best_face = None
            best_q = 0.0
            for face in self.visible_faces:
                q = fq.get(face, 0.0)
                if q > best_q:
                    best_q = q
                    best_face = face
            if best_face and best_q > 0:
                dominant_views.append(best_face)
                dominant_colors.append(self.face_colors[best_face])
            else:
                dominant_views.append('none')
                dominant_colors.append(self.NONE_COLOR)

        for i in range(n):
            ax.barh(0, 1, left=i, color=dominant_colors[i], edgecolor='none', linewidth=0)

        transitions = []
        for i in range(1, len(dominant_views)):
            if dominant_views[i] != dominant_views[i - 1] and dominant_views[i] != 'none':
                transitions.append(i)
                ax.axvline(x=i, color='white', linewidth=0.8, alpha=0.8)

        ax.set_xlim(0, n)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Frame')
        ax.set_title(f'Dominant viewpoint ({len(transitions)} transitions)')
        ax.spines['left'].set_visible(False)

        active = self._get_active_faces(track_id)
        patches = [Line2D([0], [0], color=self.face_colors[f], linewidth=4,
                          label=f.capitalize()) for f in active]
        patches.append(Line2D([0], [0], color=self.NONE_COLOR, linewidth=4, label='None'))
        ax.legend(handles=patches, loc='upper center',
                  bbox_to_anchor=(0.5, -0.25),
                  ncol=len(active) + 1, handlelength=1.0, columnspacing=0.5,
                  frameon=False)
        self._set_sparse_xticks(ax, sorted_frames, n)

    def _plot_coverage_bars(self, ax, track_id, coverage_vector):
        """Vertical bar chart of coverage for ALL faces (Page 1).

        Active faces: solid colored bars. Inactive: hatched gray slivers.
        """
        all_faces = list(self.visible_faces)
        cv = self.compute_coverage_vector(track_id)
        active_set = {f for f in all_faces if cv.get(f, 0.0) > 0}

        values = [coverage_vector.get(f, 0.0) for f in all_faces]
        x = np.arange(len(all_faces))

        for i, face in enumerate(all_faces):
            val = values[i]
            if face in active_set:
                ax.bar(x[i], val, color=self.face_colors[face], edgecolor='white',
                       linewidth=0.3, width=0.65, alpha=0.85)
                ax.text(x[i], val + 0.02, f'{val:.2f}', ha='center', va='bottom',
                        fontsize=self.typo['caption'], fontweight='bold')
            else:
                ax.bar(x[i], 0.02, color='#E0E0E0', edgecolor='#AAAAAA',
                       linewidth=0.4, width=0.65, hatch='///', alpha=0.5)
                ax.text(x[i], 0.04, '0%', ha='center', va='bottom',
                        fontsize=self.typo['micro'], color='#AAAAAA')

        active_values = [v for v in values if v > 0]
        if active_values:
            mean_cov = np.mean(active_values)
            ax.axhline(y=mean_cov, color='#666666', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.text(len(all_faces) - 0.5, mean_cov + 0.02, f'mean={mean_cov:.2f}',
                    fontsize=self.typo['micro'], color='#666666', ha='right')

        ax.set_xticks(x)
        ax.set_xticklabels([f.upper() for f in all_faces], fontweight='bold')
        for tick_label, face in zip(ax.get_xticklabels(), all_faces):
            if face in active_set:
                tick_label.set_color(self.face_colors[face])
            else:
                tick_label.set_color('#AAAAAA')
                tick_label.set_fontstyle('italic')
        ax.set_ylabel('Coverage $C_f$')
        ax.set_title('Coverage vector')
        ax.set_ylim(0, 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_summary_panel(self, ax, track_id, frame_qualities,
                            coverage_vector, diversity_index,
                            effective_visibility=None):
        """Summary table showing ALL faces. Inactive faces in gray."""
        ax.axis('off')
        ax.set_title('Summary')

        class_name = self._get_class_name(track_id)
        total_frames = len(frame_qualities)
        all_faces = list(self.visible_faces)
        cv = self.compute_coverage_vector(track_id)
        active_set = {f for f in all_faces if cv.get(f, 0.0) > 0}

        if diversity_index >= 0.7:
            div_label = 'Good'
        elif diversity_index >= 0.4:
            div_label = 'Moderate'
        else:
            div_label = 'Poor'

        lines = [
            f"Track {track_id} ({class_name})",
            f"Frames: {total_frames}    H = {diversity_index:.3f} ({div_label})",
            f"",
            f"{'Face':<7}{'MeanQ':>7}{'MaxQ':>7}{'Cov':>7}",
            f"{'-'*28}",
        ]
        for face in all_faces:
            qs = [frame_qualities[f].get(face, 0.0) for f in frame_qualities]
            non_zero = [q for q in qs if q > 0]
            mean_q = np.mean(non_zero) if non_zero else 0.0
            max_q = max(qs) if qs else 0.0
            cov = coverage_vector.get(face, 0.0)
            if face in active_set:
                lines.append(f"{face.upper():<7}{mean_q:>7.3f}{max_q:>7.3f}{cov:>7.2f}")
            else:
                lines.append(f"{face.upper():<7}{'---':>7}{'---':>7}{cov:>7.2f}")

        if effective_visibility is not None:
            lines.extend([f"", f"{'Face':<7}{'MeanE':>7}{'MaxE':>7}", f"{'-'*21}"])
            for face in all_faces:
                es = [effective_visibility.get(f, {}).get(face, 0.0) for f in effective_visibility]
                non_zero_e = [e for e in es if e > 0]
                mean_e = np.mean(non_zero_e) if non_zero_e else 0.0
                max_e = max(es) if es else 0.0
                if face in active_set:
                    lines.append(f"{face.upper():<7}{mean_e:>7.3f}{max_e:>7.3f}")
                else:
                    lines.append(f"{face.upper():<7}{'---':>7}{'---':>7}")

        ax.text(0.05, 0.95, '\n'.join(lines), ha='left', va='top',
                fontsize=self.typo['caption'] + 0.5, fontfamily='monospace',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA',
                          edgecolor='#E0E0E0', linewidth=0.5))

    # ------------------------------------------------------------------
    # Plot methods - Page 2
    # ------------------------------------------------------------------

    def _plot_quality_heatmap(self, ax, track_id, frame_qualities):
        """Horizontal quality heatmap with variable-height rows.

        Active faces get 4x row height; inactive faces get 1x (narrow slivers).
        Uses pcolormesh for non-uniform row heights.
        """
        sorted_frames = self._sorted_frames(frame_qualities)
        n = len(sorted_frames)

        cv = self.compute_coverage_vector(track_id)
        active_set = {f for f in self.visible_faces if cv.get(f, 0.0) > 0}
        all_faces, widths, y_edges, y_centers = self._compute_face_widths(active_set)
        n_faces = len(all_faces)

        # Build matrix: rows=faces, cols=frames
        matrix = np.zeros((n_faces, n))
        for j, frame_name in enumerate(sorted_frames):
            for i, face in enumerate(all_faces):
                matrix[i, j] = frame_qualities[frame_name].get(face, 0.0)

        # X edges (uniform), Y edges (non-uniform from face widths)
        x_edges = np.arange(n + 1)

        im = ax.pcolormesh(x_edges, y_edges, matrix,
                           cmap=self.QUALITY_CMAP, vmin=0, vmax=self.QUALITY_VMAX,
                           shading='flat')

        # Horizontal separators
        for edge in y_edges[1:-1]:
            ax.axhline(y=edge, color='white', linewidth=0.3, alpha=0.5)

        ax.set_yticks(y_centers)
        ax.set_yticklabels([f.upper() for f in all_faces], fontweight='bold')
        for tick_label, face in zip(ax.get_yticklabels(), all_faces):
            if face in active_set:
                tick_label.set_color(self.face_colors[face])
            else:
                tick_label.set_color('#AAAAAA')
                tick_label.set_fontstyle('italic')

        ax.invert_yaxis()
        ax.set_xlabel('Frame')
        ax.set_title('Quality heatmap (faces $\\times$ frames)')

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('$Q$')
        self._set_sparse_xticks(ax, sorted_frames, n)

    def _plot_quality_heatmap_vertical(self, ax, track_id, frame_qualities,
                                       faces=None):
        """Vertical quality heatmap with variable-width columns.

        Active faces get 4x column width; inactive faces get 1x (narrow slivers).
        Uses pcolormesh for non-uniform column widths.
        """
        sorted_frames = self._sorted_frames(frame_qualities)
        n = len(sorted_frames)

        # Determine active set for this track
        cv = self.compute_coverage_vector(track_id)
        active_set = {f for f in self.visible_faces if cv.get(f, 0.0) > 0}

        all_faces, widths, x_edges, x_centers = self._compute_face_widths(active_set)
        n_faces = len(all_faces)

        # Build matrix: rows=frames, cols=all faces
        matrix = np.zeros((n, n_faces))
        for i, frame_name in enumerate(sorted_frames):
            for j, face in enumerate(all_faces):
                matrix[i, j] = frame_qualities[frame_name].get(face, 0.0)

        # Y edges (uniform: one row per frame)
        y_edges = np.arange(n + 1)

        # pcolormesh with non-uniform x edges
        im = ax.pcolormesh(x_edges, y_edges, matrix,
                           cmap=self.QUALITY_CMAP, vmin=0, vmax=self.QUALITY_VMAX,
                           shading='flat')

        # Invert y so frame 0 is at top
        ax.invert_yaxis()

        # Thin vertical separators between columns
        for edge in x_edges[1:-1]:
            ax.axvline(x=edge, color='white', linewidth=0.3, alpha=0.5)

        # X-axis: face labels at column centers, rotated 45°
        ax.set_xticks(x_centers)
        face_labels = [f.upper() for f in all_faces]
        ax.set_xticklabels(face_labels, fontweight='bold',
                           rotation=45, ha='right')
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')

        for tick_label, face in zip(ax.get_xticklabels(), all_faces):
            if face in active_set:
                tick_label.set_color(self.face_colors[face])
            else:
                tick_label.set_color('#AAAAAA')
                tick_label.set_fontstyle('italic')

        # Y-axis: frame numbers
        ax.set_ylabel('Frame')
        step = max(1, n // 12)
        positions = [p + 0.5 for p in range(0, n, step)]
        labels = [self._frame_number(sorted_frames[p]) for p in range(0, n, step)]
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)

        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label('Quality $Q$')

    def _plot_stacked_visibility(self, ax, track_id, frame_qualities):
        """Stacked area chart: total visible quality per face stacked over time."""
        sorted_frames = self._sorted_frames(frame_qualities)
        n = len(sorted_frames)
        indices = list(range(n))
        active = self._get_active_faces(track_id)

        face_arrays = {}
        for face in active:
            face_arrays[face] = np.array([
                frame_qualities[f].get(face, 0.0) for f in sorted_frames
            ])

        bottom = np.zeros(n)
        for face in active:
            color = self.face_colors[face]
            values = face_arrays[face]
            ax.fill_between(indices, bottom, bottom + values,
                           color=color, alpha=0.6, linewidth=0)
            ax.plot(indices, bottom + values, color=color, linewidth=0.4, alpha=0.7)
            bottom = bottom + values

        ax.set_xlabel('Frame')
        ax.set_ylabel('Cumulative quality')
        ax.set_title('Stacked visibility')
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, None)

        patches = [plt.Rectangle((0, 0), 1, 1, fc=self.face_colors[f], alpha=0.6)
                   for f in active]
        ax.legend(patches, [f.capitalize() for f in active],
                  loc='upper right', ncol=len(active),
                  columnspacing=0.5, handlelength=1.0)
        self._clean_axes(ax)
        self._set_sparse_xticks(ax, sorted_frames, n)

    def _plot_coverage_bars_horiz(self, ax, track_id, coverage_vector, diversity_index):
        """Horizontal bar chart of coverage for ALL faces.

        Active faces (coverage > 0): solid filled bars with Okabe-Ito colors.
        Inactive faces (coverage = 0): hatched empty bars in light gray.
        """
        all_faces = list(self.visible_faces)
        cv = self.compute_coverage_vector(track_id)
        active_set = {f for f in all_faces if cv.get(f, 0.0) > 0}

        n = len(all_faces)
        values = [coverage_vector.get(f, 0.0) for f in all_faces]
        y_pos = np.arange(n)

        for i, face in enumerate(all_faces):
            val = values[i]
            if face in active_set:
                ax.barh(y_pos[i], val, height=0.6, color=self.face_colors[face],
                        edgecolor='white', linewidth=0.3, alpha=0.85)
            else:
                # Empty hatched bar for 0% coverage
                ax.barh(y_pos[i], 0.02, height=0.6, color='#E0E0E0',
                        edgecolor='#AAAAAA', linewidth=0.4, hatch='///',
                        alpha=0.5)

            # Value annotation
            text_x = max(val, 0.04) + 0.03
            color = self.face_colors[face] if face in active_set else '#AAAAAA'
            ax.text(text_x, y_pos[i], f'{val:.0%}',
                    va='center', ha='left', fontsize=self.typo['caption'],
                    fontweight='bold', color=color)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.upper() for f in all_faces], fontweight='bold')
        for tick_label, face in zip(ax.get_yticklabels(), all_faces):
            if face in active_set:
                tick_label.set_color(self.face_colors[face])
            else:
                tick_label.set_color('#AAAAAA')
                tick_label.set_fontstyle('italic')

        ax.set_xlim(-0.02, 1.18)
        ax.set_xlabel('Coverage $C_f$')
        ax.set_title(f'Coverage  ($H$={diversity_index:.2f})',
                     fontsize=self.typo['heading'])
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.25, linewidth=0.3)

    def _plot_cube_net(self, ax, face_mean_quality, active_set, title=None):
        """Draw cube net (unfolded cross) colored by mean quality per face.

        Layout (3-col x 4-row cross — all 6 cube faces):
                  [Top]
        [Left]   [Back]   [Right]
                [Bottom]
                 [Front]

        Active faces: colored by inferno colormap with quality value.
        Inactive / non-visible faces: light gray with hatching and 'N/A'.
        """
        face_positions = {
            'top':    (0, 1),
            'left':   (1, 0),
            'back':   (1, 1),
            'right':  (1, 2),
            'bottom': (2, 1),
            'front':  (3, 1),
        }

        cmap = plt.get_cmap(self.QUALITY_CMAP)
        norm = plt.Normalize(vmin=0, vmax=self.QUALITY_VMAX)

        ax.set_xlim(-0.08, 3.08)
        ax.set_ylim(-0.08, 4.08)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        for face, (row, col) in face_positions.items():
            mq = face_mean_quality.get(face, 0.0)
            is_active = face in active_set

            if is_active:
                facecolor = cmap(norm(mq))
                text_color = 'white' if mq > 0.20 else 'black'
                hatch = None
            else:
                facecolor = '#EEEEEE'
                text_color = '#999999'
                hatch = '///'

            rect = FancyBboxPatch(
                (col + 0.03, row + 0.03), 0.94, 0.94,
                boxstyle='round,pad=0.02',
                facecolor=facecolor, edgecolor='white', linewidth=1.5,
                hatch=hatch
            )
            ax.add_patch(rect)

            # Face label
            label_y = row + 0.38 if is_active else row + 0.45
            ax.text(col + 0.5, label_y, face.upper(),
                    ha='center', va='center',
                    fontsize=self.typo['heading'], fontweight='bold',
                    color=text_color)

            # Quality value
            if is_active:
                ax.text(col + 0.5, row + 0.65, f'{mq:.3f}',
                        ha='center', va='center',
                        fontsize=self.typo['caption'] + 1,
                        color=text_color, fontfamily='monospace')
            else:
                ax.text(col + 0.5, row + 0.65, 'N/A',
                        ha='center', va='center',
                        fontsize=self.typo['caption'],
                        color=text_color, fontstyle='italic')

        if title:
            ax.set_title(title, fontsize=self.typo['heading'],
                         fontweight='bold', pad=8)

    def _plot_radar_coverage(self, ax_polar, track_id, coverage_vector,
                             diversity_index):
        """Radar/polar chart of coverage vector (kept for PDF pages)."""
        faces = self._get_active_faces(track_id)
        n_faces = len(faces)
        angles = np.linspace(0, 2 * np.pi, n_faces, endpoint=False).tolist()
        angles += angles[:1]

        values = [coverage_vector.get(f, 0.0) for f in faces]
        values += values[:1]

        ax_polar.fill(angles, values, alpha=0.25, color='#0072B2')
        ax_polar.plot(angles, values, color='#0072B2', linewidth=1)

        for i, face in enumerate(faces):
            ax_polar.plot(angles[i], values[i], 'o', color=self.face_colors[face],
                         markersize=5, zorder=5)

        ax_polar.set_xticks(angles[:-1])
        ax_polar.set_xticklabels([f.upper() for f in faces], fontweight='bold')
        ax_polar.set_ylim(0, 1.1)
        ax_polar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_polar.set_yticklabels(['', '0.50', '', '1.0'],
                                 fontsize=self.typo['micro'], color='#999')
        ax_polar.set_title(f'Coverage    $H$ = {diversity_index:.2f}', pad=12)
        ax_polar.grid(True, alpha=0.3, linewidth=0.3)

    # ------------------------------------------------------------------
    # Plot methods - Page 3 (occlusion)
    # ------------------------------------------------------------------

    def _plot_temporal_effective_visibility(self, ax, track_id,
                                            frame_qualities, effective_visibility,
                                            show_quality_overlay=True):
        """Temporal effective visibility: E_f (solid) with optional Q overlay (dashed)."""
        all_frame_names = sorted(
            set(frame_qualities.keys()) | set(effective_visibility.keys()),
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
        )
        n = len(all_frame_names)
        indices = list(range(n))
        active = self._get_active_faces(track_id)

        for face in active:
            color = self.face_colors[face]
            e_values = [effective_visibility.get(f, {}).get(face, 0.0) for f in all_frame_names]
            ax.plot(indices, e_values, color=color, linewidth=0.8,
                    alpha=0.85, label=face.capitalize())
            if show_quality_overlay:
                q_values = [frame_qualities.get(f, {}).get(face, 0.0) for f in all_frame_names]
                ax.plot(indices, q_values, color=color, linewidth=0.5,
                        alpha=0.3, linestyle='--')

        ax.set_xlabel('Frame')
        ax.set_ylabel('Effective visibility $E$')
        ax.set_title('Effective visibility  $E_f = V \\cdot (1 - O/100)$')
        ax.set_ylim(-0.02, 0.72)

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=0.5, alpha=0.5))
        labels.append('$Q$ (no occ.)')
        ax.legend(handles, labels, loc='upper right',
                  ncol=3, columnspacing=0.6, handlelength=1.0)
        self._clean_axes(ax)
        self._set_sparse_xticks(ax, all_frame_names, n)

    def _plot_occlusion_impact(self, ax, track_id, frame_qualities, effective_visibility):
        """Bar chart comparing mean Q vs mean E per face (occlusion impact)."""
        active = self._get_active_faces(track_id)
        x = np.arange(len(active))
        width = 0.35

        mean_qs = []
        mean_es = []
        for face in active:
            qs = [frame_qualities.get(f, {}).get(face, 0.0) for f in frame_qualities]
            non_zero_q = [q for q in qs if q > 0]
            mean_qs.append(np.mean(non_zero_q) if non_zero_q else 0.0)

            es = [effective_visibility.get(f, {}).get(face, 0.0) for f in effective_visibility]
            non_zero_e = [e for e in es if e > 0]
            mean_es.append(np.mean(non_zero_e) if non_zero_e else 0.0)

        bars_q = ax.bar(x - width / 2, mean_qs, width, label='Quality $Q$',
                        color=[self.face_colors[f] for f in active], alpha=0.7,
                        edgecolor='white', linewidth=0.3)
        bars_e = ax.bar(x + width / 2, mean_es, width, label='Effective $E$',
                        color=[self.face_colors[f] for f in active], alpha=0.4,
                        edgecolor='white', linewidth=0.3, hatch='///')

        for bar, val in zip(bars_q, mean_qs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=self.typo['micro'], fontweight='bold')
        for bar, val in zip(bars_e, mean_es):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=self.typo['micro'])

        ax.set_xticks(x)
        ax.set_xticklabels([f.upper() for f in active], fontweight='bold')
        ax.set_ylabel('Mean score')
        ax.set_title('Occlusion impact: $Q$ vs $E$')
        ax.legend(loc='upper right')
        max_val = max(max(mean_qs) if mean_qs else 0, max(mean_es) if mean_es else 0)
        ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------------
    # Page generators
    # ------------------------------------------------------------------

    def _create_page1(self, pdf, track_id, video_name, fq, cv, di, ev):
        """Page 1: Temporal quality + dominant ribbon + coverage + summary."""
        fig = plt.figure(figsize=(self.fig_w_double, self.fig_w_double * 0.75))
        fig.suptitle(f'Track {track_id} \u2014 {video_name}',
                     fontsize=self.typo['title'], fontweight='bold', y=0.97)

        gs = GridSpec(3, 2, figure=fig,
                      left=0.08, right=0.95, top=0.91, bottom=0.06,
                      height_ratios=[1.0, 0.25, 0.8],
                      hspace=0.50, wspace=0.30)

        ax_quality = fig.add_subplot(gs[0, :])
        ax_ribbon = fig.add_subplot(gs[1, :])
        ax_coverage = fig.add_subplot(gs[2, 0])
        ax_summary = fig.add_subplot(gs[2, 1])

        self._panel_label(ax_quality, 'a')
        self._panel_label(ax_ribbon, 'b')
        self._panel_label(ax_coverage, 'c')
        self._panel_label(ax_summary, 'd')

        self._plot_temporal_quality(ax_quality, track_id, fq)
        self._plot_dominant_viewpoint_ribbon(ax_ribbon, track_id, fq)
        self._plot_coverage_bars(ax_coverage, track_id, cv)
        self._plot_summary_panel(ax_summary, track_id, fq, cv, di, ev)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_page2(self, pdf, track_id, video_name, fq, cv, di):
        """Page 2: Vertical quality heatmap + stacked area + coverage dots."""
        fig = plt.figure(figsize=(self.fig_w_double, self.fig_w_double * 0.75))
        fig.suptitle(f'Track {track_id} \u2014 Quality analysis    [{video_name}]',
                     fontsize=self.typo['title'], fontweight='bold', y=0.97)

        gs = GridSpec(1, 3, figure=fig,
                      left=0.07, right=0.96, top=0.91, bottom=0.06,
                      wspace=0.40, width_ratios=[1.0, 1.0, 0.8])

        ax_heatmap = fig.add_subplot(gs[0, 0])
        ax_stacked = fig.add_subplot(gs[0, 1])
        ax_dots = fig.add_subplot(gs[0, 2])

        self._panel_label(ax_heatmap, 'a')
        self._panel_label(ax_stacked, 'b')
        self._panel_label(ax_dots, 'c')

        self._plot_quality_heatmap_vertical(ax_heatmap, track_id, fq)
        self._plot_stacked_visibility(ax_stacked, track_id, fq)
        self._plot_coverage_bars_horiz(ax_dots, track_id, cv, di)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_page3_occlusion(self, pdf, track_id, video_name, fq, ev):
        """Page 3: Effective visibility + occlusion impact comparison."""
        fig = plt.figure(figsize=(self.fig_w_double, self.fig_w_double * 0.55))
        fig.suptitle(f'Track {track_id} \u2014 Occlusion analysis    [{video_name}]',
                     fontsize=self.typo['title'], fontweight='bold', y=0.97)

        gs = GridSpec(2, 1, figure=fig,
                      left=0.08, right=0.95, top=0.91, bottom=0.06,
                      hspace=0.42)

        ax_eff = fig.add_subplot(gs[0])
        ax_impact = fig.add_subplot(gs[1])

        self._panel_label(ax_eff, 'a')
        self._panel_label(ax_impact, 'b')

        self._plot_temporal_effective_visibility(ax_eff, track_id, fq, ev)
        self._plot_occlusion_impact(ax_impact, track_id, fq, ev)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Multi-track comparison page
    # ------------------------------------------------------------------

    def _create_comparison_page(self, pdf, track_ids, video_name):
        """Cross-track comparison: ribbons stacked + diversity + coverage matrix."""
        n_tracks = len(track_ids)
        if n_tracks < 2:
            return

        # Compute active set + variable widths for all faces
        active_set_cmp = set()
        for tid in track_ids:
            cv = self.compute_coverage_vector(tid)
            active_set_cmp.update(f for f in self.visible_faces if cv.get(f, 0.0) > 0)
        all_faces_cmp, widths_cmp, x_edges_cmp, x_centers_cmp = self._compute_face_widths(active_set_cmp)

        fig_h = self.fig_w_double * (0.3 + n_tracks * 0.08 + 0.35)
        fig = plt.figure(figsize=(self.fig_w_double, fig_h))
        fig.suptitle(f'Multi-track comparison    [{video_name}]',
                     fontsize=self.typo['title'], fontweight='bold', y=0.97)

        gs = GridSpec(n_tracks + 1, 2, figure=fig,
                      left=0.10, right=0.95, top=0.91, bottom=0.06,
                      height_ratios=[1.0] * n_tracks + [1.5],
                      hspace=0.45, wspace=0.30)

        for idx, track_id in enumerate(track_ids):
            ax_ribbon = fig.add_subplot(gs[idx, :])
            fq = self.compute_track_qualities(track_id)
            if not fq:
                ax_ribbon.text(0.5, 0.5, f'Track {track_id}: no data',
                              ha='center', va='center')
                ax_ribbon.axis('off')
                continue

            sorted_frames = self._sorted_frames(fq)
            n = len(sorted_frames)

            for i, frame_name in enumerate(sorted_frames):
                best_face = max(self.visible_faces,
                               key=lambda f: fq[frame_name].get(f, 0.0))
                best_q = fq[frame_name].get(best_face, 0.0)
                color = self.face_colors[best_face] if best_q > 0 else self.NONE_COLOR
                ax_ribbon.barh(0, 1, left=i, color=color, edgecolor='none')

            ax_ribbon.set_xlim(0, n)
            ax_ribbon.set_ylim(-0.5, 0.5)
            ax_ribbon.set_yticks([])
            ax_ribbon.spines['top'].set_visible(False)
            ax_ribbon.spines['right'].set_visible(False)
            ax_ribbon.spines['left'].set_visible(False)

            di = self.compute_diversity_index(track_id)
            cv = self.compute_coverage_vector(track_id)
            n_covered = sum(1 for v in cv.values() if v > 0)
            ax_ribbon.text(-0.02, 0.5, f'T{track_id}', ha='right', va='center',
                          fontweight='bold', transform=ax_ribbon.transAxes)
            ax_ribbon.text(1.01, 0.5, f'{n_covered}/{len(self.visible_faces)}  H={di:.2f}',
                          ha='left', va='center', fontfamily='monospace',
                          transform=ax_ribbon.transAxes)

            if idx < n_tracks - 1:
                ax_ribbon.set_xticks([])
            else:
                ax_ribbon.set_xlabel('Frame')
                self._set_sparse_xticks(ax_ribbon, sorted_frames, n)

            if idx == 0:
                self._panel_label(ax_ribbon, 'a')

        # Bottom-left: Diversity comparison bars
        ax_div = fig.add_subplot(gs[n_tracks, 0])
        self._panel_label(ax_div, 'b')
        diversities = [self.compute_diversity_index(tid) for tid in track_ids]
        colors_div = []
        for d in diversities:
            if d >= 0.7:
                colors_div.append('#009E73')
            elif d >= 0.4:
                colors_div.append('#E69F00')
            else:
                colors_div.append('#D55E00')

        x = np.arange(n_tracks)
        bars = ax_div.bar(x, diversities, color=colors_div, edgecolor='white',
                         linewidth=0.3, width=0.6, alpha=0.85)
        for bar, val in zip(bars, diversities):
            ax_div.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom',
                       fontsize=self.typo['caption'], fontweight='bold')

        ax_div.set_xticks(x)
        ax_div.set_xticklabels([f'T{tid}' for tid in track_ids])
        ax_div.set_ylabel('Diversity $H$')
        ax_div.set_title('Viewpoint diversity')
        ax_div.set_ylim(0, 1.15)
        ax_div.axhline(y=0.7, color='#009E73', linestyle='--', linewidth=0.4, alpha=0.4)
        ax_div.axhline(y=0.4, color='#E69F00', linestyle='--', linewidth=0.4, alpha=0.4)
        ax_div.spines['top'].set_visible(False)
        ax_div.spines['right'].set_visible(False)

        # Bottom-right: Coverage heatmap (tracks x all faces, variable width)
        ax_cov = fig.add_subplot(gs[n_tracks, 1])
        self._panel_label(ax_cov, 'c')
        cov_matrix = np.zeros((n_tracks, len(all_faces_cmp)))
        for i, tid in enumerate(track_ids):
            cv = self.compute_coverage_vector(tid)
            for j, face in enumerate(all_faces_cmp):
                cov_matrix[i, j] = cv.get(face, 0.0)

        cmap_cov = LinearSegmentedColormap.from_list('cov',
            ['#FFFFFF', '#E3F2FD', '#64B5F6', '#1E88E5', '#0D47A1'])
        y_edges_cov = np.arange(n_tracks + 1)
        im = ax_cov.pcolormesh(x_edges_cmp, y_edges_cov, cov_matrix,
                                cmap=cmap_cov, vmin=0, vmax=1, shading='flat')

        for edge in x_edges_cmp[1:-1]:
            ax_cov.axvline(x=edge, color='white', linewidth=0.3, alpha=0.5)

        for i in range(n_tracks):
            for j, face in enumerate(all_faces_cmp):
                val = cov_matrix[i, j]
                if face in active_set_cmp:
                    color = 'white' if val > 0.5 else 'black'
                    ax_cov.text(x_centers_cmp[j], i + 0.5, f'{val:.2f}',
                               ha='center', va='center',
                               fontsize=self.typo['caption'], fontweight='bold', color=color)

        ax_cov.set_xticks(x_centers_cmp)
        ax_cov.set_xticklabels([f.upper() for f in all_faces_cmp], fontweight='bold',
                                rotation=45, ha='right')
        for tick_label, face in zip(ax_cov.get_xticklabels(), all_faces_cmp):
            if face in active_set_cmp:
                tick_label.set_color(self.face_colors[face])
            else:
                tick_label.set_color('#AAAAAA')
                tick_label.set_fontstyle('italic')
        ax_cov.set_yticks([i + 0.5 for i in range(n_tracks)])
        ax_cov.set_yticklabels([f'T{tid}' for tid in track_ids])
        ax_cov.invert_yaxis()
        ax_cov.set_title('Coverage matrix')

        # Legend for ribbons (all faces) — placed below figure
        patches = [Line2D([0], [0], color=self.face_colors[f], linewidth=4,
                          label=f.capitalize()) for f in all_faces_cmp]
        fig.legend(handles=patches, loc='upper center', ncol=len(all_faces_cmp),
                  bbox_to_anchor=(0.5, 0.95), frameon=False)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Aggregate multi-track heatmap
    # ------------------------------------------------------------------

    def _build_aggregate_figure(self, track_ids, video_name):
        """Build aggregate figure: viewpoint timeline (top) + quality matrix (bottom).

        Returns (fig, valid_tids) or (None, []) if no data.
        """
        track_data = {}
        global_min_frame = float('inf')
        global_max_frame = 0
        for tid in track_ids:
            fq = self.compute_track_qualities(tid)
            if not fq:
                continue
            sorted_frames = self._sorted_frames(fq)
            frame_nums = [int(re.findall(r'\d+', f)[0]) for f in sorted_frames
                         if re.findall(r'\d+', f)]
            if frame_nums:
                global_min_frame = min(global_min_frame, min(frame_nums))
                global_max_frame = max(global_max_frame, max(frame_nums))
            track_data[tid] = {'fq': fq, 'sorted': sorted_frames,
                              'frame_nums': set(frame_nums)}

        valid_tids = [tid for tid in track_ids if tid in track_data]
        if not valid_tids:
            return None, []

        # Compute active set across all tracks + variable widths
        active_set_multi = set()
        for tid in valid_tids:
            cv = self.compute_coverage_vector(tid)
            active_set_multi.update(f for f in self.visible_faces if cv.get(f, 0.0) > 0)
        all_faces, face_widths, face_x_edges, face_x_centers = self._compute_face_widths(active_set_multi)

        total_frames = global_max_frame - global_min_frame + 1
        n_valid = len(valid_tids)
        n_faces = len(all_faces)

        # --- Figure layout ---
        # Each ribbon row ~0.22in, matrix row ~0.30in, plus margins
        ribbon_height = max(1.0, n_valid * 0.28 + 0.3)
        matrix_height = max(0.8, n_valid * 0.25 + 0.5)
        fig_height = ribbon_height + matrix_height + 0.8
        fig = plt.figure(figsize=(self.fig_w_double, fig_height))
        fig.suptitle(f'Aggregate viewpoint overview    [{video_name}]',
                     fontsize=self.typo['title'], fontweight='bold', y=0.99)

        gs = GridSpec(2, 1, figure=fig,
                      left=0.10, right=0.90, top=0.88, bottom=0.10,
                      height_ratios=[ribbon_height, matrix_height],
                      hspace=0.55)

        # ===== (a) Dominant-face ribbons =====
        ax_ribbons = fig.add_subplot(gs[0])
        self._panel_label(ax_ribbons, 'a', x=-0.06)

        # Build RGBA image: rows are repeated 3x for thicker ribbons
        repeat_factor = 3
        ribbon_rgba = np.ones((n_valid * repeat_factor, total_frames, 4))

        for t_idx, tid in enumerate(valid_tids):
            td = track_data[tid]
            fq = td['fq']
            for frame_name in td['sorted']:
                nums = re.findall(r'\d+', frame_name)
                if not nums:
                    continue
                col = int(nums[0]) - global_min_frame
                best_face = max(self.visible_faces,
                               key=lambda f: fq[frame_name].get(f, 0.0))
                best_q = fq[frame_name].get(best_face, 0.0)
                rgba = to_rgba(self.face_colors[best_face]) if best_q > 0 else to_rgba(self.NONE_COLOR)
                for r in range(repeat_factor):
                    ribbon_rgba[t_idx * repeat_factor + r, col, :] = rgba

        ax_ribbons.imshow(ribbon_rgba, aspect='auto', interpolation='nearest',
                          origin='upper')

        # Y-axis: track labels centered on each group of repeated rows
        ytick_pos = [t_idx * repeat_factor + repeat_factor // 2 for t_idx in range(n_valid)]
        ax_ribbons.set_yticks(ytick_pos)
        ax_ribbons.set_yticklabels([f'Track {tid}' for tid in valid_tids],
                                    fontweight='bold')

        # H annotations on right side
        for t_idx, tid in enumerate(valid_tids):
            di = self.compute_diversity_index(tid)
            ax_ribbons.text(total_frames + total_frames * 0.01,
                           t_idx * repeat_factor + repeat_factor // 2,
                           f'H={di:.2f}', va='center', ha='left',
                           fontsize=self.typo['caption'], fontfamily='monospace')

        # X-axis
        n_xticks = min(15, total_frames)
        step = max(1, total_frames // n_xticks)
        xtick_pos = list(range(0, total_frames, step))
        xtick_labels = [str(global_min_frame + p) for p in xtick_pos]
        ax_ribbons.set_xticks(xtick_pos)
        ax_ribbons.set_xticklabels(xtick_labels)
        ax_ribbons.set_xlabel('Frame number')
        ax_ribbons.set_title('Temporal presence & dominant viewpoint')
        ax_ribbons.spines['left'].set_visible(False)

        # Face color legend (all faces) — placed below the plot
        legend_handles = [Line2D([0], [0], color=self.face_colors[f], linewidth=4,
                                 label=f.capitalize()) for f in all_faces]
        legend_handles.append(Line2D([0], [0], color=self.NONE_COLOR, linewidth=4, label='None'))
        ax_ribbons.legend(handles=legend_handles,
                          loc='upper center', bbox_to_anchor=(0.5, -0.18),
                          ncol=len(all_faces) + 1, columnspacing=0.6,
                          handlelength=1.0, frameon=False)

        # ===== (b) Mean quality matrix (all faces, variable width) =====
        ax_matrix = fig.add_subplot(gs[1])
        self._panel_label(ax_matrix, 'b', x=-0.06)

        cov_matrix = np.zeros((n_valid, n_faces))
        for i, tid in enumerate(valid_tids):
            fq = track_data[tid]['fq']
            for j, face in enumerate(all_faces):
                qs = [fq[f].get(face, 0.0) for f in fq]
                non_zero = [q for q in qs if q > 0]
                cov_matrix[i, j] = np.mean(non_zero) if non_zero else 0.0

        # pcolormesh with variable-width x columns, uniform y rows
        y_edges = np.arange(n_valid + 1)
        im = ax_matrix.pcolormesh(face_x_edges, y_edges, cov_matrix,
                                   cmap=self.QUALITY_CMAP, vmin=0,
                                   vmax=self.QUALITY_VMAX, shading='flat')

        # Vertical separators
        for edge in face_x_edges[1:-1]:
            ax_matrix.axvline(x=edge, color='white', linewidth=0.3, alpha=0.5)

        # Value annotations at cell centers (only for active-width columns)
        for i in range(n_valid):
            for j, face in enumerate(all_faces):
                val = cov_matrix[i, j]
                if face in active_set_multi:
                    color = 'white' if val > 0.30 else '#CCCCCC'
                    ax_matrix.text(face_x_centers[j], i + 0.5, f'{val:.2f}',
                                  ha='center', va='center',
                                  fontsize=self.typo['caption'], fontweight='bold',
                                  color=color)

        ax_matrix.set_xticks(face_x_centers)
        ax_matrix.set_xticklabels([f.upper() for f in all_faces], fontweight='bold',
                                   rotation=45, ha='right')
        for tick_label, face in zip(ax_matrix.get_xticklabels(), all_faces):
            if face in active_set_multi:
                tick_label.set_color(self.face_colors[face])
            else:
                tick_label.set_color('#AAAAAA')
                tick_label.set_fontstyle('italic')

        ax_matrix.set_yticks([i + 0.5 for i in range(n_valid)])
        ax_matrix.set_yticklabels([f'Track {tid}' for tid in valid_tids],
                                   fontweight='bold')
        ax_matrix.invert_yaxis()
        ax_matrix.set_title('Mean quality per face')

        cbar = plt.colorbar(im, ax=ax_matrix, fraction=0.025, pad=0.03)
        cbar.set_label('Mean $Q$')

        return fig, valid_tids

    def _create_aggregate_heatmap(self, pdf, track_ids, video_name):
        """Aggregate view: ribbons + quality matrix, saved to PDF page."""
        if len(track_ids) < 2:
            return
        fig, valid_tids = self._build_aggregate_figure(track_ids, video_name)
        if fig is None:
            return
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def save_aggregate_image(self, output_dir, track_ids=None, video_name="", dpi=None):
        """Save aggregate multi-track overview as a standalone PNG."""
        if dpi is None:
            dpi = self.target_dpi
        if track_ids is None:
            track_ids = self.vp.labeled_tracks
        if len(track_ids) < 2:
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, valid_tids = self._build_aggregate_figure(track_ids, video_name)
        if fig is None:
            return None

        img_path = output_dir / "aggregate_heatmap.png"
        fig.savefig(str(img_path), dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  Saved aggregate heatmap: {img_path}")
        return img_path

    # ------------------------------------------------------------------
    # Standalone image export
    # ------------------------------------------------------------------

    def save_cube_net_image(self, output_dir, track_ids=None, video_name="", dpi=None):
        """Save cube net (unfolded cross) images: per-track + aggregate.

        Each face rectangle is colored by mean quality using the inferno
        colormap.  Inactive faces (0 % coverage) are hatched gray.
        """
        if dpi is None:
            dpi = self.target_dpi
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if track_ids is None:
            track_ids = self.vp.labeled_tracks

        print(f"\n  Generating cube net images...")
        saved = []

        # --- Per-track cube nets ---
        for track_id in track_ids:
            fq = self.compute_track_qualities(track_id)
            if not fq:
                continue
            cv = self.compute_coverage_vector(track_id)
            active_set = {f for f in self.visible_faces if cv.get(f, 0.0) > 0}

            face_mean_q = {}
            for face in self.visible_faces:
                qs = [fq[f].get(face, 0.0) for f in fq]
                non_zero = [q for q in qs if q > 0]
                face_mean_q[face] = float(np.mean(non_zero)) if non_zero else 0.0

            fig_size = self.fig_w_single
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 1.45))
            self._plot_cube_net(ax, face_mean_q, active_set)
            fig.suptitle(f'Track {track_id}    [{video_name}]',
                         fontsize=self.typo['title'], fontweight='bold',
                         y=0.98)

            # Colorbar
            sm = cm.ScalarMappable(cmap=self.QUALITY_CMAP,
                                   norm=plt.Normalize(0, self.QUALITY_VMAX))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.08,
                                shrink=0.55, location='bottom')
            cbar.set_label('Mean quality $\\bar{Q}$',
                           fontsize=self.typo['caption'] + 1)

            img_path = output_dir / f"track{track_id}_cubenet.png"
            fig.savefig(str(img_path), dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            saved.append(img_path)
            print(f"    Saved: {img_path}")

        # --- Aggregate cube net (mean across all tracks) ---
        if len(track_ids) >= 2:
            agg_face_q = {face: [] for face in self.visible_faces}
            agg_active = set()
            for tid in track_ids:
                fq = self.compute_track_qualities(tid)
                if not fq:
                    continue
                cv = self.compute_coverage_vector(tid)
                for face in self.visible_faces:
                    if cv.get(face, 0.0) > 0:
                        agg_active.add(face)
                    qs = [fq[f].get(face, 0.0) for f in fq]
                    non_zero = [q for q in qs if q > 0]
                    if non_zero:
                        agg_face_q[face].append(float(np.mean(non_zero)))

            face_mean_agg = {}
            for face in self.visible_faces:
                vals = agg_face_q[face]
                face_mean_agg[face] = float(np.mean(vals)) if vals else 0.0

            fig_size = self.fig_w_single
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 1.45))
            self._plot_cube_net(ax, face_mean_agg, agg_active)
            fig.suptitle(f'Aggregate cube net    [{video_name}]',
                         fontsize=self.typo['title'], fontweight='bold',
                         y=0.98)

            sm = cm.ScalarMappable(cmap=self.QUALITY_CMAP,
                                   norm=plt.Normalize(0, self.QUALITY_VMAX))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.08,
                                shrink=0.55, location='bottom')
            cbar.set_label('Mean quality $\\bar{Q}$',
                           fontsize=self.typo['caption'] + 1)

            img_path = output_dir / "aggregate_cubenet.png"
            fig.savefig(str(img_path), dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            saved.append(img_path)
            print(f"    Saved: {img_path}")

        print(f"  {len(saved)} cube net images saved")
        return saved

    def save_track_images(self, output_dir, track_ids=None, video_name="", dpi=None):
        """Save per-track PNG: heatmap + coverage bars + summary.

        Layout (Nature double-column, ~183mm wide):
        ┌──────────────────────────────────────┐
        │ (a) Quality Heatmap │ (b) Coverage   │
        │   frames (y) x      │  bar chart     │
        │   ALL faces (x)     │  (all faces)   │
        │                     ├────────────────┤
        │                     │ (c) Summary    │
        │                     │  table         │
        └──────────────────────────────────────┘
        """
        if dpi is None:
            dpi = self.target_dpi
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if track_ids is None:
            track_ids = self.vp.labeled_tracks

        print(f"\n{'='*60}")
        print("SAVING PER-TRACK IMAGES (Nature Methods style)")
        print(f"{'='*60}")
        print(f"  Tracks: {track_ids}")
        print(f"  Output: {output_dir}")
        print(f"  DPI: {dpi}")

        saved = []
        for track_id in track_ids:
            fq = self.compute_track_qualities(track_id)
            if not fq:
                print(f"  Track {track_id}: no quality data, skipping")
                continue
            cv = self.compute_coverage_vector(track_id)
            di = self.compute_diversity_index(track_id)
            n_frames = len(fq)
            class_name = self._get_class_name(track_id)

            # Double-column width, taller to avoid cramping
            fig_w = self.fig_w_double
            fig_h = min(fig_w * 0.8, max(3.5, n_frames * 0.04 + 1.5))
            fig = plt.figure(figsize=(fig_w, fig_h))
            fig.suptitle(f'Track {track_id}    [{video_name}]',
                         fontsize=self.typo['title'], fontweight='bold', y=0.97)

            # GridSpec: heatmap spans full left, bars top-right, summary bottom-right
            gs = GridSpec(2, 2, figure=fig,
                          left=0.08, right=0.92, top=0.90, bottom=0.10,
                          width_ratios=[0.7, 1.0],
                          height_ratios=[1.0, 0.7],
                          wspace=0.55, hspace=0.35)

            ax_heatmap = fig.add_subplot(gs[:, 0])
            ax_bars = fig.add_subplot(gs[0, 1])
            ax_summary = fig.add_subplot(gs[1, 1])

            self._panel_label(ax_heatmap, 'a')
            self._panel_label(ax_bars, 'b')
            self._panel_label(ax_summary, 'c', x=-0.05, y=1.08)

            # (a) Heatmap: all faces with variable widths
            self._plot_quality_heatmap_vertical(ax_heatmap, track_id, fq)

            # (b) Coverage bar chart (all faces)
            self._plot_coverage_bars_horiz(ax_bars, track_id, cv, di)

            # (c) Summary table (all faces, inactive in gray)
            ax_summary.axis('off')
            all_faces = list(self.visible_faces)
            active_set = {f for f in all_faces if cv.get(f, 0.0) > 0}
            n_covered = len(active_set)

            header = f"{class_name}  |  {n_frames} frames  |  {n_covered}/{len(all_faces)} views  |  H = {di:.2f}"
            table_lines = [header, ""]
            table_lines.append(f"{'Face':<6} {'Cov':>5} {'MeanQ':>6} {'MaxQ':>6}")
            table_lines.append(f"{'-'*25}")
            for face in all_faces:
                qs = [fq[f].get(face, 0.0) for f in fq]
                nz = [q for q in qs if q > 0]
                mq = np.mean(nz) if nz else 0.0
                mx = max(qs) if qs else 0.0
                c = cv.get(face, 0.0)
                if face in active_set:
                    table_lines.append(f"{face.upper():<6} {c:>5.0%} {mq:>6.3f} {mx:>6.3f}")
                else:
                    table_lines.append(f"{face.upper():<6} {c:>5.0%} {'---':>6} {'---':>6}")

            ax_summary.text(0.05, 0.95, '\n'.join(table_lines), ha='left', va='top',
                           fontsize=self.typo['caption'] + 0.5, fontfamily='monospace',
                           transform=ax_summary.transAxes,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAFAFA',
                                     edgecolor='#E0E0E0', linewidth=0.4))

            img_path = output_dir / f"track{track_id}_heatmap.png"
            fig.savefig(str(img_path), dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            saved.append(img_path)
            print(f"  Saved: {img_path}")

        print(f"\n  {len(saved)} images saved to {output_dir}")
        return saved

    # ------------------------------------------------------------------
    # Main PDF generation
    # ------------------------------------------------------------------

    def generate_pdf(self, output_path, track_ids=None, video_name=""):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if track_ids is None:
            track_ids = self.vp.labeled_tracks

        has_occlusion = (self.occlusion_records is not None or
                         len(self._effective_vis_cache) > 0)

        print(f"\n{'='*60}")
        print("GENERATING VIEWPOINT METRICS PDF (Nature Methods style)")
        print(f"{'='*60}")
        print(f"  Tracks: {track_ids}")
        print(f"  Occlusion data: {'yes' if has_occlusion else 'no'}")
        print(f"  Output: {output_path}")

        with PdfPages(str(output_path)) as pdf:
            for track_id in track_ids:
                print(f"  Processing track {track_id}...")
                fq = self.compute_track_qualities(track_id)
                if not fq:
                    print(f"    no quality data, skipping")
                    continue
                cv = self.compute_coverage_vector(track_id)
                di = self.compute_diversity_index(track_id)
                ev = self.compute_effective_visibility(track_id) if has_occlusion else None

                self._create_page1(pdf, track_id, video_name, fq, cv, di, ev)
                self._create_page2(pdf, track_id, video_name, fq, cv, di)

                if has_occlusion and ev is not None:
                    self._create_page3_occlusion(pdf, track_id, video_name, fq, ev)

            if len(track_ids) > 1:
                print(f"  Creating multi-track comparison...")
                self._create_comparison_page(pdf, track_ids, video_name)
                print(f"  Creating aggregate heatmap...")
                self._create_aggregate_heatmap(pdf, track_ids, video_name)

        print(f"\nSaved metrics PDF to: {output_path}")
        return output_path

    def export_json_summary(self, output_path, track_ids=None):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if track_ids is None:
            track_ids = list(self._quality_cache.keys())
        report = {
            'analysis_date': datetime.now().isoformat(),
            'version': 'metrics_viz_v3',
            'total_tracks': len(track_ids),
            'tracks': {},
        }
        for track_id in track_ids:
            fq = self._quality_cache.get(track_id, {})
            cv = self._coverage_cache.get(track_id, {})
            di = self._diversity_cache.get(track_id, 0.0)
            face_stats = {}
            for face in self.visible_faces:
                qs = [fq[f].get(face, 0.0) for f in fq]
                non_zero = [q for q in qs if q > 0]
                face_stats[face] = {
                    'coverage': round(cv.get(face, 0.0), 4),
                    'mean_quality': round(float(np.mean(non_zero)), 4) if non_zero else 0.0,
                    'max_quality': round(float(max(qs)), 4) if qs else 0.0,
                    'frames_visible': sum(1 for q in qs if q > 0),
                }
            report['tracks'][str(track_id)] = {
                'total_frames': len(fq),
                'diversity_index': round(di, 4),
                'coverage_vector': {k: round(v, 4) for k, v in cv.items()},
                'face_stats': face_stats,
            }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved JSON summary to: {output_path}")
        return output_path


# ======================================================================
# CLI ENTRY POINT
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Viewpoint Metrics Visualizer - Nature Methods publication-quality plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python viewpoint_metrics_viz.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/

    python viewpoint_metrics_viz.py \\
        --annotator_output results/zebra/scene1/corrected/ \\
        --images_dir data/zebra/scene1/images/ \\
        --occlusion_json results/.../occlusion_summary.json
        """
    )

    parser.add_argument("--annotator_output", required=True)
    parser.add_argument("--images_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--track_id", type=int, nargs='*', default=None)
    parser.add_argument("--with_occlusion", action="store_true")
    parser.add_argument("--occlusion_json", default=None)
    parser.add_argument("--ray_samples", type=int, default=8)
    parser.add_argument("--video_name", default=None)
    parser.add_argument("--json_summary", action="store_true")
    parser.add_argument("--save_images", action="store_true",
                        help="Save per-track heatmap + coverage PNG")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved images (default: 300)")
    parser.add_argument("--no_nature_style", action="store_true",
                        help="Disable Nature Methods styling (use legacy style)")
    parser.add_argument("--show_all_faces", action="store_true",
                        help="Show all faces including those with 0%% coverage")

    args = parser.parse_args()

    video_name = args.video_name
    if video_name is None:
        output_path = Path(args.annotator_output)
        if output_path.name in ("corrected_bboxes", "corrected"):
            video_name = output_path.parent.name
        else:
            video_name = output_path.name

    nature_style = not args.no_nature_style

    print("=" * 60)
    print("VIEWPOINT METRICS VISUALIZER v3 (Nature Methods)")
    print("=" * 60)
    print(f"  Video: {video_name}")
    print(f"  Style: {'Nature Methods' if nature_style else 'Legacy'}")
    print(f"  DPI: {args.dpi}")

    try:
        viz = MetricsVisualizer(
            annotator_output_dir=args.annotator_output,
            images_dir=args.images_dir,
            with_occlusion=args.with_occlusion,
            ray_samples=args.ray_samples,
            nature_style=nature_style,
            show_all_faces=args.show_all_faces,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    if args.occlusion_json:
        viz.load_occlusion_from_json(args.occlusion_json)
    elif args.with_occlusion:
        viz.run_occlusion_analysis()

    track_ids = args.track_id
    if track_ids is not None:
        invalid = [t for t in track_ids if t not in viz.vp.labeled_tracks]
        if invalid:
            print(f"  WARNING: Tracks {invalid} don't have semantic labels, skipping")
            track_ids = [t for t in track_ids if t in viz.vp.labeled_tracks]
    else:
        track_ids = viz.vp.labeled_tracks

    if not track_ids:
        print("\nERROR: No valid tracks to process")
        return 1

    pdf_path = args.output
    if pdf_path is None:
        pdf_path = (Path(args.annotator_output) / "viewpoint_analysis" /
                    f"metrics_viz_{video_name}.pdf")

    viz.generate_pdf(pdf_path, track_ids, video_name)

    if args.save_images:
        img_dir = Path(str(pdf_path).replace('.pdf', '_images'))
        viz.save_track_images(img_dir, track_ids, video_name, dpi=args.dpi)
        viz.save_cube_net_image(img_dir, track_ids, video_name, dpi=args.dpi)
        if len(track_ids) > 1:
            viz.save_aggregate_image(img_dir, track_ids, video_name, dpi=args.dpi)

    if args.json_summary:
        json_path = Path(str(pdf_path).replace('.pdf', '.json'))
        viz.export_json_summary(json_path, track_ids)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
