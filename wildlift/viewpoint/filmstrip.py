#!/usr/bin/env python3
"""
Combine filmstrips from multiple video segments of the same animal.

Loads approved_selections_track0.json from each segment, merges frames
per orientation, and generates a unified filmstrip PDF.

Usage:
    python combine_filmstrips.py \
        --segment results/.../rhin-32_2 examples/wd_data/.../rhin-32_2 \
        --segment results/.../rhin-35_1 examples/wd_data/.../rhin-35_1 \
        --output combined_filmstrip.pdf \
        --name "rhinos_cami"
"""

import sys
import json
import re
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from wildlift.viewpoint.analyzer import MaskCropExtractor, CoverageCalculator, CONFIG, CoverageResult


@dataclass
class SegmentInfo:
    """Holds data for one video segment."""
    name: str
    annotator_output: Path
    images_dir: Path
    selections: Dict[str, List[str]]  # {orientation: [frame_names]}
    crop_extractor: MaskCropExtractor


def load_segment(annotator_output: Path, images_dir: Path) -> Optional[Dict[str, List[str]]]:
    """Load approved_selections_track0.json from a segment directory."""
    viewpoint_dir = annotator_output / "viewpoint_analysis"
    sel_file = viewpoint_dir / "approved_selections_track0.json"

    if not sel_file.exists():
        print(f"  WARNING: No selections file found: {sel_file}")
        return None

    with open(sel_file, 'r') as f:
        data = json.load(f)

    selections = data.get('selections', {})
    total = sum(len(v) for v in selections.values())
    covered = sum(1 for v in selections.values() if len(v) > 0)
    print(f"  Loaded {total} frames ({covered}/5 orientations) from {sel_file.parent.parent.name}")
    return selections


def merge_selections(segments: List[SegmentInfo]) -> Dict[str, List[str]]:
    """Compute combined coverage across all segments (for summary only)."""
    merged = {}
    for orientation in CONFIG['VISIBLE_FACES']:
        frames = []
        for seg in segments:
            frames.extend(seg.selections.get(orientation, []))
        merged[orientation] = frames
    return merged


def create_summary_page(pdf: PdfPages, segments: List[SegmentInfo],
                        combined_coverage: CoverageResult, title: str):
    """Create a summary page showing per-segment and combined coverage."""
    typo = CONFIG['TYPOGRAPHY']
    orient_colors = CONFIG['ORIENTATION_COLORS']
    visible = CONFIG['VISIBLE_FACES']

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f'COMBINED VIEWPOINT COVERAGE\n{title}',
                 fontsize=typo['title'], fontweight='bold', y=0.96)

    # Per-segment table
    n_segs = len(segments)
    n_cols = len(visible)

    ax_table = fig.add_axes([0.1, 0.45, 0.8, 0.4])
    ax_table.axis('off')

    # Draw header row
    col_width = 0.8 / (n_cols + 1)
    row_height = min(0.12, 0.8 / (n_segs + 2))

    # Column headers
    for j, label in enumerate(visible):
        x = (j + 1) * col_width
        color = orient_colors[label]
        ax_table.text(x + col_width / 2, 1.0 - row_height / 2, label.upper(),
                      ha='center', va='center', fontsize=typo['heading'],
                      fontweight='bold', transform=ax_table.transAxes,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))

    # Segment rows
    calculator = CoverageCalculator()
    for i, seg in enumerate(segments):
        y = 1.0 - (i + 1.5) * row_height
        # Segment name
        ax_table.text(col_width / 2, y, seg.name,
                      ha='center', va='center', fontsize=typo['body'],
                      fontweight='bold', transform=ax_table.transAxes)
        # Frame counts per orientation
        for j, label in enumerate(visible):
            x = (j + 1) * col_width
            n = len(seg.selections.get(label, []))
            color = orient_colors[label] if n > 0 else CONFIG['MISSING_COLOR']
            alpha = 0.2 if n > 0 else 0.1
            ax_table.text(x + col_width / 2, y, str(n) if n > 0 else '-',
                          ha='center', va='center', fontsize=typo['body'],
                          transform=ax_table.transAxes,
                          bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=alpha))

    # Combined row
    y = 1.0 - (n_segs + 1.5) * row_height
    ax_table.text(col_width / 2, y, 'COMBINED',
                  ha='center', va='center', fontsize=typo['heading'],
                  fontweight='bold', transform=ax_table.transAxes)
    merged = merge_selections(segments)
    for j, label in enumerate(visible):
        x = (j + 1) * col_width
        n = len(merged.get(label, []))
        color = orient_colors[label] if n > 0 else CONFIG['MISSING_COLOR']
        alpha = 0.4 if n > 0 else 0.1
        ax_table.text(x + col_width / 2, y, str(n),
                      ha='center', va='center', fontsize=typo['heading'],
                      fontweight='bold', transform=ax_table.transAxes,
                      bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=alpha))

    # Combined stats
    ax_stats = fig.add_axes([0.1, 0.1, 0.8, 0.25])
    ax_stats.axis('off')

    stats_text = (
        f"Segments: {n_segs}    |    "
        f"Orientations: {combined_coverage.orientations_covered}/5    |    "
        f"Total Frames: {combined_coverage.total_frames}"
    )
    if combined_coverage.missing_orientations:
        stats_text += f"\nMissing: {', '.join(combined_coverage.missing_orientations)}"

    ax_stats.text(0.5, 0.6, stats_text,
                  ha='center', va='center', fontsize=typo['body'],
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                           edgecolor='#E0E0E0'))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_filmstrip_page(pdf: PdfPages, segments: List[SegmentInfo], title: str):
    """Create vertical filmstrip page: one column per orientation, segment-grouped rows."""
    typo = CONFIG['TYPOGRAPHY']
    orient_colors = CONFIG['ORIENTATION_COLORS']
    visible = CONFIG['VISIBLE_FACES']

    # Collect all frames grouped by orientation, tagged with segment index
    # frames_by_orient[label] = [(seg_idx, frame_name), ...]
    frames_by_orient = {}
    for label in visible:
        frames = []
        for seg_idx, seg in enumerate(segments):
            for frame_name in seg.selections.get(label, []):
                frames.append((seg_idx, frame_name))
        frames_by_orient[label] = frames

    max_frames = max(len(v) for v in frames_by_orient.values()) if frames_by_orient else 0
    if max_frames == 0:
        return
    max_frames = max(max_frames, 1)

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f'Combined Filmstrip - {title}',
                 fontsize=typo['subtitle'], fontweight='bold', y=0.97)

    n_cols = len(visible)
    n_rows = max_frames + 1  # +1 for header
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  height_ratios=[0.15] + [0.85 / max_frames] * max_frames,
                  left=0.02, right=0.98, top=0.92, bottom=0.02,
                  hspace=0.08, wspace=0.05)

    # Segment color palette for visual separation
    seg_colors = ['#E3F2FD', '#FFF3E0', '#E8F5E9', '#FCE4EC', '#F3E5F5']

    # Column headers
    for col_idx, label in enumerate(visible):
        ax_header = fig.add_subplot(gs[0, col_idx])
        ax_header.axis('off')
        n_total = len(frames_by_orient[label])
        color = orient_colors[label] if n_total > 0 else CONFIG['MISSING_COLOR']
        ax_header.text(0.5, 0.5, f'{label.upper()}\n({n_total})',
                       ha='center', va='center',
                       fontsize=typo['heading'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.4))

    # Frame cells
    for col_idx, label in enumerate(visible):
        frames = frames_by_orient[label]
        orientation_color = orient_colors[label]

        for row_idx in range(max_frames):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])

            if row_idx < len(frames):
                seg_idx, frame_name = frames[row_idx]
                seg = segments[seg_idx]

                # Load and display crop
                crop_image = seg.crop_extractor.extract_crop(frame_name, track_id=0)

                if crop_image is not None:
                    if crop_image.shape[-1] == 3:
                        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                    ax.imshow(crop_image)

                    # Frame number label
                    frame_num = re.findall(r'\d+', str(frame_name))
                    frame_str = frame_num[0] if frame_num else frame_name
                    ax.text(0.02, 0.98, f'F:{frame_str}',
                            transform=ax.transAxes, fontsize=typo['caption'],
                            verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.1',
                                     facecolor='white', alpha=0.8))

                    # Segment label
                    bg_color = seg_colors[seg_idx % len(seg_colors)]
                    ax.text(0.98, 0.02, seg.name,
                            transform=ax.transAxes, fontsize=typo['micro'],
                            ha='right', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.1',
                                     facecolor=bg_color, alpha=0.9, edgecolor='#BDBDBD'))

                    for spine in ax.spines.values():
                        spine.set_edgecolor(orientation_color)
                        spine.set_linewidth(3)
                else:
                    ax.set_facecolor('#EEEEEE')
                    ax.text(0.5, 0.5, 'No image', ha='center', va='center',
                            fontsize=typo['caption'], color='gray')
            else:
                ax.set_facecolor('#F8F8F8')

            ax.set_xticks([])
            ax.set_yticks([])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_horizontal_filmstrip_page(pdf: PdfPages, segments: List[SegmentInfo], title: str):
    """Create horizontal filmstrip: temporal order across all segments, 10 per row."""
    typo = CONFIG['TYPOGRAPHY']
    orient_colors = CONFIG['ORIENTATION_COLORS']
    visible = CONFIG['VISIBLE_FACES']

    # Collect all frames: (seg_idx, frame_name, orientation)
    all_frames = []
    for seg_idx, seg in enumerate(segments):
        for label in visible:
            for frame_name in seg.selections.get(label, []):
                all_frames.append((seg_idx, frame_name, label))

    if not all_frames:
        return

    # Sort by segment index, then by frame number within segment
    def sort_key(item):
        seg_idx, frame_name, _ = item
        nums = re.findall(r'\d+', str(frame_name))
        return (seg_idx, int(nums[0]) if nums else 0)

    all_frames.sort(key=sort_key)

    max_per_row = 10
    n_rows = (len(all_frames) + max_per_row - 1) // max_per_row

    fig = plt.figure(figsize=(11, max(8.5, 2 + n_rows * 1.5)))
    fig.suptitle(f'Combined Filmstrip (Temporal) - {title}',
                 fontsize=typo['subtitle'], fontweight='bold', y=0.98)

    # Segment color palette
    seg_colors = ['#E3F2FD', '#FFF3E0', '#E8F5E9', '#FCE4EC', '#F3E5F5']

    strip_top = 0.93
    strip_height = 0.88
    frame_height = strip_height / n_rows
    frames_in_row = min(len(all_frames), max_per_row)
    frame_width = 0.88 / frames_in_row

    for i, (seg_idx, frame_name, label) in enumerate(all_frames):
        row = i // max_per_row
        col = i % max_per_row
        seg = segments[seg_idx]

        x_pos = 0.05 + col * frame_width
        y_pos = strip_top - (row + 1) * frame_height

        ax = fig.add_axes([x_pos, y_pos, frame_width * 0.92, frame_height * 0.8])

        crop_image = seg.crop_extractor.extract_crop(frame_name, track_id=0)

        if crop_image is not None:
            if crop_image.shape[-1] == 3:
                crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            ax.imshow(crop_image)
        else:
            ax.set_facecolor('#F5F5F5')
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    fontsize=typo['caption'], color='gray')

        ax.axis('off')

        # Color-coded border for orientation
        color = orient_colors[label]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        # Frame number
        frame_num = re.findall(r'\d+', str(frame_name))
        frame_str = frame_num[0] if frame_num else frame_name
        ax.text(0.5, 1.08, frame_str,
                ha='center', va='bottom', fontsize=typo['micro'],
                transform=ax.transAxes)

        # Orientation + segment badge
        badge_text = f'{label.upper()[:3]}'
        ax.text(0.5, -0.06, badge_text,
                ha='center', va='top', fontsize=typo['micro'], fontweight='bold',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.6, edgecolor='none'))

        # Segment indicator
        bg_color = seg_colors[seg_idx % len(seg_colors)]
        ax.text(0.98, 0.02, seg.name,
                ha='right', va='bottom', fontsize=max(typo['micro'] - 1, 4),
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.05', facecolor=bg_color, alpha=0.85,
                         edgecolor='#BDBDBD', linewidth=0.5))

    # Legend
    legend_y = 0.01
    fig.text(0.05, legend_y, 'Segments: ', fontsize=typo['body'], fontweight='bold')
    for idx, seg in enumerate(segments):
        bg = seg_colors[idx % len(seg_colors)]
        fig.text(0.15 + idx * 0.15, legend_y, seg.name,
                 fontsize=typo['body'],
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=bg, alpha=0.9,
                          edgecolor='#BDBDBD'))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Combine filmstrips from multiple video segments of the same animal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python combine_filmstrips.py \\
        --segment results/.../rhin-32_2 examples/wd_data/.../rhin-32_2 \\
        --segment results/.../rhin-35_1 examples/wd_data/.../rhin-35_1 \\
        --output combined.pdf --name "Rhino Cami"
        """
    )
    parser.add_argument("--segment", nargs=2, action="append", required=True,
                        metavar=("ANNOTATOR_OUTPUT", "IMAGES_DIR"),
                        help="Segment: annotator_output and images_dir (repeatable)")
    parser.add_argument("--output", type=str, default="combined_filmstrip.pdf",
                        help="Output PDF path (default: combined_filmstrip.pdf)")
    parser.add_argument("--name", type=str, default=None,
                        help="Title for the PDF (auto-detected from segment dirs)")

    args = parser.parse_args()

    if len(args.segment) < 1:
        parser.error("At least one --segment is required")

    # Auto-detect title from segment directory names
    title = args.name
    if title is None:
        names = [Path(s[0]).name for s in args.segment]
        title = " + ".join(names)

    print("=" * 70)
    print("COMBINE FILMSTRIPS")
    print(f"  Segments: {len(args.segment)}")
    print(f"  Output:   {args.output}")
    print(f"  Title:    {title}")
    print("=" * 70)

    # Load all segments
    segments = []
    for annotator_output_str, images_dir_str in args.segment:
        annotator_output = Path(annotator_output_str)
        images_dir = Path(images_dir_str)
        seg_name = annotator_output.name

        print(f"\n  Loading segment: {seg_name}")

        if not annotator_output.exists():
            print(f"  ERROR: annotator_output not found: {annotator_output}")
            continue

        selections = load_segment(annotator_output, images_dir)
        if selections is None:
            continue

        crop_extractor = MaskCropExtractor(
            images_dir=images_dir,
            results_dir=annotator_output,
        )

        segments.append(SegmentInfo(
            name=seg_name,
            annotator_output=annotator_output,
            images_dir=images_dir,
            selections=selections,
            crop_extractor=crop_extractor,
        ))

    if not segments:
        print("\nERROR: No valid segments loaded.")
        return 1

    # Compute combined coverage
    merged = merge_selections(segments)
    calculator = CoverageCalculator()
    combined_coverage = calculator.calculate_coverage(merged)

    # Generate PDF
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating combined PDF...")
    with PdfPages(str(output_path)) as pdf:
        create_summary_page(pdf, segments, combined_coverage, title)
        create_filmstrip_page(pdf, segments, title)
        create_horizontal_filmstrip_page(pdf, segments, title)

    print(f"\n{'=' * 70}")
    print(f"COMBINED FILMSTRIP GENERATED")
    print(f"  Output: {output_path}")
    print(f"  Segments: {len(segments)}")
    print(f"  Coverage: {combined_coverage.orientations_covered}/5 orientations, "
          f"{combined_coverage.total_frames} total frames")
    if combined_coverage.missing_orientations:
        print(f"  Missing: {', '.join(combined_coverage.missing_orientations)}")
    print(f"{'=' * 70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
