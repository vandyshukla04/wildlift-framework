#!/usr/bin/env python3
"""
Batch generation of filmstrip (viewpoint) and occlusion analysis results.

For each sequence:
  1. Auto-generates viewpoint selections if none exist (picks top-K frames per orientation)
  2. Generates aggregate filmstrip PDF via viewpoint_analyzer_v7
  3. Runs occlusion analysis with PDF + JSON output

Output goes to: results/paper_final/wildlift_v/<animal>/<seq>/
"""

import os
import sys
import json
import numpy as np
import re
from pathlib import Path
from datetime import datetime

# Ensure CUT3R root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from wildlift.viewpoint.analyzer import (
    ViewpointAnalyzer, MaskCropExtractor, AggregateReportGenerator,
    PDFReportGenerator, CoverageCalculator, CONFIG
)
from wildlift.viewpoint.occlusion import OcclusionAnalyzer


# =============================================================================
# Configuration
# =============================================================================

SEQUENCES = [
    {
        'name': 'rhin-11',
        'animal': 'rhinos',
        'results_dir': 'results/paper_final/thursday/rhinos/rhin-11',
        'images_dir': 'examples/wd_data/rhinos/rhin-11',
    },
    {
        'name': 'rhin-10',
        'animal': 'rhinos',
        'results_dir': 'results/paper_final/thursday/rhinos/rhin-10',
        'images_dir': 'examples/wd_data/rhinos/rhin-10',
    },
    {
        'name': 'zebr-3',
        'animal': 'zebras',
        'results_dir': 'results/paper_final/thursday/zebras/zebr-3',
        'images_dir': 'examples/wd_data/zebras/zebr-3',
    },
    {
        'name': 'zebr-14_2',
        'animal': 'zebras',
        'results_dir': 'results/paper_final/thursday/zebras/zebr-14_2',
        'images_dir': 'examples/wd_data/zebras/zebr-14_2',
    },
    {
        'name': 'elep-6',
        'animal': 'elephants',
        'results_dir': 'results/paper_final/thursday/elephants/elep-6',
        'images_dir': 'examples/wd_data/elephants/elep-6',
    },
    {
        'name': 'elep-8_1',
        'animal': 'elephants',
        'results_dir': 'results/paper_final/thursday/elephants/elep-8_1',
        'images_dir': 'examples/wd_data/elephants/elep-8_1',
    },
]

ROOT = Path(__file__).parent
OUTPUT_BASE = ROOT / 'results' / 'paper_final' / 'wildlift_v'
TOP_K_PER_ORIENTATION = 5  # auto-select top K frames per orientation


# =============================================================================
# Auto-generate selections
# =============================================================================

def auto_generate_selections(analyzer, track_id, top_k=TOP_K_PER_ORIENTATION):
    """
    Automatically select the top-K quality frames per visible orientation.
    Returns (selections, frame_qualities).
    """
    frame_qualities = analyzer.compute_frame_qualities(track_id)
    if not frame_qualities:
        return {}, {}

    visible_labels = CONFIG['VISIBLE_FACES']
    selections = {}

    for orientation in visible_labels:
        candidates = analyzer.get_candidates_for_orientation(
            track_id, orientation, frame_qualities,
            min_quality=0.05,  # lower threshold to get candidates
            max_candidates=top_k,
        )
        selections[orientation] = [c.frame for c in candidates]

    return selections, frame_qualities


def ensure_selections_exist(annotator_output, images_dir, results_parent):
    """
    Check if approved_selections files exist; if not, auto-generate them.
    Returns (all_selections, all_rejected, frame_qualities_cache).
    """
    viewpoint_dir = annotator_output / 'viewpoint_analysis'

    # Try loading existing selections
    existing_selections = {}
    if viewpoint_dir.exists():
        for f in sorted(viewpoint_dir.glob('approved_selections_track*.json')):
            with open(f, 'r') as fp:
                data = json.load(fp)
                track_id = data.get('track_id')
                if track_id is not None:
                    if 'selections' in data:
                        existing_selections[track_id] = data['selections']
                    else:
                        existing_selections[track_id] = {
                            k: v for k, v in data.items()
                            if k in CONFIG['VISIBLE_FACES']
                        }

    # Initialize analyzer
    analyzer = ViewpointAnalyzer(str(annotator_output), str(images_dir))

    all_selections = {}
    all_rejected = {}
    frame_qualities_cache = {}

    for track_id in analyzer.labeled_tracks:
        if track_id in existing_selections:
            print(f"    Track {track_id}: using existing selections")
            all_selections[track_id] = existing_selections[track_id]
            all_rejected[track_id] = {}
            frame_qualities_cache[track_id] = analyzer.compute_frame_qualities(track_id)
        else:
            print(f"    Track {track_id}: auto-generating selections (top {TOP_K_PER_ORIENTATION} per orientation)")
            selections, fq = auto_generate_selections(analyzer, track_id)
            all_selections[track_id] = selections
            all_rejected[track_id] = {}
            frame_qualities_cache[track_id] = fq

            # Save the auto-generated selections
            viewpoint_dir.mkdir(parents=True, exist_ok=True)
            save_data = {
                'track_id': track_id,
                'timestamp': datetime.now().isoformat(),
                'auto_generated': True,
                'min_quality_threshold': 0.05,
                'selections': selections,
                'rejected': {},
            }
            sel_file = viewpoint_dir / f'approved_selections_track{track_id}.json'
            with open(sel_file, 'w') as fp:
                json.dump(save_data, fp, indent=2)
            print(f"      Saved to: {sel_file}")

            total = sum(len(v) for v in selections.values())
            covered = sum(1 for v in selections.values() if len(v) > 0)
            print(f"      {total} frames, {covered}/5 orientations covered")

    return analyzer, all_selections, all_rejected, frame_qualities_cache


# =============================================================================
# Main processing
# =============================================================================

def process_sequence(seq_config):
    """Process a single sequence: filmstrip + occlusion."""
    name = seq_config['name']
    animal = seq_config['animal']
    results_dir = ROOT / seq_config['results_dir']
    images_dir = ROOT / seq_config['images_dir']
    annotator_output = results_dir / 'corrected_bboxes'

    output_dir = OUTPUT_BASE / animal / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# PROCESSING: {animal}/{name}")
    print(f"#   Results: {results_dir}")
    print(f"#   Images:  {images_dir}")
    print(f"#   Output:  {output_dir}")
    print(f"{'#' * 70}")

    # Validate paths
    if not annotator_output.exists():
        print(f"  ERROR: annotator_output not found: {annotator_output}")
        return False

    if not images_dir.exists():
        print(f"  WARNING: images_dir not found: {images_dir}")
        print(f"  Will proceed without mask crops")

    # -------------------------------------------------------------------------
    # STEP 1: Filmstrip (viewpoint analysis)
    # -------------------------------------------------------------------------
    print(f"\n  --- FILMSTRIP GENERATION ---")

    try:
        analyzer, all_selections, all_rejected, fq_cache = ensure_selections_exist(
            annotator_output, images_dir, results_dir
        )

        # Set up mask crop extractor
        crop_extractor = None
        if images_dir.exists():
            crop_extractor = MaskCropExtractor(
                images_dir=images_dir,
                results_dir=results_dir,
            )

        # Generate aggregate PDF
        agg_gen = AggregateReportGenerator(
            crop_extractor=crop_extractor,
            video_name=name,
            frame_qualities_cache=fq_cache,
        )

        filmstrip_path = output_dir / f'filmstrip_{name}.pdf'
        agg_gen.generate_aggregate_report(
            all_selections, filmstrip_path,
            all_rejected, show_rejected=False,
        )

        # Also generate per-track filmstrips
        if crop_extractor:
            report_gen = PDFReportGenerator(analyzer, crop_extractor)
            for track_id in sorted(all_selections.keys()):
                per_track_path = output_dir / f'filmstrip_{name}_track{track_id}.pdf'
                report_gen.generate_report(
                    track_id, all_selections[track_id],
                    fq_cache.get(track_id, {}),
                    all_rejected.get(track_id, {}),
                    output_path=per_track_path,
                )

        # Save selections JSON summary
        sel_summary = {
            'sequence': name,
            'animal': animal,
            'timestamp': datetime.now().isoformat(),
            'tracks': {},
        }
        calculator = CoverageCalculator()
        for track_id, selections in all_selections.items():
            fq = fq_cache.get(track_id, {})
            coverage = calculator.calculate_coverage(selections, fq)
            sel_summary['tracks'][str(track_id)] = {
                'selections': selections,
                'orientations_covered': coverage.orientations_covered,
                'total_frames': coverage.total_frames,
                'mean_quality': float(coverage.mean_quality),
                'coverage_percent': float(coverage.coverage_percent),
                'missing_orientations': coverage.missing_orientations,
            }

        with open(output_dir / f'viewpoint_summary_{name}.json', 'w') as f:
            json.dump(sel_summary, f, indent=2)

        print(f"  Filmstrip complete: {filmstrip_path}")

    except Exception as e:
        print(f"  ERROR in filmstrip generation: {e}")
        import traceback
        traceback.print_exc()

    # -------------------------------------------------------------------------
    # STEP 2: Occlusion analysis
    # -------------------------------------------------------------------------
    print(f"\n  --- OCCLUSION ANALYSIS ---")

    try:
        occ_analyzer = OcclusionAnalyzer(
            annotator_output_dir=str(annotator_output),
            images_dir=str(images_dir) if images_dir.exists() else None,
            ray_samples=8,
        )

        all_records = occ_analyzer.analyze_all_frames()

        if all_records:
            # JSON summary
            occ_analyzer.generate_json_summary(
                all_records,
                output_dir / f'occlusion_summary_{name}.json',
            )

            # PDF report
            occ_analyzer.generate_pdf_report(
                all_records,
                output_dir / f'occlusion_report_{name}.pdf',
            )

            # Annotated frames (if images available)
            if images_dir.exists():
                occ_analyzer.generate_annotated_frames(
                    all_records,
                    output_dir / 'occlusion_overlays',
                    occlusion_threshold=20.0,
                )

            # Print summary
            print(f"\n  Occlusion Summary for {name}:")
            for tid, records in sorted(all_records.items()):
                summary = occ_analyzer.summarize_track(tid, records)
                if summary:
                    print(f"    Track {tid} ({summary.class_name}): "
                          f"mean vis={summary.mean_visibility_pct:.1f}%, "
                          f"best={summary.best_frame} ({summary.best_visibility:.1f}%), "
                          f"never seen: {summary.never_seen_faces or 'none'}")
        else:
            print(f"  No occlusion records generated for {name}")

    except Exception as e:
        print(f"  ERROR in occlusion analysis: {e}")
        import traceback
        traceback.print_exc()

    return True


def main():
    print("=" * 70)
    print("BATCH FILMSTRIP + OCCLUSION GENERATOR")
    print(f"Output: {OUTPUT_BASE}")
    print("=" * 70)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = {}
    for seq in SEQUENCES:
        success = process_sequence(seq)
        results[seq['name']] = success

    # Final summary
    print(f"\n\n{'=' * 70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
    print(f"\nAll outputs in: {OUTPUT_BASE}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
