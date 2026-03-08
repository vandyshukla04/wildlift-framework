"""DJI SRT log parsing utilities for WildLift.

Supports multiple DJI SRT subtitle formats commonly found in drone footage:

1. **FrameCnt with GPS (new format)** -- gimbal_heading/gimbal_pitch/gimbal_roll
   fields alongside latitude/longitude/rel_alt, keyed by ``FrameCnt``.

2. **FrameCnt with GPS (old format)** -- gb_yaw/gb_pitch/gb_roll fields with
   focal_len, latitude, longitude, rel_alt, abs_alt, keyed by ``FrameCnt``.

3. **SrtCnt KABR format** -- ``SrtCnt : X`` (space before colon) with
   gimbal_heading/gimbal_pitch/gimbal_roll and no altitude.

4. **SrtCnt fallback with altitude** -- ``SrtCnt:`` followed by rel_alt and
   gimbal_heading/gimbal_pitch/gimbal_roll.

5. **FrameCnt gimbal-only** -- ``FrameCnt:`` with rel_alt and
   gb_yaw/gb_pitch/gb_roll (no GPS coordinates).

The main entry point, :func:`parse_dji_logs`, auto-detects the format.  Use
:func:`parse_dji_logs_with_gps` when GPS data is preferred and
:func:`refine_poses_with_gps` to apply GPS-based trajectory refinement.
"""

from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Path setup so that ``dust3r.utils.gps`` can be resolved when running from
# an installed wildlift package.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backends" / "cut3r"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_srt(log_file_path: str) -> Optional[str]:
    """Read an SRT file and return its content, or *None* if the file is missing."""
    if not os.path.exists(log_file_path):
        print(f"Warning: DJI log file not found: {log_file_path}")
        return None
    with open(log_file_path, "r") as f:
        return f.read()


def _split_srt_blocks(content: str) -> List[str]:
    """Split SRT content into per-entry blocks (separated by blank lines).

    Running regex with ``re.DOTALL`` on the full multi-MB content causes
    catastrophic backtracking.  Splitting first and matching per-block is
    instant (~0.02 s for a 2.5 MB file).
    """
    return re.split(r'\n\n+', content)


def _findall_blocks(pattern: str, blocks: List[str]) -> list:
    """Run ``re.findall`` per block and collect all matches."""
    matches: list = []
    for block in blocks:
        m = re.findall(pattern, block, re.DOTALL)
        if m:
            matches.extend(m)
    return matches


def _resolve_target_frames(
    frame_indices: Optional[List[int]],
    srt_frame_indices: List[int],
) -> List[int]:
    """Decide which frame indices to use (direct matches vs. all SRT frames)."""
    if frame_indices:
        direct_matches = set(frame_indices) & set(srt_frame_indices)
        print(f"Direct frame matches: {len(direct_matches):,} out of {len(frame_indices):,}")
        if len(direct_matches) == 0:
            print("Warning: No direct matches! Using ALL SRT frames instead")
            return srt_frame_indices
        return frame_indices
    print("No frame indices provided, using ALL SRT frames")
    return srt_frame_indices


# ---------------------------------------------------------------------------
# Format-specific parsers (return dict or None)
# ---------------------------------------------------------------------------

def _try_framecnt_gps_new(blocks: List[str], frame_indices: Optional[List[int]]) -> Optional[Dict]:
    """FrameCnt with GPS -- new format (gimbal_heading/pitch/roll)."""
    pattern = (
        r'FrameCnt: (\d+).*?'
        r'\[latitude:\s*([-\d.]+)\].*?'
        r'\[longitude:\s*([-\d.]+)\].*?'
        r'\[rel_alt:\s*([\d.]+).*?'
        r'\[gimbal_heading\(degrees\):\s*([-\d.]+)\].*?'
        r'\[gimbal_pitch\(degrees\):\s*([-\d.]+)\].*?'
        r'\[gimbal_roll\(degrees\):\s*([-\d.]+)\]'
    )
    matches = _findall_blocks(pattern, blocks)
    if not matches:
        return None

    print(f"Found {len(matches):,} entries with GPS data (new format) in SRT file")
    srt_frame_indices = [int(m[0]) for m in matches]
    target_frames = set(_resolve_target_frames(frame_indices, srt_frame_indices))

    data: Dict = {}
    for frame_cnt_str, latitude, longitude, rel_alt, heading, pitch, roll in matches:
        frame_cnt = int(frame_cnt_str)
        if frame_cnt in target_frames:
            data[frame_cnt] = {
                "yaw": float(heading),
                "pitch": float(pitch),
                "roll": float(roll),
                "altitude": float(rel_alt),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
    return data or None


def _try_framecnt_gps_old(blocks: List[str], frame_indices: Optional[List[int]]) -> Optional[Dict]:
    """FrameCnt with GPS -- old format (gb_yaw/pitch/roll, focal_len)."""
    pattern = (
        r'FrameCnt: (\d+).*?'
        r'\[focal_len: ([\d.]+)\].*?'
        r'\[latitude: ([-\d.]+)\] \[longitude: ([-\d.]+)\] '
        r'\[rel_alt: ([\d.]+) abs_alt: ([\d.]+)\] '
        r'\[gb_yaw: ([-\d.]+) gb_pitch: ([-\d.]+) gb_roll: ([-\d.]+)\]'
    )
    matches = _findall_blocks(pattern, blocks)
    if not matches:
        return None

    print(f"Found {len(matches):,} entries with GPS data (old format) in SRT file")
    srt_frame_indices = [int(m[0]) for m in matches]
    target_frames = set(_resolve_target_frames(frame_indices, srt_frame_indices))

    data: Dict = {}
    for frame_cnt_str, focal_len, latitude, longitude, rel_alt, abs_alt, yaw, pitch, roll in matches:
        frame_cnt = int(frame_cnt_str)
        if frame_cnt in target_frames:
            data[frame_cnt] = {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "altitude": float(rel_alt),
                "abs_altitude": float(abs_alt),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "focal_len": float(focal_len),
            }
    return data or None


def _try_srtcnt_kabr(blocks: List[str], frame_indices: Optional[List[int]]) -> Optional[Dict]:
    """SrtCnt KABR format (space before colon, no altitude)."""
    pattern = (
        r'SrtCnt : (\d+).*?'
        r'\[gimbal_heading\(degrees\): ([-\d.]+)\].*?'
        r'\[gimbal_pitch\(degrees\): ([-\d.]+)\].*?'
        r'\[gimbal_roll\(degrees\): ([-\d.]+)\]'
    )
    matches = _findall_blocks(pattern, blocks)
    if not matches:
        return None

    print(f"Found {len(matches):,} gimbal entries (SrtCnt KABR format) in SRT file")
    srt_frame_indices = [int(m[0]) for m in matches]
    target_frames = set(_resolve_target_frames(frame_indices, srt_frame_indices))

    data: Dict = {}
    for frame_cnt_str, heading, pitch, roll in matches:
        frame_cnt = int(frame_cnt_str)
        if frame_cnt in target_frames:
            data[frame_cnt] = {
                "yaw": float(heading),
                "pitch": float(pitch),
                "roll": float(roll),
                "altitude": 0.0,
            }
    return data or None


def _try_framecnt_gimbal_only(blocks: List[str], frame_indices: Optional[List[int]]) -> Optional[Dict]:
    """FrameCnt gimbal-only with rel_alt (original format)."""
    pattern = (
        r'FrameCnt: (\d+).*?'
        r'\[rel_alt: ([\d.]+).*?'
        r'\[gb_yaw: ([-\d.]+) gb_pitch: ([-\d.]+) gb_roll: ([-\d.]+)\]'
    )
    matches = _findall_blocks(pattern, blocks)
    if not matches:
        return None

    print(f"Found {len(matches):,} gimbal entries (FrameCnt gimbal-only) in SRT file")
    srt_frame_indices = [int(m[0]) for m in matches]
    target_frames = set(_resolve_target_frames(frame_indices, srt_frame_indices))

    data: Dict = {}
    for frame_cnt_str, altitude, yaw, pitch, roll in matches:
        frame_cnt = int(frame_cnt_str)
        if frame_cnt in target_frames:
            data[frame_cnt] = {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "altitude": float(altitude),
            }
    return data or None


def _try_srtcnt_fallback(blocks: List[str], frame_indices: Optional[List[int]]) -> Optional[Dict]:
    """SrtCnt fallback with altitude (no space before colon)."""
    pattern = (
        r'SrtCnt: (\d+).*?'
        r'\[rel_alt: ([\d.]+).*?'
        r'\[gimbal_heading\(degrees\):\s*([-\d.]+)\].*?'
        r'\[gimbal_pitch\(degrees\):\s*([-\d.]+)\].*?'
        r'\[gimbal_roll\(degrees\):\s*([-\d.]+)\]'
    )
    matches = _findall_blocks(pattern, blocks)
    if not matches:
        return None

    print(f"Found {len(matches):,} gimbal entries (SrtCnt fallback) in SRT file")
    srt_frame_indices = [int(m[0]) for m in matches]
    target_frames = set(_resolve_target_frames(frame_indices, srt_frame_indices))

    data: Dict = {}
    for frame_cnt_str, altitude, heading, pitch, roll in matches:
        frame_cnt = int(frame_cnt_str)
        if frame_cnt in target_frames:
            data[frame_cnt] = {
                "yaw": float(heading),
                "pitch": float(pitch),
                "roll": float(roll),
                "altitude": float(altitude),
            }
    return data or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_dji_logs(
    log_file_path: str,
    frame_indices: Optional[List[int]] = None,
) -> Optional[Dict]:
    """Parse a DJI SRT log file, auto-detecting the format.

    Tries patterns in the following order:
    1. FrameCnt with GPS (new format)
    2. FrameCnt with GPS (old format)
    3. SrtCnt KABR format
    4. FrameCnt gimbal-only
    5. SrtCnt fallback with altitude

    Args:
        log_file_path: Path to a ``.SRT`` file produced by a DJI drone.
        frame_indices: Optional list of frame indices to extract.  When
            *None*, data for every frame in the file is returned.

    Returns:
        A dictionary mapping ``frame_cnt`` (int) to a data dict containing at
        least ``yaw``, ``pitch``, ``roll``, and ``altitude``.  GPS-enabled
        formats additionally include ``latitude`` and ``longitude``.  Returns
        *None* if the file is missing or no data could be parsed.
    """
    content = _read_srt(log_file_path)
    if content is None:
        return None

    print(f"Parsing DJI log: {log_file_path}")
    print(f"SRT file size: {len(content):,} characters")

    # Split into per-entry blocks to avoid catastrophic regex backtracking
    blocks = _split_srt_blocks(content)

    # Try each format in priority order
    parsers = [
        ("FrameCnt GPS new", _try_framecnt_gps_new),
        ("FrameCnt GPS old", _try_framecnt_gps_old),
        ("SrtCnt KABR",      _try_srtcnt_kabr),
        ("FrameCnt gimbal",  _try_framecnt_gimbal_only),
        ("SrtCnt fallback",  _try_srtcnt_fallback),
    ]

    result = None
    for label, parser in parsers:
        result = parser(blocks, frame_indices)
        if result is not None:
            print(f"Matched format: {label}")
            break

    if result is None:
        print("No gimbal data found in SRT file with any known pattern")
        print("Sample content (first 1000 chars):")
        print(content[:1000])
        print("...")
    else:
        print(f"Parsed data for {len(result):,} frames")
        # Show sample data
        sample_frames = sorted(result.keys())[:3]
        print("Sample data:")
        for frame_id in sample_frames:
            d = result[frame_id]
            gps_str = ""
            if "latitude" in d:
                gps_str = f", lat={d['latitude']:.6f}, lon={d['longitude']:.6f}"
            alt_str = f", alt={d['altitude']:.1f}m" if d.get("altitude", 0) > 0 else ""
            print(
                f"   Frame {frame_id}: yaw={d['yaw']:+6.1f}, "
                f"pitch={d['pitch']:+6.1f}, roll={d['roll']:+6.1f}"
                f"{alt_str}{gps_str}"
            )

    # Free memory from large SRT content
    del content, blocks
    gc.collect()

    return result


def parse_dji_logs_with_gps(
    log_file_path: str,
    frame_indices: Optional[List[int]] = None,
) -> Optional[Dict]:
    """Parse a DJI SRT log file, preferring GPS-enabled formats.

    Tries GPS formats first (new then old).  If neither matches, falls back
    to :func:`parse_dji_logs` which tries all remaining patterns.

    Args:
        log_file_path: Path to a ``.SRT`` file produced by a DJI drone.
        frame_indices: Optional list of frame indices to extract.

    Returns:
        Same structure as :func:`parse_dji_logs`.  GPS-enabled entries will
        include ``latitude`` and ``longitude`` keys.  Returns *None* when no
        data can be parsed.
    """
    content = _read_srt(log_file_path)
    if content is None:
        return None

    print(f"Parsing DJI log with GPS: {log_file_path}")
    print(f"SRT file size: {len(content):,} characters")

    # Split into per-entry blocks to avoid catastrophic regex backtracking
    blocks = _split_srt_blocks(content)

    # Try GPS formats first
    result = _try_framecnt_gps_new(blocks, frame_indices)
    if result is not None:
        print(f"Parsed GPS data (new format) for {len(result):,} frames")
        _print_gps_sample(result)
        del content, blocks
        gc.collect()
        return result

    result = _try_framecnt_gps_old(blocks, frame_indices)
    if result is not None:
        print(f"Parsed GPS data (old format) for {len(result):,} frames")
        _print_gps_sample(result)
        del content, blocks
        gc.collect()
        return result

    print("GPS pattern not matched in SRT file, falling back to gimbal-only parsing")
    del content, blocks
    gc.collect()

    # Fall back to the auto-detect path (gimbal-only formats)
    return parse_dji_logs(log_file_path, frame_indices)


def _print_gps_sample(data: Dict) -> None:
    """Print a few sample entries for GPS-parsed data."""
    sample_frames = sorted(data.keys())[:3]
    print("Sample GPS data:")
    for frame_id in sample_frames:
        d = data[frame_id]
        print(
            f"   Frame {frame_id}: lat={d.get('latitude', 0):.6f}, "
            f"lon={d.get('longitude', 0):.6f}, "
            f"yaw={d['yaw']:+.1f}, pitch={d['pitch']:+.1f}, "
            f"alt={d['altitude']:.1f}m"
        )


def refine_poses_with_gps(
    pr_poses,
    gps_data: Dict,
    gps_weight: float = 0.1,
    gps_velocity_weight: float = 0.05,
    gps_heading_weight: float = 0.05,
    num_iterations: int = 50,
    lr: float = 0.01,
):
    """Refine predicted camera poses using GPS constraints.

    Applies scale-invariant GPS constraints to adjust the camera trajectory:

    - Aligns XY motion direction with GPS velocity directions.
    - Adjusts relative distances to match GPS displacement ratios.
    - Aligns heading changes with GPS-derived or gimbal yaw changes.

    Args:
        pr_poses: List of ``[1, 4, 4]`` camera-to-world pose tensors.
        gps_data: Dictionary mapping ``frame_cnt`` to GPS data (as returned
            by :func:`parse_dji_logs_with_gps`).
        gps_weight: Weight for displacement ratio loss.
        gps_velocity_weight: Weight for velocity direction loss.
        gps_heading_weight: Weight for heading change loss.
        num_iterations: Number of refinement iterations.
        lr: Learning rate for optimisation.

    Returns:
        Refined list of pose tensors (same shape as *pr_poses*).
    """
    import torch

    if len(pr_poses) < 2 or not gps_data:
        return pr_poses

    # Import GPS utilities
    try:
        from dust3r.utils.gps import compute_gps_constraints, compute_gimbal_yaw_changes
    except ImportError:
        print("Warning: Could not import GPS utilities, skipping refinement")
        return pr_poses

    # Compute GPS constraints
    constraints = compute_gps_constraints(gps_data, device="cpu", min_displacement=0.1)
    if constraints is None:
        print("Warning: Could not compute GPS constraints")
        return pr_poses

    # Check if hovering
    is_hovering = constraints.get("is_hovering", False)
    total_disp = constraints.get("total_displacement", 0)
    print(f"  GPS total displacement: {total_disp:.2f}m, hovering: {is_hovering}")

    if is_hovering:
        print("  Drone is hovering - using gimbal yaw for heading only")
        gimbal_yaw_changes = compute_gimbal_yaw_changes(gps_data, device="cpu")
    else:
        gimbal_yaw_changes = None

    # Stack poses for optimization
    poses_tensor = torch.cat(pr_poses, dim=0)  # [N, 4, 4]
    n_poses = len(poses_tensor)

    # Extract initial translations
    initial_translations = poses_tensor[:, :3, 3].clone()

    # Create learnable translation offsets (small adjustments)
    translation_offsets = torch.nn.Parameter(torch.zeros(n_poses, 3))

    # Create optimizer
    optimizer = torch.optim.Adam([translation_offsets], lr=lr)

    gps_ratios = constraints.get("displacement_ratios")
    gps_directions = constraints.get("velocity_directions")
    gps_heading_changes = constraints.get("heading_changes")

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Apply offsets to translations
        adjusted_translations = initial_translations + translation_offsets

        total_loss = torch.tensor(0.0)

        # 1. Displacement ratio loss (scale-invariant)
        if gps_weight > 0 and gps_ratios is not None and len(gps_ratios) > 0:
            pred_displacements = []
            for i in range(1, min(n_poses, len(gps_ratios) + 1)):
                d_xy = torch.norm(adjusted_translations[i, :2] - adjusted_translations[i - 1, :2])
                pred_displacements.append(d_xy)

            if pred_displacements:
                pred_displacements = torch.stack(pred_displacements)
                total_pred = pred_displacements.sum() + 1e-8
                pred_ratios = pred_displacements / total_pred

                n_compare = min(len(pred_ratios), len(gps_ratios))
                disp_loss = torch.abs(pred_ratios[:n_compare] - gps_ratios[:n_compare]).mean()
                total_loss = total_loss + gps_weight * disp_loss

        # 2. Velocity direction loss
        if (
            gps_velocity_weight > 0
            and gps_directions is not None
            and len(gps_directions) > 0
            and not is_hovering
        ):
            n_vel = min(n_poses - 1, len(gps_directions))
            if n_vel > 0:
                pred_velocities = (
                    adjusted_translations[1 : n_vel + 1, :2]
                    - adjusted_translations[:n_vel, :2]
                )
                pred_magnitudes = torch.norm(pred_velocities, dim=1, keepdim=True)
                pred_dirs = pred_velocities / (pred_magnitudes + 1e-8)

                gps_dirs = gps_directions[:n_vel]
                gps_mags = torch.norm(gps_dirs, dim=1)
                valid_mask = gps_mags > 0.1

                if valid_mask.sum() > 0:
                    cos_sim = (pred_dirs * gps_dirs).sum(dim=1)
                    vel_loss = (1 - cos_sim)[valid_mask].mean()
                    total_loss = total_loss + gps_velocity_weight * vel_loss

        # 3. Heading change loss (using gimbal yaw if hovering)
        if gps_heading_weight > 0:
            if is_hovering and gimbal_yaw_changes is not None:
                target_heading_changes = gimbal_yaw_changes
            else:
                target_heading_changes = gps_heading_changes

            if target_heading_changes is not None and len(target_heading_changes) > 0:
                # For heading, we keep the original rotations.
                # This loss is informational -- rotations are not optimised here.
                pass  # Skip heading loss for translation-only refinement

        # Add regularization to prevent large offsets
        reg_loss = 0.01 * (translation_offsets**2).mean()
        total_loss = total_loss + reg_loss

        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()

    # Apply final offsets
    final_translations = initial_translations + translation_offsets.detach()

    # Reconstruct poses with adjusted translations
    refined_poses = []
    for i in range(n_poses):
        pose = poses_tensor[i].clone()
        pose[:3, 3] = final_translations[i]
        refined_poses.append(pose.unsqueeze(0))

    # Report adjustment magnitude
    offset_magnitude = translation_offsets.detach().norm(dim=1).mean().item()
    print(f"  Average translation adjustment: {offset_magnitude:.4f} units")

    return refined_poses
