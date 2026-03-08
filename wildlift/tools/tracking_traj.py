#!/usr/bin/env python3
"""
Tracking Trajectory Visualization

Creates three images per specified frame:
  1. masked_image.png          – Source image with colored instance masks
  2. pointcloud_topdown.png    – Top-down point cloud + trajectory trails
  3. pointcloud_camview.png    – Camera-aligned point cloud + projected trajectories

Usage:
    python tools/viz_tracking_trajectory.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from PIL import Image
from pycocotools import mask as mask_utils
import cv2

# ========================= Configuration =========================
RECON_DIR = '/home/shuklva/vggt/output/wd_data/zebras/zebr-14_2'
IMAGE_DIR = '/home/shuklva/CUT3R/examples/wd_data/zebras/zebr-14_2'
SAM_DIR   = os.path.join(IMAGE_DIR, 'grounded-sam')
SAVE_DIR  = '/home/shuklva/CUT3R/results/paper_final/wildift_rt'

TARGET_FRAMES = [5400, 5440, 5480, 5520, 5560, 5600, 5640, 5680, 5720]

TRACK_COLORS = [
    np.array([0.90, 0.30, 0.25]),   # Red
    np.array([0.25, 0.70, 0.40]),   # Green
    np.array([0.30, 0.50, 0.85]),   # Blue
    np.array([0.95, 0.70, 0.15]),   # Gold
    np.array([0.65, 0.35, 0.75]),   # Purple
    np.array([0.20, 0.70, 0.70]),   # Cyan
]

BG_BRIGHT    = 0.65    # How bright background points are (0=black, 1=full)
MASK_ALPHA   = 0.45    # 2D mask overlay alpha
RASTER_W     = 1500    # Top-down raster width (px)
CONF_THRESH  = 1.5     # Depth confidence threshold
ANIMAL_DILATE = 2      # Dilation for animal points in top-down raster
CAMVIEW_SCALE = 2      # Camera-view render scale vs depth resolution

# Track merges: map child track → parent track (same animal, broken tracking)
# Track 2 ends at fn 5480, Track 5 starts at fn 5476 — same zebra
TRACK_MERGES = {5: 2}

# ========================= Load Shared Data =========================
print("Loading reconstruction data...")

with open(os.path.join(RECON_DIR, 'tracking_summary.json')) as f:
    tracking = json.load(f)

with open(os.path.join(RECON_DIR, 'cameras.json')) as f:
    cameras_data = json.load(f)

with open(os.path.join(RECON_DIR, 'metadata.json')) as f:
    metadata = json.load(f)

depth_npz  = np.load(os.path.join(RECON_DIR, 'depth_maps.npz'))
all_depths = depth_npz['depth']        # (N, 294, 518)
all_confs  = depth_npz['depth_conf']   # (N, 294, 518)

frame_numbers = metadata['frame_numbers']
fn_to_idx     = {fn: i for i, fn in enumerate(frame_numbers)}

DEP_H, DEP_W = all_depths.shape[1], all_depths.shape[2]   # 294, 518
IMG_H, IMG_W = 432, 768                                     # SAM / source
SCALE_X = IMG_W / DEP_W
SCALE_Y = IMG_H / DEP_H

# Camera-view render resolution
CV_H = DEP_H * CAMVIEW_SCALE
CV_W = DEP_W * CAMVIEW_SCALE

# Per-track frame lookup
track_lookup = {}
for tid_str, track in tracking['tracks'].items():
    tid = int(tid_str)
    track_lookup[tid] = {fi: k for k, fi in enumerate(track['frames'])}

# Pixel grids (constant for all frames)
_u_flat = np.tile(np.arange(DEP_W, dtype=np.float64), DEP_H)          # (HW,)
_v_flat = np.repeat(np.arange(DEP_H, dtype=np.float64), DEP_W)        # (HW,)


# ========================= Unprojection =========================
def unproject_frame(fi):
    """
    Unproject depth → world coords.

    Convention: extrinsic [R|t] is world-to-camera.
        p_cam   = R @ p_world + t
        p_world = R^T @ (p_cam - t)  =  (p_cam - t) @ R
    """
    cam   = cameras_data['cameras'][fi]
    depth = all_depths[fi]
    conf  = all_confs[fi]

    K = np.array(cam['intrinsic'])        # 3×3
    E = np.array(cam['extrinsic'])        # 3×4

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = depth.ravel()
    x = (_u_flat - cx) * z / fx
    y = (_v_flat - cy) * z / fy

    pts_cam = np.stack([x, y, z], axis=-1)       # (HW, 3)

    R = E[:, :3]                                  # 3×3
    t = E[:, 3]                                   # 3
    pts_world = (pts_cam - t) @ R                 # (HW, 3)

    # RGB from source image resized to depth resolution
    fn  = frame_numbers[fi]
    img = np.array(Image.open(os.path.join(IMAGE_DIR, f'{fn}.jpg')))
    img_small = cv2.resize(img, (DEP_W, DEP_H), interpolation=cv2.INTER_AREA)
    colors = img_small.reshape(-1, 3).astype(np.float32) / 255.0

    conf_ok = conf.ravel() > CONF_THRESH
    return pts_world.astype(np.float32), colors, conf_ok


# ========================= Pre-compute all frames =========================
print("Pre-computing point clouds for all frames...")

frame_pts  = []   # list of (M, 3) float32
frame_cols = []   # list of (M, 3) float32

xmin_g, xmax_g = np.inf, -np.inf
zmin_g, zmax_g = np.inf, -np.inf

for fi in range(len(frame_numbers)):
    pts, cols, conf_ok = unproject_frame(fi)
    p = pts[conf_ok]
    c = cols[conf_ok]
    frame_pts.append(p)
    frame_cols.append(c)

    xmin_g = min(xmin_g, p[:, 0].min())
    xmax_g = max(xmax_g, p[:, 0].max())
    zmin_g = min(zmin_g, p[:, 2].min())
    zmax_g = max(zmax_g, p[:, 2].max())

    if (fi + 1) % 20 == 0:
        print(f"  {fi + 1}/{len(frame_numbers)}")

total_pts = sum(len(p) for p in frame_pts)
print(f"  Total confident points: {total_pts:,}")

# Scene bounds with padding
PAD = 0.03
XMIN, XMAX = xmin_g - PAD, xmax_g + PAD
ZMIN, ZMAX = zmin_g - PAD, zmax_g + PAD

aspect = (XMAX - XMIN) / (ZMAX - ZMIN)
RASTER_H = max(1, int(RASTER_W / aspect))
print(f"  Scene X=[{XMIN:.3f}, {XMAX:.3f}]  Z=[{ZMIN:.3f}, {ZMAX:.3f}]")
print(f"  Top-down raster: {RASTER_W} x {RASTER_H}")


# ========================= Top-down background raster =========================
print("Building top-down background raster...")

def pts_to_rc(pts, W=RASTER_W, H=RASTER_H):
    """3D world → raster (row, col) for top-down XZ view."""
    col = np.clip(((pts[:, 0] - XMIN) / (XMAX - XMIN) * (W - 1)).astype(np.int32), 0, W - 1)
    row = np.clip(((pts[:, 2] - ZMIN) / (ZMAX - ZMIN) * (H - 1)).astype(np.int32), 0, H - 1)
    row = H - 1 - row
    return row, col

bg_rgb = np.zeros((RASTER_H, RASTER_W, 3), dtype=np.float64)
bg_cnt = np.zeros((RASTER_H, RASTER_W),    dtype=np.float64)

for fi in range(len(frame_numbers)):
    p, c = frame_pts[fi], frame_cols[fi]
    row, col = pts_to_rc(p)
    np.add.at(bg_rgb, (row, col), c.astype(np.float64) * BG_BRIGHT)
    np.add.at(bg_cnt, (row, col), 1)

filled = bg_cnt > 0
for ch in range(3):
    bg_rgb[:, :, ch][filled] /= bg_cnt[filled]

# Light blur ONLY within the filled region (avoid bleeding into white)
bg_u8 = np.clip(bg_rgb * 255, 0, 255).astype(np.uint8)
bg_blurred = cv2.GaussianBlur(bg_u8, (3, 3), 0)
# Keep blurred values only where there were points; unfilled stays zero
filled_3 = np.stack([filled] * 3, axis=-1)
bg_u8 = np.where(filled_3, bg_blurred, bg_u8)
bg_rgb = bg_u8.astype(np.float64) / 255.0

# Unfilled pixels → pure white
bg_rgb[~filled] = 1.0

print(f"  Done ({filled.sum()}/{RASTER_H * RASTER_W} filled)")


# ========================= Helpers =========================
def resolve_tid(tid):
    """Resolve merged track IDs so child tracks use the parent's color."""
    return TRACK_MERGES.get(tid, tid)


def bbox_iou(b1, b2):
    xi = max(b1[0], b2[0]); yi = max(b1[1], b2[1])
    xa = min(b1[2], b2[2]); ya = min(b1[3], b2[3])
    inter = max(0, xa - xi) * max(0, ya - yi)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def match_sam_to_tracks(sam_anns, frame_idx):
    pairs = []
    for si, ann in enumerate(sam_anns):
        sb = ann['bbox']
        for tid, fi_to_k in track_lookup.items():
            if frame_idx not in fi_to_k:
                continue
            k = fi_to_k[frame_idx]
            tb = tracking['tracks'][str(tid)]['bbox_2d'][k]
            scaled = [tb[0] * SCALE_X, tb[1] * SCALE_Y,
                      tb[2] * SCALE_X, tb[3] * SCALE_Y]
            iou = bbox_iou(sb, scaled)
            if iou > 0.1:
                pairs.append((iou, si, tid))
    pairs.sort(reverse=True)
    matches, used_s, used_t = {}, set(), set()
    for iou, si, tid in pairs:
        resolved = resolve_tid(tid)
        if si not in used_s and resolved not in used_t:
            matches[si] = resolved     # always store the merged/parent ID
            used_s.add(si)
            used_t.add(resolved)
    return matches


def get_track_centers_up_to(frame_num):
    """Return dict: resolved_track_id → np.array of centers up to frame_num.

    Merged tracks are concatenated (sorted by frame index) under the parent ID.
    """
    raw = {}   # resolved_tid → list of (frame_idx, center)
    for tid_str, track in tracking['tracks'].items():
        tid = int(tid_str)
        resolved = resolve_tid(tid)
        if resolved not in raw:
            raw[resolved] = []
        for k, fi in enumerate(track['frames']):
            if frame_numbers[fi] <= frame_num:
                raw[resolved].append((fi, track['centers'][k]))

    out = {}
    for resolved, entries in raw.items():
        if not entries:
            continue
        # Sort by frame index, deduplicate (keep first occurrence per frame)
        entries.sort(key=lambda x: x[0])
        seen = set()
        unique = []
        for fi, c in entries:
            if fi not in seen:
                seen.add(fi)
                unique.append(c)
        out[resolved] = np.array(unique)
    return out


def build_camera_view(frame_idx, matches, sam_results):
    """
    Render the full point cloud from the current frame's camera viewpoint
    using a z-buffer, then overlay track-colored animals and projected trajectories.
    """
    cam = cameras_data['cameras'][frame_idx]
    K = np.array(cam['intrinsic'])
    E = np.array(cam['extrinsic'])
    R, t = E[:, :3], E[:, 3]

    # Intrinsics scaled to render resolution
    sx = CV_W / DEP_W
    sy = CV_H / DEP_H
    fx = K[0, 0] * sx
    fy = K[1, 1] * sy
    cx = K[0, 2] * sx
    cy = K[1, 2] * sy

    # Z-buffer + color buffer
    zbuf = np.full((CV_H, CV_W), np.inf, dtype=np.float32)
    cbuf = np.full((CV_H, CV_W, 3), 1.0, dtype=np.float32)           # pure white bg

    for fi in range(len(frame_numbers)):
        pw = frame_pts[fi]     # (M, 3) world coords
        pc = frame_cols[fi]    # (M, 3) colors

        # World → camera:  p_cam = R @ p_world + t  =  p_world @ R^T + t
        p_cam = pw @ R.T + t.astype(np.float32)    # (M, 3)

        # Keep only in front of camera
        valid = p_cam[:, 2] > 0.01
        p_cam = p_cam[valid]
        pc_v  = pc[valid]

        z  = p_cam[:, 2]
        iu = np.round(fx * p_cam[:, 0] / z + cx).astype(np.int32)
        iv = np.round(fy * p_cam[:, 1] / z + cy).astype(np.int32)

        inb = (iu >= 0) & (iu < CV_W) & (iv >= 0) & (iv < CV_H)
        iu, iv, z, pc_v = iu[inb], iv[inb], z[inb], pc_v[inb]

        # Sort far-to-near so nearest overwrites (numpy last-write-wins)
        order = np.argsort(-z)
        iu, iv, z, pc_v = iu[order], iv[order], z[order], pc_v[order]

        # Only update if closer
        closer = z < zbuf[iv, iu]
        iu_c, iv_c = iu[closer], iv[closer]

        # Among those, re-sort far-to-near for correct last-write-wins
        z_c  = z[closer]
        pc_c = pc_v[closer]
        order2 = np.argsort(-z_c)
        iu_c, iv_c, z_c, pc_c = iu_c[order2], iv_c[order2], z_c[order2], pc_c[order2]

        zbuf[iv_c, iu_c] = z_c
        cbuf[iv_c, iu_c] = pc_c * BG_BRIGHT

    # Overlay animal points for this frame with track colors
    if sam_results:
        # Re-unproject full frame so SAM masks index correctly
        pw_full, _, conf_full = unproject_frame(frame_idx)
        p_cam_full = pw_full @ R.T + t.astype(np.float32)

        for si, ann in enumerate(sam_results['annotations']):
            if si not in matches:
                continue
            tid   = matches[si]
            color = TRACK_COLORS[tid % len(TRACK_COLORS)].astype(np.float32)

            sam_mask   = mask_utils.decode(ann['segmentation'])
            mask_small = cv2.resize(sam_mask, (DEP_W, DEP_H),
                                    interpolation=cv2.INTER_NEAREST)

            sel = (mask_small.ravel() > 0) & conf_full

            pc_a = p_cam_full[sel]
            if len(pc_a) == 0:
                continue

            z_a  = pc_a[:, 2]
            iu_a = np.round(fx * pc_a[:, 0] / z_a + cx).astype(np.int32)
            iv_a = np.round(fy * pc_a[:, 1] / z_a + cy).astype(np.int32)

            inb = (iu_a >= 0) & (iu_a < CV_W) & (iv_a >= 0) & (iv_a < CV_H)
            iu_a, iv_a = iu_a[inb], iv_a[inb]

            # Dilate for visibility
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r2 = np.clip(iv_a + dr, 0, CV_H - 1)
                    c2 = np.clip(iu_a + dc, 0, CV_W - 1)
                    cbuf[r2, c2] = color

    return cbuf


# ========================= Process Each Target Frame =========================
os.makedirs(SAVE_DIR, exist_ok=True)

for frame_num in TARGET_FRAMES:
    print(f"\n{'=' * 50}")
    print(f"  Frame {frame_num}")
    frame_idx = fn_to_idx[frame_num]

    # ---- Load SAM & match ----
    sam_path = os.path.join(SAM_DIR, f'{frame_num}_results.json')
    sam_results = None
    matches = {}
    if os.path.exists(sam_path):
        with open(sam_path) as f:
            sam_results = json.load(f)
        matches = match_sam_to_tracks(sam_results['annotations'], frame_idx)
        print(f"  Matches: {matches}")

    track_centers = get_track_centers_up_to(frame_num)

    # ======== 1. MASKED IMAGE ========
    img = np.array(Image.open(os.path.join(IMAGE_DIR, f'{frame_num}.jpg')))
    masked = img.astype(np.float64)

    for si, ann in enumerate(sam_results['annotations'] if sam_results else []):
        if si not in matches:
            continue
        color = TRACK_COLORS[matches[si] % len(TRACK_COLORS)]
        m = mask_utils.decode(ann['segmentation']).astype(bool)
        for ch in range(3):
            masked[:, :, ch][m] = masked[:, :, ch][m] * (1 - MASK_ALPHA) + color[ch] * 255 * MASK_ALPHA

    masked = np.clip(masked, 0, 255).astype(np.uint8)

    for si, ann in enumerate(sam_results['annotations'] if sam_results else []):
        if si not in matches:
            continue
        color = TRACK_COLORS[matches[si] % len(TRACK_COLORS)]
        bm = mask_utils.decode(ann['segmentation'])
        contours, _ = cv2.findContours(bm.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(masked, contours, -1,
                         tuple(int(v * 255) for v in color), 2)

    # ======== 2. TOP-DOWN POINT CLOUD ========
    pc_td = bg_rgb.copy()

    # Re-unproject full frame (need all DEP_H*DEP_W entries for SAM mask indexing)
    pts_full, _, conf_full = unproject_frame(frame_idx)

    # Overlay animal points
    if sam_results:
        for si, ann in enumerate(sam_results['annotations']):
            if si not in matches:
                continue
            tid   = matches[si]
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]

            sam_mask   = mask_utils.decode(ann['segmentation'])
            mask_small = cv2.resize(sam_mask, (DEP_W, DEP_H),
                                    interpolation=cv2.INTER_NEAREST)

            sel  = (mask_small.ravel() > 0) & conf_full
            a_pts = pts_full[sel]
            if len(a_pts) == 0:
                continue

            row, col = pts_to_rc(a_pts)
            for dr in range(-ANIMAL_DILATE, ANIMAL_DILATE + 1):
                for dc in range(-ANIMAL_DILATE, ANIMAL_DILATE + 1):
                    r2 = np.clip(row + dr, 0, RASTER_H - 1)
                    c2 = np.clip(col + dc, 0, RASTER_W - 1)
                    pc_td[r2, c2] = color

    # Plot with trajectories
    fig_h = 10
    fig_w = fig_h * RASTER_W / RASTER_H
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.imshow(pc_td, extent=[XMIN, XMAX, ZMIN, ZMAX],
              aspect='equal', interpolation='bilinear')

    for tid, centers in track_centers.items():
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        if len(centers) >= 2:
            ax.plot(centers[:, 0], centers[:, 2], '-', color=color,
                    linewidth=2.5, alpha=0.9,
                    path_effects=[pe.Stroke(linewidth=5, foreground='white', alpha=0.5),
                                  pe.Normal()])
        if len(centers) >= 1:
            ax.plot(centers[-1, 0], centers[-1, 2], 'o', color=color,
                    markersize=11, markeredgecolor='white', markeredgewidth=2, zorder=10)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(ZMIN, ZMAX)
    ax.axis('off')
    plt.tight_layout(pad=0)

    # ======== 3. CAMERA-ALIGNED POINT CLOUD ========
    print("    Building camera-aligned view...")
    cv_img = build_camera_view(frame_idx, matches, sam_results)

    # Project trajectories to 2D
    cam  = cameras_data['cameras'][frame_idx]
    K    = np.array(cam['intrinsic'])
    E    = np.array(cam['extrinsic'])
    R, t = E[:, :3], E[:, 3]
    sx   = CV_W / DEP_W
    sy   = CV_H / DEP_H
    fx_c = K[0, 0] * sx
    fy_c = K[1, 1] * sy
    cx_c = K[0, 2] * sx
    cy_c = K[1, 2] * sy

    fig2_w = 10
    fig2_h = fig2_w * CV_H / CV_W
    fig2, ax2 = plt.subplots(figsize=(fig2_w, fig2_h))
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('white')

    ax2.imshow(np.clip(cv_img, 0, 1), aspect='equal', interpolation='bilinear')

    for tid, centers in track_centers.items():
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        # Project 3D centers to camera view
        p_cam = centers @ R.T + t
        valid = p_cam[:, 2] > 0.01
        if not np.any(valid):
            continue
        p_cam = p_cam[valid]
        u_proj = fx_c * p_cam[:, 0] / p_cam[:, 2] + cx_c
        v_proj = fy_c * p_cam[:, 1] / p_cam[:, 2] + cy_c

        if len(u_proj) >= 2:
            ax2.plot(u_proj, v_proj, '-', color=color, linewidth=2.5, alpha=0.9,
                     path_effects=[pe.Stroke(linewidth=5, foreground='white', alpha=0.5),
                                   pe.Normal()])
        if len(u_proj) >= 1:
            ax2.plot(u_proj[-1], v_proj[-1], 'o', color=color,
                     markersize=11, markeredgecolor='white', markeredgewidth=2, zorder=10)

    ax2.set_xlim(0, CV_W)
    ax2.set_ylim(CV_H, 0)
    ax2.axis('off')
    plt.tight_layout(pad=0)

    # ======== Save ========
    frame_dir = os.path.join(SAVE_DIR, str(frame_num))
    os.makedirs(frame_dir, exist_ok=True)

    Image.fromarray(masked).save(os.path.join(frame_dir, 'masked_image.png'))

    fig.savefig(os.path.join(frame_dir, 'pointcloud_topdown.png'),
                dpi=200, bbox_inches='tight', pad_inches=0.02,
                facecolor='white')
    plt.close(fig)

    fig2.savefig(os.path.join(frame_dir, 'pointcloud_camview.png'),
                 dpi=200, bbox_inches='tight', pad_inches=0.02,
                 facecolor='white')
    plt.close(fig2)

    print(f"  -> {frame_dir}/")

print(f"\nDone! Results in {SAVE_DIR}")
