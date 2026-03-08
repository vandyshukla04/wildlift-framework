# WildLIFT Toolkit

WildLIFT is a modular framework for **3D wildlife detection, tracking, and analysis from drone footage**. It reconstructs 3D scenes from monocular video, segments individual animals, fits oriented 3D bounding boxes, tracks identities across frames, and provides tools for viewpoint-aware analysis, annotation, and publication-quality visualization.

The 3D reconstruction backend ([CUT3R](https://github.com/vandyshukla04/CUT3R)) and the segmentation backend ([Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)) are pluggable **git submodules** under `backends/`. Either can be replaced by any system that conforms to the interface described in [Backend Interface](#backend-interface).

---

## Architecture

```
                         ┌──────────────────┐
                         │  Drone Footage   │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
             ┌──────▼───────┐           ┌───────▼──────┐
             │ Grounded-    │           │    CUT3R     │
             │  SAM-2       │           │ (3D Recon)   │
             │ (Segment.)   │           │              │
             └──────┬───────┘           └───────┬──────┘
                    │ masks                     │ depth, cameras
                    └─────────────┬─────────────┘
                                  │
                         ┌────────▼─────────┐
                         │  WildLIFT-RT     │
                         │  Reconstruction  │
                         │  & Tracking      │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  WildLIFT-A      │
                         │  Annotation      │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  WildLIFT-V      │
                         │  Viewpoint       │
                         │  Analysis        │
                         └──────────────────┘
```

Data flow summary:

```
[Grounded-SAM-2] ─ masks ─►[WildLIFT-RT]◄─ 3D recon ─ [CUT3R]
                                 │
                          [WildLIFT-A] (annotation)
                                 │
                          [WildLIFT-V] (viewpoint analysis)
```

---

## Installation

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/vandyshukla04/wildlift-framework.git
cd wildlift-framework
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Install CUT3R backend dependencies

```bash
cd backends/cut3r
pip install -r requirements.txt
cd ../..
```

### 3. Install Grounded-SAM-2 backend dependencies

Follow the setup instructions in [`backends/grounded-sam-2/README.md`](backends/grounded-sam-2/README.md), or:

```bash
cd backends/grounded-sam-2
pip install -e .
cd ../..
```

> **Note for reviewers:** The sample data includes pre-computed masks, so this step can be skipped if you are only running the provided examples. Grounded-SAM-2 is needed when generating masks for new data.

### 4. Install WildLIFT requirements

```bash
pip install -r requirements.txt
```

### 5. Download the model checkpoint

Download `cut3r_512_dpt_4_64.pth` into the CUT3R submodule (same as CUT3R's own setup):

```bash
pip install gdown
cd backends/cut3r/src
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ../../..
```

The pipeline auto-searches for the checkpoint in `backends/cut3r/src/`, `checkpoints/`, and `~/.cache/wildlift/`. If the checkpoint is in any of those locations, `--model_path` is not needed.

---

## Quick Start

The sample data includes image frames, pre-computed segmentation masks, and DJI SRT metadata. Replace `<DATA_DIR>` with the path to the sample data directory.

**Step 1 -- Run 3D reconstruction and tracking:**

```bash
python run_wildlift_rt.py \
    --seq_path <DATA_DIR>/zebr-3 \
    --mask_dir <DATA_DIR>/zebr-3/grounded-sam \
    --output_dir results/zebr-3 \
    --device cuda --size 512 --tracker kalman \
    --dji_log <DATA_DIR>/zebr-3/DJI_20240119124120_0003_V.SRT
```

**Step 2 -- Annotate and correct bounding boxes:**

```bash
python run_wildlift_a.py \
    --auto_bboxes results/zebr-3/bounding_boxes \
    --output results/zebr-3/corrected_bboxes \
    --images <DATA_DIR>/zebr-3 \
    --mask_dir <DATA_DIR>/zebr-3/grounded-sam
```

Opens a browser-based 3D editor at `http://localhost:8080`.

**Step 3 -- Analyze viewpoints:**

```bash
python run_wildlift_v.py \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir <DATA_DIR>/zebr-3 \
    --aggregate --load_saved
```

**Step 3b -- With semantic face propagation:**

```bash
python run_wildlift_v.py \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir <DATA_DIR>/zebr-3 \
    --aggregate --load_saved --semantic
```

---

## Modules

### 5.1 WildLIFT-RT: Reconstruction and Tracking

The core pipeline reconstructs per-frame 3D point clouds from monocular images, segments animals using instance masks, fits oriented 3D bounding boxes, and tracks identities across frames with a 3D Kalman filter.

**Entry point:**

```bash
python run_wildlift_rt.py [OPTIONS]
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--seq_path` | *(required)* | Directory containing the image sequence |
| `--mask_dir` | *(required)* | Directory containing Grounded-SAM mask JSON files |
| `--model_path` | *(auto-resolved)* | Path to pretrained model checkpoint. Auto-searches `backends/cut3r/src/`, `checkpoints/`, `~/.cache/wildlift/` |
| `--device` | `cuda` | Device for inference (`cuda` or `cpu`) |
| `--size` | `512` | Input image rescale resolution |
| `--tracker` | `kalman` | Tracking method: `kalman` or `simple` |
| `--dji_log` | | Path to DJI SRT log file for gimbal data |
| `--gps_refine` | off | Enable GPS-based pose refinement (flag) |
| `--revisit` | `1` | Number of revisit passes (`1` = online only, `2+` = revisiting) |
| `--output_dir` | `./demo_tmp` | Output directory |
| `--save_ply` | on | Save per-frame PLY point clouds (flag) |
| `--vis_threshold` | `1.5` | Visualization threshold for point cloud viewer |
| `--blend_mode` | `overlay` | Visualization mode: `overlay`, `highlight`, `mask_only`, `original` |

**Tracker types:**

- **`kalman`** (default) -- 3D Kalman filter with constant-velocity motion model (6-state: `[x, y, z, vx, vy, vz]`), Hungarian assignment, and dormant track re-identification for handling occlusions.
- **`simple`** -- Lightweight online tracker using centroid distance matching. Useful as a fast fallback when Kalman overhead is unnecessary.

**DJI SRT format support:**

DJI subtitle logs are auto-detected. Supported formats:

| Format | Key Field | GPS | Gimbal Fields |
|---|---|---|---|
| FrameCnt + GPS (new) | `FrameCnt` | lat/lon/rel_alt | `gimbal_heading/pitch/roll` |
| FrameCnt + GPS (old) | `FrameCnt` | lat/lon/rel_alt/abs_alt | `gb_yaw/pitch/roll` + `focal_len` |
| SrtCnt KABR | `SrtCnt : X` | -- | `gimbal_heading/pitch/roll` |
| SrtCnt + altitude | `SrtCnt:` | rel_alt | `gimbal_heading/pitch/roll` |
| FrameCnt gimbal-only | `FrameCnt:` | rel_alt | `gb_yaw/pitch/roll` |

**Species dimensions:**

Built-in dimension priors (length / width / height in meters):

| Species | Length | Width | Height |
|---|---|---|---|
| Elephant | 5.5 | 2.5 | 3.2 |
| Rhino | 3.8 | 1.5 | 1.8 |
| Zebra | 2.5 | 0.7 | 1.4 |
| Generic (fallback) | 2.0 | 0.6 | 1.2 |

#### Retracking (post-processing)

Re-run tracking on existing reconstruction outputs without re-running CUT3R:

```bash
python -m wildlift.rt.retracker \
    --result_dir results/zebr-3 \
    --source_images /path/to/zebr-3 \
    --output_subfolder retracked \
    --max_missing_frames 20 \
    --dormant_timeout 100 \
    --distance_weight 0.7 \
    --iou_weight 0.3 \
    --max_distance 8.0
```

#### Evaluation

Evaluate tracking performance against ground-truth annotations (MOTChallenge format). Computes MOTA, IDF1, ID switches, FP, FN, precision, recall, and track fragmentation.

```bash
python -m wildlift.rt.eval_tracking \
    --gt results/gt_annotations/zebras/zebr-3 \
    --pred results/zebr-3 \
    --pred_retracked results/zebr-3/retracked
```

#### Baseline comparison

Run 2D baseline trackers (ByteTrack, BotSORT, OC-SORT) on the same detections for fair comparison:

```bash
python -m wildlift.rt.run_baselines \
    --result_dir results/zebr-3 \
    --source_images /path/to/zebr-3 \
    --trackers bytetrack botsort ocsort
```

---

### 5.2 WildLIFT-A: Annotation Tools

#### 3D Bounding Box Editor

An interactive browser-based 3D annotation tool built on [Viser](https://viser.studio). Allows manual correction of auto-generated bounding boxes, semantic face labeling (front, top, left), and interpolation between keyframes.

**Entry point:**

```bash
python run_wildlift_a.py [OPTIONS]
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--auto_bboxes` | *(required)* | Directory containing auto-generated bbox JSONs |
| `--output` | *(required)* | Output directory for corrected annotations |
| `--images` | | Directory with source images for 2D visualization |
| `--mask_dir` | | Grounded-SAM mask directory (auto-detected if omitted) |
| `--no_reload` | off | Do not reload previous annotations from output directory |
| `--port` | `8080` | Viser server port |

**Workflow and controls:**

The editor launches a Viser server at `http://localhost:<port>`. GUI buttons provide the following operations:

| Action | Description |
|---|---|
| Previous / Next Frame | Navigate between frames |
| Snap to Proportions | Constrain dimensions to species-specific ratios |
| Snap to Ground | Align bbox bottom to ground plane |
| Snap All Frames to Ground | Apply ground snapping across all frames |
| Copy BBox from Previous/Next | Clone bbox from adjacent frame |
| Mark as Keyframe 1 / 2 | Set interpolation endpoints |
| Interpolate Between Keyframes | Smoothly interpolate bbox pose and dimensions between keyframes |
| Interpolate Semantic Faces Only | Propagate semantic face labels between keyframes without changing geometry |
| Propagate to All Frames | Copy current bbox to all frames |
| Next Unannotated Frame | Jump to the next frame lacking annotations |
| Save Now | Manually trigger save (auto-save also runs) |

#### GT Track ID Annotator

For creating ground-truth track identity labels used by `eval_tracking`. Produces `gt.txt` (MOTChallenge format) and `metadata.json`.

---

### 5.3 WildLIFT-V: Viewpoint Analysis

Tools for analyzing which viewpoints of each animal are captured across a sequence, generating filmstrips, measuring occlusion, and producing publication-quality metric visualizations.

#### Core analyzer

**Entry point:**

```bash
python run_wildlift_v.py [OPTIONS]
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--annotator_output` | *(required)* | Path to annotator/corrected bbox output directory |
| `--images_dir` | | Path to original images (required when using `--semantic`) |
| `--mask_dir` | | Path to Grounded-SAM mask directory |
| `--results_dir` | | Path to results directory |
| `--min_quality` | `0.3` | Minimum quality threshold |
| `--max_candidates` | | Max candidates per orientation |
| `--track_id` | | Specific track ID(s) to process (space-separated) |
| `--aggregate` | off | Generate aggregate PDF combining all tracks |
| `--load_saved` | off | Load from existing selection files (non-interactive) |
| `--select_tracks` | off | Interactively select which tracks to process |
| `--select_rejected` | off | Enable rejected frame selection in interactive mode |
| `--show_rejected` | off | Show rejected frames in output PDFs |
| `--semantic` | off | Run semantic face propagation after viewpoint analysis (requires `--images_dir`) |
| `--video_name` | | Video name for PDF title (auto-detected if omitted) |

**Examples:**

```bash
# Aggregate PDF from saved selections (recommended for batch)
python run_wildlift_v.py \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir /path/to/zebr-3 \
    --load_saved --aggregate

# Interactive selection with rejected frames
python run_wildlift_v.py \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir /path/to/zebr-3 \
    --aggregate --select_rejected

# Process specific tracks
python run_wildlift_v.py \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir /path/to/zebr-3 \
    --track_id 0 1 5 --aggregate

# With semantic face propagation
python run_wildlift_v.py \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir /path/to/zebr-3 \
    --aggregate --load_saved --semantic
```

#### Filmstrip generation

Combine per-track filmstrips from multiple segments into a single publication-quality PDF:

```bash
python -m wildlift.viewpoint.filmstrip \
    --segment results/zebr-3/corrected_bboxes results/zebr-4/corrected_bboxes \
    --output combined_filmstrip.pdf \
    --name "Zebra Study Area"
```

#### Occlusion analysis

Quantify inter-animal occlusion using 3D ray-OBB intersection and 2D mask overlap. Reports which body parts are occluded, by whom, and selects the least-occluded frame for each face.

```bash
python -m wildlift.viewpoint.occlusion \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir /path/to/zebr-3 \
    --pdf --json --annotated_frames
```

#### Publication-quality metrics visualization

Generate Nature Methods-style temporal metric plots (per-face quality, effective visibility, coverage, diversity index, dominant viewpoint ribbon, quality heatmap):

```bash
python -m wildlift.viewpoint.metrics_viz \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir /path/to/zebr-3

# With pre-computed occlusion data
python -m wildlift.viewpoint.metrics_viz \
    --annotator_output results/zebr-3/corrected_bboxes \
    --images_dir /path/to/zebr-3 \
    --occlusion_json results/zebr-3/corrected_bboxes/occlusion_analysis/occlusion_summary.json
```

#### Re-ID proof of concept

Demonstrates that viewpoint-conditioned feature matching (left-to-left, front-to-front) outperforms viewpoint-agnostic matching for discriminating between animals:

```bash
python -m wildlift.viewpoint.reid_poc \
    --results_dir results/zebr-3 \
    --images_dir /path/to/zebr-3 \
    --output_dir results/reid_poc/zebr-3

# Run across all multi-track sequences
python -m wildlift.viewpoint.reid_poc --run_all --output_dir results/reid_poc
```

---

## Data Converters

Convert WildLIFT outputs to standard 3D detection formats for training or benchmarking with external frameworks.

**Entry point:**

```bash
python -m wildlift.converters.unified [OPTIONS]
```

**Supported formats:**

| Format | Description |
|---|---|
| `kitti` | KITTI / MMDetection3D label format |
| `omni3d` | Omni3D JSON annotation format |
| `wildlife_info` | Wildlife info pickle files (`.pkl`) |

**Key arguments:**

| Argument | Description |
|---|---|
| `--format` | Target format: `kitti`, `omni3d`, or `wildlife_info` |
| `--videos-root` | Root directory containing video frame directories |
| `--results-root` | Root directory containing WildLIFT result directories |
| `--output-dir` | Output directory for converted data |
| `--target-class` | Target animal class (default: `rhino`) |
| `--config` | Optional YAML configuration file |
| `--omni3d-image-output` | Image output directory for Omni3D format |

**Example:**

```bash
python -m wildlift.converters.unified \
    --format kitti \
    --videos-root data/rhinos/ \
    --results-root results/rhinos/ \
    --output-dir converted/kitti/ \
    --target-class rhino
```

---

## Visualization Tools

The `wildlift/tools/` package contains standalone visualization utilities:

| Module | Description |
|---|---|
| `bbox_projection` | Project 3D bounding boxes onto 2D images |
| `draw_bboxes` | Draw 2D bounding boxes from JSON annotations onto images |
| `ecology_viz` | Publication-quality multi-panel figures: PCA-aligned canonical views, motion trails, instance segmentation with convex hulls, interactive Plotly HTML, GIF/video animations |
| `frame_viewer` | Interactive frame viewer with mask overlays, trajectory columns, and batch export for table creation |
| `frames_to_video` | Convert a directory of image frames into an MP4 video |
| `make_video` | Simple image-to-video conversion utility |
| `instance_viz` | Instance isolation visualization: top-down point cloud with highlighted instances, multi-view orthographic projections |
| `mask_track_viz` | Overlay instance masks with track-consistent colors on source images |
| `pointcloud_viz` | Interactive point cloud viewer (Viser-based). Supports PLY, PCD, NPY, NPZ, and RGBD sequences |
| `tracking_traj` | Tracking trajectory visualization: masked images, top-down point cloud with trails, camera-aligned projected trajectories |
| `tracklet_report` | Generate PDF tracklet quality reports with letter grades, metric breakdowns, masked animal crops, and use-case fitness assessments |

**Example usage:**

```bash
# Visualize mask tracks
python -m wildlift.tools.mask_track_viz \
    --images /path/to/zebr-3 \
    --masks results/zebr-3/instance_labels \
    --mapping results/zebr-3/mask_track_mapping.json \
    --output results/zebr-3/mask_vis

# Ecology paper figures
python -m wildlift.tools.ecology_viz \
    --result_dir results/zebr-3 \
    --output_dir figures/

# Point cloud viewer
python -m wildlift.tools.pointcloud_viz results/zebr-3/pointclouds/frame_0000.ply
```

---

## Backend Submodules

### CUT3R (3D Reconstruction)

The CUT3R submodule at `backends/cut3r/` provides the 3D reconstruction backend. Its dependencies are installed in step 2 of the installation. The model checkpoint is downloaded in step 4. For details on CUT3R itself (training, architecture, additional options), see [`backends/cut3r/README.md`](backends/cut3r/README.md).

### Grounded-SAM-2 (Segmentation)

The Grounded-SAM-2 submodule at `backends/grounded-sam-2/` is included for **reproducibility** -- it documents the exact segmentation system used to generate the instance masks. The sample data includes pre-computed masks, so **Grounded-SAM-2 does not need to be installed** to run the pipeline.

To generate masks for new data, follow the setup and usage instructions in [`backends/grounded-sam-2/README.md`](backends/grounded-sam-2/README.md).

---

## Backend Interface

WildLIFT is designed so that both the 3D reconstruction and segmentation backends can be replaced. Any replacement must conform to the following interfaces.

### 3D Reconstruction Backend

Must expose an inference function with the following signature:

```python
def inference(pairs, model, device) -> dict
```

**Returns** a dictionary containing, for each view:
- **Depth map** -- per-pixel depth (H x W array)
- **Confidence map** -- per-pixel reconstruction confidence (H x W array)
- **Camera parameters** -- intrinsic matrix and camera-to-world extrinsic pose (4x4)

The pipeline expects the backend to be importable from `backends/cut3r/` and to provide `load_images` and model loading utilities.

### Segmentation Backend

Must produce **per-frame JSON files** in the mask directory, each containing a list of detections with:

- **`class_name`** (string) -- detected object class label
- **`bbox`** (list) -- 2D bounding box `[x1, y1, x2, y2]`
- **`mask`** -- binary mask in RLE (COCO run-length encoding) or polygon format
- **`confidence`** (float) -- detection confidence score

---

## Output File Formats

### Bounding Box JSON (`bounding_boxes/<frame>.json`)

Per-frame JSON array. Each element:

```json
{
  "center": [x, y, z],
  "dimensions": [length, width, height],
  "rotation_matrix": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
  "class_name": "zebra",
  "confidence": 0.87,
  "instance_id": 2,
  "track_id": 0,
  "persistent_instance_id": 0
}
```

| Field | Type | Description |
|---|---|---|
| `center` | `float[3]` | 3D center of the bounding box |
| `dimensions` | `float[3]` | Box extents (length, width, height) |
| `rotation_matrix` | `float[3][3]` | Orientation as a 3x3 rotation matrix |
| `class_name` | `string` | Detected species class |
| `confidence` | `float` | Detection confidence |
| `instance_id` | `int` | Per-frame instance index |
| `track_id` | `int` | Cross-frame track identity (`-1` if unassigned) |
| `persistent_instance_id` | `int` | Globally persistent instance ID |

### Instance Label NPY (`instance_labels/<frame>.npy`)

NumPy array of shape `(H, W)` with integer instance labels. Background is `0`; each detected instance has a unique positive integer label.

### Camera Parameter NPZ (`camera_params/<frame>.npz`)

NumPy archive containing:

- `intrinsic` -- 3x3 intrinsic camera matrix
- `cam_c2w` -- 4x4 camera-to-world extrinsic pose

### Mask-Track Mapping (`mask_track_mapping.json`)

Maps frame names to dictionaries of `{track_id: instance_label_value}`:

```json
{
  "frame_0000": {
    "0": 1,
    "1": 2
  },
  "frame_0001": {
    "0": 1,
    "1": 3
  }
}
```

Keys are string track IDs; values are the integer instance label in the corresponding NPY file.

---

## Typical Workflows

### End-to-end pipeline

```
Drone video
  → Frame extraction
  → Grounded-SAM-2 segmentation
  → WildLIFT-RT inference (run_wildlift_rt.py)
  → Retracking if needed (wildlift.rt.retracker)
  → WildLIFT-A annotation (run_wildlift_a.py)
  → WildLIFT-V viewpoint analysis (run_wildlift_v.py)
  → Filmstrip generation (wildlift.viewpoint.filmstrip)
```

### Evaluation workflow

```
1. Create GT annotations with the GT Track ID Annotator
2. Run tracking evaluation:
     python -m wildlift.rt.eval_tracking --gt <gt_dir> --pred <pred_dir>
3. Run baseline comparison:
     python -m wildlift.rt.run_baselines --result_dir <dir> --trackers bytetrack botsort ocsort
4. Evaluate baselines:
     python -m wildlift.rt.eval_tracking --gt <gt_dir> --pred <baseline_dir>
```

### Publication figures

```
1. Run batch viewpoint analysis across sequences:
     python run_wildlift_v.py --annotator_output <dir> --images_dir <images> --aggregate --load_saved

2. Generate combined filmstrips:
     python -m wildlift.viewpoint.filmstrip \
         --segment <dir1> <dir2> --output filmstrip.pdf

3. Create metrics visualizations:
     python -m wildlift.viewpoint.metrics_viz \
         --annotator_output <dir> --images_dir <images>

4. Generate tracklet quality reports:
     python -m wildlift.tools.tracklet_report \
         --annotator_output <dir> --images_dir <images>
```

---

## Acknowledgments

WildLIFT builds on the following open-source projects:

- **[CUT3R](https://github.com/CUT3R-official/CUT3R)** -- 3D reconstruction from unposed images. WildLIFT uses CUT3R as its default 3D reconstruction backend.
- **[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)** -- Grounded segmentation combining Grounding DINO with SAM 2. WildLIFT uses Grounded-SAM-2 to generate per-frame instance segmentation masks.

We thank the authors of both projects for making their code publicly available.

---

## License

This project is released under the [MIT License](LICENSE).
