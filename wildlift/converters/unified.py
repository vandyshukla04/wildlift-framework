#!/usr/bin/env python3
"""
Unified Wildlife 3D Data Converter
Converts CUT3R wildlife detection outputs to various standard formats:
- KITTI/MMDetection3D format
- Omni3D format
- Wildlife info files (.pkl)

Consolidates functionality from:
- prepare_data.py
- rhino_to_omni3d.py
- complete_conversion.py
- create_wildlife_info.py
"""

import numpy as np
import cv2
import json
import yaml
import argparse
import pickle
import os
import glob
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConverterConfig:
    """Configuration for data conversion"""
    # Required paths
    videos_root: str
    results_root: str
    output_dir: str

    # Conversion format
    format: str = "kitti"  # "kitti", "omni3d", or "wildlife_info"

    # Optional parameters with defaults
    grounded_sam_subdir: str = "grounded-sam"
    target_width: int = 512
    target_height: int = 288
    original_width: int = 3072
    original_height: int = 1728
    train_split_ratio: float = 0.8
    min_confidence: float = 0.5
    target_class: str = "rhino"
    video_names: Optional[List[str]] = None

    # Omni3D specific
    omni3d_image_output: str = None
    category_id: int = 1000

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class BaseConverter:
    """Base class for all converters"""

    def __init__(self, config: ConverterConfig):
        self.config = config
        self.videos_root = Path(config.videos_root)
        self.results_root = Path(config.results_root)
        self.output_dir = Path(config.output_dir)

        self.processed_frames = []
        self.failed_frames = []

    def _find_results_folder(self, video_name: str) -> Optional[Path]:
        """Find corresponding results folder for a video"""
        possible_folders = []
        for folder in self.results_root.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                if folder_name.startswith(f"tmp-{video_name}-") or \
                   folder_name.startswith(f"tmp_{video_name}"):
                    possible_folders.append(folder)

        if possible_folders:
            possible_folders.sort()
            if len(possible_folders) > 1:
                logger.warning(f"Multiple results folders found for {video_name}, using: {possible_folders[0]}")
            return possible_folders[0]

        logger.warning(f"No results folder found for video: {video_name}")
        return None

    @staticmethod
    def compute_3d_corners(center, dimensions, rotation_matrix):
        """Compute 8 corners of 3D bounding box"""
        w, h, l = dimensions

        corners_local = np.array([
            [-w/2, -h/2, -l/2],
            [ w/2, -h/2, -l/2],
            [ w/2,  h/2, -l/2],
            [-w/2,  h/2, -l/2],
            [-w/2, -h/2,  l/2],
            [ w/2, -h/2,  l/2],
            [ w/2,  h/2,  l/2],
            [-w/2,  h/2,  l/2]
        ])

        R = np.array(rotation_matrix)
        corners_cam = (R @ corners_local.T).T + np.array(center)

        return corners_cam

    @staticmethod
    def project_to_2d(corners_3d, K):
        """Project 3D corners to 2D image plane"""
        corners_3d = np.array(corners_3d)
        corners_2d = (K @ corners_3d.T).T
        corners_2d[:, 0] /= corners_2d[:, 2]
        corners_2d[:, 1] /= corners_2d[:, 2]

        x_min = float(corners_2d[:, 0].min())
        y_min = float(corners_2d[:, 1].min())
        x_max = float(corners_2d[:, 0].max())
        y_max = float(corners_2d[:, 1].max())

        return [x_min, y_min, x_max, y_max]


class KITTIConverter(BaseConverter):
    """MMDetection3D/KITTI format conversion"""

    def __init__(self, config: ConverterConfig):
        super().__init__(config)
        self._create_output_dirs()

    def _create_output_dirs(self):
        """Create MMDetection3D-compatible directory structure"""
        dirs = [
            'ImageSets',
            'training/image_2',
            'training/label_2',
            'training/calib',
            'training/depth_maps',
            'validation/image_2',
            'validation/label_2',
            'validation/calib',
            'validation/depth_maps'
        ]
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def _extract_yaw_for_camera_coords(self, rotation_matrix: List[List[float]]) -> float:
        """Extract yaw angle from rotation matrix in CAMERA coordinate system"""
        R = np.array(rotation_matrix)
        yaw = np.arctan2(R[2, 0], R[0, 0])
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        return yaw

    def _create_kitti_label(self, bbox3d: Dict, bbox2d: Optional[List[float]]) -> str:
        """Convert to KITTI format label"""
        yaw = self._extract_yaw_for_camera_coords(bbox3d['rotation_matrix'])
        x, y, z = bbox3d['center']
        l, w, h = bbox3d['dimensions']

        alpha = yaw - np.arctan2(x, z)
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        if bbox2d:
            x1, y1, x2, y2 = bbox2d
        else:
            x1 = y1 = x2 = y2 = 0.0

        label = f"{self.config.target_class} "
        label += f"-1 -1 {alpha:.6f} "
        label += f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        label += f"{h:.6f} {w:.6f} {l:.6f} "
        label += f"{x:.6f} {y:.6f} {z:.6f} "
        label += f"{yaw:.6f} "
        label += f"{bbox3d.get('confidence', 1.0):.6f}"

        return label

    def _create_calibration_file(self, K: np.ndarray) -> str:
        """Create KITTI-style calibration file content"""
        P2 = np.zeros((3, 4))
        P2[:3, :3] = K
        P2_str = ' '.join(map(str, P2.flatten()))

        R0_rect = np.eye(3)
        R0_rect_str = ' '.join(map(str, R0_rect.flatten()))

        Tr_velo_to_cam = np.eye(4)[:3, :]
        Tr_velo_to_cam_str = ' '.join(map(str, Tr_velo_to_cam.flatten()))

        calib_content = f"P0: {P2_str}\n"
        calib_content += f"P1: {P2_str}\n"
        calib_content += f"P2: {P2_str}\n"
        calib_content += f"P3: {P2_str}\n"
        calib_content += f"R0_rect: {R0_rect_str}\n"
        calib_content += f"Tr_velo_to_cam: {Tr_velo_to_cam_str}\n"
        calib_content += f"Tr_imu_to_velo: {Tr_velo_to_cam_str}"

        return calib_content

    def process_frame(self, video_folder: Path, results_folder: Path,
                     frame_id: str, split: str = 'training') -> Optional[str]:
        """Process a single frame"""
        video_name = video_folder.name
        output_name = f"{video_name}_{frame_id}"

        # Load 3D annotation
        bbox3d_path = results_folder / "bounding_boxes" / f"{frame_id}.json"
        if not bbox3d_path.exists():
            return None

        with open(bbox3d_path, 'r') as f:
            bbox3d_data = json.load(f)

        if not bbox3d_data:
            return None

        bbox3d = bbox3d_data[0]

        if bbox3d.get('confidence', 1.0) < self.config.min_confidence:
            return None

        # Load and resize image
        img_path = video_folder / f"{frame_id}.jpg"
        if not img_path.exists():
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            return None

        img_resized = cv2.resize(img, (self.config.target_width, self.config.target_height))

        # Load and scale 2D bounding box
        bbox2d_scaled = None
        grounded_sam_path = video_folder / self.config.grounded_sam_subdir / f"{frame_id}_results.json"
        if grounded_sam_path.exists():
            with open(grounded_sam_path, 'r') as f:
                sam_data = json.load(f)

            for ann in sam_data.get('annotations', []):
                if ann.get('class_name') == self.config.target_class:
                    bbox = ann['bbox']
                    scale_x = self.config.target_width / sam_data['img_width']
                    scale_y = self.config.target_height / sam_data['img_height']
                    bbox2d_scaled = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                    break

        # Load depth map
        depth_path = results_folder / "depth" / f"{frame_id}.npy"
        depth = None
        if depth_path.exists():
            depth = np.load(depth_path)

        # Load camera parameters
        camera_files = list((results_folder / "camera").glob("*.npz"))
        if not camera_files:
            return None

        camera_data = np.load(camera_files[0])
        K = camera_data['intrinsics']

        # Save processed data
        cv2.imwrite(str(self.output_dir / split / 'image_2' / f'{output_name}.jpg'), img_resized)

        label = self._create_kitti_label(bbox3d, bbox2d_scaled)
        with open(self.output_dir / split / 'label_2' / f'{output_name}.txt', 'w') as f:
            f.write(label)

        if depth is not None:
            np.save(str(self.output_dir / split / 'depth_maps' / f'{output_name}.npy'), depth)

        calib_content = self._create_calibration_file(K)
        with open(self.output_dir / split / 'calib' / f'{output_name}.txt', 'w') as f:
            f.write(calib_content)

        return output_name

    def convert(self):
        """Main conversion process"""
        # Find video folders
        video_folders = []
        if self.config.video_names:
            for name in self.config.video_names:
                folder = self.videos_root / name
                if folder.exists():
                    video_folders.append(folder)
        else:
            video_folders = [f for f in self.videos_root.iterdir()
                           if f.is_dir() and list(f.glob("*.jpg"))]

        if not video_folders:
            logger.error("No video folders found!")
            return

        logger.info(f"Found {len(video_folders)} video folders")

        all_processed = []

        for video_folder in tqdm(video_folders, desc="Processing videos"):
            video_name = video_folder.name
            results_folder = self._find_results_folder(video_name)
            if results_folder is None:
                continue

            frame_files = sorted(video_folder.glob("*.jpg"))

            for frame_path in tqdm(frame_files, desc=f"  {video_name}", leave=False):
                frame_id = frame_path.stem

                try:
                    output_name = self.process_frame(
                        video_folder, results_folder, frame_id, 'training')
                    if output_name:
                        all_processed.append(output_name)
                except Exception as e:
                    logger.error(f"Error processing {video_name}/{frame_id}: {e}")
                    self.failed_frames.append(f"{video_name}/{frame_id}")

        self.processed_frames = all_processed
        self._create_splits()

        logger.info(f"✓ Processed {len(all_processed)} frames")
        logger.info(f"✗ Failed frames: {len(self.failed_frames)}")

    def _create_splits(self):
        """Create train/validation splits"""
        if not self.processed_frames:
            return

        frames = self.processed_frames.copy()
        np.random.seed(42)
        np.random.shuffle(frames)

        split_idx = int(len(frames) * self.config.train_split_ratio)
        train_frames = frames[:split_idx]
        val_frames = frames[split_idx:]

        with open(self.output_dir / 'ImageSets' / 'train.txt', 'w') as f:
            f.write('\n'.join(train_frames))

        with open(self.output_dir / 'ImageSets' / 'val.txt', 'w') as f:
            f.write('\n'.join(val_frames))

        with open(self.output_dir / 'ImageSets' / 'test.txt', 'w') as f:
            f.write('\n'.join(val_frames))

        logger.info(f"Created splits - Train: {len(train_frames)}, Val: {len(val_frames)}")


class Omni3DConverter(BaseConverter):
    """Omni3D format conversion"""

    def __init__(self, config: ConverterConfig):
        super().__init__(config)
        self.omni3d_image_output = Path(config.omni3d_image_output) if config.omni3d_image_output else self.output_dir / "images"
        self.omni3d_image_output.mkdir(parents=True, exist_ok=True)

    def _find_matched_pairs(self):
        """Automatically find and match video/result directories"""
        video_dirs = sorted(self.videos_root.glob("*"))
        video_mapping = {}
        for vdir in video_dirs:
            if vdir.is_dir():
                video_id = vdir.name.replace("rhin-", "")
                video_mapping[video_id] = vdir

        result_dirs = sorted(self.results_root.glob("tmp-*"))
        result_mapping = {}
        for rdir in result_dirs:
            basename = rdir.name
            parts = basename.replace("tmp-rhin-", "").replace("tmp-", "").split("-revisit-")
            result_id = parts[0]
            revisit_num = int(parts[1]) if len(parts) > 1 else 1

            if result_id not in result_mapping or revisit_num == 1:
                result_mapping[result_id] = rdir

        matched_pairs = []
        for vid_id in video_mapping:
            if vid_id in result_mapping:
                matched_pairs.append((vid_id, video_mapping[vid_id], result_mapping[vid_id]))

        return matched_pairs

    def convert(self):
        """Main Omni3D conversion process"""
        logger.info("="*70)
        logger.info("Converting to Omni3D format")
        logger.info("="*70)

        matched_pairs = self._find_matched_pairs()
        logger.info(f"Found {len(matched_pairs)} matched video-result pairs")

        # Split videos
        random.seed(42)
        video_ids = [vid_id for vid_id, _, _ in matched_pairs]
        random.shuffle(video_ids)

        # 8/2/2 split
        n_videos = len(video_ids)
        train_size = int(n_videos * 0.6)
        val_size = int(n_videos * 0.2)

        train_videos = video_ids[:train_size]
        val_videos = video_ids[train_size:train_size+val_size]
        test_videos = video_ids[train_size+val_size:]

        logger.info(f"Data split: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")

        splits = {
            f'{self.config.target_class.upper()}_train': train_videos,
            f'{self.config.target_class.upper()}_val': val_videos,
            f'{self.config.target_class.upper()}_test': test_videos
        }

        category_id = self.config.category_id
        categories = [{
            "id": category_id,
            "name": self.config.target_class,
            "supercategory": "animal"
        }]

        # Process each split
        for split_name, video_list in splits.items():
            logger.info(f"\nProcessing {split_name}...")

            dataset = {
                "info": {
                    "id": split_name.lower(),
                    "name": split_name,
                    "source": f"{self.config.target_class}_wildlife",
                    "known_category_ids": [category_id]
                },
                "categories": categories,
                "images": [],
                "annotations": []
            }

            image_id_counter = 1
            anno_id_counter = 1

            for vid_id in tqdm(video_list, desc=f"Videos in {split_name}"):
                video_dir = None
                result_dir = None

                for v_id, v_dir, r_dir in matched_pairs:
                    if v_id == vid_id:
                        video_dir = v_dir
                        result_dir = r_dir
                        break

                if not video_dir:
                    continue

                # Copy images and create annotations
                bbox_dir = result_dir / "bounding_boxes"
                camera_dir = result_dir / "camera"
                grounded_sam_dir = video_dir / self.config.grounded_sam_subdir

                bbox_files = sorted(bbox_dir.glob("*.json"))

                for bbox_file in bbox_files:
                    frame_name = bbox_file.stem

                    image_path = video_dir / f"{frame_name}.jpg"
                    camera_path = camera_dir / f"{frame_name}.npz"
                    sam_path = grounded_sam_dir / f"{frame_name}_results.json"

                    if not (image_path.exists() and camera_path.exists()):
                        continue

                    # Copy image
                    video_output_dir = self.omni3d_image_output / vid_id
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    output_image_path = video_output_dir / f"{frame_name}.jpg"

                    if not output_image_path.exists():
                        shutil.copy2(image_path, output_image_path)

                    # Load camera
                    cam_data = np.load(camera_path)
                    K = cam_data['intrinsics'].astype(float)
                    cam_data.close()

                    # Get image dimensions
                    img_width, img_height = 768, 432
                    if sam_path.exists():
                        with open(sam_path, 'r') as f:
                            sam_data = json.load(f)
                        img_width = sam_data.get('img_width', 768)
                        img_height = sam_data.get('img_height', 432)

                    # Create image entry
                    image_entry = {
                        "id": image_id_counter,
                        "file_path": f"{self.config.target_class}/{vid_id}/{frame_name}.jpg",
                        "dataset_id": split_name.lower(),
                        "height": img_height,
                        "width": img_width,
                        "K": K.flatten().tolist()
                    }
                    dataset["images"].append(image_entry)

                    # Load 3D boxes
                    with open(bbox_file, 'r') as f:
                        boxes_3d = json.load(f)

                    for box in boxes_3d:
                        if box.get('class_name') != self.config.target_class:
                            continue

                        center = box['center']
                        if center[2] <= 0:  # Skip if behind camera
                            continue

                        dims = box['dimensions']
                        R = box['rotation_matrix']

                        corners_3d = self.compute_3d_corners(center, dims, R).tolist()
                        bbox_2d_proj = self.project_to_2d(corners_3d, K)

                        # Get tight 2D box from SAM
                        bbox_2d_tight = bbox_2d_proj.copy()
                        if sam_path.exists():
                            with open(sam_path, 'r') as f:
                                sam_data = json.load(f)
                            for sam_anno in sam_data.get('annotations', []):
                                if sam_anno.get('class_name') == self.config.target_class:
                                    bbox_2d_tight = sam_anno['bbox']
                                    break

                        x1, y1, x2, y2 = bbox_2d_tight
                        bbox_xywh = [x1, y1, x2-x1, y2-y1]

                        annotation = {
                            "id": anno_id_counter,
                            "image_id": image_id_counter,
                            "category_id": category_id,
                            "category_name": self.config.target_class,
                            "bbox": bbox_xywh,
                            "bbox2D_proj": bbox_2d_proj,
                            "bbox2D_tight": bbox_2d_tight,
                            "bbox2D_trunc": bbox_2d_proj,
                            "center_cam": center,
                            "dimensions": dims,
                            "R_cam": R,
                            "pose": R,
                            "bbox3D_cam": corners_3d,
                            "truncation": 0.0,
                            "visibility": 1.0,
                            "behind_camera": False,
                            "valid3D": True,
                            "lidar_pts": 100,
                            "segmentation_pts": 100,
                            "depth_error": 0.0,
                            "area": bbox_xywh[2] * bbox_xywh[3],
                            "iscrowd": False,
                            "ignore": False,
                            "ignore2D": False,
                            "ignore3D": False
                        }

                        dataset["annotations"].append(annotation)
                        anno_id_counter += 1

                    image_id_counter += 1

            # Save JSON
            output_path = self.output_dir / f"{split_name}.json"
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)

            logger.info(f"  Images: {len(dataset['images'])}, Annotations: {len(dataset['annotations'])}")
            logger.info(f"  Saved to: {output_path}")

        logger.info("\nConversion complete!")


class WildlifeInfoGenerator(BaseConverter):
    """Wildlife dataset metadata generation for MMDetection3D"""

    def convert(self):
        """Generate info files"""
        logger.info("Generating wildlife info files...")

        for split in ['train', 'val']:
            logger.info(f'\nGenerating info for {split} set...')

            split_file = self.output_dir / 'ImageSets' / f'{split}.txt'
            if not split_file.exists():
                logger.warning(f"Split file not found: {split_file}")
                continue

            with open(split_file, 'r') as f:
                img_ids = [line.strip() for line in f]

            infos = []
            for img_id in tqdm(img_ids, desc=f'Processing {split}'):
                info = {}

                # Image info
                info['image'] = {
                    'image_idx': img_id,
                    'image_path': f'training/image_2/{img_id}.jpg',
                    'image_shape': np.array([self.config.target_height, self.config.target_width], dtype=np.int32)
                }

                # Calibration
                calib_path = self.output_dir / 'training' / 'calib' / f'{img_id}.txt'
                if calib_path.exists():
                    info['calib'] = self._parse_calib(calib_path)

                # Annotations
                label_path = self.output_dir / 'training' / 'label_2' / f'{img_id}.txt'
                if label_path.exists():
                    info['annos'] = self._parse_label(label_path)

                # Depth map
                depth_path = self.output_dir / 'training' / 'depth_maps' / f'{img_id}.npy'
                if depth_path.exists():
                    info['depth_map'] = {'depth_path': f'training/depth_maps/{img_id}.npy'}

                info['point_cloud'] = {'num_features': 4, 'velodyne_path': None}

                infos.append(info)

            # Save pickle
            output_file = self.output_dir / f'wildlife_infos_{split}.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(infos, f)
            logger.info(f'Saved {len(infos)} samples to {output_file}')

        # Create trainval
        logger.info('\nGenerating trainval info...')
        train_pkl = self.output_dir / 'wildlife_infos_train.pkl'
        val_pkl = self.output_dir / 'wildlife_infos_val.pkl'

        if train_pkl.exists() and val_pkl.exists():
            with open(train_pkl, 'rb') as f:
                train_infos = pickle.load(f)
            with open(val_pkl, 'rb') as f:
                val_infos = pickle.load(f)

            trainval_infos = train_infos + val_infos
            trainval_pkl = self.output_dir / 'wildlife_infos_trainval.pkl'
            with open(trainval_pkl, 'wb') as f:
                pickle.dump(trainval_infos, f)
            logger.info(f'Saved {len(trainval_infos)} samples to {trainval_pkl}')

    def _parse_calib(self, calib_path):
        """Parse calibration file"""
        calib = {}
        with open(calib_path, 'r') as f:
            lines = f.readlines()

        P2_line = lines[2].strip().split()
        P2 = np.array([float(x) for x in P2_line[1:13]]).reshape(3, 4)
        P2_extended = np.zeros((4, 4))
        P2_extended[:3, :4] = P2
        P2_extended[3, 3] = 1.0
        calib['P2'] = P2_extended

        if len(lines) > 4:
            R0_line = lines[4].strip().split()
            R0 = np.array([float(x) for x in R0_line[1:10]]).reshape(3, 3)
            R0_rect = np.zeros((4, 4))
            R0_rect[:3, :3] = R0
            R0_rect[3, 3] = 1.0
        else:
            R0_rect = np.eye(4)
        calib['R0_rect'] = R0_rect
        calib['Tr_velo_to_cam'] = np.eye(4)

        return calib

    def _parse_label(self, label_path):
        """Parse KITTI format label file"""
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return self._empty_annos()

        annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            annotations.append({
                'name': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                'rotation_y': float(parts[14]),
                'score': float(parts[15]) if len(parts) > 15 else 1.0
            })

        if not annotations:
            return self._empty_annos()

        return {
            'name': np.array([a['name'] for a in annotations], dtype='<U10'),
            'truncated': np.array([a['truncated'] for a in annotations], dtype=np.float32),
            'occluded': np.array([a['occluded'] for a in annotations], dtype=np.int64),
            'alpha': np.array([a['alpha'] for a in annotations], dtype=np.float32),
            'bbox': np.array([a['bbox'] for a in annotations], dtype=np.float32),
            'dimensions': np.array([a['dimensions'] for a in annotations], dtype=np.float32),
            'location': np.array([a['location'] for a in annotations], dtype=np.float32),
            'rotation_y': np.array([a['rotation_y'] for a in annotations], dtype=np.float32),
            'score': np.array([a['score'] for a in annotations], dtype=np.float32)
        }

    def _empty_annos(self):
        """Return empty annotation structure"""
        return {
            'name': np.array([], dtype='<U10'),
            'truncated': np.array([], dtype=np.float32),
            'occluded': np.array([], dtype=np.int64),
            'alpha': np.array([], dtype=np.float32),
            'bbox': np.zeros((0, 4), dtype=np.float32),
            'dimensions': np.zeros((0, 3), dtype=np.float32),
            'location': np.zeros((0, 3), dtype=np.float32),
            'rotation_y': np.array([], dtype=np.float32),
            'score': np.array([], dtype=np.float32)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Unified Wildlife 3D Data Converter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to KITTI format
  python unified_data_converter.py --format kitti \\
    --videos-root /path/to/videos \\
    --results-root /path/to/results \\
    --output-dir /path/to/output

  # Convert to Omni3D format
  python unified_data_converter.py --format omni3d \\
    --videos-root /path/to/videos \\
    --results-root /path/to/results \\
    --output-dir /path/to/omni3d_output \\
    --omni3d-image-output /path/to/images

  # Generate wildlife info files (after KITTI conversion)
  python unified_data_converter.py --format wildlife_info \\
    --output-dir /path/to/kitti_output
        """
    )

    parser.add_argument('--format', type=str, required=True,
                       choices=['kitti', 'omni3d', 'wildlife_info'],
                       help='Output format')
    parser.add_argument('--videos-root', type=str,
                       help='Root directory containing video folders')
    parser.add_argument('--results-root', type=str,
                       help='Root directory containing results folders')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for converted data')
    parser.add_argument('--config', type=str,
                       help='Config YAML file')
    parser.add_argument('--omni3d-image-output', type=str,
                       help='Output directory for Omni3D images')
    parser.add_argument('--target-class', type=str, default='rhino',
                       help='Target class name')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = ConverterConfig.from_yaml(args.config)
    else:
        if args.format != 'wildlife_info' and (not args.videos_root or not args.results_root):
            parser.error(f"--videos-root and --results-root are required for {args.format} format")

        config = ConverterConfig(
            videos_root=args.videos_root or "",
            results_root=args.results_root or "",
            output_dir=args.output_dir,
            format=args.format,
            omni3d_image_output=args.omni3d_image_output,
            target_class=args.target_class
        )

    # Create appropriate converter
    if config.format == 'kitti':
        converter = KITTIConverter(config)
    elif config.format == 'omni3d':
        converter = Omni3DConverter(config)
    elif config.format == 'wildlife_info':
        converter = WildlifeInfoGenerator(config)
    else:
        raise ValueError(f"Unknown format: {config.format}")

    # Run conversion
    converter.convert()

    logger.info("\n" + "="*70)
    logger.info("CONVERSION COMPLETE!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
