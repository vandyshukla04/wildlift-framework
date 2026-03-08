#!/usr/bin/env python3
"""
Enhanced Semantic Face Propagation with Tracklet-Focused Viewpoint Analysis
Comprehensive per-animal viewpoint characterization for re-identification applications
"""

import os
import json
import numpy as np
import cv2
import glob
from pathlib import Path
import re
import colorsys
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy

class TrackletViewpointAnalyzer:
    def __init__(self, output_dir, image_og_dir):
        """Initialize the tracklet viewpoint analyzer"""
        self.output_dir = Path(output_dir)
        self.image_og_dir = Path(image_og_dir)
        
        # Multi-track data structures
        self.all_bbox_data = {}  # track_id -> frame_name -> bbox_data
        self.semantic_faces = {}  # track_id -> frame_name -> {semantic_label -> face_data}
        self.labeled_tracks = []  # List of track IDs user chose to label
        
        # Load all bounding box data for all tracks
        self.load_all_bboxes()
        
        # Frame ordering
        self.frame_order = self._determine_frame_order()
        
        # Predefined track colors (BGR)
        self.track_colors = {
            0: (0, 255, 255),    # Cyan
            1: (255, 0, 255),    # Magenta  
            2: (255, 255, 0),    # Yellow
            3: (0, 255, 0),      # Green
            4: (255, 128, 0),    # Orange
            5: (128, 0, 255),    # Purple
            6: (0, 128, 255),    # Sky Blue
            7: (255, 0, 128),    # Pink
            8: (128, 255, 0),    # Lime
            9: (0, 255, 128),    # Spring Green
        }
        self.arrow_thickness = 4
        
        # Semantic face colors for visualization
        self.semantic_face_colors = {
            'front': '#FF4444',   # Red
            'back': '#44FF44',    # Green
            'left': '#4444FF',    # Blue
            'right': '#FFAA44',   # Orange
            'top': '#FF44FF',     # Magenta
            'bottom': '#44FFFF',  # Cyan
        }
    
    def _determine_frame_order(self):
        """Determine the temporal order of frames"""
        all_frame_names = set()
        for track_data in self.all_bbox_data.values():
            all_frame_names.update(track_data.keys())
        
        def extract_numeric_part(frame_name):
            numbers = re.findall(r'\d+', str(frame_name))
            return int(numbers[0]) if numbers else float('inf')
        
        sorted_frames = sorted(list(all_frame_names), key=extract_numeric_part)
        print(f"Frame order: {sorted_frames[:5]}..." if len(sorted_frames) > 5 else sorted_frames)
        return sorted_frames
    
    def _resize_image(self, img, size, maintain_aspect_ratio=True):
        """Resize image to specified size"""
        if img is None:
            return None
        
        h, w = img.shape[:2]
        
        if isinstance(size, tuple):
            new_w, new_h = size
        elif isinstance(size, int):
            if maintain_aspect_ratio:
                if w > h:
                    new_w = size
                    new_h = int(h * (size / w))
                else:
                    new_h = size
                    new_w = int(w * (size / h))
            else:
                new_w = new_h = size
        else:
            raise ValueError("size must be tuple (width, height) or int")
        
        if new_w < w or new_h < h:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        
        return cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        
    def load_all_bboxes(self):
        """Load all bounding box JSON files for ALL tracks"""
        bbox_files = sorted(glob.glob(str(self.output_dir / "bounding_boxes" / "*.json")))
        
        for bbox_file in bbox_files:
            frame_name = Path(bbox_file).stem
            
            with open(bbox_file, 'r') as f:
                frame_bboxes = json.load(f)
            
            for bbox in frame_bboxes:
                track_id = bbox.get('track_id')
                if track_id is None:
                    continue
                
                if track_id not in self.all_bbox_data:
                    self.all_bbox_data[track_id] = {}
                
                self.all_bbox_data[track_id][frame_name] = {
                    'center': np.array(bbox['center']),
                    'dimensions': np.array(bbox['dimensions']),
                    'rotation_matrix': np.array(bbox['rotation_matrix']),
                    'track_id': bbox['track_id'],
                    'class_name': bbox.get('class_name', 'animal')
                }
        
        total_frames = sum(len(frames) for frames in self.all_bbox_data.values())
        print(f"Loaded bounding boxes: {len(self.all_bbox_data)} tracks, {total_frames} total frames")
    
    def discover_available_tracks(self):
        """Find all available track IDs"""
        return sorted(list(self.all_bbox_data.keys()))
    
    def select_tracks_to_label(self):
        """Interactive track selection"""
        available_tracks = self.discover_available_tracks()
        
        if not available_tracks:
            print("No tracks found!")
            return []
        
        print(f"\nAvailable tracks: {available_tracks}")
        
        while True:
            user_input = input("Enter track IDs to label (comma-separated, e.g., '3, 5, 7'): ").strip()
            
            if not user_input:
                print("No tracks selected.")
                return []
            
            try:
                selected = [int(x.strip()) for x in user_input.split(',')]
                
                invalid = [t for t in selected if t not in available_tracks]
                if invalid:
                    print(f"Invalid track IDs: {invalid}. Please try again.")
                    continue
                
                print(f"Selected tracks: {selected}")
                return selected
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers.")
    
    def get_track_color(self, track_id):
        """Get color for a track"""
        if track_id in self.track_colors:
            return self.track_colors[track_id]
        else:
            hue = (track_id * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
            return (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
    
    def get_bbox_corners(self, center, dimensions, rotation_matrix):
        """Get 8 corner points of a 3D bounding box"""
        l, w, h = dimensions
        
        corners_local = np.array([
            [-l/2, -w/2, -h/2],  # 0: bottom-back-left
            [+l/2, -w/2, -h/2],  # 1: bottom-back-right
            [+l/2, +w/2, -h/2],  # 2: bottom-front-right
            [-l/2, +w/2, -h/2],  # 3: bottom-front-left
            [-l/2, -w/2, +h/2],  # 4: top-back-left
            [+l/2, -w/2, +h/2],  # 5: top-back-right
            [+l/2, +w/2, +h/2],  # 6: top-front-right
            [-l/2, +w/2, +h/2],  # 7: top-front-left
        ])
        
        corners_world = (rotation_matrix @ corners_local.T).T + center
        return corners_world
    
    def compute_face_from_corners(self, corners, indices, box_center):
        """Compute face properties from corner indices"""
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
        
        return {
            'center': face_center,
            'normal': normal,
            'corners': face_corners,
            'area': area
        }
    
    def get_all_faces_from_bbox(self, bbox_data):
        """Get all 6 faces from bbox data"""
        corners = self.get_bbox_corners(
            bbox_data['center'], 
            bbox_data['dimensions'], 
            bbox_data['rotation_matrix']
        )
        box_center = np.mean(corners, axis=0)
        
        face_indices = {
            'f0': [0, 1, 2, 3],  # bottom
            'f1': [4, 5, 6, 7],  # top
            'f2': [0, 1, 5, 4],  # back
            'f3': [2, 3, 7, 6],  # front
            'f4': [0, 3, 7, 4],  # left
            'f5': [1, 2, 6, 5],  # right
        }
        
        faces = {}
        for face_id, indices in face_indices.items():
            faces[face_id] = self.compute_face_from_corners(corners, indices, box_center)
        
        return faces
    
    def _load_camera_params(self, frame_name):
        """Load camera parameters for projection"""
        camera_file = self.output_dir / "camera" / f"{frame_name}.npz"
        if not camera_file.exists():
            return None
            
        camera_data = np.load(camera_file)
        return {
            'K': camera_data['intrinsics'],
            'R': camera_data['pose'][:3, :3],
            't': camera_data['pose'][:3, 3]
        }
    
    def _project_face_to_2d(self, face_data, camera_params, img_shape):
        """Project a 3D face to 2D image"""
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
    
    # def _calculate_face_visibility(self, face_data, camera_params):
    #     """Calculate if a semantic face is visible to the camera"""
    #     try:
    #         # Get camera position
    #         R = camera_params['R']
    #         t = camera_params['t']
    #         camera_pose = np.eye(4)
    #         camera_pose[:3, :3] = R
    #         camera_pose[:3, 3] = t
    #         camera_pos = np.linalg.inv(camera_pose)[:3, 3]
            
    #         # Vector from face to camera
    #         face_to_camera = camera_pos - face_data['center']
    #         face_to_camera = face_to_camera / np.linalg.norm(face_to_camera)
            
    #         # Dot product with face normal (positive = facing camera)
    #         visibility_score = np.dot(face_data['normal'], face_to_camera)
            
    #         # Face is "visible" if it's somewhat facing the camera (threshold at 0)
    #         return visibility_score > 0, visibility_score
    #     except:
    #         return False, 0.0
    
    
    def _calculate_face_visibility(self, face_data, camera_params):
        """Calculate if a semantic face is visible to the camera"""
        try:
            # CORRECTED: Direct translation for camera-to-world pose convention
            t = camera_params['t']
            camera_pos = t  # No inversion needed
            
            # Vector from face to camera  
            face_to_camera = camera_pos - face_data['center']
            face_to_camera = face_to_camera / np.linalg.norm(face_to_camera)
            
            # Dot product with face normal (positive = facing camera)
            visibility_score = np.dot(face_data['normal'], face_to_camera)
            
            return visibility_score > 0, visibility_score
        except:
            return False, 0.0
        
    def _calculate_face_quality(self, face_data, camera_params, img_shape):
        """Calculate comprehensive face quality score for frame ranking"""
        try:
            # Visibility component
            is_visible, visibility_score = self._calculate_face_visibility(face_data, camera_params)
            if not is_visible:
                return 0.0
            
            # Projection quality
            corners_2d = self._project_face_to_2d(face_data, camera_params, img_shape)
            if corners_2d is None:
                return 0.0
            
            # Face area in image (larger = better for feature extraction)
            face_area_2d = cv2.contourArea(corners_2d)
            max_possible_area = img_shape[0] * img_shape[1]
            area_score = min(face_area_2d / max_possible_area, 1.0)
            
            # Distance from center (closer to center = better)
            img_center = np.array([img_shape[1]/2, img_shape[0]/2])
            face_center_2d = np.mean(corners_2d, axis=0)
            max_distance = np.sqrt(img_shape[0]**2 + img_shape[1]**2) / 2
            distance_score = 1.0 - np.linalg.norm(face_center_2d - img_center) / max_distance
            
            # Aspect ratio score (closer to square = better)
            x_span = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
            y_span = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])
            if min(x_span, y_span) > 0:
                aspect_ratio = min(x_span, y_span) / max(x_span, y_span)
            else:
                aspect_ratio = 0.0
            
            # Combined quality score
            quality_score = (
                0.4 * abs(visibility_score) +  # Face orientation toward camera
                0.3 * area_score +              # Size in image
                0.2 * distance_score +          # Central positioning
                0.1 * aspect_ratio              # Shape preservation
            )
            
            return quality_score
        except:
            return 0.0
    
    def _highlight_projected_face(self, img, corners_2d, color, thickness=3, alpha=0.3):
        """Draw face on image"""
        if corners_2d is None or len(corners_2d) != 4:
            return
            
        overlay = img.copy()
        pts = corners_2d.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        
        for corner in corners_2d:
            cv2.circle(img, tuple(corner), 6, color, -1)
    
    def manual_face_labeling(self, track_id, frame_name=None):
        """Interactive face labeling for a specific track"""
        bbox_data = self.all_bbox_data[track_id]
        
        if frame_name is None:
            track_frames = sorted(bbox_data.keys(), key=lambda x: int(re.findall(r'\d+', x)[0]))
            frame_name = track_frames[0] if track_frames else None
            if not frame_name:
                return False
        
        print(f"\nSEMANTIC FACE LABELING for Track {track_id}, Frame {frame_name}")
        
        img_path = self.output_dir / "annotated_2d" / f"{frame_name}_tracked.png"
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return False
            
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        
        bbox = bbox_data[frame_name]
        all_faces = self.get_all_faces_from_bbox(bbox)
        
        camera_params = self._load_camera_params(frame_name)
        if not camera_params:
            print("Cannot load camera parameters")
            return False
        
        projected_faces = {}
        face_colors = [(255,100,100), (100,255,100), (100,100,255), 
                      (255,255,100), (255,100,255), (100,255,255)]
        
        for i, (face_id, face_data) in enumerate(all_faces.items()):
            corners_2d = self._project_face_to_2d(face_data, camera_params, img.shape)
            if corners_2d is not None:
                projected_faces[face_id] = {
                    'data': face_data,
                    'corners_2d': corners_2d,
                    'color': face_colors[i]
                }
        
        if not projected_faces:
            print("No faces could be projected")
            return False
        
        face_dir = self.output_dir / "face_selection"
        face_dir.mkdir(exist_ok=True)
        
        print("\n'f' = FRONT | 'l' = LEFT | 't' = TOP | 'n' = SKIP | 'q' = QUIT")
        
        semantic_assignments = {}
        
        for i, (face_id, proj_face) in enumerate(projected_faces.items()):
            display_img = img.copy()
            self._highlight_projected_face(display_img, proj_face['corners_2d'], 
                                          proj_face['color'], thickness=4, alpha=0.4)
            
            face_img_path = face_dir / f"track{track_id}_face_{i+1:02d}_{frame_name}.png"
            cv2.imwrite(str(face_img_path), display_img)
            
            print(f"\nFace {i+1}/{len(projected_faces)}")
            print(f"Saved: {face_img_path}")
            
            while True:
                user_input = input("Label this face: ").lower().strip()
                
                if user_input == 'q':
                    return False
                elif user_input == 'f' and 'front' not in semantic_assignments:
                    semantic_assignments['front'] = proj_face['data']
                    print("Labeled as FRONT")
                    break
                elif user_input == 'l' and 'left' not in semantic_assignments:
                    semantic_assignments['left'] = proj_face['data']
                    print("Labeled as LEFT")
                    break
                elif user_input == 't' and 'top' not in semantic_assignments:
                    semantic_assignments['top'] = proj_face['data']
                    print("Labeled as TOP")
                    break
                elif user_input == 'n':
                    print("Skipped")
                    break
                else:
                    print("Invalid input or label already assigned")
        
        if not all(k in semantic_assignments for k in ['front', 'left', 'top']):
            print("Missing required labels")
            return False
        
        self._infer_opposite_faces(semantic_assignments, all_faces)
        
        if track_id not in self.semantic_faces:
            self.semantic_faces[track_id] = {}
        self.semantic_faces[track_id][frame_name] = semantic_assignments
        
        print(f"\nLabeling complete for Track {track_id}, Frame {frame_name}")
        for label in semantic_assignments:
            print(f"  {label}: stored")
        
        return True
    
    def _infer_opposite_faces(self, semantic_assignments, all_faces):
        """Infer opposite faces based on normal similarity"""
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
                
                best_match = None
                best_score = 999
                
                for face_id, test_face in unassigned_faces.items():
                    dot = np.dot(face_data['normal'], test_face['normal'])
                    if dot < best_score:
                        best_score = dot
                        best_match = test_face
                
                if best_match is not None and best_score < -0.8:
                    semantic_assignments[opposite_label] = best_match
                    for fid in list(unassigned_faces.keys()):
                        if np.allclose(best_match['center'], unassigned_faces[fid]['center']):
                            del unassigned_faces[fid]
                            break
    
    def propagate_semantics_temporal(self, track_id):
        """Propagate semantic labels through time for a specific track"""
        if track_id not in self.semantic_faces or not self.semantic_faces[track_id]:
            print(f"No reference frame labeled for track {track_id}")
            return
        
        reference_frame = list(self.semantic_faces[track_id].keys())[0]
        print(f"\nPropagating from frame {reference_frame} for track {track_id}")
        
        bbox_data = self.all_bbox_data[track_id]
        track_frames = sorted(bbox_data.keys(), key=lambda x: int(re.findall(r'\d+', x)[0]))
        
        successful = 0
        
        for i, frame_name in enumerate(track_frames):
            if frame_name == reference_frame:
                continue
            
            prev_frame = None
            for j in range(i-1, -1, -1):
                if track_frames[j] in self.semantic_faces[track_id]:
                    prev_frame = track_frames[j]
                    break
            
            if not prev_frame:
                continue
            
            print(f"Frame {frame_name} (from {prev_frame})")
            
            current_bbox = bbox_data[frame_name]
            current_faces = self.get_all_faces_from_bbox(current_bbox)
            
            prev_semantic_faces = self.semantic_faces[track_id][prev_frame]
            
            semantic_mapping = self._match_faces_temporal(
                prev_semantic_faces, 
                current_faces,
                frame_name
            )
            
            if semantic_mapping:
                self.semantic_faces[track_id][frame_name] = semantic_mapping
                successful += 1
                print(f"  Successfully labeled {len(semantic_mapping)} faces")
            else:
                print(f"  Failed to match faces")
        
        print(f"\nPropagated to {successful}/{len(track_frames)-1} frames for track {track_id}")
    
    def _match_faces_temporal(self, prev_semantic_faces, current_faces, frame_name):
        """Match semantic faces from previous frame to current frame faces"""
        primary_labels = ['front', 'left', 'top']
        semantic_mapping = {}
        used_faces = set()
        
        for semantic_label in primary_labels:
            if semantic_label not in prev_semantic_faces:
                continue
                
            prev_face = prev_semantic_faces[semantic_label]
            best_match = None
            best_similarity = -1
            best_face_id = None
            
            for face_id, curr_face in current_faces.items():
                if face_id in used_faces:
                    continue
                
                similarity = self._compute_face_similarity(prev_face, curr_face)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = curr_face
                    best_face_id = face_id
            
            if best_similarity > 0.3:
                semantic_mapping[semantic_label] = best_match
                used_faces.add(best_face_id)
                print(f"    {semantic_label} matched (similarity: {best_similarity:.3f})")
        
        if len(semantic_mapping) == 3:
            unmatched_faces = {fid: face for fid, face in current_faces.items() 
                             if fid not in used_faces}
            self._infer_opposite_faces(semantic_mapping, current_faces)
        
        return semantic_mapping if len(semantic_mapping) >= 3 else None
    
    def _compute_face_similarity(self, face1, face2):
        """Compute similarity between two faces"""
        dist = np.linalg.norm(face1['center'] - face2['center'])
        dist_sim = max(0, 1 - dist / 2.0)
        
        normal_sim = abs(np.dot(face1['normal'], face2['normal']))
        
        if max(face1['area'], face2['area']) > 0:
            area_sim = min(face1['area'], face2['area']) / max(face1['area'], face2['area'])
        else:
            area_sim = 0
        
        return 0.5 * dist_sim + 0.4 * normal_sim + 0.1 * area_sim
    
    # ========================================================================
    # ENHANCED: TRACKLET-FOCUSED VIEWPOINT CHARACTERIZATION
    # ========================================================================
    
    def compute_tracklet_viewpoint_profiles(self):
        """Comprehensive per-tracklet viewpoint characterization"""
        print("\n" + "="*70)
        print("COMPUTING TRACKLET VIEWPOINT PROFILES")
        print("="*70)
        
        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        visible_labels = ['front', 'back', 'left', 'right', 'top']  # Bottom N/A for drone footage
        tracklet_profiles = {}
        
        for track_id in self.labeled_tracks:
            if track_id not in self.semantic_faces:
                continue
            
            print(f"\nAnalyzing Track {track_id}...")
            
            # Initialize data structures
            visibility_matrix = {label: [] for label in semantic_labels}
            quality_matrix = {label: [] for label in semantic_labels}
            frame_quality_scores = {}
            
            # Analyze each frame
            for frame_name in sorted(self.semantic_faces[track_id].keys(), 
                                    key=lambda x: int(re.findall(r'\d+', x)[0])):
                camera_params = self._load_camera_params(frame_name)
                if not camera_params:
                    continue
                
                # Load image for quality assessment
                img_path = self.image_og_dir / f"{frame_name}.jpg"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img_shape = img.shape if img is not None else (480, 640, 3)
                else:
                    img_shape = (480, 640, 3)
                
                semantic_faces = self.semantic_faces[track_id][frame_name]
                frame_qualities = {}
                
                for label in semantic_labels:
                    if label == 'bottom':
                        # Bottom is N/A for drone footage
                        visibility_matrix[label].append(0)
                        quality_matrix[label].append(0.0)
                        frame_qualities[label] = 0.0
                    elif label in semantic_faces:
                        # Visibility analysis
                        is_visible, visibility_score = self._calculate_face_visibility(
                            semantic_faces[label], camera_params
                        )
                        visibility_matrix[label].append(1 if is_visible else 0)
                        
                        # Quality analysis
                        quality_score = self._calculate_face_quality(
                            semantic_faces[label], camera_params, img_shape
                        )
                        quality_matrix[label].append(quality_score)
                        frame_qualities[label] = quality_score
                    else:
                        visibility_matrix[label].append(0)
                        quality_matrix[label].append(0.0)
                        frame_qualities[label] = 0.0
                
                frame_quality_scores[frame_name] = frame_qualities
            
            # Compute viewpoint coverage vector (only for visible labels)
            total_frames = len(self.semantic_faces[track_id])
            coverage_vector = {
                label: sum(visibility_matrix[label]) / total_frames if total_frames > 0 else 0.0
                for label in semantic_labels
            }
            # Mark bottom as N/A
            coverage_vector['bottom'] = float('nan')
            
            # Compute viewpoint diversity index (entropy) - only for visible labels
            coverage_values = [coverage_vector[label] for label in visible_labels]
            # Add small epsilon to avoid log(0)
            coverage_values = [max(val, 1e-10) for val in coverage_values]
            coverage_sum = sum(coverage_values)
            if coverage_sum > 0:
                normalized_coverage = [val / coverage_sum for val in coverage_values]
                diversity_index = entropy(normalized_coverage)
                max_entropy = entropy([1/len(visible_labels)] * len(visible_labels))  # Max entropy for 5 visible faces
                normalized_diversity = diversity_index / max_entropy
            else:
                diversity_index = 0.0
                normalized_diversity = 0.0
            
            # Compute coverage completeness score (only for visible labels)
            visible_coverage = [coverage_vector[label] for label in visible_labels]
            min_coverage = min(visible_coverage)
            coverage_threshold = 0.1  # Minimum acceptable coverage per orientation
            completeness_score = max(0.0, min(1.0, min_coverage / coverage_threshold))
            
            # Find optimal exemplar frames per orientation
            optimal_frames = {}
            for label in semantic_labels:
                if label == 'bottom':
                    optimal_frames[label] = {'frame': None, 'quality_score': float('nan')}
                elif any(q > 0 for q in quality_matrix[label]):
                    best_quality = max(quality_matrix[label])
                    best_frame_idx = quality_matrix[label].index(best_quality)
                    frame_names = sorted(self.semantic_faces[track_id].keys(), 
                                        key=lambda x: int(re.findall(r'\d+', x)[0]))
                    if best_frame_idx < len(frame_names):
                        optimal_frames[label] = {
                            'frame': frame_names[best_frame_idx],
                            'quality_score': best_quality
                        }
                else:
                    optimal_frames[label] = {'frame': None, 'quality_score': 0.0}
            
            # Compute re-identification readiness score (only considering visible labels)
            visible_avg_qualities = []
            for label in visible_labels:
                if any(q > 0 for q in quality_matrix[label]):
                    visible_avg_qualities.append(np.mean([q for q in quality_matrix[label] if q > 0]))
                else:
                    visible_avg_qualities.append(0.0)
            
            reid_score = (
                0.4 * normalized_diversity +      # Viewpoint diversity
                0.3 * completeness_score +        # Coverage completeness
                0.3 * np.mean(visible_avg_qualities)  # Average quality
            )
            
            # Store comprehensive profile
            tracklet_profiles[track_id] = {
                'coverage_vector': coverage_vector,
                'diversity_index': diversity_index,
                'normalized_diversity': normalized_diversity,
                'completeness_score': completeness_score,
                'reid_readiness_score': reid_score,
                'total_frames': total_frames,
                'visible_frames_per_orientation': {
                    label: sum(visibility_matrix[label]) if label != 'bottom' else 0
                    for label in semantic_labels
                },
                'average_quality_per_orientation': {
                    label: (np.mean([q for q in quality_matrix[label] if q > 0]) 
                        if label != 'bottom' and any(q > 0 for q in quality_matrix[label]) 
                        else (float('nan') if label == 'bottom' else 0.0))
                    for label in semantic_labels
                },
                'optimal_exemplar_frames': optimal_frames,
                'frame_quality_scores': frame_quality_scores,
                'coverage_gaps': [label for label in visible_labels  # Only check visible labels
                                if coverage_vector[label] < coverage_threshold],
                'bottom_note': 'N/A - not visible in drone footage'
            }
            
            visible_coverage_str = [f'{coverage_vector[l]:.2f}' for l in visible_labels]
            print(f"  Coverage Vector (visible): {visible_coverage_str}")
            print(f"  (Bottom: N/A - drone footage)")
            print(f"  Diversity Index: {normalized_diversity:.3f}")
            print(f"  Completeness Score: {completeness_score:.3f}")
            print(f"  Re-ID Readiness Score: {reid_score:.3f}")
            if tracklet_profiles[track_id]['coverage_gaps']:
                print(f"  Coverage Gaps: {tracklet_profiles[track_id]['coverage_gaps']}")
        
        return tracklet_profiles

    def create_tracklet_viewpoint_dashboard(self):
        """Create comprehensive tracklet-focused viewpoint analysis dashboard"""
        print("\n" + "="*70)
        print("GENERATING TRACKLET VIEWPOINT CHARACTERIZATION DASHBOARD")
        print("="*70)
        
        viz_dir = self.output_dir / "tracklet_viewpoint_analysis"
        viz_dir.mkdir(exist_ok=True)
        
        tracklet_profiles = self.compute_tracklet_viewpoint_profiles()
        
        if not tracklet_profiles:
            print("No tracklet profiles available. Run semantic propagation first.")
            return
        
        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        track_ids = sorted(tracklet_profiles.keys())
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Per-tracklet coverage vectors (radar chart)
        ax1 = plt.subplot(2, 4, 1, projection='polar')
        angles = np.linspace(0, 2*np.pi, len(semantic_labels), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, track_id in enumerate(track_ids):
            values = [tracklet_profiles[track_id]['coverage_vector'][label] for label in semantic_labels]
            values += values[:1]  # Complete the circle
            
            color = plt.cm.Set3(i / len(track_ids))
            ax1.plot(angles, values, 'o-', linewidth=2, label=f'Track {track_id}', color=color)
            ax1.fill(angles, values, alpha=0.25, color=color)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([label.capitalize() for label in semantic_labels])
        ax1.set_ylim(0, 1)
        ax1.set_title('Coverage Vectors per Tracklet', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax1.grid(True)
        
        # 2. Viewpoint diversity comparison
        ax2 = plt.subplot(2, 4, 2)
        diversity_scores = [tracklet_profiles[tid]['normalized_diversity'] for tid in track_ids]
        colors = [plt.cm.viridis(score) for score in diversity_scores]
        
        bars = ax2.bar(range(len(track_ids)), diversity_scores, color=colors)
        ax2.set_xlabel('Track ID', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Normalized Diversity Index', fontsize=12, fontweight='bold')
        ax2.set_title('Viewpoint Diversity per Tracklet', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(track_ids)))
        ax2.set_xticklabels([f'T{tid}' for tid in track_ids])
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars, diversity_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. Coverage completeness scores
        ax3 = plt.subplot(2, 4, 3)
        completeness_scores = [tracklet_profiles[tid]['completeness_score'] for tid in track_ids]
        reid_scores = [tracklet_profiles[tid]['reid_readiness_score'] for tid in track_ids]
        
        x = np.arange(len(track_ids))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, completeness_scores, width, label='Coverage Completeness', 
                       color='lightcoral', alpha=0.8)
        bars2 = ax3.bar(x + width/2, reid_scores, width, label='Re-ID Readiness', 
                       color='lightblue', alpha=0.8)
        
        ax3.set_xlabel('Track ID', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax3.set_title('Completeness & Re-ID Readiness', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'T{tid}' for tid in track_ids])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Coverage gaps analysis
        ax4 = plt.subplot(2, 4, 4)
        gap_matrix = np.zeros((len(track_ids), len(semantic_labels)))
        
        for i, track_id in enumerate(track_ids):
            gaps = tracklet_profiles[track_id]['coverage_gaps']
            for j, label in enumerate(semantic_labels):
                gap_matrix[i, j] = 1 if label in gaps else 0
        
        im = ax4.imshow(gap_matrix, cmap='Reds', aspect='auto')
        ax4.set_xlabel('Semantic Face', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Track ID', fontsize=12, fontweight='bold')
        ax4.set_title('Coverage Gaps (Red = Gap)', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(semantic_labels)))
        ax4.set_xticklabels([l.capitalize() for l in semantic_labels], rotation=45)
        ax4.set_yticks(range(len(track_ids)))
        ax4.set_yticklabels([f'T{tid}' for tid in track_ids])
        
        # 5. Quality heatmap per orientation
        ax5 = plt.subplot(2, 4, 5)
        quality_matrix = np.array([[tracklet_profiles[tid]['average_quality_per_orientation'][label] 
                                   for label in semantic_labels] for tid in track_ids])
        
        im2 = ax5.imshow(quality_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax5.set_xlabel('Semantic Face', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Track ID', fontsize=12, fontweight='bold')
        ax5.set_title('Average Quality per Orientation', fontsize=14, fontweight='bold')
        ax5.set_xticks(range(len(semantic_labels)))
        ax5.set_xticklabels([l.capitalize() for l in semantic_labels], rotation=45)
        ax5.set_yticks(range(len(track_ids)))
        ax5.set_yticklabels([f'T{tid}' for tid in track_ids])
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax5)
        cbar2.set_label('Quality Score', fontsize=10, fontweight='bold')
        
        # 6. Frame count distribution
        ax6 = plt.subplot(2, 4, 6)
        frame_counts = [tracklet_profiles[tid]['total_frames'] for tid in track_ids]
        
        bars3 = ax6.bar(range(len(track_ids)), frame_counts, color='lightgreen', alpha=0.8)
        ax6.set_xlabel('Track ID', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Total Frames', fontsize=12, fontweight='bold')
        ax6.set_title('Frame Count per Tracklet', fontsize=14, fontweight='bold')
        ax6.set_xticks(range(len(track_ids)))
        ax6.set_xticklabels([f'T{tid}' for tid in track_ids])
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars3, frame_counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 7. Quality distribution violin plot
        ax7 = plt.subplot(2, 4, 7)
        
        quality_data = []
        positions = []
        labels_plot = []
        
        for i, label in enumerate(semantic_labels):
            orientation_qualities = []
            for track_id in track_ids:
                avg_quality = tracklet_profiles[track_id]['average_quality_per_orientation'][label]
                if avg_quality > 0:
                    orientation_qualities.append(avg_quality)
            
            if orientation_qualities:
                quality_data.append(orientation_qualities)
                positions.append(i)
                labels_plot.append(label.capitalize())
        
        if quality_data:
            parts = ax7.violinplot(quality_data, positions=positions, widths=0.7,
                                  showmeans=True, showmedians=True)
            
            colors = [self.semantic_face_colors[semantic_labels[pos]] for pos in positions]
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
        
        ax7.set_xlabel('Semantic Face', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
        ax7.set_title('Quality Distribution per Orientation', fontsize=14, fontweight='bold')
        ax7.set_xticks(positions)
        ax7.set_xticklabels(labels_plot, rotation=45)
        ax7.grid(axis='y', alpha=0.3)
        ax7.set_ylim(0, 1)
        
        # 8. Summary statistics
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        summary_text = [
            "TRACKLET VIEWPOINT ANALYSIS SUMMARY",
            "="*45,
            f"Total Analyzed Tracklets: {len(tracklet_profiles)}",
            "",
            "DIVERSITY RANKING (Normalized):",
            "-"*45,
        ]
        
        # Sort tracklets by diversity
        sorted_tracks_diversity = sorted(track_ids, 
                                       key=lambda x: tracklet_profiles[x]['normalized_diversity'], 
                                       reverse=True)
        
        for rank, track_id in enumerate(sorted_tracks_diversity, 1):
            diversity = tracklet_profiles[track_id]['normalized_diversity']
            summary_text.append(f"{rank}. Track {track_id}: {diversity:.3f}")
        
        summary_text.extend([
            "",
            "RE-ID READINESS RANKING:",
            "-"*45,
        ])
        
        # Sort tracklets by re-ID readiness
        sorted_tracks_reid = sorted(track_ids, 
                                  key=lambda x: tracklet_profiles[x]['reid_readiness_score'], 
                                  reverse=True)
        
        for rank, track_id in enumerate(sorted_tracks_reid, 1):
            reid_score = tracklet_profiles[track_id]['reid_readiness_score']
            summary_text.append(f"{rank}. Track {track_id}: {reid_score:.3f}")
        
        summary_text.extend([
            "",
            "COVERAGE ISSUES:",
            "-"*45,
        ])
        
        for track_id in track_ids:
            gaps = tracklet_profiles[track_id]['coverage_gaps']
            if gaps:
                summary_text.append(f"Track {track_id}: Missing {', '.join(gaps)}")
            else:
                summary_text.append(f"Track {track_id}: Complete coverage")
        
        text_str = '\n'.join(summary_text)
        ax8.text(0.05, 0.95, text_str, transform=ax8.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout(pad=2.0)
        
        # Save dashboard
        output_path = viz_dir / "tracklet_viewpoint_dashboard.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved tracklet viewpoint dashboard to: {output_path}")
        plt.close()
        
        # Save detailed analysis
        self._save_tracklet_analysis_json(tracklet_profiles, viz_dir)
        self._generate_optimal_frame_report(tracklet_profiles, viz_dir)
        self.create_temporal_viewpoint_visualization(tracklet_profiles)
        return tracklet_profiles
    
    def _save_tracklet_analysis_json(self, tracklet_profiles, viz_dir):
        """Save comprehensive tracklet analysis to JSON"""
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_data = convert_numpy_types(tracklet_profiles)
        
        output_path = viz_dir / "tracklet_viewpoint_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved detailed tracklet analysis to: {output_path}")
    
    def _generate_optimal_frame_report(self, tracklet_profiles, viz_dir):
        """Generate report of optimal frames for each tracklet and orientation"""
        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        report_lines = [
            "OPTIMAL EXEMPLAR FRAMES REPORT",
            "="*60,
            "",
            "This report identifies the best frame for each orientation",
            "of each tracklet based on visibility quality, face area,",
            "central positioning, and shape preservation.",
            "",
        ]
        
        for track_id in sorted(tracklet_profiles.keys()):
            profile = tracklet_profiles[track_id]
            report_lines.extend([
                f"TRACK {track_id}:",
                "-"*20,
                f"Diversity Index: {profile['normalized_diversity']:.3f}",
                f"Re-ID Readiness: {profile['reid_readiness_score']:.3f}",
                f"Total Frames: {profile['total_frames']}",
                "",
                "Optimal Frames by Orientation:",
            ])
            
            for label in semantic_labels:
                exemplar = profile['optimal_exemplar_frames'][label]
                if exemplar['frame'] is not None:
                    report_lines.append(
                        f"  {label.capitalize():<8}: Frame {exemplar['frame']} "
                        f"(Quality: {exemplar['quality_score']:.3f})"
                    )
                else:
                    report_lines.append(f"  {label.capitalize():<8}: No good frames available")
            
            if profile['coverage_gaps']:
                report_lines.extend([
                    "",
                    f"Coverage Gaps: {', '.join(profile['coverage_gaps'])}",
                ])
            
            report_lines.extend(["", ""])
        
        # Summary statistics
        report_lines.extend([
            "DATASET SUMMARY:",
            "="*30,
            "",
            f"Total Tracklets: {len(tracklet_profiles)}",
            "",
            "Average Scores:",
        ])
        
        avg_diversity = np.mean([p['normalized_diversity'] for p in tracklet_profiles.values()])
        avg_reid = np.mean([p['reid_readiness_score'] for p in tracklet_profiles.values()])
        avg_completeness = np.mean([p['completeness_score'] for p in tracklet_profiles.values()])
        
        report_lines.extend([
            f"  Diversity Index: {avg_diversity:.3f}",
            f"  Re-ID Readiness: {avg_reid:.3f}",
            f"  Coverage Completeness: {avg_completeness:.3f}",
            "",
        ])
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "="*20,
            "",
        ])
        
        # Find tracklets with low diversity
        low_diversity = [tid for tid, p in tracklet_profiles.items() 
                        if p['normalized_diversity'] < 0.5]
        if low_diversity:
            report_lines.append(f"• Collect more diverse viewpoints for tracks: {', '.join(map(str, low_diversity))}")
        
        # Find common coverage gaps
        all_gaps = []
        for p in tracklet_profiles.values():
            all_gaps.extend(p['coverage_gaps'])
        
        if all_gaps:
            from collections import Counter
            gap_counts = Counter(all_gaps)
            common_gaps = [gap for gap, count in gap_counts.items() if count >= len(tracklet_profiles) // 2]
            if common_gaps:
                report_lines.append(f"• Systematic coverage gaps detected: {', '.join(common_gaps)}")
        
        # Save report
        output_path = viz_dir / "optimal_frames_report.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved optimal frames report to: {output_path}")
    
    def print_tracklet_analysis_summary(self):
        """Print concise tracklet analysis summary"""
        tracklet_profiles = self.compute_tracklet_viewpoint_profiles()
        
        if not tracklet_profiles:
            print("No tracklet profiles available.")
            return
        
        print("\n" + "="*70)
        print("TRACKLET VIEWPOINT CHARACTERIZATION SUMMARY")
        print("="*70)
        
        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        print(f"\nANALYZED TRACKLETS: {len(tracklet_profiles)}")
        
        for track_id in sorted(tracklet_profiles.keys()):
            profile = tracklet_profiles[track_id]
            print(f"\n  TRACK {track_id}:")
            print(f"    Frames: {profile['total_frames']}")
            coverage_values = [f'{profile["coverage_vector"][l]:.2f}' for l in semantic_labels]
            print(f"    Coverage: {coverage_values}")
            print(f"    Diversity Index: {profile['normalized_diversity']:.3f}")
            print(f"    Re-ID Readiness: {profile['reid_readiness_score']:.3f}")
            if profile['coverage_gaps']:
                print(f"    Coverage Gaps: {', '.join(profile['coverage_gaps'])}")
        
        # Dataset-level statistics
        avg_diversity = np.mean([p['normalized_diversity'] for p in tracklet_profiles.values()])
        avg_reid = np.mean([p['reid_readiness_score'] for p in tracklet_profiles.values()])
        
        print(f"\nDATASET AVERAGES:")
        print(f"  Diversity Index: {avg_diversity:.3f}")
        print(f"  Re-ID Readiness: {avg_reid:.3f}")
        
        print("\n" + "="*70)
    
    def save_manual_labels_for_viewpoint_analyzer(self):
        """Save semantic face labels as manual_labels.json for viewpoint_analyzer_v7.py.

        Converts the in-memory semantic_faces (which store face geometry) to face
        indices using the viewpoint_analyzer_v7 face ordering convention:
            f0: [0,1,5,4]  f1: [2,3,7,6]  f2: [0,3,7,4]
            f3: [1,2,6,5]  f4: [4,5,6,7]  f5: [0,1,2,3]
        """
        # viewpoint_analyzer_v7's face index definition
        va_face_indices = {
            'f0': [0, 1, 5, 4], 'f1': [2, 3, 7, 6], 'f2': [0, 3, 7, 4],
            'f3': [1, 2, 6, 5], 'f4': [4, 5, 6, 7], 'f5': [0, 1, 2, 3],
        }

        manual_labels = {}

        for track_id, frames_data in self.semantic_faces.items():
            track_key = str(track_id)
            manual_labels[track_key] = {}

            for frame_name, semantic_assignments in frames_data.items():
                bbox_data = self.all_bbox_data[track_id].get(frame_name)
                if bbox_data is None:
                    continue

                # Recompute faces using viewpoint_analyzer's ordering
                corners = self.get_bbox_corners(
                    bbox_data['center'], bbox_data['dimensions'], bbox_data['rotation_matrix']
                )
                box_center = np.mean(corners, axis=0)
                va_faces = {}
                for face_id, indices in va_face_indices.items():
                    va_faces[face_id] = self.compute_face_from_corners(corners, indices, box_center)

                frame_labels = {}
                for semantic_label, face_data in semantic_assignments.items():
                    # Find which viewpoint_analyzer face index matches this face
                    best_face_idx = None
                    best_dist = float('inf')
                    for face_id, va_face in va_faces.items():
                        dist = np.linalg.norm(face_data['center'] - va_face['center'])
                        if dist < best_dist:
                            best_dist = dist
                            best_face_idx = int(face_id[1])  # 'f3' -> 3

                    if best_face_idx is not None and best_dist < 1e-3:
                        frame_labels[semantic_label] = best_face_idx

                if frame_labels:
                    manual_labels[track_key][frame_name] = frame_labels

        # Save to corrected_labels/semantic_faces/manual_labels.json
        output_dir = self.output_dir / "corrected_labels" / "semantic_faces"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "manual_labels.json"

        with open(output_path, 'w') as f:
            json.dump(manual_labels, f, indent=2)

        total_frames = sum(len(frames) for frames in manual_labels.values())
        print(f"\nSaved manual_labels.json for viewpoint_analyzer_v7:")
        print(f"  Path: {output_path}")
        print(f"  Tracks: {len(manual_labels)}, Frames: {total_frames}")
        return output_path

    def run_tracklet_viewpoint_pipeline(self):
        """Run the complete tracklet viewpoint characterization pipeline"""
        print("\n" + "="*70)
        print("TRACKLET VIEWPOINT CHARACTERIZATION PIPELINE")
        print("="*70)

        # Step 1: Track selection
        print("\nStep 1: Track selection...")
        selected_tracks = self.select_tracks_to_label()

        if not selected_tracks:
            print("No tracks selected. Exiting.")
            return False

        self.labeled_tracks = selected_tracks

        # Step 2: Label each track
        for track_id in selected_tracks:
            print(f"\n{'='*70}")
            print(f"LABELING TRACK {track_id}")
            print(f"{'='*70}")

            print(f"\nStep 2.{track_id}.1: Manual labeling...")
            if not self.manual_face_labeling(track_id):
                print(f"Failed to label track {track_id}. Skipping.")
                continue

            print(f"\nStep 2.{track_id}.2: Propagating...")
            self.propagate_semantics_temporal(track_id)

        # Step 3: Save manual_labels.json for viewpoint_analyzer_v7
        print("\n" + "="*70)
        print("Step 3: Saving manual labels...")
        self.save_manual_labels_for_viewpoint_analyzer()

        # Step 4: Tracklet viewpoint characterization
        print("\n" + "="*70)
        print("Step 4: Tracklet viewpoint characterization...")
        self.print_tracklet_analysis_summary()
        self.create_tracklet_viewpoint_dashboard()

        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        return True

    def create_temporal_viewpoint_visualization(self, tracklet_profiles=None):
        """Create temporal viewpoint distribution visualizations - one per tracklet plus summary"""
        print("\n" + "="*70)
        print("GENERATING TEMPORAL VIEWPOINT ANALYSIS")
        print("="*70)
        
        viz_dir = self.output_dir / "tracklet_viewpoint_analysis"
        viz_dir.mkdir(exist_ok=True)
        
        if tracklet_profiles is None:
            tracklet_profiles = self.compute_tracklet_viewpoint_profiles()
        
        if not tracklet_profiles:
            print("No tracklet profiles available.")
            return
        
        track_ids = sorted(tracklet_profiles.keys())
        
        # Create individual temporal analysis for each tracklet
        for track_id in track_ids:
            self._create_single_tracklet_temporal_viz(track_id, tracklet_profiles[track_id], viz_dir)
        
        # Create a compact summary overview comparing all tracklets
        if len(track_ids) > 1:
            self._create_temporal_summary_overview(tracklet_profiles, viz_dir)
        
        print(f"\nGenerated {len(track_ids)} individual temporal visualizations")
        return viz_dir


    def _create_single_tracklet_temporal_viz(self, track_id, profile, viz_dir):
        """Create comprehensive temporal visualization for a single tracklet"""
        
        if track_id not in self.semantic_faces:
            return
        
        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        visible_labels = ['front', 'back', 'left', 'right', 'top']
        
        # Color map - bottom grayed out
        face_cmap = {
            'front': '#E74C3C', 'back': '#2ECC71', 'left': '#3498DB',
            'right': '#F39C12', 'top': '#9B59B6', 'bottom': '#CCCCCC',
        }
        
        frame_quality_scores = profile['frame_quality_scores']
        sorted_frames = sorted(frame_quality_scores.keys(), 
                            key=lambda x: int(re.findall(r'\d+', x)[0]))
        n_frames = len(sorted_frames)
        
        if n_frames == 0:
            return
        
        frame_numbers = [int(re.findall(r'\d+', f)[0]) for f in sorted_frames]
        
        # Build quality matrix
        quality_matrix = np.zeros((len(semantic_labels), n_frames))
        
        for frame_idx, frame_name in enumerate(sorted_frames):
            camera_params = self._load_camera_params(frame_name)
            if camera_params is None:
                continue
            
            semantic_faces_frame = self.semantic_faces[track_id].get(frame_name, {})
            
            for label_idx, label in enumerate(semantic_labels):
                if label == 'bottom':
                    quality_matrix[label_idx, frame_idx] = np.nan
                elif label in semantic_faces_frame:
                    quality_matrix[label_idx, frame_idx] = frame_quality_scores[frame_name].get(label, 0)
        
        # Create figure
        fig = plt.figure(figsize=(18, 14))
        
        # Title
        fig.suptitle(f'Track {track_id} - Temporal Viewpoint Analysis\n({n_frames} frames)', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Layout: 3 rows, with different configurations
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 0.5, 1], 
                            hspace=0.35, wspace=0.3,
                            left=0.08, right=0.92, top=0.90, bottom=0.08)
        
        # ===== Panel 1: Quality Heatmap (top, spans 2 columns) =====
        ax_heatmap = fig.add_subplot(gs[0, :2])
        
        quality_matrix_masked = np.ma.array(quality_matrix, mask=np.isnan(quality_matrix))
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color='#E0E0E0')
        
        im = ax_heatmap.imshow(quality_matrix_masked, aspect='auto', cmap=cmap, 
                            vmin=0, vmax=1, interpolation='nearest')
        
        # Add hatching for bottom row
        bottom_idx = semantic_labels.index('bottom')
        ax_heatmap.add_patch(plt.Rectangle((-0.5, bottom_idx - 0.5), n_frames, 1, 
                                        fill=True, facecolor='#E0E0E0', 
                                        edgecolor='gray', linewidth=1,
                                        hatch='///', alpha=0.7, zorder=2))
        
        # Y-axis labels
        y_labels = [f'{l.capitalize()}' if l != 'bottom' else 'Bottom (N/A)' for l in semantic_labels]
        ax_heatmap.set_yticks(range(len(semantic_labels)))
        ax_heatmap.set_yticklabels(y_labels, fontsize=11)
        ytick_labels = ax_heatmap.get_yticklabels()
        ytick_labels[-1].set_color('gray')
        ytick_labels[-1].set_fontstyle('italic')
        
        # X-axis
        tick_step = max(1, n_frames // 12)
        tick_positions = list(range(0, n_frames, tick_step))
        ax_heatmap.set_xticks(tick_positions)
        ax_heatmap.set_xticklabels([str(frame_numbers[i]) for i in tick_positions], fontsize=9)
        ax_heatmap.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax_heatmap.set_ylabel('Face', fontsize=12, fontweight='bold')
        ax_heatmap.set_title('Face Quality Over Time', fontsize=14, fontweight='bold', pad=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8, pad=0.02)
        cbar.set_label('Quality Score', fontsize=11)
        
        # ===== Panel 2: Stats Summary (top right) =====
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis('off')
        
        stats_text = [
            f"TRACK {track_id} STATISTICS",
            "─" * 25,
            f"Total Frames: {n_frames}",
            f"",
            f"Diversity Index: {profile['normalized_diversity']:.3f}",
            f"Completeness: {profile['completeness_score']:.3f}",
            f"Re-ID Readiness: {profile['reid_readiness_score']:.3f}",
            f"",
            "Avg Quality per Face:",
        ]
        
        for label in visible_labels:
            avg_q = profile['average_quality_per_orientation'][label]
            if not np.isnan(avg_q):
                stats_text.append(f"  {label.capitalize():<8}: {avg_q:.3f}")
            else:
                stats_text.append(f"  {label.capitalize():<8}: N/A")
        
        stats_text.append(f"  {'Bottom':<8}: N/A (drone)")
        
        if profile['coverage_gaps']:
            stats_text.extend(["", f"Coverage Gaps:", f"  {', '.join(profile['coverage_gaps'])}"])
        
        ax_stats.text(0.1, 0.95, '\n'.join(stats_text), transform=ax_stats.transAxes,
                    fontsize=10, fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        # ===== Panel 3: Dominant Viewpoint Timeline (middle) =====
        ax_dominant = fig.add_subplot(gs[1, :])
        
        # Calculate dominant views
        dominant_views = []
        dominant_colors = []
        
        for frame_idx in range(n_frames):
            qualities = {label: quality_matrix[semantic_labels.index(label), frame_idx] 
                        for label in visible_labels}
            valid_qualities = {k: v for k, v in qualities.items() if not np.isnan(v) and v > 0}
            
            if valid_qualities:
                dominant_label = max(valid_qualities, key=valid_qualities.get)
                dominant_views.append(dominant_label)
                dominant_colors.append(face_cmap[dominant_label])
            else:
                dominant_views.append('none')
                dominant_colors.append('#AAAAAA')
        
        # Draw ribbon
        for frame_idx in range(n_frames):
            ax_dominant.barh(0, 1, left=frame_idx, color=dominant_colors[frame_idx], 
                            edgecolor='none', linewidth=0)
        
        # Add transition markers
        transitions = []
        for i in range(1, len(dominant_views)):
            if dominant_views[i] != dominant_views[i-1] and dominant_views[i] != 'none':
                transitions.append(i)
                ax_dominant.axvline(x=i, color='white', linestyle='-', linewidth=2, alpha=0.8)
        
        ax_dominant.set_xlim(0, n_frames)
        ax_dominant.set_ylim(-0.5, 0.5)
        ax_dominant.set_yticks([])
        ax_dominant.set_xlabel('Frame Index', fontsize=11)
        ax_dominant.set_title(f'Dominant Viewpoint Timeline ({len(transitions)} transitions)', 
                            fontsize=13, fontweight='bold')
        
        # Add legend below timeline
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=face_cmap[l], edgecolor='black', 
                                        label=l.capitalize()) for l in visible_labels]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='#CCCCCC', edgecolor='gray',
                                            hatch='///', label='Bottom (N/A)'))
        ax_dominant.legend(handles=legend_elements, loc='upper center', 
                        bbox_to_anchor=(0.5, -0.3), ncol=6, fontsize=9)
        
        # ===== Panel 4: Line Plot (bottom left) =====
        ax_lines = fig.add_subplot(gs[2, :2])
        
        for label in visible_labels:
            label_idx = semantic_labels.index(label)
            quality_values = quality_matrix[label_idx]
            ax_lines.plot(frame_numbers, quality_values, color=face_cmap[label], 
                        linewidth=2, label=label.capitalize(), marker='o', 
                        markersize=2, alpha=0.8)
        
        ax_lines.axhspan(-0.02, 0.02, color='#E0E0E0', alpha=0.5, zorder=0)
        ax_lines.text(frame_numbers[0], 0, ' Bottom N/A', fontsize=8, color='gray', 
                    fontstyle='italic', va='center')
        
        ax_lines.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax_lines.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
        ax_lines.set_title('Quality Trends by Face', fontsize=13, fontweight='bold')
        ax_lines.legend(loc='upper right', ncol=3, fontsize=9)
        ax_lines.grid(True, alpha=0.3)
        ax_lines.set_ylim(-0.05, 1.05)
        ax_lines.set_xlim(frame_numbers[0], frame_numbers[-1])
        
        # ===== Panel 5: Pie Chart + Transitions (bottom right) =====
        ax_pie = fig.add_subplot(gs[2, 2])
        
        # Pie chart of viewpoint distribution
        view_counts = {label: sum(1 for v in dominant_views if v == label) for label in visible_labels}
        view_counts = {k: v for k, v in view_counts.items() if v > 0}
        
        if view_counts:
            colors_pie = [face_cmap[l] for l in view_counts.keys()]
            wedges, texts, autotexts = ax_pie.pie(
                view_counts.values(),
                labels=[l.capitalize() for l in view_counts.keys()],
                colors=colors_pie,
                autopct='%1.1f%%',
                pctdistance=0.75,
                startangle=90,
                textprops={'fontsize': 9}
            )
            ax_pie.set_title('Viewpoint Distribution', fontsize=12, fontweight='bold')
        else:
            ax_pie.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax_pie.set_title('Viewpoint Distribution', fontsize=12, fontweight='bold')
        
        # Save
        output_path = viz_dir / f"temporal_track_{track_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved: {output_path.name}")


    def _create_temporal_summary_overview(self, tracklet_profiles, viz_dir):
        """Create a compact summary comparing temporal patterns across all tracklets"""
        
        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        visible_labels = ['front', 'back', 'left', 'right', 'top']
        track_ids = sorted(tracklet_profiles.keys())
        n_tracks = len(track_ids)
        
        face_cmap = {
            'front': '#E74C3C', 'back': '#2ECC71', 'left': '#3498DB',
            'right': '#F39C12', 'top': '#9B59B6', 'bottom': '#CCCCCC',
        }
        
        fig = plt.figure(figsize=(16, 4 + 2 * n_tracks))
        
        # Create grid
        gs = fig.add_gridspec(n_tracks + 1, 4, height_ratios=[0.8] + [1] * n_tracks,
                            hspace=0.4, wspace=0.3,
                            left=0.08, right=0.92, top=0.92, bottom=0.08)
        
        # ===== Header Row: Comparison Charts =====
        
        # Diversity comparison
        ax_diversity = fig.add_subplot(gs[0, 0])
        diversity_scores = [tracklet_profiles[tid]['normalized_diversity'] for tid in track_ids]
        colors = [plt.cm.viridis(score) for score in diversity_scores]
        bars = ax_diversity.bar(range(n_tracks), diversity_scores, color=colors)
        ax_diversity.set_xticks(range(n_tracks))
        ax_diversity.set_xticklabels([f'T{tid}' for tid in track_ids], fontsize=9)
        ax_diversity.set_ylabel('Score', fontsize=10)
        ax_diversity.set_title('Diversity', fontsize=11, fontweight='bold')
        ax_diversity.set_ylim(0, 1)
        ax_diversity.grid(axis='y', alpha=0.3)
        
        # Re-ID readiness comparison
        ax_reid = fig.add_subplot(gs[0, 1])
        reid_scores = [tracklet_profiles[tid]['reid_readiness_score'] for tid in track_ids]
        colors_reid = [plt.cm.plasma(score) for score in reid_scores]
        ax_reid.bar(range(n_tracks), reid_scores, color=colors_reid)
        ax_reid.set_xticks(range(n_tracks))
        ax_reid.set_xticklabels([f'T{tid}' for tid in track_ids], fontsize=9)
        ax_reid.set_ylabel('Score', fontsize=10)
        ax_reid.set_title('Re-ID Readiness', fontsize=11, fontweight='bold')
        ax_reid.set_ylim(0, 1)
        ax_reid.grid(axis='y', alpha=0.3)
        
        # Frame count comparison
        ax_frames = fig.add_subplot(gs[0, 2])
        frame_counts = [tracklet_profiles[tid]['total_frames'] for tid in track_ids]
        ax_frames.bar(range(n_tracks), frame_counts, color='lightgreen', edgecolor='darkgreen')
        ax_frames.set_xticks(range(n_tracks))
        ax_frames.set_xticklabels([f'T{tid}' for tid in track_ids], fontsize=9)
        ax_frames.set_ylabel('Frames', fontsize=10)
        ax_frames.set_title('Frame Count', fontsize=11, fontweight='bold')
        ax_frames.grid(axis='y', alpha=0.3)
        
        # Legend
        ax_legend = fig.add_subplot(gs[0, 3])
        ax_legend.axis('off')
        
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=face_cmap[l], edgecolor='black', 
                                        label=l.capitalize()) for l in visible_labels]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='#CCCCCC', edgecolor='gray',
                                            hatch='///', label='Bottom (N/A)'))
        ax_legend.legend(handles=legend_elements, loc='center', ncol=2, fontsize=9,
                        title='Viewpoints', title_fontsize=10)
        
        # ===== Per-Tracklet Dominant View Ribbons =====
        for track_idx, track_id in enumerate(track_ids):
            ax_ribbon = fig.add_subplot(gs[track_idx + 1, :])
            
            profile = tracklet_profiles[track_id]
            frame_quality_scores = profile['frame_quality_scores']
            sorted_frames = sorted(frame_quality_scores.keys(), 
                                key=lambda x: int(re.findall(r'\d+', x)[0]))
            n_frames = len(sorted_frames)
            
            if n_frames == 0:
                ax_ribbon.text(0.5, 0.5, f'Track {track_id}: No frames', 
                            ha='center', va='center', fontsize=12)
                ax_ribbon.axis('off')
                continue
            
            # Build quality for this track
            quality_matrix = np.zeros((len(semantic_labels), n_frames))
            for frame_idx, frame_name in enumerate(sorted_frames):
                for label_idx, label in enumerate(semantic_labels):
                    if label == 'bottom':
                        quality_matrix[label_idx, frame_idx] = np.nan
                    else:
                        quality_matrix[label_idx, frame_idx] = frame_quality_scores[frame_name].get(label, 0)
            
            # Dominant views
            dominant_views = []
            for frame_idx in range(n_frames):
                qualities = {label: quality_matrix[semantic_labels.index(label), frame_idx] 
                            for label in visible_labels}
                valid_qualities = {k: v for k, v in qualities.items() if not np.isnan(v) and v > 0}
                
                if valid_qualities:
                    dominant_views.append(max(valid_qualities, key=valid_qualities.get))
                else:
                    dominant_views.append('none')
            
            # Draw ribbon
            for frame_idx in range(n_frames):
                color = face_cmap.get(dominant_views[frame_idx], '#AAAAAA')
                ax_ribbon.barh(0, 1, left=frame_idx, color=color, edgecolor='none')
            
            # Transitions
            transitions = sum(1 for i in range(1, len(dominant_views)) 
                            if dominant_views[i] != dominant_views[i-1] and dominant_views[i] != 'none')
            
            ax_ribbon.set_xlim(0, n_frames)
            ax_ribbon.set_ylim(-0.5, 0.5)
            ax_ribbon.set_yticks([])
            ax_ribbon.set_xlabel('Frame Index', fontsize=9)
            
            # Track info on the left
            ax_ribbon.set_ylabel(f'T{track_id}', fontsize=12, fontweight='bold', rotation=0, 
                                labelpad=30, va='center')
            
            # Stats annotation on right
            stats_str = f'n={n_frames} | div={profile["normalized_diversity"]:.2f} | trans={transitions}'
            ax_ribbon.text(1.01, 0.5, stats_str, transform=ax_ribbon.transAxes,
                        fontsize=9, va='center', fontfamily='monospace')
        
        fig.suptitle('Temporal Viewpoint Summary - All Tracklets\n(Bottom face N/A - drone footage)', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        output_path = viz_dir / "temporal_summary_overview.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved summary overview: {output_path.name}")


    def _create_individual_temporal_plots(self, tracklet_profiles, viz_dir):
        """Create detailed individual temporal plots with transition analysis for each tracklet"""
        
        semantic_labels = ['front', 'back', 'left', 'right', 'top', 'bottom']
        visible_labels = ['front', 'back', 'left', 'right', 'top']
        
        face_cmap = {
            'front': '#E74C3C', 'back': '#2ECC71', 'left': '#3498DB',
            'right': '#F39C12', 'top': '#9B59B6', 'bottom': '#CCCCCC',
        }
        
        for track_id in sorted(tracklet_profiles.keys()):
            if track_id not in self.semantic_faces:
                continue
            
            profile = tracklet_profiles[track_id]
            frame_quality_scores = profile['frame_quality_scores']
            
            sorted_frames = sorted(frame_quality_scores.keys(), 
                                key=lambda x: int(re.findall(r'\d+', x)[0]))
            n_frames = len(sorted_frames)
            
            if n_frames < 2:
                continue
            
            frame_numbers = [int(re.findall(r'\d+', f)[0]) for f in sorted_frames]
            
            # Build quality matrix
            quality_matrix = np.zeros((len(semantic_labels), n_frames))
            for frame_idx, frame_name in enumerate(sorted_frames):
                for label_idx, label in enumerate(semantic_labels):
                    if label == 'bottom':
                        quality_matrix[label_idx, frame_idx] = np.nan
                    else:
                        quality_matrix[label_idx, frame_idx] = frame_quality_scores[frame_name].get(label, 0)
            
            # Dominant views (excluding bottom)
            dominant_views = []
            for frame_idx in range(n_frames):
                qualities = {label: quality_matrix[semantic_labels.index(label), frame_idx] 
                            for label in visible_labels}
                valid_qualities = {k: v for k, v in qualities.items() if not np.isnan(v) and v > 0}
                
                if valid_qualities:
                    dominant_views.append(max(valid_qualities, key=valid_qualities.get))
                else:
                    dominant_views.append('none')
            
            # Create figure - transition analysis focused
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Track {track_id} - Transition Analysis\n(Bottom N/A - drone footage)', 
                        fontsize=14, fontweight='bold')
            
            # ===== Panel 1: Stacked Area Chart =====
            ax1 = axes[0, 0]
            
            # Prepare data for visible labels only
            visible_quality = np.zeros((len(visible_labels), n_frames))
            for i, label in enumerate(visible_labels):
                orig_idx = semantic_labels.index(label)
                visible_quality[i] = np.nan_to_num(quality_matrix[orig_idx], nan=0.0)
            
            # Smooth slightly
            from scipy.ndimage import uniform_filter1d
            smooth_window = max(1, n_frames // 20)
            smoothed = np.zeros_like(visible_quality)
            for i in range(len(visible_labels)):
                if n_frames > 3:
                    smoothed[i] = uniform_filter1d(visible_quality[i], size=smooth_window, mode='nearest')
                else:
                    smoothed[i] = visible_quality[i]
            
            ax1.stackplot(range(n_frames), smoothed,
                        labels=[l.capitalize() for l in visible_labels],
                        colors=[face_cmap[l] for l in visible_labels],
                        alpha=0.8)
            
            ax1.set_xlim(0, n_frames - 1)
            ax1.set_xlabel('Frame Index', fontsize=11)
            ax1.set_ylabel('Cumulative Quality', fontsize=11)
            ax1.set_title('Stacked Quality Over Time', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=8, ncol=3)
            ax1.grid(axis='y', alpha=0.3)
            
            # ===== Panel 2: Viewpoint Distribution Pie =====
            ax2 = axes[0, 1]
            
            view_counts = {label: sum(1 for v in dominant_views if v == label) for label in visible_labels}
            view_counts = {k: v for k, v in view_counts.items() if v > 0}
            
            if view_counts:
                colors_pie = [face_cmap[l] for l in view_counts.keys()]
                wedges, texts, autotexts = ax2.pie(
                    view_counts.values(),
                    labels=[l.capitalize() for l in view_counts.keys()],
                    colors=colors_pie,
                    autopct='%1.1f%%',
                    pctdistance=0.75,
                    startangle=90,
                    textprops={'fontsize': 10}
                )
            ax2.set_title('Dominant Viewpoint Distribution', fontsize=12, fontweight='bold')
            
            # ===== Panel 3: Transition Matrix =====
            ax3 = axes[1, 0]
            
            transition_matrix = np.zeros((len(visible_labels), len(visible_labels)))
            for i in range(1, len(dominant_views)):
                if dominant_views[i-1] != 'none' and dominant_views[i] != 'none':
                    from_idx = visible_labels.index(dominant_views[i-1])
                    to_idx = visible_labels.index(dominant_views[i])
                    transition_matrix[from_idx, to_idx] += 1
            
            # Normalize
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            transition_matrix_norm = transition_matrix / row_sums
            
            im = ax3.imshow(transition_matrix_norm, cmap='Blues', vmin=0, vmax=1)
            ax3.set_xticks(range(len(visible_labels)))
            ax3.set_yticks(range(len(visible_labels)))
            ax3.set_xticklabels([l.capitalize() for l in visible_labels], fontsize=9, rotation=45)
            ax3.set_yticklabels([l.capitalize() for l in visible_labels], fontsize=9)
            ax3.set_xlabel('To', fontsize=11, fontweight='bold')
            ax3.set_ylabel('From', fontsize=11, fontweight='bold')
            ax3.set_title('Transition Probability Matrix', fontsize=12, fontweight='bold')
            
            # Annotations
            for i in range(len(visible_labels)):
                for j in range(len(visible_labels)):
                    if transition_matrix[i, j] > 0:
                        ax3.text(j, i, f'{int(transition_matrix[i, j])}',
                                ha='center', va='center', fontsize=9,
                                color='white' if transition_matrix_norm[i, j] > 0.5 else 'black')
            
            plt.colorbar(im, ax=ax3, shrink=0.8)
            
            # ===== Panel 4: Transition Timeline =====
            ax4 = axes[1, 1]
            
            # Find transition points
            transition_points = []
            transition_labels = []
            for i in range(1, len(dominant_views)):
                if dominant_views[i] != dominant_views[i-1] and dominant_views[i] != 'none':
                    transition_points.append(i)
                    transition_labels.append(f'{dominant_views[i-1][:2]}→{dominant_views[i][:2]}')
            
            if transition_points:
                # Plot transitions as vertical lines with labels
                ax4.vlines(transition_points, 0, 1, colors='red', linewidth=2, alpha=0.7)
                
                for i, (pt, lbl) in enumerate(zip(transition_points, transition_labels)):
                    y_offset = 0.3 + (i % 3) * 0.25  # Stagger labels
                    ax4.annotate(lbl, (pt, y_offset), fontsize=8, ha='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
                
                ax4.set_xlim(0, n_frames)
                ax4.set_ylim(0, 1)
                ax4.set_xlabel('Frame Index', fontsize=11)
                ax4.set_title(f'Transition Points ({len(transition_points)} total)', fontsize=12, fontweight='bold')
                ax4.set_yticks([])
                
                # Background showing dominant view
                for frame_idx in range(n_frames):
                    color = face_cmap.get(dominant_views[frame_idx], '#EEEEEE')
                    ax4.axvspan(frame_idx, frame_idx + 1, color=color, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No transitions detected', ha='center', va='center', fontsize=12)
                ax4.set_title('Transition Points (0 total)', fontsize=12, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            output_path = viz_dir / f"temporal_track_{track_id}_transitions.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Saved transition analysis: temporal_track_{track_id}_transitions.png")

def main():
    # Change these in main():
    output_dir = "/home/shuklva/CUT3R/results/paper_final/thursday/rhinos_cami/rhin-35_3"
    image_og_dir = "/home/shuklva/CUT3R/examples/wd_data/rhinos_cami/rhin-35_3"

    
    analyzer = TrackletViewpointAnalyzer(output_dir, image_og_dir)
    
    if not analyzer.all_bbox_data:
        print("No bounding box data found!")
        return
    
    analyzer.run_tracklet_viewpoint_pipeline()


if __name__ == "__main__":
    main()