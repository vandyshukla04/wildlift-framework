#!/usr/bin/env python3
"""
3D to 2D Bounding Box Projection for Images
"""

import numpy as np
import cv2
import os
import json
from copy import deepcopy

class BoundingBox2DProjector:
    """Projects 3D bounding boxes onto 2D images"""
    
    def __init__(self):
        # Define 3D bounding box edge connections (same as in your BoundingBox3D class)
        self.bbox_edges = [
            # Bottom face
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face  
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # Class colors (matching your existing color scheme)
        self.class_colors = {
            'zebra': (255, 153, 51),     # Orange (BGR format for OpenCV)
            'ground': (51, 255, 51),     # Green
            'sky': (255, 178, 76),       # Blue
            'person': (153, 51, 255),    # Pink
            'car': (255, 51, 204),       # Purple
            'building': (51, 255, 255),  # Yellow
            'tree': (102, 204, 0),       # Forest green
            'rhino': (25, 128, 230),     # Orange-brown
            'rhinoceros': (25, 128, 230), # Same as rhino
        }
    
    def project_3d_to_2d(self, points_3d, camera_matrix, pose_matrix):
        """Project 3D points to 2D image coordinates with robust dimension handling"""
        
        print(f"🐛 DEBUG: Input points_3d shape: {points_3d.shape}")
        print(f"🐛 DEBUG: Input points_3d dtype: {points_3d.dtype}")
        
        # Handle various input shapes and force to (N, 3)
        if points_3d.ndim == 3:
            # Could be (N, 1, 3) or (1, N, 3) - reshape to (N, 3)
            if points_3d.shape[1] == 1:
                points_3d = points_3d.squeeze(1)  # (N, 1, 3) → (N, 3)
            elif points_3d.shape[0] == 1:
                points_3d = points_3d.squeeze(0)  # (1, N, 3) → (N, 3)
            else:
                points_3d = points_3d.reshape(-1, 3)  # Flatten to (N, 3)
        elif points_3d.ndim == 1:
            points_3d = points_3d.reshape(-1, 3)  # (9,) → (3, 3) etc.
        
        # Ensure it's (N, 3)
        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValueError(f"Cannot reshape points_3d to (N, 3), got shape {points_3d.shape}")
        
        print(f"🐛 DEBUG: After reshape points_3d shape: {points_3d.shape}")
        
        # Convert to homogeneous coordinates (N, 4)
        ones_array = np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)
        print(f"🐛 DEBUG: ones_array shape: {ones_array.shape}")
        
        points_3d_homo = np.concatenate([points_3d, ones_array], axis=1)
        print(f"🐛 DEBUG: points_3d_homo shape: {points_3d_homo.shape}")
        
        # Transform to camera coordinates
        if pose_matrix is not None:
            # Apply inverse pose transformation (world to camera)
            pose_inv = np.linalg.inv(pose_matrix)
            points_camera_homo = (pose_inv @ points_3d_homo.T).T
            points_camera = points_camera_homo[:, :3]
        else:
            points_camera = points_3d
        
        # Project to image plane
        points_2d_homo = (camera_matrix @ points_camera.T).T
        
        # Convert from homogeneous coordinates and get depths
        depths = points_2d_homo[:, 2]
        
        # Avoid division by zero
        valid_mask = np.abs(depths) > 1e-6
        points_2d = np.zeros((len(points_3d), 2))
        
        if np.any(valid_mask):
            points_2d[valid_mask] = points_2d_homo[valid_mask, :2] / depths[valid_mask, np.newaxis]
        
        return points_2d, depths

    def get_class_color_bgr(self, class_name):
        """Get BGR color for a class (for OpenCV)"""
        if class_name.lower() in self.class_colors:
            return self.class_colors[class_name.lower()]
        else:
            # Generate color based on hash
            hash_val = hash(class_name.lower()) % 1000
            r = int((hash_val * 0.618) % 1.0 * 255)
            g = int(((hash_val * 0.618) * 2) % 1.0 * 255)
            b = int(((hash_val * 0.618) * 3) % 1.0 * 255)
            return (b, g, r)  # BGR format
    
    def draw_bbox_on_image(self, image, bbox, camera_matrix, pose_matrix, 
                          line_thickness=1, draw_label=False):
        """
        Draw a single 3D bounding box on an image
        
        Args:
            image: Input image (numpy array)
            bbox: BoundingBox3D object
            camera_matrix: (3, 3) camera intrinsic matrix
            pose_matrix: (4, 4) camera pose matrix
            line_thickness: Thickness of bounding box lines
            draw_label: Whether to draw class label
        
        Returns:
            image: Image with bounding box drawn
        """
        # Get 3D corner points
        corners_3d = bbox.get_corners()
        
        # Project to 2D
        corners_2d, depths = self.project_3d_to_2d(corners_3d, camera_matrix, pose_matrix)
        
        # Check if bounding box is in front of camera
        if np.any(depths <= 0):
            # Some points are behind the camera, skip this bbox
            return image
        
        # Check if projected points are within image bounds (with some margin)
        h, w = image.shape[:2]
        margin = 50  # Allow some points to be slightly outside
        
        if (np.all(corners_2d[:, 0] < -margin) or np.all(corners_2d[:, 0] > w + margin) or
            np.all(corners_2d[:, 1] < -margin) or np.all(corners_2d[:, 1] > h + margin)):
            # Bounding box is completely outside image
            return image
        
        # Get class color
        color = self.get_class_color_bgr(bbox.class_name)
        
        # Draw edges
        for edge in self.bbox_edges:
            pt1 = tuple(map(int, corners_2d[edge[0]]))
            pt2 = tuple(map(int, corners_2d[edge[1]]))
            
            # Only draw if both points are reasonable (not too far outside image)
            if (abs(pt1[0]) < w + margin and abs(pt1[1]) < h + margin and
                abs(pt2[0]) < w + margin and abs(pt2[1]) < h + margin):
                cv2.line(image, pt1, pt2, color, line_thickness)
        
        # Draw class label
        if draw_label:
            # Find the top-center of the bounding box for label placement
            top_center_3d = bbox.center + np.array([0, 0, bbox.dimensions[2]/2])
            label_2d, label_depth = self.project_3d_to_2d(
                top_center_3d.reshape(1, 3), camera_matrix, pose_matrix
            )
            
            if label_depth[0] > 0:  # In front of camera
                label_pos = tuple(map(int, label_2d[0]))
                label_text = f"{bbox.class_name} {bbox.confidence:.2f}"
                
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                
                # Background rectangle
                bg_pt1 = (label_pos[0] - 2, label_pos[1] - text_h - baseline - 2)
                bg_pt2 = (label_pos[0] + text_w + 2, label_pos[1] + 2)
                cv2.rectangle(image, bg_pt1, bg_pt2, color, -1)
                
                # Text
                text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                cv2.putText(image, label_text, label_pos, font, font_scale, text_color, font_thickness)
        
        return image
    
    def project_all_bboxes_to_images(self, original_images, bounding_boxes, cam_dict, 
                                   output_dir, line_thickness=1, save_images=True):
        """
        Project all 3D bounding boxes onto their corresponding images
        
        Args:
            original_images: List of original images
            bounding_boxes: List of lists of BoundingBox3D objects (per frame)
            cam_dict: Camera parameters dictionary
            output_dir: Directory to save annotated images
            line_thickness: Thickness of bounding box lines
            save_images: Whether to save the annotated images
        
        Returns:
            annotated_images: List of images with 3D bboxes projected
        """
        annotated_images = []
        
        # Create output directory
        if save_images:
            bbox_2d_dir = os.path.join(output_dir, "images_with_3d_bboxes")
            os.makedirs(bbox_2d_dir, exist_ok=True)
        
        for frame_idx, (image, frame_bboxes) in enumerate(zip(original_images, bounding_boxes)):
            # Convert image to numpy array if it's a tensor
            if hasattr(image, 'cpu'):  # PyTorch tensor
                if image.dim() == 4:  # [B, C, H, W]
                    img_np = image[0].permute(1, 2, 0).cpu().numpy()
                elif image.dim() == 3:  # [C, H, W]  
                    img_np = image.permute(1, 2, 0).cpu().numpy()
                else:
                    img_np = image.cpu().numpy()
            else:
                img_np = image.copy()
            
            # Ensure image is in correct format
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:  # Normalized to [0, 1]
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            
            # Ensure BGR format for OpenCV
            if img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Get camera parameters for this frame
            focal = cam_dict["focal"][frame_idx]
            pp = cam_dict["pp"][frame_idx]
            
            # Construct camera intrinsic matrix
            camera_matrix = np.array([
                [focal, 0, pp[0]],
                [0, focal, pp[1]], 
                [0, 0, 1]
            ])
            
            # Get camera pose (convert from camera-to-world to world-to-camera)
            R = cam_dict["R"][frame_idx]
            t = cam_dict["t"][frame_idx]
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = R
            pose_matrix[:3, 3] = t
            
            print(f"📷 Frame {frame_idx}: Processing {len(frame_bboxes)} bounding boxes")
            
            # Draw each bounding box
            for bbox in frame_bboxes:
                img_np = self.draw_bbox_on_image(
                    img_np, bbox, camera_matrix, pose_matrix, 
                    line_thickness=line_thickness, draw_label=False
                )
            
            # Convert back to RGB for consistency
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            annotated_images.append(img_rgb)
            
            # Save annotated image
            if save_images:
                output_path = os.path.join(bbox_2d_dir, f"{frame_idx:06d}.png")
                cv2.imwrite(output_path, img_np)  # Save in BGR format
                print(f"  ✅ Saved: {output_path}")
        
        print(f"\n🎯 Projected 3D bounding boxes onto {len(annotated_images)} images!")
        if save_images:
            print(f"📁 Saved annotated images to: {os.path.join(output_dir, 'images_with_3d_bboxes')}")
        
        return annotated_images


def add_2d_projection_to_demo(original_images, bounding_boxes, cam_dict, output_dir):
    """
    Add 2D projection functionality to your existing demo
    Call this function after your existing bounding box computation
    """
    if bounding_boxes is None or not any(frame_bboxes for frame_bboxes in bounding_boxes):
        print("⚠️  No 3D bounding boxes found for 2D projection")
        return None
    
    print("\n🎯 Starting 3D → 2D Bounding Box Projection...")
    
    projector = BoundingBox2DProjector()
    annotated_images = projector.project_all_bboxes_to_images(
        original_images=original_images,
        bounding_boxes=bounding_boxes, 
        cam_dict=cam_dict,
        output_dir=output_dir,
        line_thickness=3,
        save_images=True
    )
    
    return annotated_images


# Example integration with your existing code:
"""
# Add this to your prepare_output_overlay_with_bboxes function, after computing bounding boxes:

if bounding_boxes is not None:
    # Project 3D bounding boxes to 2D images
    annotated_images = add_2d_projection_to_demo(
        original_images=original_images,
        bounding_boxes=bounding_boxes,
        cam_dict=cam_dict,
        output_dir=outdir
    )
"""