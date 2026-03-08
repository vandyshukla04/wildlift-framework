"""Simple online multi-object tracker for animals (fallback tracker).

Uses 3D spatial distance, 2D mask IoU, and class consistency to associate
detections across frames via the Hungarian algorithm.  This module is
self-contained and does not depend on any other wildlift components.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class AnimalTracker:
    """Multi-object tracker for animals using 3D spatial distance, 2D mask IoU, and class consistency"""

    def __init__(self, max_distance_threshold=8.0, mask_iou_threshold=0.15, max_missing_frames=3):
        self.max_distance_threshold = max_distance_threshold
        self.mask_iou_threshold = mask_iou_threshold
        self.max_missing_frames = max_missing_frames

        self.active_tracks = {}  # track_id -> track_info
        self.next_track_id = 0
        self.frame_count = 0

        # For debugging
        self.debug = True

        print(f"🔗 AnimalTracker initialized:")
        print(f"   Max distance: {max_distance_threshold:.1f}")
        print(f"   Min IoU: {mask_iou_threshold:.2f}")
        print(f"   Max missing frames: {max_missing_frames}")

    def compute_mask_iou(self, mask1, mask2):
        """Compute IoU between two binary masks"""
        try:
            if mask1 is None or mask2 is None:
                return 0.0

            # Ensure masks are boolean numpy arrays
            if not isinstance(mask1, np.ndarray):
                return 0.0
            if not isinstance(mask2, np.ndarray):
                return 0.0

            mask1 = mask1.astype(bool)
            mask2 = mask2.astype(bool)

            # Handle size mismatches by resizing smaller mask
            if mask1.shape != mask2.shape:
                import cv2
                if mask1.size < mask2.size:
                    mask1 = cv2.resize(mask1.astype(np.uint8), mask2.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask2 = cv2.resize(mask2.astype(np.uint8), mask1.shape[::-1],
                                     interpolation=cv2.INTER_NEAREST).astype(bool)

            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)

            union_area = np.sum(union)
            if union_area == 0:
                return 1.0 if np.sum(intersection) == 0 else 0.0

            iou = np.sum(intersection) / union_area
            return float(iou)

        except Exception as e:
            if self.debug:
                print(f"    ⚠️ IoU computation failed: {e}")
            return 0.0

    def compute_distance_3d(self, center1, center2):
        """Compute Euclidean distance between two 3D centers"""
        try:
            center1 = np.array(center1).flatten()
            center2 = np.array(center2).flatten()

            if len(center1) != 3 or len(center2) != 3:
                return float('inf')

            distance = np.linalg.norm(center1 - center2)
            return float(distance)

        except Exception as e:
            if self.debug:
                print(f"    ⚠️ Distance computation failed: {e}")
            return float('inf')

    def update(self, detections, frame_idx):
        """
        Update tracker with new detections

        Args:
            detections: List of BoundingBox3D objects
            frame_idx: Current frame index

        Returns:
            List of BoundingBox3D objects with track_id assigned
        """
        self.frame_count = frame_idx

        if self.debug:
            print(f"\n🔗 Tracker update - Frame {frame_idx}")
            print(f"   Input: {len(detections)} detections, {len(self.active_tracks)} active tracks")

        # Handle empty detections
        if not detections:
            self._update_missing_tracks()
            return []

        # Handle first frame or no active tracks
        if not self.active_tracks:
            return self._initialize_tracks(detections, frame_idx)

        # Compute cost matrix between detections and active tracks
        cost_matrix, valid_assignments = self._compute_cost_matrix(detections)

        # Handle case where no valid assignments exist
        if not valid_assignments:
            if self.debug:
                print("   ⚠️ No valid assignments possible - treating all as new tracks")
            self._update_missing_tracks()
            new_tracks = self._create_new_tracks(detections, frame_idx)
            return self._get_all_active_detections(detections)

        # Solve assignment problem
        try:
            det_indices, track_indices = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            if self.debug:
                print(f"   ❌ Assignment failed: {e}")
                print(f"   Cost matrix shape: {cost_matrix.shape}")
                print(f"   Cost matrix finite values: {np.sum(np.isfinite(cost_matrix))}")

            # Fallback: treat all detections as new tracks
            self._update_missing_tracks()
            new_tracks = self._create_new_tracks(detections, frame_idx)
            return self._get_all_active_detections(detections)

        # Process assignments
        matched_detections = set()
        matched_tracks = set()

        for det_idx, track_idx in zip(det_indices, track_indices):
            cost = cost_matrix[det_idx, track_idx]

            # Only accept assignment if cost is reasonable
            if np.isfinite(cost) and cost < float('inf'):
                track_id = list(self.active_tracks.keys())[track_idx]
                detection = detections[det_idx]

                # Update track
                self._update_track(track_id, detection, frame_idx)

                # Assign track ID to detection
                detection.track_id = track_id
                detection.persistent_instance_id = track_id

                matched_detections.add(det_idx)
                matched_tracks.add(track_id)

                if self.debug:
                    print(f"   ✓ Matched detection {det_idx} ({detection.class_name}) -> track {track_id} (cost: {cost:.3f})")

        # Handle unmatched tracks (mark as missing)
        for track_id in self.active_tracks:
            if track_id not in matched_tracks:
                self.active_tracks[track_id]['frames_missing'] += 1
                if self.debug:
                    missing = self.active_tracks[track_id]['frames_missing']
                    print(f"   📭 Track {track_id} missing (frames: {missing})")

        # Remove tracks that have been missing too long
        self._remove_stale_tracks()

        # Create new tracks for unmatched detections
        unmatched_detections = [det for i, det in enumerate(detections) if i not in matched_detections]
        new_tracks = self._create_new_tracks(unmatched_detections, frame_idx)

        # Return all detections with track IDs assigned
        return self._get_all_active_detections(detections)

    def _compute_cost_matrix(self, detections):
        """Compute cost matrix between detections and active tracks"""
        num_detections = len(detections)
        num_tracks = len(self.active_tracks)
        track_ids = list(self.active_tracks.keys())

        cost_matrix = np.full((num_detections, num_tracks), float('inf'))
        valid_assignments = False

        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                track_info = self.active_tracks[track_id]

                # Only consider tracks of the same class
                if track_info['class_name'].lower() != detection.class_name.lower():
                    continue

                # Compute 3D distance
                distance_3d = self.compute_distance_3d(
                    detection.center,
                    track_info['last_center']
                )

                # Skip if distance is too large
                if distance_3d > self.max_distance_threshold:
                    continue

                # Compute mask IoU if masks are available
                mask_iou = 0.0
                if hasattr(detection, 'mask') and detection.mask is not None:
                    mask_iou = self.compute_mask_iou(
                        detection.mask,
                        track_info.get('last_mask', None)
                    )

                # Skip if IoU is too low
                if mask_iou < self.mask_iou_threshold:
                    # But be more lenient with distance if IoU computation failed
                    if mask_iou == 0.0 and distance_3d > self.max_distance_threshold * 0.7:
                        continue

                # Compute combined cost (lower is better)
                # Weight: 70% distance, 30% IoU (inverted)
                distance_cost = distance_3d / self.max_distance_threshold
                iou_cost = 1.0 - mask_iou

                total_cost = 0.7 * distance_cost + 0.3 * iou_cost
                cost_matrix[det_idx, track_idx] = total_cost
                valid_assignments = True

                if self.debug:
                    print(f"     Det {det_idx} -> Track {track_id}: dist={distance_3d:.2f}, iou={mask_iou:.3f}, cost={total_cost:.3f}")

        return cost_matrix, valid_assignments

    def _initialize_tracks(self, detections, frame_idx):
        """Initialize tracks for the first frame"""
        if self.debug:
            print(f"   🆕 Initializing {len(detections)} tracks")

        for detection in detections:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.active_tracks[track_id] = {
                'class_name': detection.class_name,
                'last_center': detection.center.copy(),
                'last_mask': getattr(detection, 'mask', None),
                'last_rotation': detection.rotation_matrix.copy(),  # ADD THIS LINE
                'first_frame': frame_idx,
                'last_frame': frame_idx,
                'frames_missing': 0,
                'detection_count': 1
            }

            detection.track_id = track_id
            detection.persistent_instance_id = track_id

            if self.debug:
                print(f"     ✓ Created track {track_id} for {detection.class_name}")

        return detections

    def _update_track(self, track_id, detection, frame_idx):
        """Update an existing track with new detection"""
        self.active_tracks[track_id]['last_center'] = detection.center.copy()
        self.active_tracks[track_id]['last_mask'] = getattr(detection, 'mask', None)
        self.active_tracks[track_id]['last_frame'] = frame_idx
        self.active_tracks[track_id]['frames_missing'] = 0
        self.active_tracks[track_id]['detection_count'] += 1
        self.active_tracks[track_id]['last_rotation'] = detection.rotation_matrix.copy()

    def _update_missing_tracks(self):
        """Update missing frame counts for all tracks"""
        for track_id in self.active_tracks:
            self.active_tracks[track_id]['frames_missing'] += 1

    def _remove_stale_tracks(self):
        """Remove tracks that have been missing for too long"""
        stale_tracks = []
        for track_id, track_info in self.active_tracks.items():
            if track_info['frames_missing'] > self.max_missing_frames:
                stale_tracks.append(track_id)

        for track_id in stale_tracks:
            if self.debug:
                track_info = self.active_tracks[track_id]
                duration = track_info['last_frame'] - track_info['first_frame'] + 1
                print(f"   🗑️ Removing stale track {track_id} ({track_info['class_name']}, duration: {duration} frames)")
            del self.active_tracks[track_id]

    def _create_new_tracks(self, unmatched_detections, frame_idx):
        """Create new tracks for unmatched detections"""
        new_tracks = []

        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.active_tracks[track_id] = {
                'class_name': detection.class_name,
                'last_center': detection.center.copy(),
                'last_mask': getattr(detection, 'mask', None),
                'first_frame': frame_idx,
                'last_frame': frame_idx,
                'frames_missing': 0,
                'detection_count': 1,
                'last_rotation': detection.rotation_matrix.copy()  # ADD THIS LINE
            }

            detection.track_id = track_id
            detection.persistent_instance_id = track_id
            new_tracks.append(detection)

            if self.debug:
                print(f"   🆕 Created new track {track_id} for unmatched {detection.class_name}")

        return new_tracks

    def _get_all_active_detections(self, detections):
        """Return all detections with track IDs assigned"""
        result = []
        for detection in detections:
            if hasattr(detection, 'track_id') and detection.track_id is not None:
                result.append(detection)
        return result

    def get_tracklets(self):
        """Get all tracklets (for analysis/export)"""
        # This is a simplified version - you might want to store full history
        tracklets = {}
        for track_id, track_info in self.active_tracks.items():
            tracklets[track_id] = [{
                'class_name': track_info['class_name'],
                'first_frame': track_info['first_frame'],
                'last_frame': track_info['last_frame'],
                'detection_count': track_info['detection_count']
            }]
        return tracklets

    def get_tracking_statistics(self):
        """Get tracking statistics"""
        tracklets = self.get_tracklets()

        stats = {
            'total_tracks': len(tracklets),
            'active_tracks': len(self.active_tracks),
            'avg_tracklet_length': 0,
            'class_stats': {}
        }

        if tracklets:
            tracklet_lengths = []
            for track_info_list in tracklets.values():
                for track_info in track_info_list:
                    length = track_info['detection_count']
                    tracklet_lengths.append(length)

                    class_name = track_info['class_name']
                    if class_name not in stats['class_stats']:
                        stats['class_stats'][class_name] = {'count': 0, 'lengths': []}
                    stats['class_stats'][class_name]['count'] += 1
                    stats['class_stats'][class_name]['lengths'].append(length)

            stats['avg_tracklet_length'] = np.mean(tracklet_lengths)

        return stats
