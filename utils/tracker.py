"""
tracker.py

Player tracking utilities for associating detections across video frames.

This module provides the PlayerTracker class for tracking player detections across frames using IoU-based association. It includes utilities for updating tracks, filtering valid tracks, and running tracking on a sequence of detections.

Functions:
    calculate_iou: Compute the Intersection over Union (IoU) between two bounding boxes.
    track_players: Convenience function to run tracking on a list of detections.

Classes:
    PlayerTracker: Tracks player detections across frames using IoU and configurable thresholds.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_iou(bbox1: Tuple[int, int, int, int], 
                 bbox2: Tuple[int, int, int, int],
                 device: str = 'cpu') -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1 (Tuple[int, int, int, int]): First bounding box (x1, y1, x2, y2).
        bbox2 (Tuple[int, int, int, int]): Second bounding box (x1, y1, x2, y2).
        device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.
    Returns:
        float: IoU value between 0 and 1.
    """
    if device == 'cuda' and torch.cuda.is_available():
        bb1 = torch.tensor(bbox1, dtype=torch.float32, device='cuda')
        bb2 = torch.tensor(bbox2, dtype=torch.float32, device='cuda')
        x1 = torch.max(bb1[0], bb2[0])
        y1 = torch.max(bb1[1], bb2[1])
        x2 = torch.min(bb1[2], bb2[2])
        y2 = torch.min(bb1[3], bb2[3])
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        union_area = bb1_area + bb2_area - inter_area
        return (inter_area / union_area).item() if union_area > 0 else 0.0
    else:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

class PlayerTracker:
    """
    Tracks player detections across video frames using IoU-based association.

    Args:
        iou_threshold (float): Minimum IoU required to associate a detection with an existing track.
        max_lost_frames (int): Maximum number of frames a track can be lost before removal.
        min_track_length (int): Minimum number of frames for a track to be considered valid.
        device (str): 'cpu' or 'cuda'.
    """
    def __init__(self, 
                 iou_threshold: float = 0.5,
                 max_lost_frames: int = 10,
                 min_track_length: int = 5,
                 device: str = 'cpu'):
        """
        Initialize the PlayerTracker.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_track_length = min_track_length
        self.device = device
        self.next_id = 0
        self.tracks = {}
        logger.info(f"PlayerTracker initialized with IoU threshold: {iou_threshold}")

    def _find_best_match(self, detection: Dict[str, Any], 
                        frame_idx: int) -> Tuple[int, float]:
        """
        Find the best matching track for a detection based on IoU.

        Args:
            detection (Dict[str, Any]): Detection dictionary with 'bbox'.
            frame_idx (int): Current frame index.
        Returns:
            Tuple[int, float]: (Best matching track ID or None, IoU score).
        """
        best_iou = 0.0
        best_track_id = None
        for track_id, track in self.tracks.items():
            if frame_idx - track['last_seen'] > self.max_lost_frames:
                continue
            current_iou = calculate_iou(detection['bbox'], 
                                       track['bbox'], 
                                       self.device)
            if current_iou > self.iou_threshold and current_iou > best_iou:
                best_iou = current_iou
                best_track_id = track_id
        return best_track_id, best_iou

    def _create_new_track(self, detection: Dict[str, Any], 
                         frame_idx: int) -> int:
        """
        Create a new track for an unmatched detection.

        Args:
            detection (Dict[str, Any]): Detection dictionary with 'bbox'.
            frame_idx (int): Current frame index.
        Returns:
            int: New track ID.
        """
        track_id = self.next_id
        self.tracks[track_id] = {
            'bbox': detection['bbox'],
            'last_seen': frame_idx,
            'track': [(frame_idx, detection['bbox'])],
            'confidence_history': [detection.get('conf', 1.0)]
        }
        self.next_id += 1
        return track_id

    def _update_track(self, track_id: int, detection: Dict[str, Any], 
                     frame_idx: int) -> None:
        """
        Update an existing track with a new detection.

        Args:
            track_id (int): Track ID to update.
            detection (Dict[str, Any]): Detection dictionary.
            frame_idx (int): Current frame index.
        """
        track = self.tracks[track_id]
        track['bbox'] = detection['bbox']
        track['last_seen'] = frame_idx
        track['track'].append((frame_idx, detection['bbox']))
        track['confidence_history'].append(detection.get('conf', 1.0))

    def update(self, detections: List[Dict[str, Any]], 
              frame_idx: int) -> Dict[int, Dict[str, Any]]:
        """
        Update the tracker with detections for a single frame.

        Args:
            detections (List[Dict[str, Any]]): List of detections for the frame.
            frame_idx (int): Current frame index.
        Returns:
            Dict[int, Dict[str, Any]]: Updated tracks.
        """
        matched_tracks = set()
        for detection in detections:
            track_id, iou_score = self._find_best_match(detection, frame_idx)
            if track_id is not None and track_id not in matched_tracks:
                self._update_track(track_id, detection, frame_idx)
                matched_tracks.add(track_id)
            else:
                self._create_new_track(detection, frame_idx)
        return self.tracks

    def get_active_tracks(self, frame_idx: int) -> Dict[int, Dict[str, Any]]:
        """
        Get tracks that are currently active (not lost for too many frames).

        Args:
            frame_idx (int): Current frame index.
        Returns:
            Dict[int, Dict[str, Any]]: Active tracks.
        """
        active_tracks = {}
        for track_id, track in self.tracks.items():
            if frame_idx - track['last_seen'] <= self.max_lost_frames:
                active_tracks[track_id] = track
        return active_tracks

    def get_valid_tracks(self) -> Dict[int, Dict[str, Any]]:
        """
        Get tracks that are considered valid (long enough).

        Returns:
            Dict[int, Dict[str, Any]]: Valid tracks.
        """
        valid_tracks = {}
        for track_id, track in self.tracks.items():
            if len(track['track']) >= self.min_track_length:
                valid_tracks[track_id] = track
        return valid_tracks

    def track_video(self, detections: List[List[Dict[str, Any]]], 
                   progress_interval: int = 50) -> Dict[int, Dict[str, Any]]:
        """
        Run tracking on a sequence of detections for all frames in a video.

        Args:
            detections (List[List[Dict[str, Any]]]): List of detections for each frame.
            progress_interval (int): Interval for logging progress. Defaults to 50.
        Returns:
            Dict[int, Dict[str, Any]]: Valid tracks after tracking.
        """
        total_detections = sum(len(frame) for frame in detections)
        logger.info(f"Tracking players across {len(detections)} frames...")
        logger.info(f"Total detections: {total_detections}")
        for frame_idx, frame_detections in enumerate(detections):
            self.update(frame_detections, frame_idx)
            if frame_idx % progress_interval == 0 and frame_idx > 0:
                active_tracks = self.get_active_tracks(frame_idx)
                logger.info(f"Frame {frame_idx}: {len(active_tracks)} active tracks")
        valid_tracks = self.get_valid_tracks()
        logger.info(f"Tracking complete: {len(self.tracks)} total tracks, "
                   f"{len(valid_tracks)} valid tracks")
        return valid_tracks

def track_players(detections: List[List[Dict[str, Any]]], 
                 iou_threshold: float = 0.5,
                 max_lost_frames: int = 10,
                 min_track_length: int = 5,
                 device: str = 'cpu') -> Dict[int, Dict[str, Any]]:
    """
    Convenience function to run player tracking on a list of detections.

    Args:
        detections (List[List[Dict[str, Any]]]): List of detections for each frame.
        iou_threshold (float): Minimum IoU required to associate a detection with an existing track.
        max_lost_frames (int): Maximum number of frames a track can be lost before removal.
        min_track_length (int): Minimum number of frames for a track to be considered valid.
        device (str): 'cpu' or 'cuda'.
    Returns:
        Dict[int, Dict[str, Any]]: Valid tracks after tracking.
    """
    tracker = PlayerTracker(
        iou_threshold=iou_threshold,
        max_lost_frames=max_lost_frames,
        min_track_length=min_track_length,
        device=device
    )
    return tracker.track_video(detections)

iou = calculate_iou

