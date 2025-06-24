"""
detector.py

Player detection module using YOLO and OpenCV.

This module provides the PlayerDetector class for detecting players in video frames or video files using a YOLO model. It includes utilities for filtering detections to only player-related classes and for running detection on entire videos.

Classes:
    PlayerDetector: Loads a YOLO model and provides methods to detect players in frames and videos.

Functions:
    run_detection: Convenience function to run detection on a video file.
"""

import torch
from ultralytics import YOLO
import cv2
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerDetector:
    """
    PlayerDetector loads a YOLO model and provides methods to detect players in frames and videos.

    Args:
        model_path (str): Path to the YOLO model file.
        device (str): Device to use ('auto', 'cuda', or 'cpu'). Defaults to 'auto'.
    """
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the PlayerDetector with a model path and device.
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = self._load_model()
        self.class_names = self.model.names
        
        logger.info(f"PlayerDetector initialized with device: {self.device}")
        logger.info(f"Model classes: {self.class_names}")
    
    def _setup_device(self, device: str) -> str:
        """
        Determine the device to use for inference.

        Args:
            device (str): Requested device ('auto', 'cuda', or 'cpu').
        Returns:
            str: Selected device.
        """
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        else:
            return device
    
    def _load_model(self) -> YOLO:
        """
        Load the YOLO model from the specified path.

        Returns:
            YOLO: Loaded YOLO model.
        Raises:
            Exception: If the model fails to load.
        """
        try:
            model = YOLO(self.model_path)
            if self.device == 'cuda':
                model.to('cuda')
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def _is_player_class(self, class_id: int, class_name: str) -> bool:
        """
        Determine if a detected class corresponds to a player.

        Args:
            class_id (int): Class ID from the model.
            class_name (str): Class name from the model.
        Returns:
            bool: True if the class is considered a player, False otherwise.
        """
        if class_name:
            class_lower = class_name.lower()
            player_keywords = ['player', 'goalkeeper', 'person']
            if any(keyword in class_lower for keyword in player_keywords):
                return True
        
        return class_id in [1, 2]
    
    def detect_frame(self, frame: Any, confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect players in a single video frame.

        Args:
            frame (Any): Input image/frame (numpy array).
            confidence_threshold (float): Minimum confidence for detections. Defaults to 0.3.
        Returns:
            List[Dict[str, Any]]: List of detections with bounding boxes, confidence, class name, and class ID.
        """
        if self.device == 'cuda':
            results = self.model(frame, device='cuda', verbose=False)
        else:
            results = self.model(frame, verbose=False)
        
        frame_detections = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls)
                class_name = self.class_names.get(cls, 'unknown')
                conf = float(box.conf)
                
                if (self._is_player_class(cls, class_name) and 
                    conf > confidence_threshold):
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    frame_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class': class_name,
                        'class_id': cls
                    })
        
        return frame_detections
    
    def detect_video(self, video_path: str, 
                    confidence_threshold: float = 0.3,
                    progress_interval: int = 30) -> List[List[Dict[str, Any]]]:
        """
        Detect players in all frames of a video file.

        Args:
            video_path (str): Path to the video file.
            confidence_threshold (float): Minimum confidence for detections. Defaults to 0.3.
            progress_interval (int): Interval for logging progress. Defaults to 30.
        Returns:
            List[List[Dict[str, Any]]]: List of detections for each frame.
        Raises:
            ValueError: If the video file cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        detections = []
        frame_count = 0
        
        logger.info(f"Starting detection on video: {video_path}")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_detections = self.detect_frame(frame, confidence_threshold)
                detections.append(frame_detections)
                frame_count += 1
                
                if frame_count % progress_interval == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Detection complete. Total frames: {frame_count}")
        return detections


def run_detection(video_path: str, model_path: str, 
                 device: str = 'auto',
                 confidence_threshold: float = 0.3) -> List[List[Dict[str, Any]]]:
    """
    Convenience function to run player detection on a video file.

    Args:
        video_path (str): Path to the video file.
        model_path (str): Path to the YOLO model file.
        device (str): Device to use ('auto', 'cuda', or 'cpu'). Defaults to 'auto'.
        confidence_threshold (float): Minimum confidence for detections. Defaults to 0.3.
    Returns:
        List[List[Dict[str, Any]]]: List of detections for each frame.
    """
    detector = PlayerDetector(model_path, device)
    return detector.detect_video(video_path, confidence_threshold)

