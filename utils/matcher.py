import numpy as np
import torch
from typing import Dict, Any, Tuple, Set, Optional
import logging
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float],
                      device: str = 'cpu') -> float:
    if device == 'cuda' and torch.cuda.is_available():
        p1_tensor = torch.tensor(point1, dtype=torch.float32, device='cuda')
        p2_tensor = torch.tensor(point2, dtype=torch.float32, device='cuda')
        return torch.norm(p1_tensor - p2_tensor).item()
    else:
        return np.linalg.norm(np.array(point1) - np.array(point2))

class CrossViewMatcher:
    def __init__(self, 
                 spatial_threshold: float = 100.0,
                 min_overlap_frames: int = 5,
                 device: str = 'cpu'):
        self.spatial_threshold = spatial_threshold
        self.min_overlap_frames = min_overlap_frames
        self.device = device
        logger.info(f"CrossViewMatcher initialized with spatial threshold: {spatial_threshold}")

    def _get_track_frames(self, track: Dict[str, Any]) -> Set[int]:
        return set(frame_idx for frame_idx, _ in track['track'])

    def _get_bbox_at_frame(self, track: Dict[str, Any], 
                          frame_idx: int) -> Optional[Tuple[int, int, int, int]]:
        for f_idx, bbox in track['track']:
            if f_idx == frame_idx:
                return bbox
        return None

    def _calculate_track_similarity(self, track1: Dict[str, Any], 
                                   track2: Dict[str, Any]) -> Tuple[float, int]:
        frames1 = self._get_track_frames(track1)
        frames2 = self._get_track_frames(track2)
        common_frames = frames1 & frames2
        if len(common_frames) < self.min_overlap_frames:
            return float('inf'), 0
        total_distance = 0.0
        valid_frames = 0
        for frame_idx in common_frames:
            bbox1 = self._get_bbox_at_frame(track1, frame_idx)
            bbox2 = self._get_bbox_at_frame(track2, frame_idx)
            if bbox1 is not None and bbox2 is not None:
                center1 = calculate_bbox_center(bbox1)
                center2 = calculate_bbox_center(bbox2)
                distance = calculate_distance(center1, center2, self.device)
                total_distance += distance
                valid_frames += 1
        if valid_frames == 0:
            return float('inf'), 0
        avg_distance = total_distance / valid_frames
        return avg_distance, valid_frames

    def _greedy_matching(self, tracks1: Dict[int, Dict[str, Any]], 
                        tracks2: Dict[int, Dict[str, Any]]) -> Dict[int, int]:
        mapping = {}
        used_tracks1 = set()
        sorted_tracks2 = sorted(tracks2.items(), 
                               key=lambda x: len(x[1]['track']), 
                               reverse=True)
        for track2_id, track2 in sorted_tracks2:
            best_match = None
            best_score = float('inf')
            best_overlap = 0
            for track1_id, track1 in tracks1.items():
                if track1_id in used_tracks1:
                    continue
                avg_distance, overlap_frames = self._calculate_track_similarity(
                    track1, track2)
                if (avg_distance < self.spatial_threshold and 
                    avg_distance < best_score and
                    overlap_frames >= self.min_overlap_frames):
                    best_score = avg_distance
                    best_match = track1_id
                    best_overlap = overlap_frames
            if best_match is not None:
                mapping[track2_id] = best_match
                used_tracks1.add(best_match)
                logger.debug(f"Matched track {track2_id} -> {best_match} "
                           f"(distance: {best_score:.2f}, overlap: {best_overlap})")
        return mapping

    def _hungarian_matching(self, tracks1: Dict[int, Dict[str, Any]], 
                           tracks2: Dict[int, Dict[str, Any]]) -> Dict[int, int]:
        track1_ids = list(tracks1.keys())
        track2_ids = list(tracks2.keys())
        if not track1_ids or not track2_ids:
            return {}
        cost_matrix = np.full((len(track2_ids), len(track1_ids)), 
                             self.spatial_threshold * 2)
        for i, track2_id in enumerate(track2_ids):
            for j, track1_id in enumerate(track1_ids):
                track1 = tracks1[track1_id]
                track2 = tracks2[track2_id]
                avg_distance, overlap_frames = self._calculate_track_similarity(
                    track1, track2)
                if (avg_distance < self.spatial_threshold and 
                    overlap_frames >= self.min_overlap_frames):
                    cost_matrix[i, j] = avg_distance
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        mapping = {}
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < self.spatial_threshold:
                track2_id = track2_ids[i]
                track1_id = track1_ids[j]
                mapping[track2_id] = track1_id
                logger.debug(f"Hungarian matched track {track2_id} -> {track1_id} "
                           f"(cost: {cost_matrix[i, j]:.2f})")
        return mapping

    def match_tracks(self, tracks1: Dict[int, Dict[str, Any]], 
                    tracks2: Dict[int, Dict[str, Any]],
                    method: str = 'greedy') -> Dict[int, int]:
        if not tracks1 or not tracks2:
            logger.warning("One or both track sets are empty")
            return {}
        logger.info(f"Matching {len(tracks2)} tracks to {len(tracks1)} tracks "
                   f"using {method} method")
        if method == 'hungarian':
            mapping = self._hungarian_matching(tracks1, tracks2)
        else:
            mapping = self._greedy_matching(tracks1, tracks2)
        logger.info(f"Successfully matched {len(mapping)} track pairs")
        return mapping

    def evaluate_matching(self, tracks1: Dict[int, Dict[str, Any]], 
                         tracks2: Dict[int, Dict[str, Any]],
                         mapping: Dict[int, int]) -> Dict[str, float]:
        if not mapping:
            return {
                'match_rate': 0.0,
                'avg_distance': float('inf'),
                'avg_overlap': 0.0
            }
        total_distance = 0.0
        total_overlap = 0
        for track2_id, track1_id in mapping.items():
            track1 = tracks1[track1_id]
            track2 = tracks2[track2_id]
            avg_distance, overlap_frames = self._calculate_track_similarity(
                track1, track2)
            total_distance += avg_distance
            total_overlap += overlap_frames
        return {
            'match_rate': len(mapping) / len(tracks2),
            'avg_distance': total_distance / len(mapping),
            'avg_overlap': total_overlap / len(mapping)
        }

def match_players_across_views(tracks1: Dict[int, Dict[str, Any]], 
                              tracks2: Dict[int, Dict[str, Any]],
                              spatial_threshold: float = 100.0,
                              min_overlap_frames: int = 5,
                              device: str = 'cpu',
                              method: str = 'greedy') -> Dict[int, int]:
    matcher = CrossViewMatcher(
        spatial_threshold=spatial_threshold,
        min_overlap_frames=min_overlap_frames,
        device=device
    )
    mapping = matcher.match_tracks(tracks1, tracks2, method)
    metrics = matcher.evaluate_matching(tracks1, tracks2, mapping)
    logger.info(f"Matching results: {metrics}")
    return mapping

center = calculate_bbox_center

