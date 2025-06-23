import numpy as np
import torch

def iou(bb1, bb2, device='cpu'):
    if device == 'cuda' and torch.cuda.is_available():
        bb1 = torch.tensor(bb1, dtype=torch.float32, device='cuda')
        bb2 = torch.tensor(bb2, dtype=torch.float32, device='cuda')
        x1 = torch.max(bb1[0], bb2[0])
        y1 = torch.max(bb1[1], bb2[1])
        x2 = torch.min(bb1[2], bb2[2])
        y2 = torch.min(bb1[3], bb2[3])
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        union_area = bb1_area + bb2_area - inter_area
        return (inter_area / union_area).item() if union_area > 0 else 0
    else:
        x1 = max(bb1[0], bb2[0])
        y1 = max(bb1[1], bb2[1])
        x2 = min(bb1[2], bb2[2])
        y2 = min(bb1[3], bb2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        union_area = bb1_area + bb2_area - inter_area
        return inter_area / union_area if union_area else 0

def track_players(detections, iou_threshold=0.5, device='cpu'):
    next_id = 0
    tracks = {}
    for frame_idx, frame in enumerate(detections):
        for det in frame:
            matched = False
            for tid, track in tracks.items():
                if frame_idx - track['last_seen'] > 10: continue
                if iou(det['bbox'], track['bbox'], device=device) > iou_threshold:
                    track['bbox'] = det['bbox']
                    track['last_seen'] = frame_idx
                    track['track'].append((frame_idx, det['bbox']))
                    matched = True
                    break
            if not matched:
                tracks[next_id] = {
                    'bbox': det['bbox'],
                    'last_seen': frame_idx,
                    'track': [(frame_idx, det['bbox'])]
                }
                next_id += 1
    return tracks
