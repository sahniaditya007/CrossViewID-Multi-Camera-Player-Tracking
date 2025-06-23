import numpy as np
import torch

def center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def match_players_across_views(tracks1, tracks2, spatial_thresh=50, device='cpu'):
    mapping = {}
    for tid2, t2 in tracks2.items():
        best_match = None
        best_score = float('inf')
        for tid1, t1 in tracks1.items():
            common_frames = set(f for f, _ in t2['track']) & set(f for f, _ in t1['track'])
            if not common_frames: continue

            total_dist = 0
            for f in common_frames:
                c1 = center(next(b for f1, b in t1['track'] if f1 == f))
                c2 = center(next(b for f2, b in t2['track'] if f2 == f))
                # Use torch tensors for GPU computation if device is cuda
                if device == 'cuda' and torch.cuda.is_available():
                    c1_tensor = torch.tensor(c1, dtype=torch.float32, device='cuda')
                    c2_tensor = torch.tensor(c2, dtype=torch.float32, device='cuda')
                    dist = torch.norm(c1_tensor - c2_tensor).item()
                else:
                    dist = np.linalg.norm(np.array(c1) - np.array(c2))
                total_dist += dist
            avg_dist = total_dist / len(common_frames)
            if avg_dist < best_score and avg_dist < spatial_thresh:
                best_score = avg_dist
                best_match = tid1
        if best_match is not None:
            mapping[tid2] = best_match
    return mapping
