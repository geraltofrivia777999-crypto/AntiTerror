from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger
from shapely.geometry import Polygon

from .config import AssociationConfig


def bbox_center(box: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2), float((y1 + y2) / 2)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    poly_a = Polygon([(a[0], a[1]), (a[0], a[3]), (a[2], a[3]), (a[2], a[1])])
    poly_b = Polygon([(b[0], b[1]), (b[0], b[3]), (b[2], b[3]), (b[2], b[1])])
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union == 0:
        return 0.0
    return float(inter / union)


@dataclass
class Link:
    person_track: int
    bag_track: int
    stable_frames: int = 0


class AssociationEngine:
    def __init__(self, cfg: AssociationConfig):
        self.cfg = cfg
        self.links: Dict[Tuple[int, int], Link] = {}
        logger.info("Association engine ready")

    def associate(self, person_tracks, bag_tracks) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        # naive nearest neighbor with IoU support
        for b in bag_tracks:
            b_center = bbox_center(b.box)
            best_pid = None
            best_score = 1e9
            for p in person_tracks:
                p_center = bbox_center(p.box)
                dist = np.linalg.norm(np.array(b_center) - np.array(p_center))
                overlap = iou(b.box, p.box)
                if overlap >= self.cfg.iou_threshold:
                    dist *= 0.5  # favor overlaps
                if dist < best_score and dist <= self.cfg.max_link_distance_px:
                    best_score = dist
                    best_pid = p.track_id
            if best_pid is not None:
                key = (best_pid, b.track_id)
                link = self.links.get(key, Link(person_track=best_pid, bag_track=b.track_id))
                link.stable_frames += 1
                self.links[key] = link
                if link.stable_frames >= self.cfg.time_consistency_frames:
                    assignments[b.track_id] = best_pid
        # prune stale links not seen this frame
        current_pairs = set((p.track_id, b.track_id) for b in bag_tracks for p in person_tracks)
        stale_keys = [k for k in self.links if k not in current_pairs]
        for k in stale_keys:
            del self.links[k]
        return assignments
