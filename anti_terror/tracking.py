from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from loguru import logger
from supervision import Detections
from supervision.tracker.byte_tracker.core import ByteTrack

from .config import TrackingConfig


@dataclass
class Track:
    track_id: int
    box: np.ndarray  # xyxy
    score: float
    cls: int


class Tracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.byte_tracker = ByteTrack(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            frame_rate=cfg.frame_rate,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
        )
        logger.info("Initialized ByteTrack tracker")

    def update(self, detections_xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> List[Track]:
        if len(detections_xyxy) == 0:
            self.byte_tracker.update_with_detections(Detections.empty())
            return []
        det = Detections(
            xyxy=detections_xyxy,
            confidence=scores,
            class_id=classes,
        )
        tracks = self.byte_tracker.update_with_detections(det)
        result: List[Track] = []
        for idx, track_id in enumerate(tracks.tracker_id):
            result.append(
                Track(
                    track_id=int(track_id),
                    box=tracks.xyxy[idx],
                    score=float(tracks.confidence[idx]),
                    cls=int(tracks.class_id[idx]),
                )
            )
        return result
