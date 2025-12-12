from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger
from ultralytics import YOLO

from .config import DetectionConfig, select_device


@dataclass
class DetectionResult:
    boxes: np.ndarray  # (N,4) xyxy
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,)


class Detector:
    def __init__(self, cfg: DetectionConfig):
        device = select_device(cfg.device)
        logger.info(f"Loading YOLO model {cfg.model_path} on {device}")
        self.model = YOLO(cfg.model_path)
        self.model.to(device)
        self.cfg = cfg

    def __call__(self, frame) -> List[DetectionResult]:
        res = self.model.predict(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            device=self.model.device,
            verbose=False,
        )[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        return DetectionResult(boxes=boxes, scores=scores, classes=classes)
