"""
Thin wrapper around an Ultralytics YOLO model.

Kept deliberately narrow (load + detect) so that a different detector
backend - e.g. a specialized crowd-counting model used later for the
comparison task - can be dropped in behind the same `detect()` interface
without touching main.py or the rest of the pipeline.
"""

from __future__ import annotations

from typing import List

import numpy as np
from ultralytics import YOLO

from core.config import DetectionConfig
from core.types import Detection


class PersonDetector:
    def __init__(self, model_path: str, cfg: DetectionConfig, task: str = "detect"):
        self.model = YOLO(model_path, task=task)
        self.cfg = cfg

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg.tracker,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            classes=[self.cfg.target_class],
            verbose=False,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                track_id = int(box.id.item()) if box.id is not None else None
                detections.append(Detection(float(x1), float(y1), float(x2), float(y2), conf, track_id))

        return detections
