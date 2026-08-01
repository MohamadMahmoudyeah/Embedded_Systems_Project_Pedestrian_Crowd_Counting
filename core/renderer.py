"""Drawing helpers shared by both output streams (detection-only and heatmap)."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from core.types import Detection

BOX_COLOR = (0, 255, 0)
POINT_COLOR = (0, 165, 255)
ID_COLOR = (255, 255, 0)
HUD_BG = (0, 0, 0)
COUNT_COLOR = (0, 255, 255)
FPS_COLOR = (0, 255, 0)


def draw_detections(frame: np.ndarray, detections: List[Detection], as_points: bool = False) -> np.ndarray:
    """
    Draws each detection. `as_points=True` (used for P2PNet, which
    localizes people as single points rather than real boxes) draws a
    small filled marker circle at the box center instead of a rectangle,
    so point-based and box-based backends are visually distinguishable.
    No ID label is ever drawn for a detection with track_id=None, which
    covers all point-based detections since P2PNet runs per-frame with
    no tracker.
    """
    out = frame.copy()
    for det in detections:
        if as_points:
            cx, cy = det.center
            cv2.circle(out, (int(cx), int(cy)), 5, POINT_COLOR, -1)
            cv2.circle(out, (int(cx), int(cy)), 5, (0, 0, 0), 1)
        else:
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            cv2.rectangle(out, p1, p2, BOX_COLOR, 2)
            if det.track_id is not None:
                cv2.putText(out, f"ID {det.track_id}", (p1[0], p1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ID_COLOR, 2)
    return out


def draw_hud(frame: np.ndarray, count: int, fps: float) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (10, 10), (320, 100), HUD_BG, -1)
    cv2.putText(out, f"Count: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, COUNT_COLOR, 2)
    cv2.putText(out, f"FPS: {fps:.1f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, FPS_COLOR, 2)
    return out
