"""Small shared data types with no heavy dependencies (no ultralytics/torch import)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: Optional[int]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def feet_point(self) -> tuple:
        """Bottom-center of the box: the person's approximate ground contact point."""
        return (self.x1 + self.x2) / 2.0, self.y2

    @classmethod
    def from_point(cls, cx: float, cy: float, half_size: float = 12.0) -> "Detection":
        """
        Build a pseudo-detection from a single (x, y) point, for point-output
        models (e.g. P2PNet) that localize people without giving a box size.
        A small fixed-size marker box is used so it can flow through the same
        heatmap/rendering code as real bounding boxes; there is no track ID
        since point-based models here run per-frame with no tracker.
        """
        return cls(cx - half_size, cy - half_size, cx + half_size, cy + half_size, 1.0, None)
