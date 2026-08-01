"""Exponential moving average smoothing for the displayed crowd count."""

from __future__ import annotations

from core.config import CountingConfig


class CountSmoother:
    def __init__(self, cfg: CountingConfig):
        self.alpha = cfg.smoothing
        self.value = 0.0

    def update(self, raw_count: int) -> float:
        self.value = (1 - self.alpha) * self.value + self.alpha * raw_count
        return self.value

    @property
    def rounded(self) -> int:
        return round(self.value)
