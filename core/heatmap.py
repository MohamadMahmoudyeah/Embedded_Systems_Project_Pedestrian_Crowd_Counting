"""
Spatial crowd-density heatmap.

Three modes are supported. A single "whole-frame" blob (one flat color
derived only from the person count, with no spatial information) was
intentionally left out, because it cannot show *where* on the sidewalk
people are, which is exactly what the assignment asks for.

  instantaneous : density map is rebuilt from scratch every frame, feet-
                  anchored, blurred, normalized against a slowly-adapting
                  reference ceiling. Answers "how crowded is it right
                  now, and where" with a color scale that stays
                  comparable across frames.
  cumulative    : same as instantaneous but the density map persists
                  across frames with exponential decay instead of being
                  rebuilt every frame. Answers "which spots see
                  sustained/repeated foot traffic over time" (a
                  dwell/hotspot map).
  classic       : rebuilt every frame, box-CENTER anchored, no blur, and
                  normalized with per-frame min-max (like the reference
                  script this was modeled on). This trades away
                  cross-frame comparability - a single person can look
                  as saturated as a packed crowd - in exchange for a
                  heatmap that always looks visually "full" and vivid
                  regardless of how many people are present. Kept as an
                  explicit, separate option rather than the default,
                  since instantaneous/cumulative are the
                  quantitatively-meaningful choices.

Anchor point (instantaneous/cumulative): each person contributes a
Gaussian blob centered at the BOTTOM-CENTER of their bounding box (feet
point), not the box center and not an assumed head position. The feet
point is the only part of a bounding box that actually corresponds to a
location on the ground plane of the sidewalk; the detector was trained
on full-body boxes, not head boxes, so a "head" position would just be
a guess. Classic mode uses the box center instead, matching the
reference script it reproduces.

Point-based backends (P2PNet): a point has no real box size to derive a
Gaussian radius from - P2PNet's pseudo-detections use a small, arbitrary
fixed marker size (see core/types.Detection.from_point) purely so they
can flow through the same drawing code as real boxes. Deriving sigma from
that arbitrary marker size (as box-based backends do) produces
noticeably smaller/patchier blobs than deriving it from a size that
actually looks right for a head-scale point. So when
`use_point_sigma=True`, every mode uses a single fixed
`cfg.point_sigma` (default 25px) instead of a box-derived sigma, in all
three heatmap modes - this was tuned to match the visual quality of a
reference point-heatmap script (scatter points, blur with a large fixed
sigma) that turned out to look substantially better for P2PNet's output
than a box-derived sigma did.

Normalization (instantaneous/cumulative): raw per-frame min-max
normalization is deliberately NOT used, because it would make a single
pedestrian look exactly as "hot" (fully red) as a packed crowd - every
frame gets stretched to its own local max regardless of how many people
are actually present. Instead, intensity is normalized against a
reference ceiling that starts at a configured value and slowly adapts
upward when it observes a busier frame, then decays back down slowly.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from core.config import HeatmapConfig
from core.types import Detection


def blend_density_map(
    frame: np.ndarray,
    density_map: np.ndarray,
    ref_max: float,
    overlay_alpha: float,
    ref_decay: float = 0.995,
    floor: float = 1e-3,
    noise_floor: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Colorize `density_map` and blend it on top of `frame` (which the caller
    may have already darkened). Normalizes against a slowly-adapting
    reference ceiling rather than the current frame's own max, so color
    intensity stays comparable across frames. Returns (blended, new_ref_max).
    Shared by HeatmapGenerator (instantaneous/cumulative) and the native
    density-map renderer below, so all backends look and behave consistently.

    `noise_floor` (0-1, fraction of ref_max) zeroes out low-level speckle in
    the displayed map before colorizing. It only cleans up faint background
    haze - it does NOT and cannot remove genuine false-positive activations
    on textured clutter (e.g. a model reacting to a flower basket or a pile
    of bicycles as if it were a head of hair): those score just as "real" to
    the network as an actual person, so no threshold on the output can tell
    them apart. See core/csrnet.py's CSRNetConfig.count_method="peak_detect"
    for a mitigation that targets that failure mode directly, instead of
    trying to fix it in the display layer.
    """
    current_max = float(density_map.max()) if density_map.size else 0.0
    new_ref_max = max(current_max, ref_max * ref_decay, floor)

    normalized = np.clip(density_map / new_ref_max, 0.0, 1.0)
    if noise_floor > 0:
        normalized = np.where(normalized >= noise_floor, normalized, 0.0)

    heatmap_u8 = (normalized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    alpha = (normalized * overlay_alpha)[..., None]
    blended = frame.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha
    return blended.astype(np.uint8), new_ref_max


class NativeDensityRenderer:
    """
    Renders a heatmap from a density map that a model already produces
    natively (CSRNet, DM-Count) rather than one built from Gaussian blobs
    around bounding boxes (HeatmapGenerator, above). Reuses the same
    adaptive-ceiling normalization for a consistent look across backends.
    """

    def __init__(self, overlay_alpha: float = 0.65, initial_ceiling: float = 1.0, noise_floor: float = 0.0):
        self.overlay_alpha = overlay_alpha
        self.ref_max = max(initial_ceiling, 1e-3)
        self.noise_floor = noise_floor

    def render(self, frame: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        blended, self.ref_max = blend_density_map(
            frame, density_map, self.ref_max, self.overlay_alpha, noise_floor=self.noise_floor
        )
        return blended


class HeatmapGenerator:
    def __init__(self, cfg: HeatmapConfig, frame_w: int, frame_h: int, use_point_sigma: bool = False):
        self.cfg = cfg
        self.w = frame_w
        self.h = frame_h
        self.density_map = np.zeros((frame_h, frame_w), dtype=np.float32)
        self.ref_max = max(cfg.intensity_ceiling, 1e-3)
        self.use_point_sigma = use_point_sigma  # True for point-based backends (P2PNet) - see module docstring

        k = cfg.blur_kernel
        self._blur_kernel = k if k % 2 == 1 else k + 1

    def update(self, detections: List[Detection]) -> None:
        if self.cfg.mode == "cumulative":
            self.density_map *= self.cfg.decay
        else:
            # instantaneous and classic both rebuild from scratch every frame
            self.density_map[:] = 0.0

        for det in detections:
            if self.use_point_sigma:
                cx, cy = det.center
                sigma = self.cfg.point_sigma
            elif self.cfg.mode == "classic":
                cx, cy = det.center
                sigma = max(1.0, max(det.width, det.height) * self.cfg.classic_sigma_ratio)
            else:
                cx, cy = det.feet_point
                sigma = max(1.0, det.width * self.cfg.sigma_ratio)
            self._add_gaussian(cx, cy, sigma)

    def _add_gaussian(self, cx: float, cy: float, sigma: float) -> None:
        radius = int(3 * sigma) or 1

        y0 = max(0, int(cy) - radius)
        y1 = min(self.h, int(cy) + radius + 1)
        x0 = max(0, int(cx) - radius)
        x1 = min(self.w, int(cx) + radius + 1)

        if y1 <= y0 or x1 <= x0:
            return

        xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        patch = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        self.density_map[y0:y1, x0:x1] += patch

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the current density map on top of `frame` and return the
        result. The caller decides whether `frame` has already been
        darkened - this method only does the overlay math.
        """
        if self.cfg.mode == "classic":
            # Reference-script behavior: no blur, raw per-frame min-max
            # normalization, fixed alpha. Intentionally NOT comparable
            # across frames - see module docstring.
            if np.max(self.density_map) > 0:
                heatmap_u8 = cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                heatmap_u8 = np.zeros((self.h, self.w), dtype=np.uint8)

            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            alpha = (heatmap_u8.astype(np.float32) / 255.0 * self.cfg.overlay_alpha)[..., None]
            blended = frame.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha
            return blended.astype(np.uint8)

        blurred = cv2.GaussianBlur(self.density_map, (self._blur_kernel, self._blur_kernel), 0)
        blended, self.ref_max = blend_density_map(
            frame, blurred, self.ref_max, self.cfg.overlay_alpha, noise_floor=self.cfg.noise_floor
        )
        return blended
