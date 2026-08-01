"""
Specialized crowd-counting backend: CSRNet (Li, Zhang & Chen, 2018),
loaded from a checkpoint fine-tuned on ShanghaiTech Part A (see the
csrnet-pytorch-parta-*-run-two notebooks for the training pipeline this
checkpoint came from).

Architecture is reproduced exactly from the training notebook so the
state_dict keys line up: a VGG16 frontend (first 13 conv layers) feeding
a dilated-convolution backend, ending in a single 1x1 conv that outputs
one density value per output pixel (output resolution is 1/8 of the
input on each axis, from the three max-pools in the frontend).

Unlike the YOLO backend, CSRNet has no notion of individual bounding
boxes, track IDs, or a confidence threshold - it directly regresses a
density map from the whole image, and the estimated crowd count is
derived from that map (see CSRNetConfig.count_method below). This is why
the CLI does not ask for a heatmap mode or a detection-only/combined view
when this backend is selected: there are no boxes to draw, and the
model's own native output *is* the heatmap.

`CSRNet(load_weights=True)` is used deliberately, even though the
original training script's default is `load_weights=False` (which
downloads ImageNet-pretrained VGG16 weights from the internet). At
inference time every weight is about to be overwritten by the checkpoint
anyway, so skipping that download avoids a hard internet dependency -
important for offline use on the embedded board.

Observed failure mode and mitigation
-------------------------------------
On footage visually unlike ShanghaiTech Part A (this checkpoint's
training set - dense, angled crowd photos), CSRNet can activate on
textured, roughly head-sized clutter that has nothing to do with people -
a flower basket or a cluster of parked bicycle wheels, for example. This
is a genuine domain-shift/generalization limitation, not a rendering bug:
the network has no semantic notion of "person", only a learned
correlation between local image texture and crowd density, so
sufficiently person-texture-like clutter scores just as "real" to it as
an actual head does. See REPORT.md for the full discussion, including
concrete examples.

`CSRNetConfig.count_method = "peak_detect"` is a mitigation for this:
instead of summing the whole density map (where a diffuse patch of
clutter-driven density mass counts just as much as a concentrated,
person-shaped peak), it counts the number of distinct local maxima in a
smoothed density map. Clutter tends to produce more diffuse, lower, and
fewer peaks per unit of density mass than genuine heads do, so this
typically undercounts clutter relative to real people - but it is a
heuristic, not a guarantee: clutter that happens to produce one sharp,
head-sized peak is still counted as a person by this method too. No
post-processing on the model's output can fully eliminate this failure
mode; a real fix requires fine-tuning on the target camera's own footage.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from core.config import CSRNetConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =====================================================================
# Model architecture (must match the training notebook exactly)
# =====================================================================

class CSRNet(nn.Module):
    def __init__(self, load_weights: bool = False):
        super().__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            from torchvision import models
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(list(self.frontend.state_dict().items()))):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels: int = 3, batch_norm: bool = False, dilation: bool = False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# =====================================================================
# Post-processing helpers (the peak_detect mitigation)
# =====================================================================

def _smooth_density(density: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-blur the density map, acting as a matched filter for
    head-sized features: sub-head-scale spikes get averaged down more
    than head-sized blobs do. The sum of the map is approximately
    preserved by the blur, so density_sum counting is not significantly
    affected by this step - it mainly matters for peak_detect.
    """
    if sigma <= 0:
        return density
    ksize = int(sigma * 6) | 1   # 6-sigma rule, ensure odd
    ksize = max(ksize, 3)
    return cv2.GaussianBlur(density, (ksize, ksize), sigma)


def _count_peaks(density: np.ndarray, min_height: float, nms_radius: int) -> int:
    """
    Count local maxima above `min_height` in the (already-smoothed)
    density map. A pixel counts as a peak if it equals the max value in
    its own circular neighborhood of radius `nms_radius` (implemented via
    grayscale dilation) and clears the minimum height.
    """
    if density.size == 0 or density.max() < min_height:
        return 0

    ksize = 2 * nms_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(density, kernel)

    peaks = (density >= dilated) & (density >= min_height)
    return int(np.count_nonzero(peaks))


# =====================================================================
# Estimator (inference + post-processing)
# =====================================================================

class CSRNetEstimator:
    """
    Loads a trained checkpoint and turns a BGR frame into
    (estimated_count, density_map_full_res).

    Post-processing (applied at native 1/8 resolution, before upscaling
    for display):
      1. Clip negatives.
      2. Zero out values below cfg.density_threshold (mild noise floor).
      3. Gaussian-blur with sigma=cfg.blur_sigma (head-scale matched filter).
      4. Count via cfg.count_method: "density_sum" (sum the map) or
         "peak_detect" (count local maxima - see module docstring).
    """

    def __init__(self, checkpoint_path: str, cfg: CSRNetConfig | None = None, device: str | None = None):
        self.cfg = cfg or CSRNetConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CSRNet(load_weights=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._printed_diagnostics = False

    def estimate(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns (estimated_count, density_map) where density_map is resized to frame_bgr's H, W."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for c in range(3):
            rgb[..., c] = (rgb[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
        density_raw = output.squeeze().detach().cpu().numpy()

        # ---- Post-processing (density-map space, 1/8 resolution) ----
        density = np.clip(density_raw, 0, None).astype(np.float32)
        raw_sum = float(density.sum())

        if self.cfg.density_threshold > 0:
            density[density < self.cfg.density_threshold] = 0.0

        density = _smooth_density(density, self.cfg.blur_sigma)

        if self.cfg.count_method == "peak_detect":
            count = float(_count_peaks(density, self.cfg.peak_min_height, self.cfg.peak_nms_radius))
        else:
            count = float(density.sum())

        if not self._printed_diagnostics:
            self._printed_diagnostics = True
            peak_count = _count_peaks(density, self.cfg.peak_min_height, self.cfg.peak_nms_radius)
            print(
                f"\n[CSRNet diagnostics, first frame] "
                f"raw density sum={raw_sum:.2f}  smoothed sum={float(density.sum()):.2f}  "
                f"peak count={peak_count}  count_method={self.cfg.count_method}\n"
                f"  If the sum looks inflated relative to visible people (e.g. by clutter), "
                f"try count_method: \"peak_detect\" in config/settings.yaml's csrnet section.\n"
            )

        h, w = frame_bgr.shape[:2]
        density_vis = cv2.resize(density, (w, h), interpolation=cv2.INTER_CUBIC)
        density_vis = np.clip(density_vis, 0, None).astype(np.float32)

        return count, density_vis
