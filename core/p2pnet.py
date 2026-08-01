"""
Specialized crowd-counting backend: P2PNet
(TencentYoutuResearch/CrowdCounting-P2PNet). Unlike CSRNet/DM-Count
(pure density regression, no localization at all), P2PNet outputs a set
of individual (x, y) head-point locations with confidence scores - a
middle ground between YOLO's full bounding boxes and CSRNet's fully
implicit density map. That per-instance localization is why P2PNet can
plug into the exact same instance-based pipeline as YOLO (heatmap modes,
`combined`/`detection`/`heatmap` output streams) instead of being
restricted to a single native-density `heatmap` stream like CSRNet/DM-Count.

As with DM-Count (see core/dmcount.py), P2PNet's architecture is not
reproduced by hand here - it is imported from a local clone of the
upstream repo via `from models import build_model`, exactly as the
repo's own scripts do.

Setup:
  1. Clone https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
     and point `external_repos.p2pnet` at that folder.
  2. `build_model` internally loads an ImageNet-pretrained vgg16_bn
     backbone via torchvision *unless* it is already present in torch's
     model cache. If the machine has no internet access at run time,
     pre-download it once:
         mkdir -p ~/.cache/torch/hub/checkpoints
         wget https://download.pytorch.org/models/vgg16_bn-6c64b313.pth \\
             -O ~/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth
     (that is torchvision's own default cache location; setting the
     TORCH_HOME environment variable also works if a different location
     is preferred).
  3. Select a P2PNet `.pth` checkpoint (e.g. `SHTechA.pth`) as the model.

Input size requirement: P2PNet's backbone needs both spatial dimensions
to be a multiple of 128; frames are resized up to the nearest multiple of
128 before inference and detected points are scaled back to the original
frame size afterward, so output coordinates line up with the real frame.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

from core.types import Detection

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class P2PNetEstimator:
    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        backbone: str = "vgg16_bn",
        row: int = 2,
        line: int = 2,
        threshold: float = 0.5,
        marker_half_size: float = 12.0,
        device: str | None = None,
    ):
        if not repo_path:
            raise ValueError(
                "P2PNet requires a local clone of TencentYoutuResearch/CrowdCounting-P2PNet. "
                "Set external_repos.p2pnet to that folder's path."
            )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from models import build_model  # noqa: import from the cloned P2PNet repo, not this project
        except ImportError as e:
            raise ImportError(
                f"Could not import P2PNet's model code from '{repo_path}'. "
                f"Make sure this path is a clone of CrowdCounting-P2PNet "
                f"(it must contain a models/ package)."
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.marker_half_size = marker_half_size

        args = SimpleNamespace(backbone=backbone, row=row, line=line)
        self.model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Same shape as PersonDetector.detect(): a list of pseudo-detections, one per found head."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        width, height = pil_img.size
        new_width = max(128, (width // 128) * 128)
        new_height = max(128, (height // 128) * 128)
        pil_resized = pil_img.resize((new_width, new_height), Image.LANCZOS)

        tensor = self.transform(pil_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
        points = outputs["pred_points"][0]

        mask = scores > self.threshold
        kept_points = points[mask].detach().cpu().numpy()

        scale_x = frame_bgr.shape[1] / new_width
        scale_y = frame_bgr.shape[0] / new_height

        detections: List[Detection] = []
        for x, y in kept_points:
            detections.append(Detection.from_point(float(x) * scale_x, float(y) * scale_y, self.marker_half_size))

        return detections
