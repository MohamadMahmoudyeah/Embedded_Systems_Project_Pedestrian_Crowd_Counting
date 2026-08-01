"""
Specialized crowd-counting backend: DM-Count
(cvlab-stonybrook/DM-Count), a VGG19-based density-regression model,
similar in spirit to CSRNet (§ core/csrnet.py) but a different
architecture and training recipe (Optimal-Transport loss rather than
pixel-wise MSE against a Gaussian-blurred target).

Unlike CSRNet, this project does NOT reproduce DM-Count's architecture
by hand. CSRNet's frontend+backend is small and fully specified in the
training notebook, so vendoring it directly was safe. DM-Count's model
code lives in the upstream repo and is not trivial enough to safely
retype from memory without risking a subtle, silent mismatch against the
actual trained checkpoint - a wrong reproduction wouldn't necessarily
crash, it would just quietly give wrong numbers, which is worse. So this
module inserts a local clone of the repo onto `sys.path` and imports the
model-building function directly from it (`from models import vgg19`),
exactly as running the script from inside the cloned repo would.

Setup: clone https://github.com/cvlab-stonybrook/DM-Count and point
`external_repos.dmcount` (config/settings.yaml, or the CLI prompt) at
that folder. A `.pth` checkpoint (e.g. `pretrained_models/model_qnrf.pth`
from the repo's own release) is selected separately as the model file.
"""

from __future__ import annotations

import sys
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class DMCountEstimator:
    def __init__(self, repo_path: str, checkpoint_path: str, device: str | None = None):
        if not repo_path:
            raise ValueError(
                "DM-Count requires a local clone of cvlab-stonybrook/DM-Count. "
                "Set external_repos.dmcount to that folder's path."
            )
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from models import vgg19  # noqa: import from the cloned DM-Count repo, not this project
        except ImportError as e:
            raise ImportError(
                f"Could not import DM-Count's model code from '{repo_path}'. "
                f"Make sure this path is a clone of cvlab-stonybrook/DM-Count "
                f"(it must contain a models/ package)."
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = vgg19()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # DM-Count's own inference scripts use plain ToTensor with no
        # ImageNet mean/std normalization - matched here deliberately.
        self.transform = transforms.ToTensor()

    def estimate(self, frame_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            density, _ = self.model(tensor)

        density_map = density[0, 0].detach().cpu().numpy()
        count = float(density_map.sum())

        h, w = frame_bgr.shape[:2]
        density_resized = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_CUBIC)
        density_resized = np.clip(density_resized, 0, None).astype(np.float32)

        return count, density_resized
