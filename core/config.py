"""
Typed configuration objects for the pipeline.

A RunConfig is built by:
  1. Reading config/settings.yaml for defaults.
  2. Optionally overriding fields interactively (CLI wizard) or from a
     saved run-config JSON file (headless / embedded execution).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@dataclass
class DetectionConfig:
    conf_threshold: float = 0.60         # YOLO: box confidence. P2PNet: point-score threshold (same slot, same meaning: "how sure before we count it").
    iou_threshold: float = 0.45          # YOLO only.
    target_class: int = 0                # YOLO only.
    tracker: str = "bytetrack.yaml"      # YOLO only.


@dataclass
class HeatmapConfig:
    mode: str = "instantaneous"          # "instantaneous" | "cumulative" | "classic"
    sigma_ratio: float = 0.35            # used by instantaneous/cumulative (feet-anchored), box-based backends only
    classic_sigma_ratio: float = 0.25    # used by classic mode (box-center anchored), box-based backends only
    point_sigma: float = 25.0            # fixed Gaussian radius (pixels) for point-based backends (P2PNet) in ALL modes -
                                          # a point has no box size to derive sigma from, so this is set directly rather
                                          # than scaled from a (somewhat arbitrary) pseudo-box size.
    decay: float = 0.97                  # cumulative mode only
    blur_kernel: int = 51                # instantaneous/cumulative only (classic skips blurring)
    intensity_ceiling: float = 5.0
    overlay_alpha: float = 0.65
    background_darken: float = 0.6       # frame brightness multiplier for heatmap-containing streams
    noise_floor: float = 0.0             # zero out sub-threshold speckle (fraction of ref_max); cosmetic only


@dataclass
class CSRNetConfig:
    """
    Post-processing parameters specific to CSRNet, developed in response to
    the false-positive failure mode documented in REPORT.md (density
    regression models activating on head-shaped clutter like flower
    baskets on footage unlike their ShanghaiTech training data).

    Two counting methods are available:

      density_sum (the standard CSRNet method):
        count = integral (sum) of the density map. Accurate when the
        deployment footage matches the training domain; on domain-
        mismatched footage, any head-shaped clutter contributes density
        mass and inflates the count.

      peak_detect:
        count = number of local maxima ("peaks") in a Gaussian-smoothed
        density map, each counted as one head. A diffuse blob of density
        mass spread over a flower basket tends to collapse into fewer,
        broader peaks after smoothing than the same density mass spread
        over several distinct, sharp person-shaped peaks would - so this
        typically undercounts diffuse clutter relative to genuine heads.
        This is a heuristic, not a guarantee: clutter that happens to
        produce a single sharp, head-sized peak (rather than a diffuse
        blob) will still be counted by this method too - see REPORT.md
        for the full discussion of what this can and can't fix.
    """
    density_threshold: float = 0.001     # zero out density below this (mild noise floor, not related to the clutter issue)
    blur_sigma: float = 1.5              # Gaussian smoothing sigma, in density-map pixels (1 density-px = 8 frame-px at CSRNet's 1/8 output stride)
    count_method: str = "density_sum"    # "density_sum" | "peak_detect"
    peak_min_height: float = 0.005       # [peak_detect only] minimum peak value to count as a person
    peak_nms_radius: int = 1             # [peak_detect only] two peaks must be at least 2*radius+1 density-map pixels apart
    intensity_ceiling: float = 0.01      # heatmap starting reference - CSRNet's raw density values are tiny (~0.001-0.05),
                                          # much smaller than the generic heatmap.intensity_ceiling default (tuned for YOLO's Gaussian-blob sums)


@dataclass
class CountingConfig:
    smoothing: float = 0.15


@dataclass
class OutputConfig:
    video_fps: int = 30
    video_codec: str = "mp4v"
    image_format: str = "jpg"
    save_as: str = "video"               # "video" | "images"


@dataclass
class SourceConfig:
    kind: str = "video"                  # "camera" | "video" | "images"
    path: str = ""                       # camera index / URL, video file path, or image folder path


@dataclass
class ExternalReposConfig:
    """
    Paths to locally-cloned upstream repos for backends whose model code is
    not (and should not be) reproduced by hand in this project - DM-Count
    and P2PNet both have non-trivial architectures beyond what can be
    safely copied from memory, unlike CSRNet's small, fully-specified
    frontend+backend, which is vendored directly in core/csrnet.py.
    A backend using one of these repos inserts its path onto sys.path and
    imports the repo's own `models` package, exactly like running the
    scripts from inside the cloned repo would.
    """
    dmcount: str = ""   # path to a local clone of cvlab-stonybrook/DM-Count
    p2pnet: str = ""    # path to a local clone of TencentYoutuResearch/CrowdCounting-P2PNet


@dataclass
class P2PNetConfig:
    backbone: str = "vgg16_bn"
    row: int = 2
    line: int = 2
    marker_half_size: float = 12.0       # pseudo-detection box half-size for heatmap/rendering (P2PNet gives points, not boxes)


@dataclass
class RunConfig:
    model_path: str = ""
    backend: str = "yolo"                # "yolo" | "csrnet" | "dmcount" | "p2pnet"
    source: SourceConfig = field(default_factory=SourceConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    csrnet: CSRNetConfig = field(default_factory=CSRNetConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    external_repos: ExternalReposConfig = field(default_factory=ExternalReposConfig)
    p2pnet: P2PNetConfig = field(default_factory=P2PNetConfig)
    headless: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load_defaults(settings_path: str | Path = DEFAULT_SETTINGS_PATH) -> "RunConfig":
        with open(settings_path, "r") as f:
            raw = yaml.safe_load(f)

        return RunConfig(
            detection=DetectionConfig(**raw.get("detection", {})),
            heatmap=HeatmapConfig(**raw.get("heatmap", {})),
            csrnet=CSRNetConfig(**raw.get("csrnet", {})),
            counting=CountingConfig(**raw.get("counting", {})),
            output=OutputConfig(**{k: v for k, v in raw.get("output", {}).items()
                                    if k in OutputConfig.__dataclass_fields__}),
            external_repos=ExternalReposConfig(**raw.get("external_repos", {})),
            p2pnet=P2PNetConfig(**raw.get("p2pnet", {})),
        )

    @staticmethod
    def load_from_json(path: str | Path) -> "RunConfig":
        with open(path, "r") as f:
            raw = json.load(f)

        return RunConfig(
            model_path=raw.get("model_path", ""),
            backend=raw.get("backend", "yolo"),
            source=SourceConfig(**raw.get("source", {})),
            detection=DetectionConfig(**raw.get("detection", {})),
            heatmap=HeatmapConfig(**raw.get("heatmap", {})),
            csrnet=CSRNetConfig(**raw.get("csrnet", {})),
            counting=CountingConfig(**raw.get("counting", {})),
            output=OutputConfig(**raw.get("output", {})),
            external_repos=ExternalReposConfig(**raw.get("external_repos", {})),
            p2pnet=P2PNetConfig(**raw.get("p2pnet", {})),
            headless=raw.get("headless", False),
        )
