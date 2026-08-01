"""
Interactive terminal wizard used when main.py is run manually (laptop
demo for the TA). None of this runs on the embedded board - there,
main.py is launched with --config pointing at a saved run_config.json
instead (see deployment/crowd_monitor.service).
"""

from __future__ import annotations

import glob
import os
from typing import List

from core.config import OutputConfig, RunConfig, SourceConfig

BACKEND_EXTENSIONS = {
    "yolo": ("*.pt", "*.onnx"),
    "csrnet": ("*.pth.tar", "*.tar"),
    "dmcount": ("*.pth",),
    "p2pnet": ("*.pth",),
}

BACKEND_LABELS = {
    "yolo": "YOLO (bounding boxes + tracking)",
    "csrnet": "CSRNet (specialized, native density map)",
    "dmcount": "DM-Count (specialized, native density map)",
    "p2pnet": "P2PNet (specialized, point locations)",
}

# Backends that give per-instance localization (boxes or points) and so
# support the instantaneous/cumulative/classic heatmap-mode choice and the
# combined/detection/heatmap output streams, same as YOLO.
INSTANCE_BACKENDS = ("yolo", "p2pnet")
# Backends whose network outputs a density map directly - no per-instance
# localization, no heatmap-mode choice, only a single heatmap output stream.
NATIVE_DENSITY_BACKENDS = ("csrnet", "dmcount")
# Backends that need a local clone of their own repo (see core/dmcount.py, core/p2pnet.py).
EXTERNAL_REPO_BACKENDS = ("dmcount", "p2pnet")


def _prompt_choice(question: str, options: List[str]) -> int:
    print(f"\n{question}")
    for i, opt in enumerate(options, start=1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input(f"Select 1-{len(options)}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("Invalid choice, try again.")


def _prompt_text(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{question}{suffix}: ").strip()
    return raw if raw else default


def _prompt_float(question: str, default: float) -> float:
    raw = input(f"{question} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Not a number, keeping default.")
        return default


def select_backend() -> str:
    order = ["yolo", "csrnet", "dmcount", "p2pnet"]
    idx = _prompt_choice("Select a backend:", [BACKEND_LABELS[b] for b in order])
    return order[idx]


def select_model(backend: str, weights_dir: str = "weights") -> str:
    backend_dir = os.path.join(weights_dir, backend)
    candidates = set()
    for pattern in BACKEND_EXTENSIONS[backend]:
        candidates.update(glob.glob(os.path.join(backend_dir, pattern)))

    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No model files found in '{backend_dir}/' for backend '{backend}'. "
            f"Place a checkpoint there first (see weights/README.txt)."
        )

    idx = _prompt_choice("Select a model file:", [os.path.basename(c) for c in candidates])
    return candidates[idx]


def select_external_repo(backend: str, default_path: str) -> str:
    print(
        f"\n{BACKEND_LABELS[backend]} needs a local clone of its own GitHub repo "
        f"(its model architecture is imported from there, not reproduced in this project)."
    )
    path = _prompt_text("Path to the cloned repo", default_path)
    if not os.path.isdir(path):
        print(f"Warning: '{path}' does not exist yet - this run will fail at model load time.")
    return path


def select_source() -> SourceConfig:
    idx = _prompt_choice("Select input source:", ["Camera (webcam / RTSP stream)", "Video file", "Image folder"])

    if idx == 0:
        path = _prompt_text("Camera index (e.g. 0) or stream URL (e.g. rtsp://...)", "0")
        return SourceConfig(kind="camera", path=path)
    if idx == 1:
        path = _prompt_text("Path to video file", "videos/sample.mp4")
        return SourceConfig(kind="video", path=path)

    path = _prompt_text("Path to image folder", "images/sample")
    return SourceConfig(kind="images", path=path)


def select_heatmap_mode() -> str:
    print(
        "\nHeatmap mode:\n"
        "  Instantaneous : rebuilt every frame, feet-anchored - shows crowd density right now,\n"
        "                  with a color scale that stays comparable across frames.\n"
        "  Cumulative    : persists with decay - shows sustained hotspots over time.\n"
        "  Classic       : rebuilt every frame, always normalized to look visually \"full\".\n"
        "                  Looks punchier, but a single person can look as intense as a\n"
        "                  crowd since each frame is stretched to its own local max."
    )
    idx = _prompt_choice(
        "Select heatmap mode:",
        ["Instantaneous (per-frame, comparable colors)", "Cumulative (temporal decay)", "Classic (vivid, per-frame normalized)"],
    )
    return ["instantaneous", "cumulative", "classic"][idx]


def select_output_format() -> str:
    idx = _prompt_choice("Output format:", ["Video file (.mp4)", "Image sequence (numbered frames)"])
    return "video" if idx == 0 else "images"


def maybe_edit_parameters(run_config: RunConfig) -> None:
    idx = _prompt_choice("Adjust advanced parameters?", ["Keep defaults", "Edit parameters"])
    if idx == 0:
        return

    if run_config.backend == "yolo":
        print("\n--- Detection ---")
        print("conf_threshold: minimum confidence to accept a detection.")
        print("  Higher = fewer false positives (e.g. shadows), but may miss partially-visible people.")
        run_config.detection.conf_threshold = _prompt_float(
            "conf_threshold", run_config.detection.conf_threshold
        )
    elif run_config.backend == "p2pnet":
        print("\n--- Point detection ---")
        print("conf_threshold: minimum point-confidence to accept a detected head.")
        print("  Higher = fewer false positives, but may miss partially-visible/small heads.")
        run_config.detection.conf_threshold = _prompt_float(
            "conf_threshold", run_config.detection.conf_threshold
        )
    elif run_config.backend == "csrnet":
        print("\n--- Counting method ---")
        print("CSRNet can miscount head-shaped clutter (e.g. a flower basket) as people on")
        print("footage unlike its training data. 'peak_detect' mitigates this - see REPORT.md.")
        idx = _prompt_choice(
            "count_method:", ["density_sum (standard CSRNet)", "peak_detect (clutter-robustness mitigation)"]
        )
        run_config.csrnet.count_method = "density_sum" if idx == 0 else "peak_detect"

    print("\n--- Counting ---")
    print("smoothing: EMA factor for the displayed count.")
    print("  Higher = the number reacts faster to change, but flickers more between frames.")
    run_config.counting.smoothing = _prompt_float(
        "smoothing", run_config.counting.smoothing
    )

    if run_config.backend in INSTANCE_BACKENDS and run_config.heatmap.mode == "cumulative":
        print("\n--- Heatmap (cumulative mode) ---")
        print("decay: fraction of past density kept every frame (0-1).")
        print("  Closer to 1 = hotspots fade slowly and reflect a longer time window.")
        print("  Closer to 0 = map behaves almost like instantaneous mode.")
        run_config.heatmap.decay = _prompt_float("decay", run_config.heatmap.decay)


def run_wizard(defaults: RunConfig) -> RunConfig:
    print("=" * 60)
    print("Crowd Density Estimation - Setup")
    print("=" * 60)

    defaults.backend = select_backend()
    defaults.model_path = select_model(defaults.backend)

    if defaults.backend in EXTERNAL_REPO_BACKENDS:
        current_default = getattr(defaults.external_repos, defaults.backend)
        repo_path = select_external_repo(defaults.backend, current_default)
        setattr(defaults.external_repos, defaults.backend, repo_path)

    defaults.source = select_source()

    if defaults.backend in INSTANCE_BACKENDS:
        defaults.heatmap.mode = select_heatmap_mode()
        print(
            "\nThree files will be produced every run: combined (markers+count+heatmap,\n"
            "shown live below), detection-only (markers+count), and heatmap-only (count+heatmap)."
        )
    else:
        print(
            f"\n{BACKEND_LABELS[defaults.backend]} regresses a density map directly from the image,\n"
            "so there are no per-person markers and no heatmap-mode choice - its own native\n"
            "density output is used as-is. One output file will be produced (count+heatmap)."
        )

    save_as = select_output_format()
    defaults.output = OutputConfig(
        video_fps=defaults.output.video_fps,
        video_codec=defaults.output.video_codec,
        image_format=defaults.output.image_format,
        save_as=save_as,
    )
    maybe_edit_parameters(defaults)

    return defaults
