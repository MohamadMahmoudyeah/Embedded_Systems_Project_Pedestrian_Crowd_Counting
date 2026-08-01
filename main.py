"""
Crowd Density Estimation - entry point.

Two ways to run this:

  1. Interactive (manual demo on a laptop):
         python main.py
     Walks through backend / model / source / heatmap / output-format
     choices in the terminal. Every run always writes all output streams
     the selected backend supports (see below) - there is no "what to
     save" choice.

  2. Headless (embedded board / systemd service):
         python main.py --config path/to/run_config.json --no-display
     Every setting is read from a JSON file (the same file that gets written
     automatically to outputs/run_<timestamp>/run_config.json after every
     interactive run). No terminal prompts, no GUI window.

Backends and their output streams:
  yolo, p2pnet   -> per-instance localization (boxes or points), so all
                    three streams are produced: combined (markers+count+
                    heatmap, also the live preview), detection
                    (markers+count, full brightness), heatmap (count+
                    heatmap only, darkened background).
  csrnet, dmcount -> native density regression, no per-instance markers,
                     so only one stream is produced: heatmap (count+
                     heatmap, darkened background).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List

import cv2
import numpy as np

from core.cli import INSTANCE_BACKENDS, run_wizard
from core.config import RunConfig
from core.counter import CountSmoother
from core.output_writer import OutputWriter
from core.renderer import draw_detections, draw_hud
from core.source import build_source
from core.types import Detection

# Starting heatmap-color reference per native-density backend - their raw
# output scales are very different (CSRNet's density values are tiny,
# ~0.001-0.05; DM-Count's are a different, untuned scale), so each gets
# its own adaptive-ceiling starting point rather than sharing one default.
NATIVE_DENSITY_INITIAL_CEILING = {
    "csrnet": None,  # filled in from run_config.csrnet.intensity_ceiling at build time
    "dmcount": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sidewalk crowd density estimation")
    parser.add_argument("--config", type=str, default=None,
                         help="Path to a saved run_config.json for headless execution")
    parser.add_argument("--no-display", action="store_true",
                         help="Never open a preview window, even in interactive mode")
    return parser.parse_args()


def build_run_config(args) -> RunConfig:
    if args.config:
        cfg = RunConfig.load_from_json(args.config)
        cfg.headless = True
        return cfg

    defaults = RunConfig.load_defaults()
    return run_wizard(defaults)


def darken(frame: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def build_get_detections(run_config: RunConfig) -> Callable[[np.ndarray], List[Detection]]:
    """Returns a frame -> list[Detection] callable for either instance-based backend."""
    if run_config.backend == "yolo":
        from core.detector import PersonDetector
        detector = PersonDetector(run_config.model_path, run_config.detection)
        return detector.detect

    if run_config.backend == "p2pnet":
        from core.p2pnet import P2PNetEstimator
        estimator = P2PNetEstimator(
            repo_path=run_config.external_repos.p2pnet,
            checkpoint_path=run_config.model_path,
            backbone=run_config.p2pnet.backbone,
            row=run_config.p2pnet.row,
            line=run_config.p2pnet.line,
            threshold=run_config.detection.conf_threshold,
            marker_half_size=run_config.p2pnet.marker_half_size,
        )
        return estimator.detect

    raise ValueError(f"Not an instance-based backend: {run_config.backend}")


def build_density_estimator(run_config: RunConfig):
    """Returns an object with .estimate(frame) -> (count, density_map) for a native-density backend."""
    if run_config.backend == "csrnet":
        from core.csrnet import CSRNetEstimator
        return CSRNetEstimator(run_config.model_path, cfg=run_config.csrnet)

    if run_config.backend == "dmcount":
        from core.dmcount import DMCountEstimator
        return DMCountEstimator(run_config.external_repos.dmcount, run_config.model_path)

    raise ValueError(f"Not a native-density backend: {run_config.backend}")


def run_instance_based(get_detections: Callable[[np.ndarray], List[Detection]], run_config: RunConfig,
                        source, frame_w: int, frame_h: int, counter: CountSmoother, writer: OutputWriter,
                        show_display: bool) -> int:
    from core.heatmap import HeatmapGenerator

    is_point_based = run_config.backend == "p2pnet"
    heatmap_gen = HeatmapGenerator(run_config.heatmap, frame_w, frame_h, use_point_sigma=is_point_based)
    darken_factor = run_config.heatmap.background_darken

    frame_count = 0
    while True:
        frame = source.read()
        if frame is None:
            break

        start = time.time()
        detections = get_detections(frame)
        counter.update(len(detections))
        heatmap_gen.update(detections)
        fps = 1.0 / (time.time() - start) if time.time() > start else 0.0

        heat_base = heatmap_gen.render(darken(frame, darken_factor))

        detection_frame = draw_hud(draw_detections(frame, detections, is_point_based), counter.rounded, fps)
        heatmap_frame = draw_hud(heat_base, counter.rounded, fps)
        combined_frame = draw_hud(draw_detections(heat_base.copy(), detections, is_point_based), counter.rounded, fps)

        writer.write("detection", detection_frame)
        writer.write("heatmap", heatmap_frame)
        writer.write("combined", combined_frame)

        frame_count += 1

        if show_display:
            try:
                cv2.imshow("Crowd Density Estimation", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break
            except cv2.error:
                show_display = False

    return frame_count


def run_native_density(estimator, run_config: RunConfig, source, frame_w: int, frame_h: int,
                        counter: CountSmoother, writer: OutputWriter, show_display: bool) -> int:
    from core.heatmap import NativeDensityRenderer

    if run_config.backend == "csrnet":
        initial_ceiling = run_config.csrnet.intensity_ceiling
    else:
        initial_ceiling = NATIVE_DENSITY_INITIAL_CEILING.get(run_config.backend, 1.0)

    renderer = NativeDensityRenderer(
        overlay_alpha=run_config.heatmap.overlay_alpha,
        initial_ceiling=initial_ceiling,
        noise_floor=run_config.heatmap.noise_floor,
    )
    darken_factor = run_config.heatmap.background_darken

    frame_count = 0
    while True:
        frame = source.read()
        if frame is None:
            break

        start = time.time()
        count, density = estimator.estimate(frame)
        counter.update(count)
        fps = 1.0 / (time.time() - start) if time.time() > start else 0.0

        heat_frame = renderer.render(darken(frame, darken_factor), density)
        heat_frame = draw_hud(heat_frame, counter.rounded, fps)

        writer.write("heatmap", heat_frame)
        frame_count += 1

        if show_display:
            try:
                cv2.imshow("Crowd Density Estimation", heat_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break
            except cv2.error:
                show_display = False

    return frame_count


def main():
    args = parse_args()
    run_config = build_run_config(args)

    print(f"\nBackend: {run_config.backend}")
    print(f"Model: {run_config.model_path}")
    print(f"Opening source ({run_config.source.kind}): {run_config.source.path}")

    source = build_source(run_config.source.kind, run_config.source.path)
    frame_w, frame_h = source.get_dims()

    counter = CountSmoother(run_config.counting)
    is_instance_based = run_config.backend in INSTANCE_BACKENDS
    streams = ["combined", "detection", "heatmap"] if is_instance_based else ["heatmap"]
    writer = OutputWriter(run_config.output, run_config, frame_w, frame_h, streams=streams)

    show_display = not run_config.headless and not args.no_display

    print(f"Saving results to: {writer.run_dir}")
    print("Press 'q' in the preview window to stop early.\n" if show_display else "Running headless...\n")

    frame_count = 0
    try:
        if is_instance_based:
            get_detections = build_get_detections(run_config)
            frame_count = run_instance_based(get_detections, run_config, source, frame_w, frame_h,
                                              counter, writer, show_display)
        else:
            estimator = build_density_estimator(run_config)
            frame_count = run_native_density(estimator, run_config, source, frame_w, frame_h,
                                              counter, writer, show_display)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        writer.release()
        source.release()
        if show_display:
            cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames.")
    print(f"Results saved in: {writer.run_dir}")


if __name__ == "__main__":
    sys.exit(main())
