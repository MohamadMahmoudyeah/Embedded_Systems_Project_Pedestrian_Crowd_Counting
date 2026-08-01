"""
Handles where and how results are saved.

Every run gets its own timestamped folder under outputs/, e.g.:

    outputs/run_20260729_143210/
        run_config.json          <- exact settings used, for reproducibility
        combined_20260729_143210.mp4
        detection_20260729_143210.mp4
        heatmap_20260729_143210.mp4

or, if "images" was selected as the save format, a subfolder of numbered
frames per stream instead of a single video file.

Which streams get produced is decided by the backend (YOLO produces all
three: "combined", "detection", "heatmap"; CSRNet produces only
"heatmap", since it has no bounding boxes/IDs) - it is not a choice the
user makes, so this class just writes whatever stream names it is given.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

from core.config import OutputConfig, RunConfig


class OutputWriter:
    def __init__(self, cfg: OutputConfig, run_config: RunConfig, frame_w: int, frame_h: int,
                 streams: List[str], outputs_root: str = "outputs"):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(outputs_root, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.active_streams = list(streams)

        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._image_dirs: Dict[str, str] = {}
        self._image_counters: Dict[str, int] = {}

        for stream in self.active_streams:
            if cfg.save_as == "video":
                path = os.path.join(self.run_dir, f"{stream}_{self.timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*cfg.video_codec)
                self._video_writers[stream] = cv2.VideoWriter(path, fourcc, cfg.video_fps, (frame_w, frame_h))
            else:
                stream_dir = os.path.join(self.run_dir, f"{stream}_frames")
                os.makedirs(stream_dir, exist_ok=True)
                self._image_dirs[stream] = stream_dir
                self._image_counters[stream] = 0

        run_config.save(os.path.join(self.run_dir, "run_config.json"))

    def write(self, stream: str, frame: np.ndarray) -> None:
        if stream not in self.active_streams:
            return

        if self.cfg.save_as == "video":
            self._video_writers[stream].write(frame)
        else:
            idx = self._image_counters[stream]
            filename = f"frame_{idx:06d}.{self.cfg.image_format}"
            cv2.imwrite(os.path.join(self._image_dirs[stream], filename), frame)
            self._image_counters[stream] = idx + 1

    def release(self) -> None:
        for writer in self._video_writers.values():
            writer.release()
