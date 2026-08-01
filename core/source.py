"""
Unified frame source abstraction.

The rest of the pipeline only ever calls `.read()` and `.release()`, so
swapping between a laptop webcam, a video file, a folder of images, or -
later, on the embedded board - a CCTV/RTSP stream requires no changes
outside this file. A CCTV camera is just another `CameraSource` pointed
at a stream URL instead of a local device index.
"""

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


class FrameSource(ABC):
    """Common interface for anything that produces a sequence of frames."""

    fps: float = 30.0

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None when the source is exhausted."""
        raise NotImplementedError

    @abstractmethod
    def get_dims(self) -> Tuple[int, int]:
        """Return (width, height) of the frames produced by this source."""
        raise NotImplementedError

    def release(self) -> None:
        pass


class CameraSource(FrameSource):
    """
    Live source backed by cv2.VideoCapture.

    `identifier` accepts either an integer device index (local USB webcam,
    e.g. 0) or a string (RTSP/HTTP stream URL from an IP/CCTV camera).
    This is the same class that would be used on the embedded board once
    a real camera is attached - only the identifier value changes.
    """

    def __init__(self, identifier):
        try:
            identifier = int(identifier)
        except (TypeError, ValueError):
            pass  # keep as string (stream URL)

        self.cap = cv2.VideoCapture(identifier)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera/stream source: {identifier}")

        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported_fps if reported_fps and reported_fps > 1 else 30.0

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def get_dims(self) -> Tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self) -> None:
        self.cap.release()


class VideoFileSource(FrameSource):
    """Source backed by a video file on disk."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported_fps if reported_fps and reported_fps > 1 else 30.0

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def get_dims(self) -> Tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self) -> None:
        self.cap.release()


class ImageFolderSource(FrameSource):
    """
    Source backed by a directory of still images (e.g. MOT-style img1/
    sequences, or periodic snapshots transmitted by a camera as described
    in the assignment). Frames are played back at a synthetic fps for the
    purposes of video export / FPS display.
    """

    def __init__(self, folder: str, synthetic_fps: float = 30.0):
        if not os.path.isdir(folder):
            raise NotADirectoryError(f"Image folder not found: {folder}")

        paths = []
        for pattern in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(folder, pattern)))
        self.paths = sorted(paths)

        if not self.paths:
            raise FileNotFoundError(f"No images found in: {folder}")

        self.index = 0
        self.fps = synthetic_fps

        first = cv2.imread(self.paths[0])
        self._h, self._w = first.shape[:2]

    def read(self) -> Optional[np.ndarray]:
        if self.index >= len(self.paths):
            return None
        frame = cv2.imread(self.paths[self.index])
        self.index += 1
        return frame

    def get_dims(self) -> Tuple[int, int]:
        return self._w, self._h

    def __len__(self) -> int:
        return len(self.paths)


def build_source(kind: str, path: str) -> FrameSource:
    """Factory used by both the CLI wizard and headless/JSON-config runs."""
    if kind == "camera":
        return CameraSource(path)
    if kind == "video":
        return VideoFileSource(path)
    if kind == "images":
        return ImageFolderSource(path)
    raise ValueError(f"Unknown source kind: {kind}")
