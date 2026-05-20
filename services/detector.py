"""
Face detector using OpenCV Haar Cascade.
Fast, offline, no network — deterministic latency.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.entities import BoundingBox
from core.interfaces import IFaceDetector
from utils.exceptions import RecognitionError
from utils.logger import get_logger

logger = get_logger(__name__)

_CASCADE_LOCK = threading.Lock()


class HaarFaceDetector(IFaceDetector):
    """
    Thread-safe Haar cascade detector.
    One instance per thread recommended; or use the lock.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (60, 60),
        cascade_path: Optional[str] = None,
    ) -> None:
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

        path = cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not Path(path).exists():
            raise RecognitionError(f"Cascade file not found: {path}")

        with _CASCADE_LOCK:
            self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RecognitionError(f"Failed to load cascade: {path}")

        logger.info("HaarFaceDetector initialised", extra={"x_path": path})

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improve detection in varied lighting

        try:
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=self._min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error as exc:
            raise RecognitionError("detectMultiScale failed", cause=exc) from exc

        if faces is None or len(faces) == 0:
            return []

        return [
            BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h))
            for (x, y, w, h) in faces
        ]
