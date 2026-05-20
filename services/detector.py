"""
Face detector using OpenCV Haar Cascade.
Fast, offline, no network — deterministic latency.
Supports both frontal and profile face detection for robust pose tolerance.
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
    Thread-safe Haar cascade detector with optional profile support.
    Detects both frontal and profile faces to improve pose tolerance.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (60, 60),
        cascade_path: Optional[str] = None,
        use_profile: bool = False,
    ) -> None:
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._use_profile = use_profile

        # Load frontal cascade
        frontal_path = cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not Path(frontal_path).exists():
            raise RecognitionError(f"Cascade file not found: {frontal_path}")

        with _CASCADE_LOCK:
            self._frontal_cascade = cv2.CascadeClassifier(frontal_path)
        if self._frontal_cascade.empty():
            raise RecognitionError(f"Failed to load frontal cascade: {frontal_path}")

        logger.info("HaarFaceDetector initialised", extra={"x_path": frontal_path})

        # Try to load profile cascade if requested
        self._profile_cascade = None
        if use_profile:
            profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            if Path(profile_path).exists():
                with _CASCADE_LOCK:
                    self._profile_cascade = cv2.CascadeClassifier(profile_path)
                if not self._profile_cascade.empty():
                    logger.info("Profile Face Cascade loaded", extra={"x_path": profile_path})
                else:
                    logger.warning("Profile cascade found but failed to load")
                    self._profile_cascade = None
            else:
                logger.warning("Profile cascade not available on this system")

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improve detection in varied lighting

        boxes = []

        # Detect frontal faces
        try:
            frontal_faces = self._frontal_cascade.detectMultiScale(
                gray,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=self._min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error as exc:
            raise RecognitionError("frontal detectMultiScale failed", cause=exc) from exc

        if frontal_faces is not None and len(frontal_faces) > 0:
            boxes.extend([
                BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h))
                for (x, y, w, h) in frontal_faces
            ])

        # Detect profile faces if enabled
        if self._use_profile and self._profile_cascade is not None:
            try:
                profile_faces = self._profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self._scale_factor,
                    minNeighbors=self._min_neighbors,
                    minSize=self._min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                if profile_faces is not None and len(profile_faces) > 0:
                    profile_boxes = [
                        BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h))
                        for (x, y, w, h) in profile_faces
                    ]
                    # Filter out duplicates (faces detected by both cascades)
                    boxes.extend(self._filter_overlaps(boxes, profile_boxes, iou_threshold=0.3))
            except cv2.error as exc:
                logger.warning("profile detectMultiScale failed: %s", exc)

        return boxes

    @staticmethod
    def _filter_overlaps(
        existing: list[BoundingBox],
        new_boxes: list[BoundingBox],
        iou_threshold: float = 0.3,
    ) -> list[BoundingBox]:
        """Filter out new boxes that overlap too much with existing ones."""
        filtered = []
        for new_box in new_boxes:
            is_duplicate = False
            for existing_box in existing:
                if HaarFaceDetector._iou(existing_box, new_box) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(new_box)
        return filtered

    @staticmethod
    def _iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute intersection over union between two bounding boxes."""
        x1_min, y1_min = box1.x, box1.y
        x1_max, y1_max = box1.x + box1.w, box1.y + box1.h
        x2_min, y2_min = box2.x, box2.y
        x2_max, y2_max = box2.x + box2.w, box2.y + box2.h

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = box1.w * box1.h
        box2_area = box2.w * box2.h
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
