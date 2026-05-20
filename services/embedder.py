"""
Face embedder using the `face_recognition` library (dlib under the hood).
Produces 128-dim float32 vectors. Fully offline — no cloud calls.
"""
from __future__ import annotations

import threading

import numpy as np

from core.entities import BoundingBox
from core.interfaces import IEmbedder
from utils.exceptions import EmbeddingError
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy import — dlib/face_recognition is heavy, load once
_IMPORT_LOCK = threading.Lock()
_face_recognition = None


def _get_face_recognition():
    global _face_recognition
    if _face_recognition is None:
        with _IMPORT_LOCK:
            if _face_recognition is None:
                try:
                    import face_recognition as fr
                    _face_recognition = fr
                    logger.info("face_recognition library loaded")
                except ImportError as exc:
                    raise EmbeddingError(
                        "face_recognition not installed. Run: pip install face-recognition",
                        cause=exc,
                    ) from exc
    return _face_recognition


class FaceRecognitionEmbedder(IEmbedder):
    """
    Wraps face_recognition.face_encodings() with explicit bbox injection.
    Thread-safe — dlib models are stateless per call.
    """

    _EMBEDDING_DIM = 128

    def __init__(self, num_jitters: int = 1, model: str = "small") -> None:
        """
        Args:
            num_jitters: Extra jitter passes — higher = more accurate, slower.
                         1 = fast (default), 100 = very accurate.
            model: 'small' (5-point, faster) or 'large' (68-point, slower).
        """
        self._num_jitters = num_jitters
        self._model = model
        # Pre-load the library on construction to fail fast
        _get_face_recognition()
        logger.info(
            "FaceRecognitionEmbedder ready",
            extra={"x_jitters": num_jitters, "x_model": model},
        )

    def embed(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        fr = _get_face_recognition()

        # Convert BGR (OpenCV) to RGB (face_recognition) and ensure contiguous memory
        rgb = frame[:, :, ::-1]
        rgb = np.ascontiguousarray(rgb)

        # face_recognition expects (top, right, bottom, left) order
        # Ensure coordinates are plain Python ints (not numpy types)
        top = int(bbox.y)
        right = int(bbox.x + bbox.w)
        bottom = int(bbox.y + bbox.h)
        left = int(bbox.x)

        known_face_locations = [(top, right, bottom, left)]

        try:
            encodings = fr.face_encodings(
                rgb,
                known_face_locations=known_face_locations,
                num_jitters=self._num_jitters,
                model=self._model,
            )
        except Exception as exc:
            raise EmbeddingError(f"Encoding failed: {exc}", cause=exc) from exc

        if not encodings:
            raise EmbeddingError("face_encodings returned empty list for provided bbox")

        return encodings[0].astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._EMBEDDING_DIM
