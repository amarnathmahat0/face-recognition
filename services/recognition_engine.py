"""
Recognition engine: cosine similarity matching against registered identities.
Strategy pattern — swap to ANN index for large-scale without touching other code.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from core.entities import FaceIdentity
from core.interfaces import IRecognitionEngine
from utils.logger import get_logger

logger = get_logger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]. Both vectors must be non-zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class CosineRecognitionEngine(IRecognitionEngine):
    """
    Brute-force cosine similarity.
    Suitable for up to ~1000 identities at real-time frame rates.

    For larger datasets, swap to FAISS / sklearn BallTree without
    changing any other code (same IRecognitionEngine interface).
    """

    def __init__(self, similarity_threshold: float) -> None:
        self._threshold = similarity_threshold
        self._identities: list[FaceIdentity] = []
        # Pre-computed mean embeddings for fast matching
        self._mean_embeddings: list[np.ndarray] = []
        self._lock = threading.RLock()

    def reload(self, identities: list[FaceIdentity]) -> None:
        """Rebuild the index from a fresh list of identities."""
        means = []
        for ident in identities:
            try:
                means.append(ident.mean_embedding())
            except ValueError:
                logger.warning(
                    "Identity %r has no embeddings — skipping",
                    ident.identity_id,
                )
                means.append(None)

        with self._lock:
            self._identities = list(identities)
            self._mean_embeddings = means
        logger.info(
            "RecognitionEngine index reloaded",
            extra={"x_identity_count": len(identities)},
        )

    def match(
        self,
        embedding: np.ndarray,
        identities: list[FaceIdentity],
    ) -> tuple[Optional[FaceIdentity], float]:
        """
        Find the closest identity to `embedding`.
        Returns (best_identity, similarity) or (None, 0.0).
        """
        with self._lock:
            index_ids = self._identities
            index_embs = self._mean_embeddings

        if not index_ids:
            return None, 0.0

        best_identity: Optional[FaceIdentity] = None
        best_score = -1.0

        for identity, mean_emb in zip(index_ids, index_embs):
            if mean_emb is None:
                continue
            score = _cosine_similarity(embedding, mean_emb)
            if score > best_score:
                best_score = score
                best_identity = identity

        if best_score < self._threshold:
            return None, best_score

        logger.debug(
            "Match: %s (%.3f)",
            best_identity.identity_id if best_identity else "unknown",
            best_score,
        )
        return best_identity, best_score

    def compute_liveness(
        self,
        current_frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        motion_threshold: float = 0.5,
    ) -> float:
        """
        Motion-variance liveness hint.
        Returns 0.0 (likely static/spoof) to 1.0 (live).
        Pure frame delta — no ML required.
        """
        if prev_frame is None or current_frame.shape != prev_frame.shape:
            return 1.0  # assume live if no prior frame

        import cv2
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(curr_gray - prev_gray)
        mean_diff = float(diff.mean())

        # Normalise: < threshold → likely static photo
        score = min(1.0, mean_diff / (motion_threshold * 255.0))
        return score
