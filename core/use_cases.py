"""
Application use-cases — pure orchestration logic.
No I/O, no UI imports, no OpenCV. Fully unit-testable with mocks.
"""
from __future__ import annotations

import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from core.entities import (
    BoundingBox,
    ConfidenceBand,
    FaceIdentity,
    FaceMatch,
    RegistrationSession,
    SampleResult,
    SampleStatus,
    VerificationResult,
)
from core.interfaces import IEmbedder, IFaceDetector, IFaceRepository, IRecognitionEngine
from utils.exceptions import (
    FaceBlurryError,
    FaceTooSmallError,
    InsufficientSamplesError,
    MultipleFacesError,
    NoFaceDetectedError,
    TrainingMemoryError,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9_]")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("_", name.lower().strip()).strip("_") or "identity"


class RegisterFaceUseCase:
    """
    Orchestrates multi-sample face registration.

    Lifecycle:
        session = use_case.begin_session("Alice")
        for frame in frames:
            result = use_case.process_frame(session, frame)
        identity = use_case.finalise(session)
        repository.save(identity)
    """

    def __init__(
        self,
        detector: IFaceDetector,
        embedder: IEmbedder,
        min_face_size: int,
        blur_threshold: float,
        sample_frame_gap: int,
        target_samples: int,
        max_samples: int,
        compute_blur: Callable[[np.ndarray], float],
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self._min_face_size = min_face_size
        self._blur_threshold = blur_threshold
        self._sample_frame_gap = sample_frame_gap
        self._target_samples = target_samples
        self._max_samples = max_samples
        self._compute_blur = compute_blur

    def begin_session(self, display_name: str) -> RegistrationSession:
        identity_id = _slugify(display_name)
        return RegistrationSession(
            identity_id=identity_id,
            display_name=display_name,
            target_samples=self._target_samples,
        )

    def process_frame(
        self,
        session: RegistrationSession,
        frame: np.ndarray,
    ) -> SampleResult:
        """Validate and (if passing) capture one embedding sample."""
        session.frame_counter += 1

        # Enforce frame gap
        if session.frame_counter % self._sample_frame_gap != 0:
            return SampleResult(
                status=SampleStatus.REJECTED_FRAME_GAP,
                frame_index=session.frame_counter,
                rejection_reason="frame gap skip",
            )

        if session.is_ready:
            return SampleResult(
                status=SampleStatus.ACCEPTED,
                frame_index=session.frame_counter,
                rejection_reason="already complete",
            )

        # Detect faces
        boxes = self._detector.detect(frame)
        if len(boxes) == 0:
            session.rejected_count += 1
            return SampleResult(
                status=SampleStatus.REJECTED_NO_FACE,
                frame_index=session.frame_counter,
                rejection_reason="no face detected",
            )
        if len(boxes) > 1:
            session.rejected_count += 1
            return SampleResult(
                status=SampleStatus.REJECTED_MULTIPLE_FACES,
                frame_index=session.frame_counter,
                rejection_reason=f"{len(boxes)} faces in frame",
            )

        bbox = boxes[0]

        # Size check
        if bbox.min_side < self._min_face_size:
            session.rejected_count += 1
            return SampleResult(
                status=SampleStatus.REJECTED_TOO_SMALL,
                frame_index=session.frame_counter,
                rejection_reason=f"face {bbox.min_side}px < min {self._min_face_size}px",
                bbox=bbox,
            )

        # Blur check
        roi = _extract_roi(frame, bbox)
        blur_score = self._compute_blur(roi)
        if blur_score < self._blur_threshold:
            session.rejected_count += 1
            return SampleResult(
                status=SampleStatus.REJECTED_BLUR,
                frame_index=session.frame_counter,
                rejection_reason=f"blur {blur_score:.1f} < threshold {self._blur_threshold}",
                bbox=bbox,
            )

        # Extract embedding
        embedding = self._embedder.embed(frame, bbox).astype(np.float32)

        # FIFO eviction if at max
        if len(session.accepted_samples) >= self._max_samples:
            session.accepted_samples.pop(0)

        session.accepted_samples.append(embedding)
        return SampleResult(
            status=SampleStatus.ACCEPTED,
            frame_index=session.frame_counter,
            bbox=bbox,
        )

    def finalise(self, session: RegistrationSession) -> FaceIdentity:
        """Build a FaceIdentity from a completed session."""
        if session.accepted_count < 1:
            raise InsufficientSamplesError(
                f"Session for {session.display_name!r} has 0 accepted samples"
            )

        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            return FaceIdentity(
                identity_id=session.identity_id,
                display_name=session.display_name,
                embeddings=list(session.accepted_samples),  # copy
                sample_count=session.accepted_count,
                created_at=now,
                updated_at=now,
            )
        except MemoryError as exc:
            raise TrainingMemoryError(
                "OOM while building FaceIdentity", cause=exc
            ) from exc


class VerifyFaceUseCase:
    """
    Real-time face verification against registered identities.
    Stateless — call verify() per frame.
    """

    def __init__(
        self,
        detector: IFaceDetector,
        embedder: IEmbedder,
        engine: IRecognitionEngine,
        similarity_threshold: float,
        confidence_high: float,
        confidence_medium: float,
        compute_blur: Callable[[np.ndarray], float],
        compute_liveness: Callable[[np.ndarray, Optional[np.ndarray]], float],
        liveness_threshold: float = 0.30,
        liveness_window: int = 5,
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self._engine = engine
        self._threshold = similarity_threshold
        self._high = confidence_high
        self._medium = confidence_medium
        self._compute_blur = compute_blur
        self._compute_liveness = compute_liveness
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_index = 0
        self._liveness_threshold = liveness_threshold
        self._liveness_history: deque[float] = deque(maxlen=liveness_window)

    def verify(
        self,
        frame: np.ndarray,
        identities: list[FaceIdentity],
    ) -> VerificationResult:
        start = time.perf_counter()
        self._frame_index += 1

        boxes = self._detector.detect(frame)
        matches: list[FaceMatch] = []

        raw_liveness = self._compute_liveness(frame, self._prev_frame)
        self._prev_frame = frame  # keep reference for next frame
        self._liveness_history.append(raw_liveness)
        smooth_liveness = float(sum(self._liveness_history) / len(self._liveness_history))
        if smooth_liveness < self._liveness_threshold:
            logger.debug(
                "Low temporal liveness: %.3f (threshold %.3f)",
                smooth_liveness,
                self._liveness_threshold,
            )

        for bbox in boxes:
            try:
                embedding = self._embedder.embed(frame, bbox).astype(np.float32)
            except Exception:
                # Embedding failed for this face — skip silently (logged upstream)
                continue

            if identities:
                matched_id, similarity = self._engine.match(embedding, identities)
            else:
                matched_id, similarity = None, 0.0

            confidence = self._band(similarity, bool(matched_id))

            matches.append(
                FaceMatch(
                    bbox=bbox,
                    identity_id=matched_id.identity_id if matched_id else None,
                    display_name=matched_id.display_name if matched_id else None,
                    similarity=similarity,
                    confidence=confidence,
                    liveness_score=smooth_liveness,
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return VerificationResult(
            matches=matches,
            inference_ms=elapsed_ms,
            frame_index=self._frame_index,
        )

    def _band(self, similarity: float, has_match: bool) -> ConfidenceBand:
        if not has_match or similarity < self._threshold:
            return ConfidenceBand.UNKNOWN
        if similarity >= self._high:
            return ConfidenceBand.HIGH
        if similarity >= self._medium:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW


def _extract_roi(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Safely extract and return face ROI from frame."""
    h, w = frame.shape[:2]
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(w, bbox.x + bbox.w)
    y2 = min(h, bbox.y + bbox.h)
    return frame[y1:y2, x1:x2]
