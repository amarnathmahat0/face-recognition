"""
Unit tests for RegisterFaceUseCase and VerifyFaceUseCase.
No camera, no UI, no disk I/O — fully isolated with mock implementations.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.entities import (
    BoundingBox, ConfidenceBand, FaceIdentity, SampleStatus,
)
from core.interfaces import IEmbedder, IFaceDetector, IRecognitionEngine
from core.use_cases import RegisterFaceUseCase, VerifyFaceUseCase


# ── Mock implementations ───────────────────────────────────────────────────────

class MockDetector(IFaceDetector):
    def __init__(self, boxes: list[list[BoundingBox]] | None = None) -> None:
        self._boxes = boxes or []
        self._call_count = 0

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        idx = min(self._call_count, len(self._boxes) - 1)
        result = self._boxes[idx] if self._boxes else []
        self._call_count += 1
        return result


class MockEmbedder(IEmbedder):
    def __init__(self, embedding_val: float = 0.5) -> None:
        self._val = embedding_val

    def embed(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        return np.full(128, self._val, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return 128


class MockEngine(IRecognitionEngine):
    def __init__(self, result=None) -> None:
        self._result = result

    def match(self, embedding, identities):
        if self._result:
            return self._result
        return (None, 0.0)

    def reload(self, identities):
        pass


def _make_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _good_bbox() -> BoundingBox:
    return BoundingBox(x=100, y=100, w=120, h=120)


def _blur_good(_roi) -> float:
    return 200.0  # above threshold


def _blur_bad(_roi) -> float:
    return 10.0   # below threshold


def _liveness_good(cur, prev) -> float:
    return 1.0


# ── RegisterFaceUseCase tests ──────────────────────────────────────────────────

def _make_register_use_case(detector, embedder, blur_fn=_blur_good):
    return RegisterFaceUseCase(
        detector=detector,
        embedder=embedder,
        min_face_size=80,
        blur_threshold=100.0,
        sample_frame_gap=1,
        target_samples=3,
        max_samples=5,
        compute_blur=blur_fn,
    )


class TestRegisterFaceUseCase:
    def test_begin_session_creates_slug(self):
        uc = _make_register_use_case(MockDetector(), MockEmbedder())
        session = uc.begin_session("Alice Johnson")
        assert session.identity_id == "alice_johnson"
        assert session.display_name == "Alice Johnson"

    def test_process_frame_gap_skip(self):
        uc = RegisterFaceUseCase(
            detector=MockDetector([[_good_bbox()]]),
            embedder=MockEmbedder(),
            min_face_size=80,
            blur_threshold=100.0,
            sample_frame_gap=5,
            target_samples=3,
            max_samples=5,
            compute_blur=_blur_good,
        )
        session = uc.begin_session("Test")
        # frame_counter becomes 1 after first call → 1 % 5 != 0 → skip
        result = uc.process_frame(session, _make_frame())
        assert result.status == SampleStatus.REJECTED_FRAME_GAP

    def test_process_frame_no_face(self):
        detector = MockDetector([[]])  # returns empty list always
        uc = _make_register_use_case(detector, MockEmbedder())
        session = uc.begin_session("Test")
        result = uc.process_frame(session, _make_frame())
        assert result.status == SampleStatus.REJECTED_NO_FACE
        assert session.rejected_count == 1

    def test_process_frame_multiple_faces(self):
        detector = MockDetector([[_good_bbox(), _good_bbox()]])
        uc = _make_register_use_case(detector, MockEmbedder())
        session = uc.begin_session("Test")
        result = uc.process_frame(session, _make_frame())
        assert result.status == SampleStatus.REJECTED_MULTIPLE_FACES

    def test_process_frame_too_small(self):
        small_box = BoundingBox(x=0, y=0, w=40, h=40)  # below min_face_size=80
        detector = MockDetector([[small_box]])
        uc = _make_register_use_case(detector, MockEmbedder())
        session = uc.begin_session("Test")
        result = uc.process_frame(session, _make_frame())
        assert result.status == SampleStatus.REJECTED_TOO_SMALL

    def test_process_frame_blur_fail(self):
        detector = MockDetector([[_good_bbox()]])
        uc = _make_register_use_case(detector, MockEmbedder(), blur_fn=_blur_bad)
        session = uc.begin_session("Test")
        result = uc.process_frame(session, _make_frame())
        assert result.status == SampleStatus.REJECTED_BLUR

    def test_accepted_sample_increments_count(self):
        detector = MockDetector([[_good_bbox()]] * 10)
        uc = _make_register_use_case(detector, MockEmbedder())
        session = uc.begin_session("Test")
        result = uc.process_frame(session, _make_frame())
        assert result.status == SampleStatus.ACCEPTED
        assert session.accepted_count == 1

    def test_fifo_eviction_at_max(self):
        detector = MockDetector([[_good_bbox()]] * 100)
        uc = RegisterFaceUseCase(
            detector=detector,
            embedder=MockEmbedder(),
            min_face_size=80,
            blur_threshold=100.0,
            sample_frame_gap=1,
            target_samples=20,
            max_samples=3,  # low max for test
            compute_blur=_blur_good,
        )
        session = uc.begin_session("Test")
        for _ in range(10):
            uc.process_frame(session, _make_frame())
        assert session.accepted_count <= 3

    def test_finalise_builds_identity(self):
        detector = MockDetector([[_good_bbox()]] * 3)
        uc = _make_register_use_case(detector, MockEmbedder())
        session = uc.begin_session("Bob")
        for _ in range(3):
            uc.process_frame(session, _make_frame())
        identity = uc.finalise(session)
        assert identity.identity_id == "bob"
        assert identity.display_name == "Bob"
        assert len(identity.embeddings) == 3
        assert identity.embeddings[0].dtype == np.float32

    def test_finalise_empty_session_raises(self):
        from utils.exceptions import InsufficientSamplesError
        uc = _make_register_use_case(MockDetector([[]]), MockEmbedder())
        session = uc.begin_session("Empty")
        with pytest.raises(InsufficientSamplesError):
            uc.finalise(session)

    def test_mean_embedding_shape(self):
        identity = FaceIdentity(
            identity_id="test",
            display_name="Test",
            embeddings=[np.ones(128, dtype=np.float32) * 0.5],
        )
        mean = identity.mean_embedding()
        assert mean.shape == (128,)
        assert mean.dtype == np.float32


# ── VerifyFaceUseCase tests ────────────────────────────────────────────────────

def _make_verify_use_case(detector, embedder, engine, threshold=0.6):
    return VerifyFaceUseCase(
        detector=detector,
        embedder=embedder,
        engine=engine,
        similarity_threshold=threshold,
        confidence_high=0.80,
        confidence_medium=0.65,
        compute_blur=_blur_good,
        compute_liveness=_liveness_good,
    )


def _make_identity(identity_id="alice", val=0.9) -> FaceIdentity:
    return FaceIdentity(
        identity_id=identity_id,
        display_name=identity_id.capitalize(),
        embeddings=[np.full(128, val, dtype=np.float32)],
    )


class TestVerifyFaceUseCase:
    def test_no_face_returns_empty_matches(self):
        uc = _make_verify_use_case(
            MockDetector([[]]), MockEmbedder(), MockEngine()
        )
        result = uc.verify(_make_frame(), [])
        assert result.matches == []
        assert not result.has_faces

    def test_known_face_returns_match(self):
        identity = _make_identity("alice", 0.9)
        engine = MockEngine(result=(identity, 0.92))
        uc = _make_verify_use_case(
            MockDetector([[_good_bbox()]]), MockEmbedder(0.9), engine
        )
        result = uc.verify(_make_frame(), [identity])
        assert result.has_faces
        best = result.best_match
        assert best is not None
        assert best.identity_id == "alice"
        assert best.similarity == 0.92

    def test_high_confidence_band(self):
        identity = _make_identity()
        engine = MockEngine(result=(identity, 0.95))
        uc = _make_verify_use_case(
            MockDetector([[_good_bbox()]]), MockEmbedder(0.9), engine
        )
        result = uc.verify(_make_frame(), [identity])
        assert result.best_match.confidence == ConfidenceBand.HIGH

    def test_unknown_band_below_threshold(self):
        engine = MockEngine(result=(None, 0.3))
        uc = _make_verify_use_case(
            MockDetector([[_good_bbox()]]), MockEmbedder(0.1), engine
        )
        result = uc.verify(_make_frame(), [])
        best = result.best_match
        assert best.confidence == ConfidenceBand.UNKNOWN

    def test_multiple_faces_multiple_matches(self):
        boxes = [_good_bbox(), BoundingBox(300, 100, 120, 120)]
        engine = MockEngine(result=(None, 0.0))
        uc = _make_verify_use_case(
            MockDetector([boxes]), MockEmbedder(), engine
        )
        result = uc.verify(_make_frame(), [])
        assert len(result.matches) == 2

    def test_inference_ms_positive(self):
        uc = _make_verify_use_case(
            MockDetector([[]]), MockEmbedder(), MockEngine()
        )
        result = uc.verify(_make_frame(), [])
        assert result.inference_ms >= 0.0
