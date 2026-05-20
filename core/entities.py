"""
Core domain entities — pure Python dataclasses.
Zero imports from services, ui, or utils (except stdlib).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


class ConfidenceBand(Enum):
    HIGH = auto()      # similarity >= confidence_high threshold
    MEDIUM = auto()    # similarity >= similarity_threshold
    LOW = auto()       # below threshold but a face was found
    UNKNOWN = auto()   # no registered identities or no match at all


class SampleStatus(Enum):
    ACCEPTED = auto()
    REJECTED_NO_FACE = auto()
    REJECTED_MULTIPLE_FACES = auto()
    REJECTED_BLUR = auto()
    REJECTED_TOO_SMALL = auto()
    REJECTED_FRAME_GAP = auto()  # skipped due to frame-gap enforcement


class ThreadStatus(Enum):
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    CRASHED = auto()
    RESTARTING = auto()


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def min_side(self) -> int:
        return min(self.w, self.h)


@dataclass
class FaceIdentity:
    """A registered identity with stored embeddings."""
    identity_id: str                     # unique slug, e.g. "john_doe"
    display_name: str                    # human-readable label
    embeddings: list[np.ndarray]         # list of float32 (128,) vectors
    sample_count: int = 0
    created_at: str = ""                 # ISO-8601 timestamp
    updated_at: str = ""

    def mean_embedding(self) -> np.ndarray:
        """Return mean embedding across all samples."""
        if not self.embeddings:
            raise ValueError(f"Identity {self.identity_id!r} has no embeddings")
        arr = np.stack(self.embeddings, axis=0)  # (N, 128)
        return arr.mean(axis=0).astype(np.float32)


@dataclass(frozen=True)
class SampleResult:
    """Result of a single frame during registration sampling."""
    status: SampleStatus
    frame_index: int
    rejection_reason: str = ""
    bbox: Optional[BoundingBox] = None


@dataclass(frozen=True)
class FaceMatch:
    """A single face detection result with identity match attempt."""
    bbox: BoundingBox
    identity_id: Optional[str]           # None if unknown
    display_name: Optional[str]
    similarity: float                    # cosine similarity score [0, 1]
    confidence: ConfidenceBand
    liveness_score: float = 1.0         # 0=static/spoof hint, 1=live


@dataclass(frozen=True)
class VerificationResult:
    """Result of one verification frame — may contain multiple faces."""
    matches: list[FaceMatch]
    inference_ms: float                  # time taken for this frame
    frame_index: int

    @property
    def has_faces(self) -> bool:
        return len(self.matches) > 0

    @property
    def best_match(self) -> Optional[FaceMatch]:
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.similarity)


@dataclass
class RegistrationSession:
    """Mutable state for an ongoing registration workflow."""
    identity_id: str
    display_name: str
    target_samples: int
    accepted_samples: list[np.ndarray] = field(default_factory=list)
    rejected_count: int = 0
    frame_counter: int = 0
    completed: bool = False
    error: Optional[str] = None

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_samples)

    @property
    def progress(self) -> float:
        return min(1.0, self.accepted_count / self.target_samples)

    @property
    def is_ready(self) -> bool:
        return self.accepted_count >= self.target_samples
