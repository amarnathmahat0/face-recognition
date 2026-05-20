"""
Abstract interfaces (protocols) for all service layer contracts.
Core layer depends only on these — never on concrete implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional
import numpy as np

from core.entities import BoundingBox, FaceIdentity


class ICameraSource(ABC):
    """Contract for any frame source (webcam, file, mock)."""

    @abstractmethod
    def start(self) -> None:
        """Start capturing frames."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and release resources."""

    @abstractmethod
    def latest_frame(self) -> Optional[np.ndarray]:
        """Return most recent BGR frame, or None if unavailable."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return True if the camera thread is running and healthy."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Measured frames-per-second of the capture stream."""


class IFaceDetector(ABC):
    """Contract for face detection (bounding boxes only, no embedding)."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """
        Detect faces in a BGR frame.
        Returns list of BoundingBox (may be empty).
        Raises RecognitionError on hard failure.
        """


class IEmbedder(ABC):
    """Contract for face embedding extraction."""

    @abstractmethod
    def embed(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Extract a float32 embedding vector from the face ROI.
        Raises EmbeddingError if extraction fails.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""


class IFaceRepository(ABC):
    """Contract for persistent face identity storage."""

    @abstractmethod
    def save(self, identity: FaceIdentity) -> None:
        """
        Persist a FaceIdentity atomically.
        Raises StorageError on failure.
        """

    @abstractmethod
    def load(self, identity_id: str) -> FaceIdentity:
        """
        Load a FaceIdentity by ID.
        Raises IdentityNotFoundError, CorruptedDataError.
        """

    @abstractmethod
    def load_all(self) -> list[FaceIdentity]:
        """
        Load all registered identities.
        Skips corrupted entries (logs warning), returns rest.
        """

    @abstractmethod
    def delete(self, identity_id: str) -> None:
        """Remove a FaceIdentity and its files."""

    @abstractmethod
    def exists(self, identity_id: str) -> bool:
        """Return True if identity_id is stored."""

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return all stored identity IDs."""


class IRecognitionEngine(ABC):
    """Contract for matching an embedding against registered identities."""

    @abstractmethod
    def match(
        self,
        embedding: np.ndarray,
        identities: list[FaceIdentity],
    ) -> tuple[Optional[FaceIdentity], float]:
        """
        Find the best matching identity.
        Returns (identity, similarity_score) or (None, 0.0) if no match.
        """

    @abstractmethod
    def reload(self, identities: list[FaceIdentity]) -> None:
        """Update the internal identity index."""
