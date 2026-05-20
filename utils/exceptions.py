"""
Custom exception hierarchy for FaceID application.
Never raise bare Exception — always use one of these.
"""
from __future__ import annotations


class FaceAppError(Exception):
    """Root exception for all application errors."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        if self.cause:
            return f"{base} | caused by: {type(self.cause).__name__}: {self.cause}"
        return base


# ── Camera ────────────────────────────────────────────────────────────────────

class CameraError(FaceAppError):
    """Any camera I/O failure."""


class CameraUnavailableError(CameraError):
    """Camera device not found or cannot be opened."""


class CameraDisconnectedError(CameraError):
    """Camera was connected but lost mid-session."""


class CameraReconnectExhaustedError(CameraError):
    """All reconnect attempts failed."""


# ── Recognition ───────────────────────────────────────────────────────────────

class RecognitionError(FaceAppError):
    """Errors in face detection or embedding."""


class NoFaceDetectedError(RecognitionError):
    """Frame contains no detectable face."""


class MultipleFacesError(RecognitionError):
    """Frame contains more faces than expected (registration context)."""


class FaceBlurryError(RecognitionError):
    """Face ROI failed blur/quality threshold."""


class FaceTooSmallError(RecognitionError):
    """Detected face bounding box is below minimum size."""


class EmbeddingError(RecognitionError):
    """Embedding extraction failed."""


# ── Storage ───────────────────────────────────────────────────────────────────

class StorageError(FaceAppError):
    """Errors in reading/writing face data."""


class CorruptedDataError(StorageError):
    """Stored embeddings or metadata are malformed."""


class IdentityNotFoundError(StorageError):
    """Requested identity does not exist in storage."""


class AtomicWriteError(StorageError):
    """Atomic file write (tmp → rename) failed."""


# ── Training ──────────────────────────────────────────────────────────────────

class TrainingError(FaceAppError):
    """Errors during the registration/training workflow."""


class InsufficientSamplesError(TrainingError):
    """Not enough valid samples collected to register identity."""


class TrainingMemoryError(TrainingError):
    """OOM or memory pressure during training."""


class DuplicateIdentityError(TrainingError):
    """Identity with this name already exists."""


# ── Config ────────────────────────────────────────────────────────────────────

class ConfigError(FaceAppError):
    """Configuration file missing, malformed, or invalid values."""
