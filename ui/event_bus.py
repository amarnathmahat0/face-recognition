"""
Lightweight Qt signal-based event bus.
UI components and services communicate only through this bus —
never via direct method calls across layers.

Usage:
    bus = EventBus.instance()
    bus.frame_ready.connect(my_slot)
    bus.frame_ready.emit(frame)
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np

from core.entities import (
    SampleResult,
    VerificationResult,
    ThreadStatus,
)


class EventBus(QObject):
    """
    Singleton Qt QObject that owns all application-level signals.
    All signals are safe to emit from any thread — Qt queues them
    to the main thread automatically when connected with Qt.QueuedConnection.
    """

    # ── Camera ────────────────────────────────────────────────────────────────
    frame_ready = pyqtSignal(object)           # np.ndarray (BGR frame)
    camera_fps_updated = pyqtSignal(float)
    camera_status_changed = pyqtSignal(str)    # human-readable status string
    camera_error = pyqtSignal(str)             # error message

    # ── Registration ──────────────────────────────────────────────────────────
    registration_sample_result = pyqtSignal(object)   # SampleResult
    registration_progress = pyqtSignal(int, int)       # (accepted, total)
    registration_completed = pyqtSignal(str)           # display_name
    registration_failed = pyqtSignal(str)              # error message
    registration_training_started = pyqtSignal()
    registration_training_done = pyqtSignal(str)       # identity_id

    # ── Verification ──────────────────────────────────────────────────────────
    verification_result = pyqtSignal(object)           # VerificationResult
    verification_error = pyqtSignal(str)

    # ── System ────────────────────────────────────────────────────────────────
    identities_reloaded = pyqtSignal(int)              # count of loaded identities
    identity_deleted = pyqtSignal(str)                 # identity_id
    thread_status_changed = pyqtSignal(str, object)    # (thread_name, ThreadStatus)
    app_error = pyqtSignal(str)                        # generic error for status bar
    latency_updated = pyqtSignal(float, float, float)  # (p50, p95, p99) ms

    _instance: "EventBus | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def instance(cls) -> "EventBus":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """For tests only."""
        with cls._lock:
            cls._instance = None
