"""
Singleton configuration loader.
Reads config.yaml once; exposes typed, validated attributes.
No global mutable state — the singleton itself is immutable after init.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import yaml

from utils.exceptions import ConfigError

_LOCK = threading.Lock()
_INSTANCE: "AppConfig | None" = None


class _CameraConfig:
    __slots__ = (
        "index", "width", "height", "ring_buffer_size",
        "reconnect_attempts", "reconnect_backoff_seconds",
    )

    def __init__(self, d: dict[str, Any]) -> None:
        self.index: int = int(d.get("index", 0))
        self.width: int = int(d.get("width", 640))
        self.height: int = int(d.get("height", 480))
        self.ring_buffer_size: int = int(d.get("ring_buffer_size", 5))
        self.reconnect_attempts: int = int(d.get("reconnect_attempts", 3))
        self.reconnect_backoff_seconds: float = float(d.get("reconnect_backoff_seconds", 2.0))


class _RecognitionConfig:
    __slots__ = (
        "similarity_threshold", "min_face_size", "blur_threshold",
        "samples_per_identity", "sample_frame_gap",
        "max_samples_per_identity", "confidence_high", "confidence_medium",
        "verify_frame_gap", "liveness_threshold", "liveness_window",
    )

    def __init__(self, d: dict[str, Any]) -> None:
        self.similarity_threshold: float = float(d.get("similarity_threshold", 0.60))
        self.min_face_size: int = int(d.get("min_face_size", 80))
        self.blur_threshold: float = float(d.get("blur_threshold", 100.0))
        self.samples_per_identity: int = int(d.get("samples_per_identity", 15))
        self.sample_frame_gap: int = int(d.get("sample_frame_gap", 5))
        self.max_samples_per_identity: int = int(d.get("max_samples_per_identity", 20))
        self.confidence_high: float = float(d.get("confidence_high", 0.80))
        self.confidence_medium: float = float(d.get("confidence_medium", 0.65))
        self.verify_frame_gap: int = int(d.get("verify_frame_gap", 1))
        self.liveness_threshold: float = float(d.get("liveness_threshold", 0.30))
        self.liveness_window: int = int(d.get("liveness_window", 5))


class _StorageConfig:
    __slots__ = ("data_dir", "embeddings_ext", "metadata_ext")

    def __init__(self, d: dict[str, Any]) -> None:
        self.data_dir: Path = Path(d.get("data_dir", "./data/faces"))
        self.embeddings_ext: str = str(d.get("embeddings_ext", ".npy"))
        self.metadata_ext: str = str(d.get("metadata_ext", ".json"))


class _LoggingConfig:
    __slots__ = ("level", "file", "max_bytes", "backup_count")

    def __init__(self, d: dict[str, Any]) -> None:
        self.level: str = str(d.get("level", "INFO")).upper()
        self.file: Path = Path(d.get("file", "./logs/app.log"))
        self.max_bytes: int = int(d.get("max_bytes", 5 * 1024 * 1024))
        self.backup_count: int = int(d.get("backup_count", 3))


class _UIConfig:
    __slots__ = (
        "window_title", "min_width", "min_height",
        "fps_rolling_window", "debug_mode",
    )

    def __init__(self, d: dict[str, Any]) -> None:
        self.window_title: str = str(d.get("window_title", "FaceID"))
        self.min_width: int = int(d.get("min_width", 1024))
        self.min_height: int = int(d.get("min_height", 640))
        self.fps_rolling_window: int = int(d.get("fps_rolling_window", 30))
        self.debug_mode: bool = bool(d.get("debug_mode", False))


class AppConfig:
    """Immutable application configuration. Use get_config() to obtain the singleton."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Malformed YAML in {path}", cause=exc) from exc

        self.camera = _CameraConfig(raw.get("camera", {}))
        self.recognition = _RecognitionConfig(raw.get("recognition", {}))
        self.storage = _StorageConfig(raw.get("storage", {}))
        self.logging = _LoggingConfig(raw.get("logging", {}))
        self.ui = _UIConfig(raw.get("ui", {}))

        self._validate()

    def _validate(self) -> None:
        if not (0.0 < self.recognition.similarity_threshold < 1.0):
            raise ConfigError("recognition.similarity_threshold must be in (0, 1)")
        if self.camera.ring_buffer_size < 1:
            raise ConfigError("camera.ring_buffer_size must be >= 1")
        if self.recognition.samples_per_identity < 3:
            raise ConfigError("recognition.samples_per_identity must be >= 3")
        if self.recognition.verify_frame_gap < 1:
            raise ConfigError("recognition.verify_frame_gap must be >= 1")
        if self.recognition.liveness_window < 1:
            raise ConfigError("recognition.liveness_window must be >= 1")
        if not (0.0 <= self.recognition.liveness_threshold <= 1.0):
            raise ConfigError("recognition.liveness_threshold must be between 0 and 1")


def get_config(path: str | Path | None = None) -> AppConfig:
    """Return the singleton AppConfig, initialising it on first call."""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                cfg_path = Path(path) if path else _find_config()
                _INSTANCE = AppConfig(cfg_path)
    return _INSTANCE


def _find_config() -> Path:
    """Walk up from CWD looking for config.yaml."""
    candidates = [
        Path("config.yaml"),
        Path(__file__).parent.parent / "config.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise ConfigError(
        "config.yaml not found. Run from the project root or pass --config."
    )


def reset_config() -> None:
    """Reset singleton — for tests only."""
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None
