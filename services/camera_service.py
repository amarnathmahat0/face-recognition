"""
Camera service: threaded frame capture with:
- Fixed-size ring buffer (deque maxlen=N) — O(1), no unbounded growth
- Rolling FPS measurement
- Auto-reconnect with backoff
- Clean thread lifecycle via threading.Event
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from core.interfaces import ICameraSource
from utils.exceptions import (
    CameraDisconnectedError,
    CameraReconnectExhaustedError,
    CameraUnavailableError,
)
from utils.logger import get_logger
from utils.metrics import FPSCounter

logger = get_logger(__name__)


class CameraService(ICameraSource):
    """
    Daemon thread that continuously captures frames into a bounded ring buffer.
    The producer (camera) never blocks — oldest frames are dropped on overflow.
    """

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        ring_buffer_size: int,
        reconnect_attempts: int = 3,
        reconnect_backoff: float = 2.0,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._ring_buffer_size = ring_buffer_size
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_backoff = reconnect_backoff

        # Ring buffer — maxlen enforces O(1) bounded memory
        self._buffer: deque[np.ndarray] = deque(maxlen=ring_buffer_size)
        self._buffer_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps_counter = FPSCounter(window=30)
        self._read_failures = 0

        self._connected = threading.Event()
        self._error: Optional[Exception] = None

    # ── ICameraSource ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True,
        )
        logger.info("CameraService starting", extra={"x_camera_index": self._index})
        self._thread.start()

    def stop(self) -> None:
        logger.info("CameraService stopping")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("Camera thread did not stop within timeout")
        self._release_cap()

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._buffer_lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()  # return copy to avoid cross-thread mutation

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def last_error(self) -> Optional[Exception]:
        return self._error

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        try:
            self._open_camera()
        except CameraUnavailableError as exc:
            self._error = exc
            logger.error("Camera unavailable at start: %s", exc)
            return

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._read_failures += 1
                if self._read_failures < 3:
                    time.sleep(0.05)
                    continue

                logger.warning(
                    "Frame read failed (%d consecutive) — attempting reconnect",
                    self._read_failures,
                )
                self._read_failures = 0
                self._connected.clear()
                try:
                    self._reconnect()
                except CameraReconnectExhaustedError as exc:
                    self._error = exc
                    logger.error("Camera reconnect exhausted: %s", exc)
                    break
                continue

            self._read_failures = 0

            # Resize if needed
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                frame = cv2.resize(frame, (self._width, self._height))

            # Push into ring buffer — deque.append is O(1) and thread-safe for GIL
            with self._buffer_lock:
                self._buffer.append(frame)

            self._fps_counter.tick()

        self._release_cap()
        self._connected.clear()
        logger.info("CameraService capture loop exited")

    def _open_camera(self) -> None:
        self._release_cap()
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise CameraUnavailableError(
                f"Cannot open camera index {self._index}"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize internal buffering latency
        except Exception as exc:  # pragma: no cover
            logger.warning("Camera buffer-size property unsupported: %s", exc)
        self._cap = cap
        self._connected.set()
        self._error = None
        self._fps_counter.reset()
        logger.info(
            "Camera opened",
            extra={"x_index": self._index, "x_w": self._width, "x_h": self._height},
        )

    def _reconnect(self) -> None:
        for attempt in range(1, self._reconnect_attempts + 1):
            if self._stop_event.is_set():
                return
            wait = self._reconnect_backoff * attempt
            logger.info(
                "Reconnect attempt %d/%d in %.1fs",
                attempt,
                self._reconnect_attempts,
                wait,
                extra={"x_attempt": attempt},
            )
            self._stop_event.wait(timeout=wait)
            try:
                self._open_camera()
                logger.info("Camera reconnected on attempt %d", attempt)
                return
            except CameraUnavailableError:
                continue
        raise CameraReconnectExhaustedError(
            f"Camera {self._index} not recoverable after "
            f"{self._reconnect_attempts} attempts"
        )

    def _release_cap(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
