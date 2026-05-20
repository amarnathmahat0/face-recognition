"""
Lightweight metrics primitives.
- RollingWindow: fixed-size deque of floats, thread-safe
- LatencyTracker: wraps RollingWindow, exposes p50/p95/p99/mean
- FPSCounter: rolling FPS measurement
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Sequence


class RollingWindow:
    """Thread-safe fixed-size circular buffer of float samples."""

    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, value: float) -> None:
        with self._lock:
            self._buf.append(value)

    def snapshot(self) -> list[float]:
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class LatencyTracker:
    """
    Tracks inference/processing latency with percentile support.
    Usage:
        with tracker.measure():
            do_work()
        p99 = tracker.p99()
    """

    def __init__(self, window: int = 100) -> None:
        self._window = RollingWindow(window)

    class _Ctx:
        def __init__(self, tracker: "LatencyTracker") -> None:
            self._tracker = tracker
            self._start = 0.0

        def __enter__(self) -> "_Ctx":
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_: object) -> None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000.0
            self._tracker._window.push(elapsed_ms)

    def measure(self) -> "_Ctx":
        return self._Ctx(self)

    def _percentile(self, samples: list[float], pct: float) -> float:
        if not samples:
            return 0.0
        sorted_s = sorted(samples)
        # Ceiling index: ensures p99 of [1,2,3,100] returns 100, not 3
        idx = min(len(sorted_s) - 1, int(len(sorted_s) * pct / 100.0 + 0.9999))
        return sorted_s[idx]

    def p50(self) -> float:
        return self._percentile(self._window.snapshot(), 50)

    def p95(self) -> float:
        return self._percentile(self._window.snapshot(), 95)

    def p99(self) -> float:
        return self._percentile(self._window.snapshot(), 99)

    def mean(self) -> float:
        samples = self._window.snapshot()
        return sum(samples) / len(samples) if samples else 0.0

    def sample_count(self) -> int:
        return len(self._window)


class FPSCounter:
    """
    Rolling-window FPS counter.
    Call tick() on each frame; read fps property.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            span = self._timestamps[-1] - self._timestamps[0]
            if span <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / span

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()
