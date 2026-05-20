"""Unit tests for metrics primitives — no I/O, no UI, no camera."""
import time
import pytest
from utils.metrics import RollingWindow, LatencyTracker, FPSCounter


class TestRollingWindow:
    def test_maxlen_enforced(self):
        w = RollingWindow(maxlen=3)
        for i in range(10):
            w.push(float(i))
        snap = w.snapshot()
        assert len(snap) == 3
        assert snap == [7.0, 8.0, 9.0]

    def test_empty_snapshot(self):
        w = RollingWindow(maxlen=5)
        assert w.snapshot() == []

    def test_thread_safe_push(self):
        import threading
        w = RollingWindow(maxlen=1000)
        threads = [
            threading.Thread(target=lambda: [w.push(float(i)) for i in range(100)])
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(w) <= 1000

    def test_invalid_maxlen(self):
        with pytest.raises(ValueError):
            RollingWindow(maxlen=0)


class TestLatencyTracker:
    def test_measure_records_latency(self):
        tracker = LatencyTracker(window=10)
        with tracker.measure():
            time.sleep(0.01)  # 10ms
        assert tracker.sample_count() == 1
        assert tracker.mean() >= 5.0  # at least 5ms

    def test_p99_single_sample(self):
        tracker = LatencyTracker(window=10)
        with tracker.measure():
            time.sleep(0.005)
        # p99 of 1 sample = that sample
        assert tracker.p99() > 0

    def test_p99_ordering(self):
        tracker = LatencyTracker(window=100)
        # Push known values via internal window
        for val in [1.0, 2.0, 3.0, 100.0]:
            tracker._window.push(val)
        p99 = tracker.p99()
        assert p99 == 100.0

    def test_mean_empty(self):
        tracker = LatencyTracker(window=10)
        assert tracker.mean() == 0.0

    def test_p50_median(self):
        tracker = LatencyTracker(window=100)
        for v in range(1, 11):
            tracker._window.push(float(v))
        p50 = tracker.p50()
        assert 5.0 <= p50 <= 6.0


class TestFPSCounter:
    def test_fps_zero_on_single_tick(self):
        counter = FPSCounter(window=30)
        counter.tick()
        assert counter.fps == 0.0

    def test_fps_approximation(self):
        counter = FPSCounter(window=60)
        for _ in range(30):
            counter.tick()
            time.sleep(1 / 30)
        fps = counter.fps
        assert 20.0 < fps < 40.0, f"Expected ~30 fps, got {fps:.1f}"

    def test_reset(self):
        counter = FPSCounter(window=10)
        for _ in range(5):
            counter.tick()
        counter.reset()
        assert counter.fps == 0.0
