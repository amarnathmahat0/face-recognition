"""
MainWindow — top-level application window.
Wires all UI components together; owns the background worker threads;
routes EventBus signals to appropriate UI slots.

Threading model:
  CameraThread  → ring buffer → InferenceThread → EventBus signals → UI (main thread)
  TrainingThread (QThread) runs during registration
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, QObject, QMetaObject, Q_ARG,
)
from PyQt6.QtGui import QFont, QColor, QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QStatusBar, QMessageBox, QFileDialog,
    QStackedWidget, QFrame, QSizePolicy,
)

from core.entities import (
    ConfidenceBand, FaceIdentity, RegistrationSession,
    SampleResult, SampleStatus, VerificationResult,
)
from core.use_cases import RegisterFaceUseCase, VerifyFaceUseCase
from services.camera_service import CameraService
from services.detector import HaarFaceDetector
from services.embedder import FaceRecognitionEmbedder
from services.face_repository import FileFaceRepository
from services.recognition_engine import CosineRecognitionEngine
from ui.event_bus import EventBus
from ui.register_panel import RegisterPanel
from ui.verify_panel import VerifyPanel
from ui.video_widget import VideoWidget
from utils.config import get_config
from utils.exceptions import (
    CameraUnavailableError, FaceAppError, StorageError,
)
from utils.logger import get_logger
from utils.metrics import LatencyTracker

logger = get_logger(__name__)


def _laplacian_blur(roi: np.ndarray) -> float:
    import cv2
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ══════════════════════════════════════════════════════════════════════════════
# Background worker: inference loop
# ══════════════════════════════════════════════════════════════════════════════

class InferenceWorker(QObject):
    """
    Runs in a dedicated QThread.
    Pulls frames from a bounded queue, runs verification, emits results.
    Bounded queue (maxsize=2) prevents memory accumulation under slow inference.
    """

    result_ready = pyqtSignal(object)        # VerificationResult
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        verify_use_case: VerifyFaceUseCase,
        frame_queue: "queue.Queue[Optional[np.ndarray]]",
        latency_tracker: LatencyTracker,
    ) -> None:
        super().__init__()
        self._use_case = verify_use_case
        self._queue = frame_queue
        self._tracker = latency_tracker
        self._stop = threading.Event()
        self._identities: list[FaceIdentity] = []
        self._identities_lock = threading.Lock()

    def set_identities(self, identities: list[FaceIdentity]) -> None:
        with self._identities_lock:
            self._identities = list(identities)
        self._use_case._engine.reload(identities)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        logger.info("InferenceWorker started", extra={"x_thread": "InferenceWorker"})
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame is None:
                break
            try:
                with self._tracker.measure():
                    with self._identities_lock:
                        ids = list(self._identities)
                    result = self._use_case.verify(frame, ids)
                self.result_ready.emit(result)
            except Exception as exc:
                logger.error("Inference error: %s", exc, exc_info=True)
                self.error_occurred.emit(str(exc))
            finally:
                del frame  # release reference immediately

        logger.info("InferenceWorker stopped")


# ══════════════════════════════════════════════════════════════════════════════
# Background worker: training / save
# ══════════════════════════════════════════════════════════════════════════════

class TrainingWorker(QObject):
    """Saves a completed FaceIdentity to disk in a QThread."""

    finished = pyqtSignal(str)      # identity_id on success
    failed = pyqtSignal(str)        # error message on failure

    def __init__(self, identity: FaceIdentity, repository: FileFaceRepository) -> None:
        super().__init__()
        self._identity = identity
        self._repository = repository

    def run(self) -> None:
        logger.info(
            "TrainingWorker saving %r (%d samples)",
            self._identity.identity_id,
            self._identity.sample_count,
        )
        try:
            self._repository.save(self._identity)
            self.finished.emit(self._identity.identity_id)
        except MemoryError as exc:
            logger.error("OOM during training save: %s", exc)
            self.failed.emit(f"Out of memory: {exc}")
        except FaceAppError as exc:
            logger.error("Storage error during training: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:
            logger.error("Unexpected training error: %s", exc, exc_info=True)
            self.failed.emit(f"Unexpected error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Window
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self._cfg = get_config()
        self._bus = EventBus.instance()
        self._setup_services()
        self._setup_ui()
        self._setup_timers()
        self._connect_signals()
        self._load_identities()
        self._start_camera()

    # ── Service wiring ────────────────────────────────────────────────────────

    def _setup_services(self) -> None:
        cfg = self._cfg

        self._repository = FileFaceRepository(cfg.storage.data_dir)
        self._detector = HaarFaceDetector(min_size=(cfg.recognition.min_face_size,) * 2)
        self._embedder = FaceRecognitionEmbedder(num_jitters=1, model="small")
        self._engine = CosineRecognitionEngine(cfg.recognition.similarity_threshold)

        self._register_use_case = RegisterFaceUseCase(
            detector=self._detector,
            embedder=self._embedder,
            min_face_size=cfg.recognition.min_face_size,
            blur_threshold=cfg.recognition.blur_threshold,
            sample_frame_gap=cfg.recognition.sample_frame_gap,
            target_samples=cfg.recognition.samples_per_identity,
            max_samples=cfg.recognition.max_samples_per_identity,
            compute_blur=_laplacian_blur,
        )

        self._verify_use_case = VerifyFaceUseCase(
            detector=self._detector,
            embedder=self._embedder,
            engine=self._engine,
            similarity_threshold=cfg.recognition.similarity_threshold,
            confidence_high=cfg.recognition.confidence_high,
            confidence_medium=cfg.recognition.confidence_medium,
            compute_blur=_laplacian_blur,
            compute_liveness=self._engine.compute_liveness,
            liveness_threshold=cfg.recognition.liveness_threshold,
            liveness_window=cfg.recognition.liveness_window,
        )

        self._camera = CameraService(
            index=cfg.camera.index,
            width=cfg.camera.width,
            height=cfg.camera.height,
            ring_buffer_size=cfg.camera.ring_buffer_size,
            reconnect_attempts=cfg.camera.reconnect_attempts,
            reconnect_backoff=cfg.camera.reconnect_backoff_seconds,
        )

        # Bounded inference queue — maxsize=2 prevents memory bloat
        self._inference_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=2)
        self._latency_tracker = LatencyTracker(window=100)

        self._inference_worker = InferenceWorker(
            self._verify_use_case, self._inference_queue, self._latency_tracker
        )
        self._verify_frame_gap = max(1, cfg.recognition.verify_frame_gap)
        self._verify_frame_counter = 0
        self._inference_thread = QThread()
        self._inference_worker.moveToThread(self._inference_thread)
        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.result_ready.connect(self._on_verification_result)
        self._inference_worker.error_occurred.connect(self._on_inference_error)

        self._identities: list[FaceIdentity] = []
        self._registration_session: Optional[RegistrationSession] = None
        self._mode = "verify"  # "verify" | "register"

        self._training_thread: Optional[QThread] = None
        self._training_worker: Optional[TrainingWorker] = None

        self._io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DiskIO")

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        cfg = self._cfg.ui
        self.setWindowTitle(cfg.window_title)
        self.setMinimumSize(cfg.min_width, cfg.min_height)
        self.setStyleSheet("QMainWindow { background: #0d0d1a; }")

        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(central)

        # ── Video area ────────────────────────────────────────────────────────
        video_container = QWidget()
        video_container.setStyleSheet("background: #0d0d1a;")
        v_layout = QVBoxLayout(video_container)
        v_layout.setContentsMargins(8, 8, 8, 8)

        self._video_widget = VideoWidget()
        v_layout.addWidget(self._video_widget)

        root_layout.addWidget(video_container, stretch=3)

        # ── Sidebar ───────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("background: #12122a; border-left: 1px solid #222244;")
        s_layout = QVBoxLayout(sidebar)
        s_layout.setContentsMargins(0, 0, 0, 0)
        s_layout.setSpacing(0)

        # Mode toggle
        toggle_frame = QFrame()
        toggle_frame.setStyleSheet("background: #0a0a1e; border-bottom: 1px solid #222244;")
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(8, 8, 8, 8)
        toggle_layout.setSpacing(4)

        self._register_tab_btn = QPushButton("📷  Register")
        self._verify_tab_btn = QPushButton("🔍  Verify")
        for btn in (self._register_tab_btn, self._verify_tab_btn):
            btn.setCheckable(True)
            btn.setStyleSheet(self._tab_btn_style())
        self._verify_tab_btn.setChecked(True)

        self._register_tab_btn.clicked.connect(lambda: self._switch_mode("register"))
        self._verify_tab_btn.clicked.connect(lambda: self._switch_mode("verify"))

        toggle_layout.addWidget(self._register_tab_btn)
        toggle_layout.addWidget(self._verify_tab_btn)
        s_layout.addWidget(toggle_frame)

        # Stacked panels
        self._panel_stack = QStackedWidget()

        self._register_panel = RegisterPanel()
        self._verify_panel = VerifyPanel()

        self._panel_stack.addWidget(self._verify_panel)    # index 0
        self._panel_stack.addWidget(self._register_panel)  # index 1
        s_layout.addWidget(self._panel_stack)

        root_layout.addWidget(sidebar)

        # ── Status bar ────────────────────────────────────────────────────────
        status_bar = self.statusBar()
        status_bar.setStyleSheet(
            "QStatusBar { background: #0a0a1e; color: #888; font-family: Consolas; font-size: 10px; }"
        )

        self._fps_status = QLabel("FPS: —")
        self._fps_status.setStyleSheet("color: #4caf50; padding: 0 8px;")

        self._camera_status = QLabel("Camera: —")
        self._camera_status.setStyleSheet("color: #888; padding: 0 8px;")

        self._latency_status = QLabel("P99: —")
        self._latency_status.setStyleSheet("color: #888; padding: 0 8px;")

        self._error_status = QLabel("")
        self._error_status.setStyleSheet("color: #dc3232; padding: 0 8px;")

        status_bar.addPermanentWidget(self._fps_status)
        status_bar.addPermanentWidget(_vsep())
        status_bar.addPermanentWidget(self._camera_status)
        status_bar.addPermanentWidget(_vsep())
        status_bar.addPermanentWidget(self._latency_status)
        status_bar.addWidget(self._error_status)

        self._switch_mode("verify")

    # ── Timers ─────────────────────────────────────────────────────────────────

    def _setup_timers(self) -> None:
        # Frame pump: polls camera ring buffer, feeds UI and inference queue
        self._frame_timer = QTimer()
        self._frame_timer.setInterval(33)  # ~30fps pump rate
        self._frame_timer.timeout.connect(self._pump_frame)
        self._frame_timer.start()

        # Status bar refresh
        self._status_timer = QTimer()
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Register panel
        self._register_panel.start_requested.connect(self._on_register_start)
        self._register_panel.cancel_requested.connect(self._on_register_cancel)

        # Verify panel
        self._verify_panel.delete_identity_requested.connect(self._on_delete_identity)
        self._verify_panel.export_requested.connect(self._on_export)

    # ── Camera lifecycle ──────────────────────────────────────────────────────

    def _start_camera(self) -> None:
        try:
            self._camera.start()
            self._inference_thread.start()
            self._camera_status.setText("Camera: Connected")
            self._camera_status.setStyleSheet("color: #4caf50; padding: 0 8px;")
        except CameraUnavailableError as exc:
            self._video_widget.set_status_text(
                f"⚠ Camera not available\n{exc}\n\nCheck connection and restart."
            )
            self._show_camera_error(str(exc))

    def _show_camera_error(self, msg: str) -> None:
        self._camera_status.setText("Camera: ERROR")
        self._camera_status.setStyleSheet("color: #dc3232; padding: 0 8px;")
        self._error_status.setText(f"Camera: {msg[:60]}")
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setWindowTitle("Camera Error")
        dlg.setText(f"Camera unavailable:\n{msg}")
        retry_btn = dlg.addButton("Retry", QMessageBox.ButtonRole.AcceptRole)
        dlg.addButton("Continue without camera", QMessageBox.ButtonRole.RejectRole)
        dlg.exec()
        if dlg.clickedButton() == retry_btn:
            self._start_camera()

    # ── Frame pump (main thread, timer-driven) ────────────────────────────────

    def _pump_frame(self) -> None:
        frame = self._camera.latest_frame()
        if frame is None:
            if not self._camera.is_connected:
                self._video_widget.set_status_text("⚠ Camera disconnected…\nAttempting reconnect")
            return

        # Always update the video widget
        if self._mode == "register" and self._registration_session:
            result = self._register_use_case.process_frame(
                self._registration_session, frame
            )
            bbox = result.bbox if result else None
            self._register_panel.on_sample_result(
                result, self._cfg.recognition.samples_per_identity
            )
            self._register_panel._progress_bar.setValue(
                self._registration_session.accepted_count
            )
            self._video_widget.set_frame(frame)
            self._video_widget.set_registration_sample(bbox, result)

            if self._registration_session.is_ready:
                self._on_register_samples_complete()
            return

        self._video_widget.set_frame(frame)

        if self._mode == "verify":
            self._verify_frame_counter += 1
            if self._verify_frame_gap <= 1 or self._verify_frame_counter % self._verify_frame_gap == 0:
                try:
                    self._inference_queue.put_nowait(frame.copy())
                except queue.Full:
                    logger.debug("Inference queue full — frame dropped")
            else:
                logger.debug(
                    "Skipping inference on frame %d/%d",
                    self._verify_frame_counter,
                    self._verify_frame_gap,
                )

    # ── Verification result (main thread via Qt signal) ───────────────────────

    def _on_verification_result(self, result: VerificationResult) -> None:
        self._video_widget.set_verification_result(result)
        self._verify_panel.update_result(result)

    def _on_inference_error(self, msg: str) -> None:
        self._error_status.setText(f"Inference: {msg[:60]}")

    # ── Registration workflow ─────────────────────────────────────────────────

    def _on_register_start(self, display_name: str) -> None:
        if self._repository.exists(_slugify(display_name)):
            ans = QMessageBox.question(
                self, "Identity Exists",
                f"'{display_name}' is already registered. Overwrite?",
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        self._registration_session = self._register_use_case.begin_session(display_name)
        self._register_panel.set_running_state(
            display_name, self._cfg.recognition.samples_per_identity
        )
        logger.info("Registration started for %r", display_name)

    def _on_register_cancel(self) -> None:
        self._registration_session = None
        self._video_widget.clear_overlays()
        logger.info("Registration cancelled")

    def _on_register_samples_complete(self) -> None:
        if not self._registration_session:
            return
        session = self._registration_session
        self._registration_session = None

        self._register_panel.set_training_state()

        try:
            identity = self._register_use_case.finalise(session)
        except FaceAppError as exc:
            self._register_panel.set_failed_state(str(exc))
            return

        # Save asynchronously in QThread
        worker = TrainingWorker(identity, self._repository)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda iid: self._on_training_done(iid, thread))
        worker.failed.connect(lambda msg: self._on_training_failed(msg, thread))
        self._training_worker = worker
        self._training_thread = thread
        thread.start()

    def _on_training_done(self, identity_id: str, thread: QThread) -> None:
        thread.quit()
        thread.wait(2000)
        self._load_identities()
        name = next(
            (i.display_name for i in self._identities if i.identity_id == identity_id),
            identity_id,
        )
        self._register_panel.set_completed_state(name)
        self._video_widget.clear_overlays()
        logger.info("Training completed for %r", identity_id)

    def _on_training_failed(self, msg: str, thread: QThread) -> None:
        thread.quit()
        thread.wait(2000)
        self._register_panel.set_failed_state(msg)
        logger.error("Training failed: %s", msg)

    # ── Identity management ───────────────────────────────────────────────────

    def _load_identities(self) -> None:
        identities = self._repository.load_all()
        self._identities = identities
        self._inference_worker.set_identities(identities)
        self._verify_panel.update_identities(identities)
        logger.info("Identities loaded: %d", len(identities))

    def _on_delete_identity(self, identity_id: str) -> None:
        ans = QMessageBox.question(
            self, "Delete Identity",
            f"Delete identity '{identity_id}'? This cannot be undone.",
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            self._repository.delete(identity_id)
            self._load_identities()
        except StorageError as exc:
            QMessageBox.critical(self, "Delete Failed", str(exc))

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Identities", "faceid_export.zip", "ZIP (*.zip)"
        )
        if not path:
            return
        try:
            data_dir = self._cfg.storage.data_dir
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                for entry in data_dir.iterdir():
                    if entry.is_dir():
                        for f in entry.iterdir():
                            zf.write(f, arcname=f.relative_to(data_dir.parent))
            QMessageBox.information(self, "Export Complete", f"Exported to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    # ── Mode switching ────────────────────────────────────────────────────────

    def _switch_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == "verify":
            self._panel_stack.setCurrentIndex(0)
            self._verify_tab_btn.setChecked(True)
            self._register_tab_btn.setChecked(False)
            self._inference_thread.start() if not self._inference_thread.isRunning() else None
        else:
            self._panel_stack.setCurrentIndex(1)
            self._register_tab_btn.setChecked(True)
            self._verify_tab_btn.setChecked(False)
            self._video_widget.clear_overlays()
            self._registration_session = None

    # ── Status bar refresh ────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        fps = self._camera.fps
        self._fps_status.setText(f"FPS: {fps:.1f}")
        if fps < 5 and self._camera.is_alive():
            self._fps_status.setStyleSheet("color: #ffcc00; padding: 0 8px;")
        else:
            self._fps_status.setStyleSheet("color: #4caf50; padding: 0 8px;")

        if self._camera.is_connected:
            self._camera_status.setText("Camera: ✓")
            self._camera_status.setStyleSheet("color: #4caf50; padding: 0 8px;")
        else:
            self._camera_status.setText("Camera: ✗")
            self._camera_status.setStyleSheet("color: #dc3232; padding: 0 8px;")

        if self._cfg.ui.debug_mode:
            p99 = self._latency_tracker.p99()
            mean = self._latency_tracker.mean()
            self._latency_status.setText(f"P99: {p99:.0f}ms  Mean: {mean:.0f}ms")
        else:
            self._latency_status.setText("")

        err = self._camera.last_error
        if err:
            self._error_status.setText(f"⚠ {str(err)[:60]}")

    # ── Style helpers ─────────────────────────────────────────────────────────

    def _tab_btn_style(self) -> str:
        return """
            QPushButton {
                background: #1a1a3a;
                color: #888;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:checked {
                background: #2962ff;
                color: white;
                border: 1px solid #2962ff;
            }
            QPushButton:hover:!checked { background: #252545; }
        """

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802
        logger.info("Application shutting down")
        self._frame_timer.stop()
        self._status_timer.stop()

        # Stop inference
        self._inference_worker.stop()
        try:
            self._inference_queue.put_nowait(None)
        except queue.Full:
            pass
        self._inference_thread.quit()
        self._inference_thread.wait(3000)

        # Stop camera
        self._camera.stop()

        # Shutdown executor
        self._io_executor.shutdown(wait=False)

        event.accept()


def _slugify(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip()).strip("_") or "identity"


def _vsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setStyleSheet("color: #333355;")
    return f
