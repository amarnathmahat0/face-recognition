"""
Registration panel — sidebar UI for face registration workflow.
Emits signals via EventBus; never calls services directly.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QFrame, QSizePolicy, QScrollArea,
)

from core.entities import SampleResult, SampleStatus


class RegisterPanel(QWidget):
    """
    UI panel for the registration workflow.
    Signals emitted to parent/controller — no direct service calls.
    """

    # Controller connects these
    start_requested = pyqtSignal(str)    # display_name
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._set_idle_state()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Title
        title = QLabel("Register New Face")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0ff;")
        layout.addWidget(title)

        layout.addWidget(_separator())

        # Name input
        name_label = QLabel("Full Name / ID")
        name_label.setStyleSheet("color: #aaaacc; font-size: 11px;")
        layout.addWidget(name_label)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g. Alice Johnson")
        self._name_input.setStyleSheet(_input_style())
        self._name_input.returnPressed.connect(self._on_start)
        layout.addWidget(self._name_input)

        # Start / Cancel buttons
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start Registration")
        self._start_btn.setStyleSheet(_btn_style("#2962ff", "#1a3fbf"))
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("✕  Cancel")
        self._cancel_btn.setStyleSheet(_btn_style("#555", "#333"))
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        layout.addLayout(btn_row)

        layout.addWidget(_separator())

        # Progress
        progress_label = QLabel("Sample Progress")
        progress_label.setStyleSheet("color: #aaaacc; font-size: 11px;")
        layout.addWidget(progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(_progress_style())
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        # Counters
        counter_row = QHBoxLayout()
        self._accepted_label = QLabel("Accepted: 0")
        self._accepted_label.setStyleSheet("color: #00dc50; font-size: 11px;")
        self._rejected_label = QLabel("Rejected: 0")
        self._rejected_label.setStyleSheet("color: #dc3232; font-size: 11px;")
        counter_row.addWidget(self._accepted_label)
        counter_row.addStretch()
        counter_row.addWidget(self._rejected_label)
        layout.addLayout(counter_row)

        layout.addWidget(_separator())

        # Live feedback log
        feedback_label = QLabel("Live Feedback")
        feedback_label.setStyleSheet("color: #aaaacc; font-size: 11px;")
        layout.addWidget(feedback_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(180)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #333; border-radius: 4px; }")

        self._feedback_container = QWidget()
        self._feedback_layout = QVBoxLayout(self._feedback_container)
        self._feedback_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._feedback_layout.setSpacing(2)
        self._feedback_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self._feedback_container)
        layout.addWidget(scroll)

        # Status message
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #ffcc00; font-size: 11px; padding: 4px;")
        layout.addWidget(self._status_label)

        layout.addStretch()

        self._accepted = 0
        self._rejected = 0

    # ── State transitions ──────────────────────────────────────────────────────

    def _set_idle_state(self) -> None:
        self._name_input.setEnabled(True)
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("0 / 0")
        self._status_label.setText("")

    def set_running_state(self, display_name: str, total: int) -> None:
        self._name_input.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat(f"0 / {total}")
        self._accepted = 0
        self._rejected = 0
        self._accepted_label.setText("Accepted: 0")
        self._rejected_label.setText("Rejected: 0")
        self._clear_feedback()
        self._status_label.setText(f"Registering: {display_name}")

    def set_training_state(self) -> None:
        self._status_label.setText("⏳ Saving to disk…")
        self._cancel_btn.setEnabled(False)

    def set_completed_state(self, display_name: str) -> None:
        self._status_label.setStyleSheet("color: #00dc50; font-size: 11px; padding: 4px;")
        self._status_label.setText(f"✓ {display_name!r} registered successfully!")
        self._set_idle_state()
        self._status_label.setStyleSheet("color: #00dc50; font-size: 11px; padding: 4px;")

    def set_failed_state(self, error: str) -> None:
        self._status_label.setStyleSheet("color: #dc3232; font-size: 11px; padding: 4px;")
        self._status_label.setText(f"✗ Error: {error}")
        self._set_idle_state()
        self._status_label.setStyleSheet("color: #dc3232; font-size: 11px; padding: 4px;")

    # ── Sample feedback ────────────────────────────────────────────────────────

    def on_sample_result(self, result: SampleResult, total: int, multi_angle_hint: bool = False) -> None:
        if result.status == SampleStatus.REJECTED_FRAME_GAP:
            return  # Don't spam the log with every skipped frame

        if result.status == SampleStatus.ACCEPTED:
            self._accepted += 1
            self._accepted_label.setText(f"Accepted: {self._accepted}")
            color = "#00dc50"
            icon = "✓"
            text = f"Sample #{self._accepted} accepted"
        else:
            self._rejected += 1
            self._rejected_label.setText(f"Rejected: {self._rejected}")
            color = "#dc3232"
            icon = "✗"
            text = result.rejection_reason or result.status.name

        self._progress_bar.setValue(self._accepted)
        self._progress_bar.setFormat(f"{self._accepted} / {total}")

        entry = QLabel(f"{icon} {text}")
        entry.setStyleSheet(f"color: {color}; font-size: 10px; font-family: Consolas;")
        self._feedback_layout.addWidget(entry)

        # Keep max 50 entries to avoid memory growth
        if self._feedback_layout.count() > 50:
            item = self._feedback_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def _clear_feedback(self) -> None:
        while self._feedback_layout.count():
            item = self._feedback_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        name = self._name_input.text().strip()
        if not name:
            self._status_label.setText("⚠ Please enter a name first.")
            return
        self.start_requested.emit(name)

    def _on_cancel(self) -> None:
        self.cancel_requested.emit()
        self._set_idle_state()

    def get_name(self) -> str:
        return self._name_input.text().strip()


# ── Style helpers ──────────────────────────────────────────────────────────────

def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #333355;")
    return line


def _input_style() -> str:
    return """
        QLineEdit {
            background: #0d0d1a;
            border: 1px solid #334;
            border-radius: 4px;
            color: #e0e0ff;
            padding: 6px 8px;
            font-size: 12px;
        }
        QLineEdit:focus { border: 1px solid #2962ff; }
    """


def _btn_style(bg: str, hover: str) -> str:
    return f"""
        QPushButton {{
            background: {bg};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 7px 12px;
            font-size: 11px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background: {hover}; }}
        QPushButton:disabled {{ background: #333; color: #666; }}
    """


def _progress_style() -> str:
    return """
        QProgressBar {
            background: #0d0d1a;
            border: 1px solid #334;
            border-radius: 4px;
            color: #fff;
            text-align: center;
            font-size: 11px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #1565c0, stop:1 #2962ff);
            border-radius: 3px;
        }
    """
