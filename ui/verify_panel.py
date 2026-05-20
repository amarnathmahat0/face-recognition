"""
Verification panel — sidebar UI for real-time face verification.
Displays confidence, identity, and registered identity management.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QListWidget, QListWidgetItem, QSizePolicy,
)

from core.entities import ConfidenceBand, FaceIdentity, VerificationResult


_BAND_STYLE = {
    ConfidenceBand.HIGH: ("color: #00dc50;", "HIGH"),
    ConfidenceBand.MEDIUM: ("color: #ffc800;", "MEDIUM"),
    ConfidenceBand.LOW: ("color: #ff7800;", "LOW"),
    ConfidenceBand.UNKNOWN: ("color: #dc3232;", "UNKNOWN"),
}


class VerifyPanel(QWidget):
    """Sidebar panel for verification mode."""

    delete_identity_requested = pyqtSignal(str)   # identity_id
    export_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Title
        title = QLabel("Verification")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0ff;")
        layout.addWidget(title)

        layout.addWidget(_sep())

        # Live result display
        result_title = QLabel("Live Match Result")
        result_title.setStyleSheet("color: #aaaacc; font-size: 11px;")
        layout.addWidget(result_title)

        self._name_label = QLabel("—")
        self._name_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self._name_label.setStyleSheet("color: #e0e0ff;")
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._name_label)

        conf_row = QHBoxLayout()
        self._conf_label = QLabel("Confidence:")
        self._conf_label.setStyleSheet("color: #aaaacc; font-size: 11px;")
        self._band_label = QLabel("—")
        self._band_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self._band_label.setStyleSheet("color: #aaaacc;")
        conf_row.addStretch()
        conf_row.addWidget(self._conf_label)
        conf_row.addWidget(self._band_label)
        conf_row.addStretch()
        layout.addLayout(conf_row)

        self._score_label = QLabel("Score: —")
        self._score_label.setStyleSheet("color: #888; font-size: 10px; font-family: Consolas;")
        self._score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._score_label)

        self._liveness_label = QLabel("Liveness: —")
        self._liveness_label.setStyleSheet("color: #888; font-size: 10px; font-family: Consolas;")
        self._liveness_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._liveness_label)

        self._face_count_label = QLabel("Faces in frame: 0")
        self._face_count_label.setStyleSheet("color: #888; font-size: 10px;")
        self._face_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._face_count_label)

        layout.addWidget(_sep())

        # Registered identities
        reg_title = QLabel("Registered Identities")
        reg_title.setStyleSheet("color: #aaaacc; font-size: 11px;")
        layout.addWidget(reg_title)

        self._id_count_label = QLabel("0 identities")
        self._id_count_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._id_count_label)

        self._identity_list = QListWidget()
        self._identity_list.setMaximumHeight(160)
        self._identity_list.setStyleSheet(_list_style())
        layout.addWidget(self._identity_list)

        # Delete button
        del_btn = QPushButton("🗑  Delete Selected")
        del_btn.setStyleSheet(_btn_style("#8b0000", "#5a0000"))
        del_btn.clicked.connect(self._on_delete)
        layout.addWidget(del_btn)

        # Export button
        export_btn = QPushButton("📦  Export All to ZIP")
        export_btn.setStyleSheet(_btn_style("#1b5e20", "#0a3d0a"))
        export_btn.clicked.connect(self.export_requested.emit)
        layout.addWidget(export_btn)

        layout.addWidget(_sep())

        self._no_data_label = QLabel("⚠ No registered faces.\nSwitch to Register mode to add faces.")
        self._no_data_label.setWordWrap(True)
        self._no_data_label.setStyleSheet("color: #ffcc00; font-size: 11px; padding: 4px;")
        self._no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_data_label.hide()
        layout.addWidget(self._no_data_label)

        layout.addStretch()

        self._identities: list[FaceIdentity] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_identities(self, identities: list[FaceIdentity]) -> None:
        self._identities = identities
        self._identity_list.clear()
        for ident in identities:
            item = QListWidgetItem(f"👤  {ident.display_name}  ({ident.sample_count} samples)")
            item.setData(Qt.ItemDataRole.UserRole, ident.identity_id)
            item.setForeground(QColor("#c0c0e0"))
            self._identity_list.addItem(item)
        self._id_count_label.setText(f"{len(identities)} identit{'y' if len(identities)==1 else 'ies'}")
        self._no_data_label.setVisible(len(identities) == 0)

    def update_result(self, result: Optional[VerificationResult]) -> None:
        if result is None or not result.matches:
            self._name_label.setText("No Face")
            self._band_label.setText("—")
            self._band_label.setStyleSheet("color: #888;")
            self._score_label.setText("Score: —")
            self._liveness_label.setText("Liveness: —")
            self._face_count_label.setText("Faces in frame: 0")
            return

        self._face_count_label.setText(f"Faces in frame: {len(result.matches)}")

        best = result.best_match
        if best is None:
            return

        self._name_label.setText(best.display_name or "Unknown")
        style, band_text = _BAND_STYLE.get(best.confidence, ("color: #888;", "—"))
        self._band_label.setStyleSheet(style + " font-size: 12px; font-family: Consolas; font-weight: bold;")
        self._band_label.setText(band_text)
        self._score_label.setText(f"Score: {best.similarity * 100:.1f}%")
        live_pct = best.liveness_score * 100
        live_color = "#00dc50" if live_pct > 50 else "#ffcc00" if live_pct > 20 else "#dc3232"
        self._liveness_label.setText(f"Liveness: {live_pct:.0f}%")
        self._liveness_label.setStyleSheet(f"color: {live_color}; font-size: 10px; font-family: Consolas;")

    def set_no_identities_warning(self, show: bool) -> None:
        self._no_data_label.setVisible(show)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_delete(self) -> None:
        selected = self._identity_list.selectedItems()
        if not selected:
            return
        identity_id = selected[0].data(Qt.ItemDataRole.UserRole)
        if identity_id:
            self.delete_identity_requested.emit(identity_id)


# ── Style helpers ──────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #333355;")
    return line


def _btn_style(bg: str, hover: str) -> str:
    return f"""
        QPushButton {{
            background: {bg};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 10px;
            font-size: 11px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background: {hover}; }}
        QPushButton:disabled {{ background: #333; color: #666; }}
    """


def _list_style() -> str:
    return """
        QListWidget {
            background: #0d0d1a;
            border: 1px solid #334;
            border-radius: 4px;
            color: #c0c0e0;
            font-size: 11px;
        }
        QListWidget::item:selected {
            background: #1a3fbf;
        }
        QListWidget::item:hover {
            background: #1a1a3a;
        }
    """
