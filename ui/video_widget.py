"""
VideoWidget — QLabel subclass for live webcam display.
Renders frames via QImage, paints overlays (bounding boxes, labels, status)
directly in paintEvent — zero threading issues since painting is always main thread.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import (
    QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPixmap,
)
from PyQt6.QtWidgets import QLabel, QSizePolicy

from core.entities import (
    BoundingBox,
    ConfidenceBand,
    FaceMatch,
    SampleResult,
    SampleStatus,
    VerificationResult,
)


_BAND_COLORS = {
    ConfidenceBand.HIGH: QColor(0, 220, 80),      # green
    ConfidenceBand.MEDIUM: QColor(255, 200, 0),   # yellow
    ConfidenceBand.LOW: QColor(255, 120, 0),      # orange
    ConfidenceBand.UNKNOWN: QColor(220, 50, 50),  # red
}

_BOX_THICKNESS = 2
_LABEL_FONT_SIZE = 11
_OVERLAY_ALPHA = 200


class VideoWidget(QLabel):
    """
    Displays live webcam frames with painted overlays.
    Thread-safe: frames are set via set_frame() (main thread only).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a2e;")

        self._pixmap: Optional[QPixmap] = None
        self._verification_result: Optional[VerificationResult] = None
        self._registration_bbox: Optional[BoundingBox] = None
        self._registration_sample: Optional[SampleResult] = None
        self._status_text: str = "Waiting for camera…"
        self._show_status = True

        font = QFont("Consolas", _LABEL_FONT_SIZE)
        font.setBold(True)
        self._label_font = font

    # ── Public API (main thread only) ─────────────────────────────────────────

    def set_frame(self, frame: np.ndarray) -> None:
        """Convert BGR numpy array to QPixmap and trigger repaint."""
        h, w, ch = frame.shape
        assert ch == 3, "Expected BGR 3-channel frame"
        # BGR → RGB
        rgb = frame[:, :, ::-1].copy()
        img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img)
        self._show_status = False
        self.update()

    def set_verification_result(self, result: Optional[VerificationResult]) -> None:
        self._verification_result = result
        self._registration_bbox = None
        self._registration_sample = None
        self.update()

    def set_registration_sample(
        self,
        bbox: Optional[BoundingBox],
        sample: Optional[SampleResult],
    ) -> None:
        self._registration_bbox = bbox
        self._registration_sample = sample
        self._verification_result = None
        self.update()

    def set_status_text(self, text: str) -> None:
        self._status_text = text
        self._show_status = True
        self._pixmap = None
        self.update()

    def clear_overlays(self) -> None:
        self._verification_result = None
        self._registration_bbox = None
        self._registration_sample = None
        self.update()

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_w = self.width()
        widget_h = self.height()

        if self._pixmap:
            scaled = self._pixmap.scaled(
                widget_w,
                widget_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x_off = (widget_w - scaled.width()) // 2
            y_off = (widget_h - scaled.height()) // 2
            painter.drawPixmap(x_off, y_off, scaled)

            scale_x = scaled.width() / self._pixmap.width()
            scale_y = scaled.height() / self._pixmap.height()

            # Paint verification overlays
            if self._verification_result is not None:
                self._paint_verification(
                    painter, self._verification_result,
                    x_off, y_off, scale_x, scale_y,
                )

            # Paint registration bbox
            if self._registration_bbox is not None:
                self._paint_registration_bbox(
                    painter, self._registration_bbox, self._registration_sample,
                    x_off, y_off, scale_x, scale_y,
                )
        else:
            # Dark background with status text
            painter.fillRect(0, 0, widget_w, widget_h, QColor(26, 26, 46))
            if self._show_status:
                painter.setPen(QColor(140, 140, 180))
                painter.setFont(QFont("Consolas", 14))
                painter.drawText(
                    QRect(0, 0, widget_w, widget_h),
                    Qt.AlignmentFlag.AlignCenter,
                    self._status_text,
                )

        painter.end()

    def _paint_verification(
        self,
        painter: QPainter,
        result: VerificationResult,
        x_off: int, y_off: int,
        scale_x: float, scale_y: float,
    ) -> None:
        if not result.matches:
            # No face overlay — suggest positioning
            painter.setPen(QColor(255, 180, 0))
            painter.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
            painter.drawText(
                QRect(x_off + 10, y_off + 20, 400, 40),
                Qt.AlignmentFlag.AlignLeft,
                "📷 Face not detected\nTurn toward camera or improve lighting",
            )
            return

        for match in result.matches:
            color = _BAND_COLORS.get(match.confidence, QColor(200, 200, 200))
            self._draw_bbox(painter, match.bbox, color, x_off, y_off, scale_x, scale_y)
            self._draw_label(
                painter, match, color, x_off, y_off, scale_x, scale_y
            )

        # Liveness warning
        best = result.best_match
        if best and best.liveness_score < 0.3:
            painter.setPen(QColor(255, 80, 80))
            painter.setFont(QFont("Consolas", 11))
            painter.drawText(
                QRect(x_off, y_off + 40, 300, 25),
                Qt.AlignmentFlag.AlignLeft,
                "⚠ Move slowly for liveness check",
            )

    def _paint_registration_bbox(
        self,
        painter: QPainter,
        bbox: BoundingBox,
        sample: Optional[SampleResult],
        x_off: int, y_off: int,
        scale_x: float, scale_y: float,
    ) -> None:
        from core.entities import SampleStatus
        if sample and sample.status == SampleStatus.ACCEPTED:
            color = QColor(0, 220, 80)
        elif sample and sample.status == SampleStatus.REJECTED_FRAME_GAP:
            color = QColor(80, 80, 200)
        else:
            color = QColor(220, 50, 50)

        self._draw_bbox(painter, bbox, color, x_off, y_off, scale_x, scale_y)

        if sample and sample.rejection_reason:
            painter.setPen(color)
            painter.setFont(QFont("Consolas", 10))
            sx = x_off + int(bbox.x * scale_x)
            sy = y_off + int(bbox.y * scale_y) - 5
            painter.drawText(QPoint(sx, max(y_off + 15, sy)), sample.rejection_reason)

    def _draw_bbox(
        self,
        painter: QPainter,
        bbox: BoundingBox,
        color: QColor,
        x_off: int, y_off: int,
        scale_x: float, scale_y: float,
    ) -> None:
        pen = QPen(color, _BOX_THICKNESS)
        painter.setPen(pen)
        rx = x_off + int(bbox.x * scale_x)
        ry = y_off + int(bbox.y * scale_y)
        rw = int(bbox.w * scale_x)
        rh = int(bbox.h * scale_y)
        painter.drawRect(rx, ry, rw, rh)

        # Corner accents for polish
        corner = min(rw, rh) // 4
        for cx, cy, dx, dy in [
            (rx, ry, 1, 1), (rx + rw, ry, -1, 1),
            (rx, ry + rh, 1, -1), (rx + rw, ry + rh, -1, -1),
        ]:
            thick_pen = QPen(color, _BOX_THICKNESS + 2)
            painter.setPen(thick_pen)
            painter.drawLine(cx, cy, cx + dx * corner, cy)
            painter.drawLine(cx, cy, cx, cy + dy * corner)

    def _draw_label(
        self,
        painter: QPainter,
        match: FaceMatch,
        color: QColor,
        x_off: int, y_off: int,
        scale_x: float, scale_y: float,
    ) -> None:
        name = match.display_name or "Unknown"
        pct = f"{match.similarity * 100:.1f}%"
        band = match.confidence.name.capitalize()
        text = f"{name}  {pct}  [{band}]"

        painter.setFont(self._label_font)
        fm = QFontMetrics(self._label_font)
        text_w = fm.horizontalAdvance(text) + 10
        text_h = fm.height() + 6

        rx = x_off + int(match.bbox.x * scale_x)
        ry = y_off + int(match.bbox.y * scale_y)
        label_y = max(y_off, ry - text_h - 2)

        # Background pill
        bg = QColor(color)
        bg.setAlpha(_OVERLAY_ALPHA)
        painter.fillRect(rx, label_y, text_w, text_h, bg)

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rx + 5, label_y + text_h - 4, text)
