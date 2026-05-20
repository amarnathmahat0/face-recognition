"""
FaceID — Production Face Recognition Desktop App
Entry point: wires dependency injection, handles CLI flags, launches QApplication.

Usage:
    python main.py                           # GUI mode
    python main.py --config /path/cfg.yaml  # custom config
    python main.py --headless --verify --image /path/to/img.jpg  # CLI mode
    python main.py --debug                   # enable P99 overlay in status bar
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import traceback
from pathlib import Path

from utils.logger import get_logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FaceID Face Recognition App")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    p.add_argument("--headless", action="store_true", help="Run without GUI")
    p.add_argument("--verify", action="store_true", help="(headless) verify an image")
    p.add_argument("--image", type=str, default=None, help="(headless) image path to verify")
    p.add_argument("--debug", action="store_true", help="Enable debug overlays")
    return p.parse_args()


def _headless_verify(image_path: str) -> int:
    """CLI verification — prints JSON result to stdout."""
    from utils.config import get_config
    from utils.logger import setup_logging

    cfg = get_config()
    setup_logging(cfg.logging.level, cfg.logging.file, cfg.logging.max_bytes, cfg.logging.backup_count)

    import cv2
    import numpy as np

    from services.detector import HaarFaceDetector
    from services.embedder import FaceRecognitionEmbedder
    from services.face_repository import FileFaceRepository
    from services.recognition_engine import CosineRecognitionEngine
    from core.use_cases import VerifyFaceUseCase

    def _blur(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _liveness(cur, prev):
        return 1.0  # CLI: no motion check

    path = Path(image_path)
    if not path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        return 1

    frame = cv2.imread(str(path))
    if frame is None:
        print(json.dumps({"error": "Cannot decode image"}))
        return 1

    detector = HaarFaceDetector(min_size=(cfg.recognition.min_face_size,) * 2)
    embedder = FaceRecognitionEmbedder()
    engine = CosineRecognitionEngine(cfg.recognition.similarity_threshold)
    repository = FileFaceRepository(cfg.storage.data_dir)
    identities = repository.load_all()
    engine.reload(identities)

    use_case = VerifyFaceUseCase(
        detector=detector,
        embedder=embedder,
        engine=engine,
        similarity_threshold=cfg.recognition.similarity_threshold,
        confidence_high=cfg.recognition.confidence_high,
        confidence_medium=cfg.recognition.confidence_medium,
        compute_blur=_blur,
        compute_liveness=_liveness,
    )

    result = use_case.verify(frame, identities)
    output = {
        "faces_detected": len(result.matches),
        "inference_ms": round(result.inference_ms, 2),
        "matches": [
            {
                "identity_id": m.identity_id,
                "display_name": m.display_name,
                "similarity": round(m.similarity, 4),
                "confidence": m.confidence.name,
                "bbox": {"x": m.bbox.x, "y": m.bbox.y, "w": m.bbox.w, "h": m.bbox.h},
            }
            for m in result.matches
        ],
    }
    print(json.dumps(output, indent=2))
    return 0


def _run_gui(debug: bool) -> int:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPalette, QColor

    from utils.config import get_config
    from utils.logger import setup_logging

    cfg = get_config()
    setup_logging(
        level=cfg.logging.level,
        log_file=cfg.logging.file,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
    )

    if debug:
        cfg.ui.debug_mode = True  # type: ignore[attr-defined]

    app = QApplication(sys.argv)
    app.setApplicationName("FaceID")
    app.setOrganizationName("FaceID")

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(13, 13, 26))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(10, 10, 20))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(18, 18, 42))
    palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(26, 26, 58))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(41, 98, 255))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    from ui.main_window import MainWindow

    def _global_exception_hook(exc_type, exc_value, exc_tb):
        from utils.logger import get_logger
        log = get_logger("uncaught")
        log.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        traceback.print_exception(exc_type, exc_value, exc_tb)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(
            None,
            "Fatal Error",
            f"{exc_type.__name__}: {exc_value}\n\nCheck logs for details.",
        )

    sys.excepthook = _global_exception_hook

    window = MainWindow()
    window.show()

    app.aboutToQuit.connect(window.close)

    def _handle_signal(signum, frame):
        logger = get_logger("signal")
        logger.info("Received signal %s, quitting application", signum)
        app.quit()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        return app.exec()
    finally:
        logging.shutdown()


def main() -> None:
    args = _parse_args()

    # Config path must be set before any import that calls get_config()
    if args.config:
        from utils.config import get_config
        get_config(args.config)

    if args.headless and args.verify and args.image:
        sys.exit(_headless_verify(args.image))
    elif args.headless:
        print("Headless mode requires --verify --image <path>")
        sys.exit(1)
    else:
        sys.exit(_run_gui(debug=args.debug))


if __name__ == "__main__":
    main()
