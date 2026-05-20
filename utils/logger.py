"""
Logger factory with JSON structured output and rotating file handler.
Call get_logger(__name__) in every module.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SETUP_LOCK = threading.Lock()
_INITIALIZED = False


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "thread": record.threadName,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        # Any extra fields attached via extra={} in the log call
        for key, val in record.__dict__.items():
            if key.startswith("x_"):
                payload[key] = val
        return json.dumps(payload, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configure root logger. Idempotent — safe to call multiple times."""
    global _INITIALIZED
    with _SETUP_LOCK:
        if _INITIALIZED:
            return

        root = logging.getLogger()
        root.setLevel(getattr(logging, level.upper(), logging.INFO))

        formatter = _JsonFormatter()

        # Console handler (plain text for readability during dev)
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(console)

        # Rotating JSON file handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

        # Suppress noisy third-party loggers
        for noisy in ("PIL", "matplotlib", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger. Call setup_logging() first."""
    return logging.getLogger(name)


def reset_logging() -> None:
    """For tests only."""
    global _INITIALIZED
    with _SETUP_LOCK:
        _INITIALIZED = False
