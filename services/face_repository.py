"""
File-system face repository.
- Embeddings → .npy (float32)
- Metadata → .json
- Atomic writes: write to .tmp then os.replace() (POSIX atomic)
- Corrupted files skipped with logged warning — never crash the app
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.entities import FaceIdentity
from core.interfaces import IFaceRepository
from utils.exceptions import (
    AtomicWriteError,
    CorruptedDataError,
    IdentityNotFoundError,
    StorageError,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_WRITE_LOCK = threading.Lock()


class FileFaceRepository(IFaceRepository):
    """
    Stores each identity in a subdirectory:
        data_dir/
            <identity_id>/
                embeddings.npy    (shape: [N, 128], dtype: float32)
                metadata.json     (display_name, sample_count, timestamps)
    """

    def __init__(self, data_dir: Path) -> None:
        self._root = Path(data_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("FileFaceRepository initialised", extra={"x_root": str(self._root)})

    # ── IFaceRepository ───────────────────────────────────────────────────────

    def save(self, identity: FaceIdentity) -> None:
        identity_dir = self._root / identity.identity_id
        identity_dir.mkdir(parents=True, exist_ok=True)

        emb_path = identity_dir / "embeddings.npy"
        meta_path = identity_dir / "metadata.json"

        # Stack embeddings to (N, 128) float32 array
        try:
            arr = np.stack(identity.embeddings, axis=0).astype(np.float32)
        except (ValueError, MemoryError) as exc:
            raise StorageError(f"Cannot stack embeddings for {identity.identity_id}", cause=exc) from exc

        now = datetime.now(tz=timezone.utc).isoformat()
        meta = {
            "identity_id": identity.identity_id,
            "display_name": identity.display_name,
            "sample_count": len(identity.embeddings),
            "created_at": identity.created_at or now,
            "updated_at": now,
        }

        with _WRITE_LOCK:
            self._atomic_write_npy(emb_path, arr)
            self._atomic_write_json(meta_path, meta)

        logger.info(
            "Identity saved",
            extra={
                "x_id": identity.identity_id,
                "x_samples": len(identity.embeddings),
            },
        )

    def load(self, identity_id: str) -> FaceIdentity:
        identity_dir = self._root / identity_id
        if not identity_dir.exists():
            raise IdentityNotFoundError(f"Identity not found: {identity_id!r}")
        return self._load_from_dir(identity_dir)

    def load_all(self) -> list[FaceIdentity]:
        identities: list[FaceIdentity] = []
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue
            try:
                ident = self._load_from_dir(entry)
                identities.append(ident)
            except CorruptedDataError as exc:
                logger.warning("Skipping corrupted identity %r: %s", entry.name, exc)
            except Exception as exc:
                logger.warning("Skipping identity %r due to error: %s", entry.name, exc)
        return identities

    def delete(self, identity_id: str) -> None:
        identity_dir = self._root / identity_id
        if not identity_dir.exists():
            raise IdentityNotFoundError(f"Cannot delete — not found: {identity_id!r}")
        import shutil
        with _WRITE_LOCK:
            shutil.rmtree(identity_dir)
        logger.info("Identity deleted", extra={"x_id": identity_id})

    def exists(self, identity_id: str) -> bool:
        return (self._root / identity_id).is_dir()

    def list_ids(self) -> list[str]:
        return [
            d.name for d in sorted(self._root.iterdir()) if d.is_dir()
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_from_dir(self, directory: Path) -> FaceIdentity:
        emb_path = directory / "embeddings.npy"
        meta_path = directory / "metadata.json"

        if not emb_path.exists() or not meta_path.exists():
            raise CorruptedDataError(
                f"Missing embeddings or metadata in {directory}"
            )

        try:
            arr = np.load(str(emb_path), allow_pickle=False)
        except Exception as exc:
            raise CorruptedDataError(
                f"Cannot load embeddings from {emb_path}", cause=exc
            ) from exc

        if arr.ndim != 2 or arr.shape[1] == 0:
            raise CorruptedDataError(
                f"Bad embedding shape {arr.shape} in {emb_path}"
            )

        arr = arr.astype(np.float32)
        embeddings = [arr[i] for i in range(arr.shape[0])]

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CorruptedDataError(
                f"Cannot parse metadata {meta_path}", cause=exc
            ) from exc

        return FaceIdentity(
            identity_id=meta.get("identity_id", directory.name),
            display_name=meta.get("display_name", directory.name),
            embeddings=embeddings,
            sample_count=meta.get("sample_count", len(embeddings)),
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
        )

    @staticmethod
    def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
        # np.save() appends .npy automatically if not present — use a .tmp dir instead
        # to avoid the double-extension problem (foo.npy.tmp -> foo.npy.tmp.npy)
        tmp = path.parent / (path.stem.replace(".npy", "") + "_tmp.npy")
        try:
            np.save(str(tmp), arr)
            os.replace(str(tmp), str(path))
        except Exception as exc:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise AtomicWriteError(f"Atomic write failed: {path}", cause=exc) from exc

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        tmp = path.parent / (path.name + ".tmp")
        try:
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(str(tmp), str(path))
        except Exception as exc:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise AtomicWriteError(f"Atomic write failed: {path}", cause=exc) from exc
