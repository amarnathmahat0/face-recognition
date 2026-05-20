"""
Storage round-trip tests for FileFaceRepository.
Uses a temp directory — no permanent disk side effects.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.entities import FaceIdentity
from services.face_repository import FileFaceRepository
from utils.exceptions import (
    AtomicWriteError,
    CorruptedDataError,
    IdentityNotFoundError,
)


def _make_identity(identity_id: str = "alice", n_samples: int = 3) -> FaceIdentity:
    return FaceIdentity(
        identity_id=identity_id,
        display_name=identity_id.capitalize(),
        embeddings=[
            np.random.rand(128).astype(np.float32) for _ in range(n_samples)
        ],
        sample_count=n_samples,
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
    )


@pytest.fixture()
def repo(tmp_path: Path) -> FileFaceRepository:
    return FileFaceRepository(tmp_path)


class TestFileFaceRepository:

    # ── save / load round-trip ─────────────────────────────────────────────────

    def test_save_and_load_roundtrip(self, repo):
        identity = _make_identity("bob", n_samples=5)
        repo.save(identity)
        loaded = repo.load("bob")

        assert loaded.identity_id == "bob"
        assert loaded.display_name == "Bob"
        assert loaded.sample_count == 5
        assert len(loaded.embeddings) == 5
        assert loaded.embeddings[0].dtype == np.float32

    def test_embeddings_values_preserved(self, repo):
        original = _make_identity("carol", 2)
        repo.save(original)
        loaded = repo.load("carol")

        for orig_emb, load_emb in zip(original.embeddings, loaded.embeddings):
            np.testing.assert_allclose(orig_emb, load_emb, rtol=1e-6)

    def test_embedding_shape_correct(self, repo):
        identity = _make_identity("dave", 4)
        repo.save(identity)
        loaded = repo.load("dave")
        for emb in loaded.embeddings:
            assert emb.shape == (128,)

    # ── exists / list_ids ──────────────────────────────────────────────────────

    def test_exists_false_before_save(self, repo):
        assert not repo.exists("nobody")

    def test_exists_true_after_save(self, repo):
        repo.save(_make_identity("eve"))
        assert repo.exists("eve")

    def test_list_ids_empty(self, repo):
        assert repo.list_ids() == []

    def test_list_ids_multiple(self, repo):
        for name in ["alice", "bob", "carol"]:
            repo.save(_make_identity(name))
        ids = repo.list_ids()
        assert sorted(ids) == ["alice", "bob", "carol"]

    # ── load_all ───────────────────────────────────────────────────────────────

    def test_load_all_returns_all(self, repo):
        for name in ["x1", "x2", "x3"]:
            repo.save(_make_identity(name))
        all_ids = repo.load_all()
        assert len(all_ids) == 3

    def test_load_all_skips_corrupted(self, repo, tmp_path):
        """Corrupted identity directory is skipped, healthy ones returned."""
        repo.save(_make_identity("healthy"))

        # Create a broken identity dir: metadata exists, embeddings is corrupt
        bad_dir = tmp_path / "broken"
        bad_dir.mkdir()
        (bad_dir / "embeddings.npy").write_bytes(b"not a numpy file at all!!!!")
        (bad_dir / "metadata.json").write_text(
            json.dumps({"identity_id": "broken", "display_name": "Broken",
                        "sample_count": 1, "created_at": "", "updated_at": ""}),
            encoding="utf-8",
        )

        result = repo.load_all()
        ids = [i.identity_id for i in result]
        assert "healthy" in ids
        assert "broken" not in ids

    def test_load_all_skips_missing_files(self, repo, tmp_path):
        """Directory with no npy/json is gracefully skipped."""
        repo.save(_make_identity("good"))
        (tmp_path / "empty_dir").mkdir()
        result = repo.load_all()
        assert len(result) == 1
        assert result[0].identity_id == "good"

    # ── delete ────────────────────────────────────────────────────────────────

    def test_delete_removes_identity(self, repo):
        repo.save(_make_identity("frank"))
        assert repo.exists("frank")
        repo.delete("frank")
        assert not repo.exists("frank")

    def test_delete_nonexistent_raises(self, repo):
        with pytest.raises(IdentityNotFoundError):
            repo.delete("nobody")

    def test_delete_then_load_all_empty(self, repo):
        repo.save(_make_identity("grace"))
        repo.delete("grace")
        assert repo.load_all() == []

    # ── error cases ───────────────────────────────────────────────────────────

    def test_load_nonexistent_raises(self, repo):
        with pytest.raises(IdentityNotFoundError):
            repo.load("nobody")

    def test_load_corrupted_npy_raises(self, repo, tmp_path):
        bad_dir = tmp_path / "corrupt"
        bad_dir.mkdir()
        (bad_dir / "embeddings.npy").write_bytes(b"\x00\x01BAD")
        (bad_dir / "metadata.json").write_text(
            json.dumps({"identity_id": "corrupt", "display_name": "C",
                        "sample_count": 1, "created_at": "", "updated_at": ""}),
            encoding="utf-8",
        )
        with pytest.raises(CorruptedDataError):
            repo.load("corrupt")

    def test_load_corrupted_json_raises(self, repo, tmp_path):
        ident = _make_identity("semi")
        repo.save(ident)
        # Corrupt the metadata
        meta_path = tmp_path / "semi" / "metadata.json"
        meta_path.write_text("{ not valid json !!!", encoding="utf-8")
        with pytest.raises(CorruptedDataError):
            repo.load("semi")

    # ── atomic write ──────────────────────────────────────────────────────────

    def test_no_tmp_files_left_after_save(self, repo, tmp_path):
        repo.save(_make_identity("han"))
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"

    def test_overwrite_updates_sample_count(self, repo):
        repo.save(_make_identity("ivan", n_samples=2))
        repo.save(_make_identity("ivan", n_samples=8))
        loaded = repo.load("ivan")
        assert loaded.sample_count == 8
        assert len(loaded.embeddings) == 8

    # ── data integrity ────────────────────────────────────────────────────────

    def test_embeddings_stored_as_float32(self, repo, tmp_path):
        identity = _make_identity("judy", 2)
        repo.save(identity)
        arr = np.load(str(tmp_path / "judy" / "embeddings.npy"), allow_pickle=False)
        assert arr.dtype == np.float32

    def test_metadata_json_readable(self, repo, tmp_path):
        repo.save(_make_identity("karl", 3))
        meta = json.loads((tmp_path / "karl" / "metadata.json").read_text())
        assert meta["identity_id"] == "karl"
        assert meta["sample_count"] == 3
        assert "created_at" in meta
        assert "updated_at" in meta
