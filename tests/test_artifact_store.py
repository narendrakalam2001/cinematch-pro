"""Tests for src/artifact_store.py (download trained artifacts from the Hugging Face Hub)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.artifact_store import ALLOW_PATTERNS, REQUIRED_PATHS, ArtifactError, ensure_artifacts, missing_artifacts


def _create_all(base: Path) -> None:
    for relative in REQUIRED_PATHS:
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


def test_missing_artifacts_lists_only_what_is_absent(tmp_path):
    assert set(missing_artifacts(tmp_path)) == set(REQUIRED_PATHS)
    _create_all(tmp_path)
    assert missing_artifacts(tmp_path) == []
    (tmp_path / "models" / "svd_model.pkl").unlink()
    assert missing_artifacts(tmp_path) == ["models/svd_model.pkl"]


def test_nothing_is_downloaded_when_everything_is_present(tmp_path):
    _create_all(tmp_path)
    with patch("huggingface_hub.snapshot_download") as download:
        assert ensure_artifacts(repo_id="u/r", base_dir=tmp_path) is False
    download.assert_not_called()


def test_without_a_configured_repo_it_warns_and_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_ARTIFACTS_REPO", raising=False)
    with patch("huggingface_hub.snapshot_download") as download:
        assert ensure_artifacts(base_dir=tmp_path) is False
    download.assert_not_called()


def test_downloads_missing_files_using_env_repo_and_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ARTIFACTS_REPO", "user/cinematch-pro-artifacts")
    monkeypatch.setenv("HF_TOKEN", "hf_read_token")

    def fake_download(**kwargs):
        _create_all(Path(kwargs["local_dir"]))

    with patch("huggingface_hub.snapshot_download", side_effect=fake_download) as download:
        assert ensure_artifacts(base_dir=tmp_path) is True
    kwargs = download.call_args.kwargs
    assert kwargs["repo_id"] == "user/cinematch-pro-artifacts" and kwargs["token"] == "hf_read_token"
    assert kwargs["repo_type"] == "model" and kwargs["local_dir"] == str(tmp_path)
    assert kwargs["allow_patterns"] == list(ALLOW_PATTERNS)
    assert missing_artifacts(tmp_path) == []


def test_download_failure_becomes_a_friendly_artifact_error(tmp_path):
    with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("401 unauthorized")):
        with pytest.raises(ArtifactError, match="HF_TOKEN"):
            ensure_artifacts(repo_id="u/private", base_dir=tmp_path)


def test_incomplete_download_is_reported(tmp_path):
    with patch("huggingface_hub.snapshot_download"):  # "succeeds" but writes nothing
        with pytest.raises(ArtifactError, match="still missing"):
            ensure_artifacts(repo_id="u/r", base_dir=tmp_path)
