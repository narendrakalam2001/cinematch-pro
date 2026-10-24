"""Tests for the deployment helpers: upload_artifacts, deploy_hf_space, .gitignore and the API startup hook."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import deploy_hf_space, upload_artifacts
from src.artifact_store import REQUIRED_PATHS

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# deploy_hf_space
# ---------------------------------------------------------------------------
def test_stage_space_files_contains_only_what_docker_needs(tmp_path):
    staged = deploy_hf_space.stage_space_files(tmp_path / "stage", root=ROOT)
    assert {"Dockerfile", "requirements.txt", "README.md", "serving/api.py", "services/movie_catalog.py", "src/artifact_store.py"} <= set(staged)
    assert not any(p.startswith(("tests/", "models/", "chromadb_store/", "frontend/", "notebooks/", "docs/")) for p in staged)
    assert not any("__pycache__" in p or p.endswith(".pyc") for p in staged)
    readme = (tmp_path / "stage" / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n") and "sdk: docker" in readme and "app_port: 8000" in readme


def test_stage_space_files_fails_clearly_when_a_folder_is_missing(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM x")
    (tmp_path / "requirements.txt").write_text("x")
    with pytest.raises(FileNotFoundError, match="src"):
        deploy_hf_space.stage_space_files(tmp_path / "stage", root=tmp_path)


def test_deploy_space_requires_a_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert deploy_hf_space.main(["--space-id", "u/api", "--artifacts-repo", "u/art"]) == 1


def test_deploy_space_creates_docker_space_uploads_and_sets_variables(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_write")
    api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=api):
        assert deploy_hf_space.main(["--space-id", "u/api", "--artifacts-repo", "u/art"]) == 0
    assert api.create_repo.call_args.kwargs["repo_type"] == "space" and api.create_repo.call_args.kwargs["space_sdk"] == "docker"
    assert api.upload_folder.call_args.kwargs["repo_id"] == "u/api"
    variables = {c.kwargs["key"]: c.kwargs["value"] for c in api.add_space_variable.call_args_list}
    assert variables["HF_ARTIFACTS_REPO"] == "u/art" and variables["ENVIRONMENT"] == "production"
    assert api.add_space_secret.call_count == 0  # secrets are added by hand, never by the script


def test_deploy_space_explains_the_402_pro_requirement(monkeypatch, caplog):
    """HF answers 402 for Docker Spaces on free hardware; the script says what to do instead of a traceback."""
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    monkeypatch.setenv("HF_TOKEN", "hf_write")
    response = httpx.Response(402, request=httpx.Request("POST", "https://huggingface.co/api/repos/create"))
    api = MagicMock()
    api.create_repo.side_effect = HfHubHTTPError("402 Payment Required", response=response)
    with patch("huggingface_hub.HfApi", return_value=api):
        assert deploy_hf_space.main(["--space-id", "u/api", "--artifacts-repo", "u/art"]) == 1
    api.upload_folder.assert_not_called()  # nothing is uploaded to a Space that was not created

    api.create_repo.side_effect = HfHubHTTPError("500 boom", response=httpx.Response(500, request=response.request))
    with patch("huggingface_hub.HfApi", return_value=api), pytest.raises(HfHubHTTPError):
        deploy_hf_space.main(["--space-id", "u/api", "--artifacts-repo", "u/art"])  # other errors still surface


# ---------------------------------------------------------------------------
# upload_artifacts
# ---------------------------------------------------------------------------
def test_upload_requires_token_and_complete_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert upload_artifacts.main(["--repo-id", "u/art"]) == 1
    monkeypatch.setenv("HF_TOKEN", "hf_write")
    with patch.object(upload_artifacts, "BASE_DIR", tmp_path):  # nothing trained in an empty folder
        assert upload_artifacts.main(["--repo-id", "u/art"]) == 1


def test_upload_creates_private_repo_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_write")
    for relative in REQUIRED_PATHS:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(b"x")
    api = MagicMock()
    with patch.object(upload_artifacts, "BASE_DIR", tmp_path), patch("huggingface_hub.HfApi", return_value=api):
        assert upload_artifacts.main(["--repo-id", "u/art"]) == 0
        assert api.create_repo.call_args.kwargs["private"] is True
        assert upload_artifacts.main(["--repo-id", "u/art", "--public"]) == 0
        assert api.create_repo.call_args.kwargs["private"] is False
    assert "models/**" in api.upload_folder.call_args.kwargs["allow_patterns"]


# ---------------------------------------------------------------------------
# .gitignore — deployment files go to GitHub, secrets and big artifacts do not
# ---------------------------------------------------------------------------
MUST_BE_COMMITTED = [
    "Dockerfile", ".dockerignore", "render.yaml", "requirements.txt", "frontend/requirements.txt",
    ".streamlit/config.toml", ".env.example", "README.md", "LICENSE", ".github/workflows/ci.yml",
    "src/config.py", "services/recommendation_service.py", "serving/api.py", "frontend/app.py",
    "scripts/run_api.py", "tests/test_api.py", "models/model_card.json",
    "lite/streamlit_app.py", "lite/requirements.txt", "src/errors.py",
    "docs/screenshots/cinebot.png", "docs/gifs/User_Portal.gif", "docs/architecture/cinematch_pro_system_architecture.svg",
    "notebooks/Kaggle_Training.ipynb", "notebooks/cinematch_pro_eda.ipynb",
]
MUST_BE_IGNORED = [
    ".env", ".env.local", ".streamlit/secrets.toml",
    "models/svd_model.pkl", "models/roberta_sentiment/model.safetensors", "models/faiss_poster.index",
    "chromadb_store/chroma.sqlite3", "data/processed/merged_movies.parquet", "data/raw/ratings.csv",
    "cinematch_artifacts.zip", "src/__pycache__/x.pyc", ".pytest_cache/v", "notebooks/.ipynb_checkpoints/a.ipynb",
]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_gitignore_keeps_deployment_files_and_blocks_secrets_and_artifacts(tmp_path):
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=tmp_path).returncode == 0

    wrongly_ignored = [p for p in MUST_BE_COMMITTED if ignored(p)]
    wrongly_tracked = [p for p in MUST_BE_IGNORED if not ignored(p)]
    assert not wrongly_ignored, f"These deployment files would NOT be pushed: {wrongly_ignored}"
    assert not wrongly_tracked, f"These files would be pushed but must not be: {wrongly_tracked}"


# ---------------------------------------------------------------------------
# API startup hook
# ---------------------------------------------------------------------------
def test_api_startup_fetches_artifacts_and_survives_failure():
    from fastapi.testclient import TestClient

    import serving.api as api_module
    from src.artifact_store import ArtifactError

    with patch.object(api_module, "ensure_artifacts", side_effect=ArtifactError("hub down")) as ensure, \
            patch.object(api_module, "get_service", return_value=MagicMock()) as warm:
        with TestClient(api_module.app):
            pass
    ensure.assert_called_once()
    warm.assert_called()  # the API still starts (degraded) when the download fails
