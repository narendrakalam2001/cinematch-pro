"""Fetch the trained artifacts from the Hugging Face Hub when they are not on disk.

The trained files (SVD model, TF-IDF, FAISS index, RoBERTa weights, ChromaDB
store, merged catalogue) total ~1 GB — far beyond GitHub's 100 MB per-file
limit — so they are kept in a (private) Hugging Face model repo. A deployed
container (Hugging Face Space, Render, ...) calls `ensure_artifacts` at
startup and pulls them once.

Configuration (environment variables):
    HF_ARTIFACTS_REPO: ``<user>/<repo>`` holding the artifacts.
    HF_TOKEN: Read token (required when the repo is private).
"""

from __future__ import annotations

import os
from pathlib import Path

from src.config import BASE_DIR, LITE_MODE, logger

# Paths (relative to the project root) the running API cannot work without.
REQUIRED_PATHS: tuple[str, ...] = (
    "models/svd_model.pkl",
    "models/tfidf_matrix.pkl",
    "models/tfidf_vectorizer.pkl",
    "models/faiss_poster.index",
    "models/faiss_poster_metadata.pkl",
    "models/roberta_sentiment/config.json",
    "models/roberta_sentiment/model.safetensors",
    "chromadb_store/chroma.sqlite3",
    "data/processed/merged_movies.parquet",
)

# What gets uploaded to / downloaded from the Hub repo (mirrors the project layout).
ALLOW_PATTERNS: tuple[str, ...] = (
    "models/**",
    "chromadb_store/**",
    "data/processed/merged_movies.parquet",
)


# Lite mode needs only the catalogue and the TF-IDF content filter (~70 MB instead of ~1.2 GB).
LITE_REQUIRED_PATHS: tuple[str, ...] = (
    "models/tfidf_matrix.pkl",
    "models/tfidf_vectorizer.pkl",
    "data/processed/merged_movies.parquet",
)
LITE_ALLOW_PATTERNS: tuple[str, ...] = (
    "models/tfidf_*.pkl",
    "models/model_card.json",
    "data/processed/merged_movies.parquet",
)


class ArtifactError(RuntimeError):
    """Raised when required artifacts are missing and cannot be downloaded."""


def missing_artifacts(base_dir: Path = BASE_DIR, lite: bool = False) -> list[str]:
    """List required artifact paths that are not present on disk.

    Args:
        base_dir: Project root to check.
        lite: Check only the small set the lite deployment needs.

    Returns:
        list[str]: Relative paths (from `REQUIRED_PATHS`, or `LITE_REQUIRED_PATHS`
        in lite mode) that do not exist.
    """
    required = LITE_REQUIRED_PATHS if lite else REQUIRED_PATHS
    return [relative for relative in required if not (base_dir / relative).exists()]


def ensure_artifacts(
    repo_id: str | None = None,
    token: str | None = None,
    base_dir: Path = BASE_DIR,
    revision: str | None = None,
    lite: bool | None = None,
) -> bool:
    """Download the trained artifacts from the Hugging Face Hub if any are missing.

    Does nothing when everything is already on disk (local development, or a
    container restart that kept its files).

    Args:
        repo_id: Hub model repo (``user/name``). Defaults to ``$HF_ARTIFACTS_REPO``.
        token: Hub access token. Defaults to ``$HF_TOKEN``.
        base_dir: Project root to download into.
        revision: Optional branch / tag / commit to pin.
        lite: Download only the lite subset. Defaults to ``CINEMATCH_LITE``.

    Returns:
        bool: ``True`` if a download happened, ``False`` if nothing was needed
        or no repo is configured (a warning is logged in that case).

    Raises:
        ArtifactError: If the download fails or files are still missing after it.
    """
    lite = LITE_MODE if lite is None else lite
    missing = missing_artifacts(base_dir, lite)
    if not missing:
        logger.info("All trained artifacts are present — nothing to download.")
        return False

    repo_id = repo_id or os.getenv("HF_ARTIFACTS_REPO")
    if not repo_id:
        logger.warning(
            "Missing artifacts (%s) and HF_ARTIFACTS_REPO is not set — the API will start with "
            "reduced functionality.",
            ", ".join(missing[:3]) + ("…" if len(missing) > 3 else ""),
        )
        return False

    logger.info("Downloading %d missing artifact(s) from Hugging Face Hub repo %s ...", len(missing), repo_id)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token or os.getenv("HF_TOKEN") or None,
            local_dir=str(base_dir),
            allow_patterns=list(LITE_ALLOW_PATTERNS if lite else ALLOW_PATTERNS),
            revision=revision,
        )
    except Exception as exc:  # noqa: BLE001 - surface any Hub/network/auth failure as one domain error
        raise ArtifactError(
            f"Could not download artifacts from '{repo_id}': {exc}. "
            "If the repo is private, set HF_TOKEN to a token with read access."
        ) from exc

    still_missing = missing_artifacts(base_dir, lite)
    if still_missing:
        raise ArtifactError(f"Download finished but these files are still missing: {', '.join(still_missing)}")
    logger.info("Artifacts downloaded from %s.", repo_id)
    return True
