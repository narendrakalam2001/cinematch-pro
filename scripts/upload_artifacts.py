"""Upload the trained artifacts to a Hugging Face Hub model repo.

GitHub cannot hold the ~1 GB of trained files, so they go to the Hub (private by
default — MovieLens' terms don't allow redistributing its data) and the deployed
API downloads them at startup (see `src/artifact_store.py`).

Usage:
    set HF_TOKEN=hf_xxx   (a token with WRITE access)
    python -m scripts.upload_artifacts --repo-id <user>/cinematch-pro-artifacts
"""

from __future__ import annotations

import argparse
import os
import sys

from src.artifact_store import ALLOW_PATTERNS, missing_artifacts
from src.config import BASE_DIR, logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Upload CineMatch Pro artifacts to the Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Target Hub repo, e.g. narendrakalam2001/cinematch-pro-artifacts")
    parser.add_argument("--public", action="store_true", help="Create the repo as public (default: private).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Create the Hub repo if needed and upload models/, chromadb_store/ and the catalogue.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        int: Exit code (0 success, 1 failure).
    """
    args = parse_args(argv)
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN is not set. Create a WRITE token at huggingface.co/settings/tokens and export it.")
        return 1
    missing = missing_artifacts(BASE_DIR)
    if missing:
        logger.error("Cannot upload — these artifacts are missing locally: %s", ", ".join(missing))
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    logger.info("Uploading artifacts to %s (this can take several minutes)...", args.repo_id)
    api.upload_folder(
        folder_path=str(BASE_DIR),
        repo_id=args.repo_id,
        repo_type="model",
        allow_patterns=list(ALLOW_PATTERNS),
        commit_message="Upload CineMatch Pro trained artifacts",
    )
    logger.info("Done. Set HF_ARTIFACTS_REPO=%s (and HF_TOKEN with read access) in your deployment.", args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
