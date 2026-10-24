"""Deploy the FastAPI backend to a free Hugging Face Space (Docker SDK).

Why a Space: the models need several GB of RAM. A Hugging Face CPU Basic Space
gives 2 vCPU / 16 GB (Render's free tier is only 512 MB). NOTE: since late 2026
Docker Spaces on CPU Basic require a Hugging Face PRO subscription ($9/month,
cancellable); without it Hugging Face answers "402 Payment Required". The script uploads only
what the Docker image needs (Dockerfile, requirements.txt, src/, services/,
serving/, scripts/) plus the Space README that tells Hugging Face to use Docker
on port 8000, then sets non-secret variables. Secrets (API keys, read token) are
added by hand in the Space's Settings page.

Usage:
    set HF_TOKEN=hf_xxx   (WRITE token)
    python -m scripts.deploy_hf_space --space-id <user>/cinematch-pro-api \
        --artifacts-repo <user>/cinematch-pro-artifacts
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from src.config import BASE_DIR, logger

SPACE_FILES: tuple[str, ...] = ("Dockerfile", "requirements.txt")
SPACE_DIRS: tuple[str, ...] = ("src", "services", "serving", "scripts")

SPACE_README = """---
title: CineMatch Pro API
emoji: 🎬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Movie recommendation + review intelligence API
---

# CineMatch Pro API

FastAPI backend for CineMatch Pro — hybrid recommendations, poster visual search,
aspect-based review sentiment and the CineBot assistant.
Interactive docs: `/docs` · health check: `/health`.
"""


def stage_space_files(destination: Path, root: Path = BASE_DIR) -> list[str]:
    """Copy the files a Docker Space needs into `destination`.

    Args:
        destination: Empty/new folder to stage into.
        root: Project root to copy from.

    Returns:
        list[str]: Staged paths relative to `destination` (sorted).

    Raises:
        FileNotFoundError: If a required file or folder is missing from `root`.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f"Required file not found: {source}")
        shutil.copy2(source, destination / name)
    for name in SPACE_DIRS:
        source = root / name
        if not source.is_dir():
            raise FileNotFoundError(f"Required folder not found: {source}")
        shutil.copytree(source, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    (destination / "README.md").write_text(SPACE_README, encoding="utf-8")
    return sorted(str(p.relative_to(destination)).replace("\\", "/") for p in destination.rglob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Deploy the CineMatch Pro API to a Hugging Face Space.")
    parser.add_argument("--space-id", required=True, help="e.g. narendrakalam2001/cinematch-pro-api")
    parser.add_argument("--artifacts-repo", required=True, help="Hub model repo holding the trained artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Create/update the Space, upload the code and set its variables.

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

    from huggingface_hub import HfApi

    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=args.space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    except HfHubHTTPError as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) == 402:
            logger.error(
                "Hugging Face refused to create the Space (402 Payment Required): Docker Spaces on the free CPU "
                "tier now need a PRO subscription (https://huggingface.co/subscribe/pro, $9/month, cancel anytime). "
                "Subscribe and re-run this command, or host the API elsewhere (see README -> Deployment)."
            )
            return 1
        raise
    with tempfile.TemporaryDirectory() as tmp:
        staged = stage_space_files(Path(tmp))
        logger.info("Uploading %d files to Space %s ...", len(staged), args.space_id)
        api.upload_folder(
            folder_path=tmp, repo_id=args.space_id, repo_type="space", commit_message="Deploy CineMatch Pro API"
        )
    for key, value in (("HF_ARTIFACTS_REPO", args.artifacts_repo), ("ENVIRONMENT", "production"), ("LOG_LEVEL", "INFO")):
        api.add_space_variable(repo_id=args.space_id, key=key, value=value)
    logger.info(
        "Uploaded. Now add these SECRETS in the Space settings: HF_TOKEN (read), TMDB_API_KEY, GEMINI_API_KEY, "
        "OMDB_API_KEY (optional). API URL: https://%s.hf.space",
        args.space_id.replace("/", "-").lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
