"""
CLI entry point: run the FastAPI backend with uvicorn.

Usage:
    python -m scripts.run_api
    python -m scripts.run_api --port 9000 --reload
"""

from __future__ import annotations

import argparse

import uvicorn

from src.config import logger, settings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the API server launcher.

    Returns:
        argparse.Namespace: Parsed host/port/reload options.
    """
    parser = argparse.ArgumentParser(description="Run CineMatch Pro's FastAPI backend.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only).",
    )
    return parser.parse_args()


def main() -> None:
    """Launch the uvicorn server hosting `serving.api:app`."""
    args = parse_args()
    logger.info(
        "Starting CineMatch Pro API on %s:%d (environment=%s, reload=%s)",
        args.host,
        args.port,
        settings.environment,
        args.reload,
    )
    uvicorn.run("serving.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
