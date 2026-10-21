"""
CLI entry point: run the Streamlit frontend.

Usage:
    python -m scripts.run_app
    python -m scripts.run_app --port 8501
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.config import BASE_DIR, logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Streamlit launcher.

    Returns:
        argparse.Namespace: Parsed port option.
    """
    parser = argparse.ArgumentParser(description="Run CineMatch Pro's Streamlit frontend.")
    parser.add_argument("--port", type=int, default=8501)
    return parser.parse_args()


def main() -> None:
    """Launch `streamlit run frontend/app.py` as a subprocess.

    Uses `subprocess.run` rather than Streamlit's internal Python API
    because Streamlit apps are designed to be launched as a CLI
    entry-point, not imported and run programmatically.
    """
    args = parse_args()
    app_path = BASE_DIR / "frontend" / "app.py"
    logger.info("Starting CineMatch Pro frontend on port %d (%s)", args.port, app_path)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)],
        check=True,
    )


if __name__ == "__main__":
    main()
