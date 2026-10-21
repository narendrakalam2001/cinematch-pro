"""
CLI entry point: fit and save the TF-IDF content-based filter.

This is cheap (about a minute on CPU), so it is fine to re-run it on the
machine that serves the app — doing so avoids any scikit-learn version
mismatch between a training machine (e.g. Kaggle) and the serving machine.

Usage:
    python -m scripts.build_content_model
    python -m scripts.build_content_model --movies-parquet data/processed/merged_movies.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import MERGED_MOVIES_PARQUET, TFIDF_MATRIX_PATH, logger
from src.content_filter import ContentBasedFilter, ContentFilterError
from src.model_card import update_model_card


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the content-filter build script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build CineMatch Pro's TF-IDF content filter.")
    parser.add_argument(
        "--movies-parquet",
        type=str,
        default=str(MERGED_MOVIES_PARQUET),
        help="merged_movies.parquet produced by src.preprocessing.",
    )
    return parser.parse_args()


def main() -> int:
    """Fit the TF-IDF vectorizer on every movie's combined features and save it.

    Returns:
        int: Process exit code (0 on success, 1 on failure).
    """
    args = parse_args()
    path = Path(args.movies_parquet)
    if not path.exists():
        logger.error("%s not found. Run `python -m src.preprocessing ...` first.", path)
        return 1

    movies = pd.read_parquet(path, columns=["movieId", "combined_features"])
    try:
        content_filter = ContentBasedFilter().fit(movies)
    except ContentFilterError as exc:
        logger.error("Content filter build failed: %s", exc)
        return 1

    content_filter.save()
    update_model_card(
        "content_filter",
        {
            "movies": int(content_filter.tfidf_matrix.shape[0]),
            "vocabulary_size": int(content_filter.tfidf_matrix.shape[1]),
        },
    )
    logger.info("Content filter saved to %s", TFIDF_MATRIX_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
