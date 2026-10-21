"""
CLI entry point: embed and ingest movie descriptions into ChromaDB for
semantic search.

Usage:
    python -m scripts.ingest_descriptions
    python -m scripts.ingest_descriptions --reset
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import MERGED_MOVIES_PARQUET, logger, settings
from src.model_card import update_model_card
from src.semantic_search import SemanticSearchEngine, SemanticSearchError


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ChromaDB ingestion script.

    Returns:
        argparse.Namespace: Parsed arguments, including whether to wipe
        the collection before re-ingesting.
    """
    parser = argparse.ArgumentParser(description="Ingest movie descriptions into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the ChromaDB collection before ingesting.",
    )
    parser.add_argument(
        "--min-overview-chars",
        type=int,
        default=40,
        help="Skip movies whose plot overview is shorter than this (genre-only "
        "documents all embed identically and pollute semantic search). 0 = keep all.",
    )
    return parser.parse_args()


def build_semantic_corpus(movies: pd.DataFrame, min_overview_chars: int = 40) -> pd.DataFrame:
    """Select and prepare the movies worth embedding for semantic search.

    Movies without a real plot overview end up with genre-only text such as
    "Mystery Thriller"; thousands of those share one identical embedding and
    outrank genuine matches. They are dropped, and the title is prepended to
    the remaining documents so title-like queries also match.

    Args:
        movies: Merged movies frame with ``overview``, ``combined_features``
            and (optionally) ``clean_title`` columns.
        min_overview_chars: Minimum overview length to keep a movie (0 keeps all).

    Returns:
        pd.DataFrame: Filtered copy with an updated ``combined_features`` column.
    """
    out = movies.copy()
    if min_overview_chars > 0 and "overview" in out.columns:
        length = out["overview"].fillna("").astype(str).str.strip().str.len()
        out = out[length >= min_overview_chars].copy()
    if "clean_title" in out.columns:
        out["combined_features"] = (
            out["clean_title"].fillna("").astype(str) + ". " + out["combined_features"].fillna("").astype(str)
        )
    return out


def main() -> int:
    """Run the ChromaDB ingestion pipeline.

    Returns:
        int: Process exit code (0 on success, 1 on failure).
    """
    args = parse_args()

    if not MERGED_MOVIES_PARQUET.exists():
        logger.error(
            "Merged movies file not found at %s. Run src/preprocessing.py "
            "(or scripts from Day 1) first.",
            MERGED_MOVIES_PARQUET,
        )
        return 1

    movies_df = pd.read_parquet(MERGED_MOVIES_PARQUET)
    total_movies = len(movies_df)
    movies_df = build_semantic_corpus(movies_df, args.min_overview_chars)
    logger.info(
        "Semantic corpus: %d of %d movies kept (overview >= %d chars).",
        len(movies_df), total_movies, args.min_overview_chars,
    )
    engine = SemanticSearchEngine()

    if args.reset:
        engine.reset()

    try:
        count = engine.ingest_movies(movies_df)
    except SemanticSearchError as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    update_model_card(
        "semantic_search",
        {
            "embedding_model": settings.semantic_search.sbert_model_name,
            "movies_ingested": int(count),
            "collection_size": int(engine.count()),
        },
    )
    logger.info("Ingested %d movies. ChromaDB collection now has %d documents.", count, engine.count())
    return 0


if __name__ == "__main__":
    sys.exit(main())