"""
CLI entry point: build the CLIP + FAISS poster visual-search index.

Pipeline:
    MovieGenre.csv (poster URLs, keyed by IMDB id)
      + MovieLens links.csv (imdbId -> movieId)
      + merged_movies.parquet (optional: TMDB poster_path fallback URLs)
    -> download posters to data/posters/<movieId>.jpg   (resumable)
    -> CLIP embeddings -> FAISS index keyed by real movieId

Usage:
    python -m scripts.build_faiss_index \\
        --genre-path ".../MovieGenre.csv" --links-path ".../ml-25m/links.csv"
    python -m scripts.build_faiss_index --limit 500            # quick smoke test
    python -m scripts.build_faiss_index --no-download          # reuse cached posters only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    FAISS_INDEX_PATH,
    MERGED_MOVIES_PARQUET,
    MOVIELENS_LINKS_CSV,
    POSTERS_DIR,
    POSTERS_METADATA_CSV,
    logger,
)
from src.data_loader import DatasetNotFoundError, load_movielens_links, load_poster_metadata
from src.model_card import update_model_card
from src.poster_utils import (
    PosterCatalogError,
    build_poster_catalog,
    collect_cached_posters,
    download_posters,
)
from src.visual_search import VisualSearchEngine, VisualSearchError

_MIN_POSTERS = 10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the FAISS index build script.

    Returns:
        argparse.Namespace: Parsed paths and tuning options.
    """
    parser = argparse.ArgumentParser(description="Build CineMatch Pro's poster FAISS index.")
    parser.add_argument("--genre-path", type=str, default=str(POSTERS_METADATA_CSV),
                        help="MovieGenre.csv (poster URLs keyed by imdbId).")
    parser.add_argument("--links-path", type=str, default=str(MOVIELENS_LINKS_CSV),
                        help="MovieLens links.csv (imdbId -> movieId).")
    parser.add_argument("--movies-parquet", type=str, default=str(MERGED_MOVIES_PARQUET),
                        help="merged_movies.parquet; its TMDB poster_path is used as a fallback URL "
                             "if the file exists.")
    parser.add_argument("--posters-dir", type=str, default=str(POSTERS_DIR),
                        help="Where to cache downloaded posters as <movieId>.jpg.")
    parser.add_argument("--imdb-col", type=str, default="imdbId", help="IMDB id column in the poster CSV.")
    parser.add_argument("--url-col", type=str, default="Poster", help="Poster URL column in the poster CSV.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: cap number of posters, for a quick smoke-test run.")
    parser.add_argument("--workers", type=int, default=32, help="Concurrent download threads.")
    parser.add_argument("--batch-size", type=int, default=64, help="CLIP embedding batch size.")
    parser.add_argument("--no-download", action="store_true",
                        help="Do not touch the network; index only posters already in --posters-dir.")
    return parser.parse_args()


def main() -> int:
    """Run the FAISS poster index build pipeline.

    Returns:
        int: Process exit code (0 on success, 1 on failure).
    """
    args = parse_args()
    posters_dir = Path(args.posters_dir)

    try:
        genre_df = load_poster_metadata(path=Path(args.genre_path))
        links_df = load_movielens_links(path=Path(args.links_path))
    except DatasetNotFoundError as exc:
        logger.error("Cannot build FAISS index: %s", exc)
        return 1

    movies_df = None
    parquet_path = Path(args.movies_parquet)
    if parquet_path.exists():
        movies_df = pd.read_parquet(parquet_path, columns=["movieId", "poster_path"])
    else:
        logger.warning("%s not found — building the catalogue from MovieGenre.csv URLs only.", parquet_path)

    try:
        catalog = build_poster_catalog(
            genre_df, links_df, movies_df, imdb_col=args.imdb_col, url_col=args.url_col
        )
    except PosterCatalogError as exc:
        logger.error("Cannot build poster catalogue: %s", exc)
        return 1

    if args.limit:
        catalog = catalog.head(args.limit)

    if args.no_download:
        available = collect_cached_posters(catalog, posters_dir)
    else:
        available = download_posters(catalog, posters_dir, workers=args.workers)

    if len(available) < _MIN_POSTERS:
        logger.error(
            "Only %d posters available (need at least %d). Check network access / --posters-dir.",
            len(available),
            _MIN_POSTERS,
        )
        return 1

    movie_ids = [movie_id for movie_id, _ in available]
    poster_paths = [path for _, path in available]

    engine = VisualSearchEngine()
    try:
        engine.build_index(poster_paths, movie_ids, batch_size=args.batch_size)
    except VisualSearchError as exc:
        logger.error("FAISS index build failed: %s", exc)
        return 1

    engine.save()
    update_model_card(
        "visual_search",
        {
            "clip_model": engine.model.name_or_path,
            "posters_requested": len(catalog),
            "posters_downloaded": len(available),
            "vectors_indexed": int(engine.index.ntotal),
            "embedding_dim": int(engine.index.d),
        },
    )
    logger.info("FAISS index with %d posters saved to %s", engine.index.ntotal, FAISS_INDEX_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
