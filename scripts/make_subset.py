"""
CLI entry point: shrink the downloaded datasets into a small, Kaggle-ready folder.

Reads your full downloaded datasets and writes a smaller copy with the SAME file
names and the SAME columns, so the Kaggle notebook works on it without changes.

What is reduced (and why it is safe):
  * ml-25m/ratings.csv      -> only the ratings of users with userId <= --users
                               (userIds are randomly assigned, so this is an unbiased user sample).
  * ml-25m/movies.csv       -> copied in full (3 MB; the recommendation catalogue needs every movie).
  * ml-25m/links.csv        -> copied in full (1 MB; maps MovieLens ids to IMDB/TMDB ids).
  * IMDB TMDB ... (1M).csv  -> only the rows that match a MovieLens movie (same matching the pipeline
                               uses) and only the columns the project reads. The pipeline throws the
                               other ~90% away anyway, so nothing it uses is lost.
  * IMDB Dataset.csv        -> all 50k reviews by default; --reviews N takes a balanced sample.
  * MovieGenre.csv          -> all rows by default; --posters N keeps the first N. Re-saved as UTF-8.
Not used by the project and therefore not copied: genome-*.csv, tags.csv, SampleMoviePosters/.

Usage (Windows example):
    python -m scripts.make_subset ^
        --source-dir "D:\\Data_Science_Capstone_Datasets\\cinematch-pro-complete" ^
        --out-dir    "D:\\Data_Science_Capstone_Datasets\\kaggle_subset" --users 20000
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.config import logger
from src.data_loader import _FULL_METADATA_COLUMNS, _read_table
from src.preprocessing import restrict_tmdb_to_movielens

_RATINGS_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}


class SubsetError(RuntimeError):
    """Raised when a source file is missing or cannot be subset."""


def find_file(source_dir: Path, predicate, what: str) -> Path:
    """Locate a dataset file anywhere under `source_dir` (case-insensitive name tests).

    Args:
        source_dir: Folder holding the downloaded datasets.
        predicate: Callable taking a lower-cased file name, returning True for a match.
        what: Human-readable description used in the error message.

    Returns:
        Path: The first matching file (shortest path first, so top-level files win).

    Raises:
        SubsetError: If nothing matches.
    """
    hits = sorted(
        (p for p in source_dir.rglob("*") if p.is_file() and predicate(p.name.lower())),
        key=lambda p: (len(p.parts), str(p)),
    )
    if not hits:
        raise SubsetError(f"Could not find {what} under '{source_dir}'.")
    return hits[0]


def subset_ratings(src: Path, dst: Path, max_user_id: int, chunksize: int = 2_000_000) -> int:
    """Stream the ratings file and keep only users with ``userId <= max_user_id``.

    Args:
        src: Full ``ratings.csv``.
        dst: Output path (parent directories are created).
        max_user_id: Highest userId to keep.
        chunksize: Rows per chunk, so the 25M-row file never sits in memory at once.

    Returns:
        int: Number of ratings written.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept, first = 0, True
    for chunk in pd.read_csv(src, chunksize=chunksize, dtype=_RATINGS_DTYPES):
        part = chunk[chunk["userId"] <= max_user_id]
        part.to_csv(dst, mode="w" if first else "a", header=first, index=False)
        first = False
        kept += len(part)
    if first:  # empty source: still produce a valid header-only file
        pd.read_csv(src, nrows=0).to_csv(dst, index=False)
    return kept


def subset_tmdb(src: Path, dst: Path, movies: pd.DataFrame, chunksize: int = 100_000) -> tuple[int, int]:
    """Keep only TMDB rows that can match a MovieLens movie, and only the used columns.

    Args:
        src: The combined TMDB/IMDB export (CSV).
        dst: Output path.
        movies: MovieLens movies table (``title`` column) to match against.
        chunksize: Rows per chunk (the full file is close to 1 GB).

    Returns:
        tuple[int, int]: ``(rows_read, rows_kept)``.

    Raises:
        SubsetError: If `src` is not a CSV file.
    """
    if src.suffix.lower() != ".csv":
        raise SubsetError(f"'{src.name}' is not a .csv; save it as CSV first (chunked reading needs CSV).")
    # encoding_errors="replace": a stray bad byte in a title must not abort a ~1 GB read.
    reader = pd.read_csv(
        src,
        usecols=lambda column: column in _FULL_METADATA_COLUMNS,
        chunksize=chunksize,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    read_rows, kept_parts = 0, []
    for chunk in reader:
        read_rows += len(chunk)
        kept_parts.append(restrict_tmdb_to_movielens(chunk, movies))
    kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else pd.DataFrame()
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(dst, index=False)
    return read_rows, len(kept)


def subset_reviews(src: Path, dst: Path, limit: int | None) -> int:
    """Copy the IMDB reviews, optionally as a class-balanced sample.

    Args:
        src: ``IMDB Dataset.csv`` (needs ``review`` and ``sentiment`` columns).
        dst: Output path.
        limit: Total reviews to keep (half positive, half negative); ``None`` keeps all.

    Returns:
        int: Number of reviews written.
    """
    reviews = _read_table(src)
    if limit and limit < len(reviews):
        per_class = max(1, limit // 2)
        # Iterating the groups (rather than groupby().apply) behaves the same in every
        # pandas version; newer ones drop the grouping column inside apply().
        parts = [
            group.sample(n=min(len(group), per_class), random_state=42)
            for _, group in reviews.groupby("sentiment")
        ]
        reviews = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(dst, index=False)
    return len(reviews)


def subset_posters(src: Path, dst: Path, limit: int | None) -> int:
    """Re-save the poster metadata as UTF-8, optionally keeping only the first rows.

    Args:
        src: ``MovieGenre.csv`` (read with an automatic Latin-1 fallback).
        dst: Output path.
        limit: Rows to keep; ``None`` keeps all.

    Returns:
        int: Number of rows written.
    """
    posters = _read_table(src)
    if limit:
        posters = posters.head(limit)
    dst.parent.mkdir(parents=True, exist_ok=True)
    posters.to_csv(dst, index=False)
    return len(posters)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Shrink the downloaded datasets for a Kaggle run.")
    parser.add_argument("--source-dir", required=True, help="Folder with your downloaded datasets.")
    parser.add_argument("--out-dir", required=True, help="Folder to write the smaller copy into.")
    parser.add_argument("--users", type=int, default=20000,
                        help="Keep ratings of users with userId <= this (MovieLens 25M has 162,541 users; "
                             "20000 is roughly 3M ratings). Default: 20000.")
    parser.add_argument("--reviews", type=int, default=None,
                        help="Balanced sample of IMDB reviews to keep. Default: all 50,000.")
    parser.add_argument("--posters", type=int, default=None,
                        help="Keep only the first N poster rows. Default: all.")
    return parser.parse_args()


def main() -> int:
    """Create the subset folder.

    Returns:
        int: 0 on success, 1 on failure.
    """
    args = parse_args()
    source, out = Path(args.source_dir), Path(args.out_dir)
    if not source.is_dir():
        logger.error("Source folder not found: %s", source)
        return 1
    if out.resolve() == source.resolve():
        logger.error("--out-dir must be a different folder from --source-dir.")
        return 1

    try:
        ratings = find_file(source, lambda n: n == "ratings.csv", "MovieLens ratings.csv")
        movies_csv = find_file(source, lambda n: n == "movies.csv", "MovieLens movies.csv")
        links_csv = find_file(source, lambda n: n == "links.csv", "MovieLens links.csv")
        tmdb = find_file(source, lambda n: "big dataset" in n and n.endswith(".csv"), "the 'Big Dataset (1M)' csv")
        imdb = find_file(source, lambda n: "imdb dataset" in n and n.endswith(".csv") and "tmdb" not in n,
                         "'IMDB Dataset.csv'")
        genre = find_file(source, lambda n: n == "moviegenre.csv", "'MovieGenre.csv'")
    except SubsetError as exc:
        logger.error("%s", exc)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    summary: list[str] = []

    (out / "ml-25m").mkdir(parents=True, exist_ok=True)
    shutil.copy2(movies_csv, out / "ml-25m" / "movies.csv")
    shutil.copy2(links_csv, out / "ml-25m" / "links.csv")
    movies = pd.read_csv(movies_csv)
    summary.append(f"movies.csv / links.csv: copied in full ({len(movies):,} movies)")

    logger.info("Subsetting ratings (users <= %d) ...", args.users)
    n_ratings = subset_ratings(ratings, out / "ml-25m" / "ratings.csv", args.users)
    summary.append(f"ratings.csv: {n_ratings:,} ratings from users 1..{args.users}")

    logger.info("Subsetting the big TMDB file (this reads ~1 GB in chunks) ...")
    read_rows, kept_rows = subset_tmdb(tmdb, out / tmdb.name, movies)
    summary.append(f"{tmdb.name}: {kept_rows:,} of {read_rows:,} rows kept (the MovieLens-matching ones)")

    n_reviews = subset_reviews(imdb, out / imdb.name, args.reviews)
    summary.append(f"{imdb.name}: {n_reviews:,} reviews")

    n_posters = subset_posters(genre, out / genre.name, args.posters)
    summary.append(f"{genre.name}: {n_posters:,} rows (saved as UTF-8)")

    (out / "SUBSET_INFO.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\nDone. Created:", out)
    for line in summary:
        print("  -", line)
    sizes = sorted(((p.stat().st_size / 1e6, p.relative_to(out)) for p in out.rglob("*") if p.is_file()), reverse=True)
    print("\nFile sizes (MB):")
    for size, rel in sizes:
        print(f"  {size:9,.1f}  {rel}")
    print("\nNext: upload this folder to Kaggle (Datasets -> New Dataset) and add it to the notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
