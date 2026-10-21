"""
Data cleaning, merging, and feature engineering for CineMatch Pro.

Takes the raw DataFrames produced by `data_loader.py` and turns them
into the two processed artifacts consumed by the rest of the pipeline:

1. ``merged_movies.parquet`` — one row per movie, combining MovieLens
   metadata, TMDB metadata/credits, and a text ``combined_features``
   column used by the content-based filter and semantic search.
2. ``cleaned_ratings.parquet`` — a deduplicated, outlier-trimmed ratings
   table ready for collaborative filtering.
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from typing import Any

import pandas as pd

from src.config import (
    CLEANED_RATINGS_PARQUET,
    MERGED_MOVIES_PARQUET,
    PROCESSED_DATA_DIR,
    logger,
)


def _safe_literal_eval(value: Any) -> list[dict[str, Any]]:
    """Safely parse a TMDB JSON-like string column (genres, keywords, cast).

    TMDB's CSV export stores nested JSON as a Python-literal string, e.g.
    ``"[{'id': 28, 'name': 'Action'}]"``. This parses it defensively,
    returning an empty list for null, malformed, or already-parsed input.

    Args:
        value: The raw cell value from a TMDB JSON-like column.

    Returns:
        list[dict[str, Any]]: The parsed list of dicts, or ``[]`` if the
        value could not be parsed.
    """
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []


def extract_names(json_str: Any, key: str = "name", limit: int | None = None) -> list[str]:
    """Extract a flat list of ``name`` values from a TMDB JSON-like column.

    Args:
        json_str: Raw cell from a column such as ``genres`` or
            ``keywords``.
        key: Dict key to extract from each entry (default ``"name"``).
        limit: Optional cap on how many names to return (e.g. top 6 cast).

    Returns:
        list[str]: Extracted names, in original order, truncated to
        ``limit`` if given.
    """
    entries = _safe_literal_eval(json_str)
    names = [entry.get(key, "") for entry in entries if isinstance(entry, dict)]
    names = [n for n in names if n]
    return names[:limit] if limit else names


def extract_director(crew_json: Any) -> str:
    """Pull the director's name out of a TMDB ``crew`` JSON column.

    Args:
        crew_json: Raw ``crew`` cell containing a list of crew dicts,
            each with a ``job`` field.

    Returns:
        str: The director's name, or ``""`` if none is listed.
    """
    crew = _safe_literal_eval(crew_json)
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def extract_music_director(crew_json: Any) -> str:
    """Pull the composer / music-department lead from a TMDB ``crew`` column.

    Args:
        crew_json: Raw ``crew`` cell.

    Returns:
        str: Name of the person credited as "Original Music Composer",
        or ``""`` if not present.
    """
    crew = _safe_literal_eval(crew_json)
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Original Music Composer":
            return member.get("name", "")
    return ""


_YEAR_SUFFIX = re.compile(r"\s*\((\d{4})\)\s*$")
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_ARTICLE_SUFFIX = re.compile(r"^(?P<main>.+),\s+(?P<article>The|A|An)$", re.IGNORECASE)


def clean_movielens_title(title: str) -> tuple[str, int | None]:
    """Split a MovieLens-style title into a clean title and release year.

    MovieLens titles look like ``"Toy Story (1995)"``, but also like
    ``"Matrix, The (1999)"`` (article moved to the end) and
    ``"City of Lost Children, The (Cite des enfants perdus, La) (1995)"``
    (alternate title in parentheses). This extracts the trailing year,
    drops trailing alternate-title parentheticals, and moves a trailing
    ", The" / ", A" / ", An" back to the front so titles can be matched
    against TMDB's natural-order titles.

    Args:
        title: Raw MovieLens title string.

    Returns:
        tuple[str, int | None]: The cleaned title and the parsed year, or
        ``None`` for the year if no ``(YYYY)`` suffix was found.
    """
    title = title.strip()
    year: int | None = None

    year_match = _YEAR_SUFFIX.search(title)
    if year_match:
        year = int(year_match.group(1))
        title = title[: year_match.start()].strip()

    # Drop trailing alternate-title parentheticals, but never reduce the
    # title to an empty string.
    while True:
        stripped = _TRAILING_PAREN.sub("", title).strip()
        if not stripped or stripped == title:
            break
        title = stripped

    article_match = _ARTICLE_SUFFIX.match(title)
    if article_match:
        title = f"{article_match.group('article')} {article_match.group('main')}"

    return title, year


def normalize_title_key(title: Any) -> str:
    """Build a punctuation/accent/case-insensitive key for title matching.

    Lower-cases, strips accents (``"Amelie"`` == ``"Amélie"``), turns ``&``
    into ``and``, and removes everything that is not a letter or digit, so
    ``"Spider-Man: Homecoming"`` and ``"spider man homecoming"`` match.

    Args:
        title: A movie title (non-strings yield an empty key).

    Returns:
        str: The normalized key; ``""`` if nothing alphanumeric remains
        (e.g. an all-CJK title), which callers must treat as "no match".
    """
    if not isinstance(title, str):
        return ""
    folded = unicodedata.normalize("NFKD", title.replace("&", " and "))
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_only.lower())


def build_combined_features(row: pd.Series) -> str:
    """Build the free-text feature string used for TF-IDF / SBERT.

    Concatenates overview, genres, keywords, top cast, and director into
    a single whitespace-joined string, which downstream vectorizers treat
    as a bag of words / sentence embedding input.

    Args:
        row: A row from the merged movies DataFrame, expected to contain
            ``overview``, ``genre_names``, ``keyword_names``,
            ``cast_names``, and ``director`` columns.

    Returns:
        str: The combined feature text for this movie.
    """
    parts: list[str] = []
    if isinstance(row.get("overview"), str):
        parts.append(row["overview"])
    for col in ("genre_names", "keyword_names", "cast_names"):
        val = row.get(col)
        if isinstance(val, list):
            parts.append(" ".join(val))
    if isinstance(row.get("director"), str):
        parts.append(row["director"])
    return " ".join(parts).strip()


def parse_string_list(value: Any) -> list[str]:
    """Safely parse a plain string-list column into a Python list of strings.

    Some pre-merged TMDB exports (e.g. the combined "TMDB Movie Metadata"
    dataset used in this project's `full-metadata` pipeline) store
    genres/keywords/cast as already-flat string lists, like
    ``"['Action', 'Science Fiction']"`` or a Cast_list of plain actor
    names — unlike the original TMDB API's list-of-dicts JSON format
    (``"[{'id': 28, 'name': 'Action'}]"``) that `_safe_literal_eval` +
    `extract_names` handle. This parses the flat form directly.

    Args:
        value: The raw cell value, expected to be a Python-literal string
            list, an already-parsed list, or null.

    Returns:
        list[str]: The parsed list of strings, or ``[]`` if `value` is
        null or cannot be parsed as a list.
    """
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(v) for v in parsed if v is not None]


def _extract_year(df: pd.DataFrame) -> pd.Series:
    """Get a nullable-integer release year for every row of a TMDB frame.

    Prefers the ``release_year`` column and fills gaps from
    ``release_date`` when that column exists.

    Args:
        df: A TMDB-style frame with ``release_year`` and/or ``release_date``.

    Returns:
        pd.Series: ``Int64`` years (``<NA>`` where unknown), aligned to `df`.
    """
    years = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "release_year" in df.columns:
        years = pd.to_numeric(df["release_year"], errors="coerce").round().astype("Int64")
    if "release_date" in df.columns:
        from_date = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        years = years.fillna(from_date.astype("Int64"))
    return years


def restrict_tmdb_to_movielens(
    tmdb_full: pd.DataFrame, movielens_movies: pd.DataFrame
) -> pd.DataFrame:
    """Keep only TMDB rows that can possibly match a MovieLens movie.

    The combined TMDB export has ~1M rows but MovieLens only has ~62k
    movies. Filtering on the normalized ``(title, year)`` key *before* the
    expensive per-row feature engineering cuts the work by more than 90%.

    Args:
        tmdb_full: Raw full TMDB metadata (needs ``title`` plus
            ``release_year`` and/or ``release_date``).
        movielens_movies: Raw MovieLens movies (``title`` column).

    Returns:
        pd.DataFrame: The subset of `tmdb_full` whose normalized title and
        year occur in MovieLens (original index preserved).
    """
    ml_parsed = movielens_movies["title"].astype(str).map(clean_movielens_title)
    ml_keys = {
        (normalize_title_key(title), year)
        for title, year in ml_parsed
        if year is not None and normalize_title_key(title)
    }

    tmdb_keys = tmdb_full["title"].astype(str).map(normalize_title_key).tolist()
    tmdb_years = _extract_year(tmdb_full).tolist()
    mask = [
        bool(key) and year is not pd.NA and (key, year) in ml_keys
        for key, year in zip(tmdb_keys, tmdb_years)
    ]
    restricted = tmdb_full[mask]
    logger.info(
        "Restricted TMDB metadata to MovieLens title/year matches: %d -> %d rows.",
        len(tmdb_full),
        len(restricted),
    )
    return restricted


def prepare_full_tmdb_metadata(tmdb_full: pd.DataFrame) -> pd.DataFrame:
    """Feature-engineer a pre-merged "full TMDB metadata" export.

    Unlike `merge_tmdb_datasets` (which expects the original TMDB API's
    two-file, list-of-dicts JSON export), this handles a single combined
    CSV where genres/keywords/cast are already flat string lists and
    director/writer/composer are already their own plain columns — the
    schema of the "IMDB TMDB Movie Metadata" Kaggle export used here.
    Adult titles (``adult`` == True) are dropped so the demo stays
    presentable.

    Args:
        tmdb_full: Raw DataFrame with (at least) ``id``, ``title``,
            ``overview``, ``genres_list``, ``keywords``, ``Cast_list``,
            ``Director`` columns. ``release_year``/``release_date``,
            ``budget``, ``original_language``, ``adult`` and
            ``overview_sentiment`` are used when present.

    Returns:
        pd.DataFrame: One row per TMDB movie with ``genre_names``,
        ``keyword_names``, ``cast_names``, ``director``,
        ``music_director``, ``clean_title``, ``title_key``, ``year`` and
        ``combined_features`` columns — the same shape
        `merge_tmdb_datasets` produces, so everything downstream
        (content filter, fusion engine) works unchanged. A
        ``sentiment_score`` column is also seeded from
        ``overview_sentiment`` (scaled to 0-100) as a placeholder for
        Row 10's "Sentiment Champions" until real per-review sentiment
        is trained and merged in — see the note in the function body.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"id", "title", "overview", "genres_list", "keywords", "Cast_list", "Director"}
    missing = required - set(tmdb_full.columns)
    if missing:
        raise ValueError(f"tmdb_full is missing required columns: {missing}")

    df = tmdb_full.copy()

    if "adult" in df.columns:
        is_adult = df["adult"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        if is_adult.any():
            logger.info("Dropping %d adult titles.", int(is_adult.sum()))
        df = df[~is_adult].copy()

    df["genre_names"] = df["genres_list"].apply(parse_string_list)
    df["keyword_names"] = df["keywords"].apply(parse_string_list)
    df["cast_names"] = df["Cast_list"].apply(lambda v: parse_string_list(v)[:6])
    df["director"] = df["Director"].fillna("")
    df["music_director"] = df.get(
        "Music_Composer", pd.Series([""] * len(df), index=df.index)
    ).fillna("")
    df["overview"] = df["overview"].fillna("")
    df["clean_title"] = df["title"].astype(str).str.strip()
    df["title_key"] = df["clean_title"].map(normalize_title_key)
    df["year"] = _extract_year(df)

    df["combined_features"] = df.apply(build_combined_features, axis=1)

    # NOTE on sentiment_score: `overview_sentiment` (if present) is a
    # polarity score on the *plot overview text*, not an aggregate of
    # real user reviews — it's a reasonable placeholder so Row 10
    # ("Sentiment Champions") has something to rank by before
    # real review sentiment exists, but it should be overwritten with
    # the real aggregate positive-review percentage (from
    # `SentimentEngine.aggregate_movie_sentiment`) once available.
    if "overview_sentiment" in df.columns:
        polarity = pd.to_numeric(df["overview_sentiment"], errors="coerce").fillna(0.0)
        df["sentiment_score"] = ((polarity.clip(-1, 1) + 1) / 2 * 100).round(1)

    before = len(df)
    df = df[df["combined_features"].str.strip() != ""].reset_index(drop=True)
    logger.info(
        "Prepared full TMDB metadata: %d -> %d movies (dropped %d with empty features).",
        before,
        len(df),
        before - len(df),
    )
    return df


def merge_movielens_with_full_tmdb(
    prepared_tmdb: pd.DataFrame, movielens_movies: pd.DataFrame
) -> pd.DataFrame:
    """Left-join MovieLens's catalogue onto a pre-merged full TMDB export.

    Mirrors `merge_with_movielens`, but joins against the output of
    `prepare_full_tmdb_metadata` on a normalized ``(title_key, year)`` key.
    Guarantees exactly one output row per MovieLens ``movieId``:

    * When several TMDB rows share a title+year (very common in a
      million-row export) only the one with the most votes (falling back
      to popularity) is kept.
    * TMDB rows with an unknown year or empty title key never match
      (pandas would otherwise join NaN keys to each other).

    Args:
        prepared_tmdb: Output of `prepare_full_tmdb_metadata`.
        movielens_movies: Raw MovieLens movies DataFrame.

    Returns:
        pd.DataFrame: One row per MovieLens ``movieId``, with the full
        TMDB metadata merged in where a match exists (NaN/empty
        otherwise, with MovieLens genres used as a `combined_features`
        fallback). ``clean_title`` is the MovieLens-derived display title.
    """
    logger.info(
        "Merging full TMDB metadata onto MovieLens catalogue (%d movies).", len(movielens_movies)
    )

    tmdb = prepared_tmdb.copy()
    if "title_key" not in tmdb.columns:
        tmdb["title_key"] = tmdb["clean_title"].map(normalize_title_key)
    tmdb = tmdb[(tmdb["title_key"] != "") & tmdb["year"].notna()]

    rank_col = next((c for c in ("vote_count", "popularity") if c in tmdb.columns), None)
    if rank_col is not None:
        tmdb = tmdb.assign(_rank=pd.to_numeric(tmdb[rank_col], errors="coerce").fillna(0))
        tmdb = tmdb.sort_values("_rank", ascending=False).drop(columns="_rank")
    tmdb = tmdb.drop_duplicates(subset=["title_key", "year"], keep="first")
    # MovieLens supplies the display title; avoid a clean_title name clash.
    tmdb = tmdb.drop(columns=["clean_title"])

    ml = movielens_movies.copy()
    ml[["clean_title", "year"]] = ml["title"].apply(lambda t: pd.Series(clean_movielens_title(t)))
    ml["year"] = pd.to_numeric(ml["year"], errors="coerce").astype("Int64")
    ml["title_key"] = ml["clean_title"].map(normalize_title_key)
    ml["ml_genres_list"] = ml["genres"].fillna("").apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    merged = ml.merge(tmdb, on=["title_key", "year"], how="left", suffixes=("_ml", "_tmdb"))
    merged = merged.drop_duplicates(subset="movieId", keep="first").reset_index(drop=True)

    missing_features = merged["combined_features"].isna()
    merged.loc[missing_features, "combined_features"] = merged.loc[
        missing_features, "ml_genres_list"
    ].apply(lambda genres: " ".join(genres))

    matched = int(merged["id"].notna().sum()) if "id" in merged.columns else 0
    logger.info(
        "Matched %d / %d MovieLens movies to full TMDB metadata (%.1f%%).",
        matched,
        len(merged),
        100 * matched / max(len(merged), 1),
    )
    return merged


def run_preprocessing_pipeline_full_metadata(
    movielens_ratings: pd.DataFrame,
    movielens_movies: pd.DataFrame,
    tmdb_full: pd.DataFrame,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the clean -> merge -> feature-engineer pipeline for the
    pre-merged "full TMDB metadata" dataset (recommended path).

    Writes to the exact same output paths as `run_preprocessing_pipeline`
    (`MERGED_MOVIES_PARQUET`, `CLEANED_RATINGS_PARQUET`), so every
    downstream module (content filter, fusion engine, semantic search
    ingestion) works unchanged regardless of which raw TMDB source was used.

    Args:
        movielens_ratings: Raw ratings DataFrame.
        movielens_movies: Raw MovieLens movies DataFrame.
        tmdb_full: Raw full TMDB metadata DataFrame (see
            `data_loader.load_tmdb_full_metadata`).
        save: If True, writes the resulting DataFrames to
            ``data/processed/`` as parquet files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(merged_movies, cleaned_ratings)``.
    """
    restricted = restrict_tmdb_to_movielens(tmdb_full, movielens_movies)
    prepared_tmdb = prepare_full_tmdb_metadata(restricted)
    merged_movies = merge_movielens_with_full_tmdb(prepared_tmdb, movielens_movies)
    cleaned_ratings = clean_ratings(movielens_ratings)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        merged_movies.to_parquet(MERGED_MOVIES_PARQUET, index=False)
        cleaned_ratings.to_parquet(CLEANED_RATINGS_PARQUET, index=False)
        logger.info("Saved processed data to %s", PROCESSED_DATA_DIR)

    return merged_movies, cleaned_ratings


def merge_tmdb_datasets(
    tmdb_movies: pd.DataFrame, tmdb_credits: pd.DataFrame
) -> pd.DataFrame:
    """Merge TMDB movie metadata with TMDB credits and engineer text features.

    Args:
        tmdb_movies: Raw TMDB movies DataFrame (from `load_tmdb_movies`).
        tmdb_credits: Raw TMDB credits DataFrame (from `load_tmdb_credits`).

    Returns:
        pd.DataFrame: One row per TMDB movie with parsed genre/keyword/cast
        lists, director, music_director, and a ``combined_features`` text
        column ready for vectorization.
    """
    logger.info(
        "Merging TMDB movies (%d rows) with credits (%d rows).",
        len(tmdb_movies),
        len(tmdb_credits),
    )
    credits_slim = tmdb_credits[["movie_id", "cast", "crew"]].rename(
        columns={"movie_id": "id"}
    )
    merged = tmdb_movies.merge(credits_slim, on="id", how="left")

    merged["genre_names"] = merged["genres"].apply(lambda x: extract_names(x))
    merged["keyword_names"] = merged["keywords"].apply(lambda x: extract_names(x))
    merged["cast_names"] = merged["cast"].apply(lambda x: extract_names(x, limit=6))
    merged["director"] = merged["crew"].apply(extract_director)
    merged["music_director"] = merged["crew"].apply(extract_music_director)

    merged["overview"] = merged["overview"].fillna("")
    merged["combined_features"] = merged.apply(build_combined_features, axis=1)

    # Drop rows with no usable text — they cannot be recommended sensibly.
    before = len(merged)
    merged = merged[merged["combined_features"].str.strip() != ""].reset_index(drop=True)
    logger.info("Dropped %d movies with empty combined features.", before - len(merged))

    return merged


def merge_with_movielens(
    tmdb_merged: pd.DataFrame, movielens_movies: pd.DataFrame
) -> pd.DataFrame:
    """Left-join engineered TMDB data onto the MovieLens movie catalogue.

    MovieLens `movieId` is the canonical ID used by the collaborative
    filter, so this keeps every MovieLens movie and attaches TMDB detail
    where a confident title match exists. Movies without a TMDB match
    keep their MovieLens genres as a fallback for `combined_features`.

    Args:
        tmdb_merged: Output of `merge_tmdb_datasets`.
        movielens_movies: Raw MovieLens movies DataFrame.

    Returns:
        pd.DataFrame: One row per MovieLens ``movieId``, with TMDB columns
        merged in where available (NaN otherwise).
    """
    logger.info("Merging TMDB data onto MovieLens catalogue (%d movies).", len(movielens_movies))

    ml = movielens_movies.copy()
    ml[["clean_title", "year"]] = ml["title"].apply(
        lambda t: pd.Series(clean_movielens_title(t))
    )
    ml["genres_list"] = ml["genres"].fillna("").apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    tmdb = tmdb_merged.copy()
    tmdb["clean_title"] = tmdb["title"].astype(str).str.strip()
    tmdb["year"] = pd.to_datetime(
        tmdb["release_date"], errors="coerce"
    ).dt.year

    merged = ml.merge(
        tmdb,
        on=["clean_title", "year"],
        how="left",
        suffixes=("_ml", "_tmdb"),
    )

    # Fallback: if TMDB combined_features is missing, use MovieLens genres.
    missing_features = merged["combined_features"].isna()
    merged.loc[missing_features, "combined_features"] = merged.loc[
        missing_features, "genres_list"
    ].apply(lambda genres: " ".join(genres))

    matched = (~merged["id"].isna()).sum() if "id" in merged.columns else 0
    logger.info(
        "Matched %d / %d MovieLens movies to TMDB metadata (%.1f%%).",
        matched,
        len(merged),
        100 * matched / max(len(merged), 1),
    )
    return merged


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and validate the raw MovieLens ratings table.

    Args:
        ratings: Raw ratings DataFrame from `load_movielens_ratings`.

    Returns:
        pd.DataFrame: Deduplicated ratings with values constrained to the
        valid MovieLens range [0.5, 5.0].
    """
    before = len(ratings)
    cleaned = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")
    cleaned = cleaned[(cleaned["rating"] >= 0.5) & (cleaned["rating"] <= 5.0)]
    logger.info(
        "Cleaned ratings: %d -> %d rows (%d duplicates/invalid removed).",
        before,
        len(cleaned),
        before - len(cleaned),
    )
    return cleaned.reset_index(drop=True)


def run_preprocessing_pipeline(
    movielens_ratings: pd.DataFrame,
    movielens_movies: pd.DataFrame,
    tmdb_movies: pd.DataFrame,
    tmdb_credits: pd.DataFrame,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full clean -> merge -> feature-engineer pipeline end to end.

    Args:
        movielens_ratings: Raw ratings DataFrame.
        movielens_movies: Raw MovieLens movies DataFrame.
        tmdb_movies: Raw TMDB movies DataFrame.
        tmdb_credits: Raw TMDB credits DataFrame.
        save: If True, writes the resulting DataFrames to
            ``data/processed/`` as parquet files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(merged_movies, cleaned_ratings)``.
    """
    tmdb_merged = merge_tmdb_datasets(tmdb_movies, tmdb_credits)
    merged_movies = merge_with_movielens(tmdb_merged, movielens_movies)
    cleaned_ratings = clean_ratings(movielens_ratings)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        merged_movies.to_parquet(MERGED_MOVIES_PARQUET, index=False)
        cleaned_ratings.to_parquet(CLEANED_RATINGS_PARQUET, index=False)
        logger.info("Saved processed data to %s", PROCESSED_DATA_DIR)

    return merged_movies, cleaned_ratings


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.data_loader import (
        DatasetNotFoundError,
        load_movielens_movies,
        load_movielens_ratings,
        load_tmdb_credits,
        load_tmdb_full_metadata,
        load_tmdb_movies,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run CineMatch Pro's preprocessing pipeline. Two modes: "
            "--tmdb-full-path (recommended) for a single pre-merged TMDB "
            "export, or --tmdb-movies-path + --tmdb-credits-path for the "
            "original two-file TMDB 5000 API export."
        )
    )
    parser.add_argument(
        "--tmdb-full-path",
        type=str,
        default=None,
        help=(
            "Path to a single pre-merged TMDB metadata CSV (columns: id, title, "
            "overview, genres_list, keywords, Cast_list, Director, ...). "
            "When given, this takes priority over --tmdb-movies-path/--tmdb-credits-path."
        ),
    )
    parser.add_argument("--tmdb-movies-path", type=str, default=None, help="Legacy: tmdb_5000_movies.csv path.")
    parser.add_argument("--tmdb-credits-path", type=str, default=None, help="Legacy: tmdb_5000_credits.csv path.")
    parser.add_argument("--ratings-path", type=str, default=None, help="MovieLens ratings.csv path.")
    parser.add_argument("--movies-path", type=str, default=None, help="MovieLens movies.csv path.")
    parser.add_argument("--nrows", type=int, default=500_000, help="Cap on ratings rows loaded.")
    args = parser.parse_args()

    try:
        ratings_df = (
            load_movielens_ratings(path=Path(args.ratings_path), nrows=args.nrows)
            if args.ratings_path
            else load_movielens_ratings(nrows=args.nrows)
        )
        movies_df = (
            load_movielens_movies(path=Path(args.movies_path))
            if args.movies_path
            else load_movielens_movies()
        )

        if args.tmdb_full_path:
            tmdb_full_df = load_tmdb_full_metadata(path=Path(args.tmdb_full_path))
            run_preprocessing_pipeline_full_metadata(ratings_df, movies_df, tmdb_full_df)
        elif args.tmdb_movies_path and args.tmdb_credits_path:
            tmdb_movies_df = load_tmdb_movies(path=Path(args.tmdb_movies_path))
            tmdb_credits_df = load_tmdb_credits(path=Path(args.tmdb_credits_path))
            run_preprocessing_pipeline(ratings_df, movies_df, tmdb_movies_df, tmdb_credits_df)
        else:
            logger.info(
                "No TMDB path given — falling back to default config.py paths "
                "(legacy tmdb_5000_movies.csv + tmdb_5000_credits.csv)."
            )
            tmdb_movies_df = load_tmdb_movies()
            tmdb_credits_df = load_tmdb_credits()
            run_preprocessing_pipeline(ratings_df, movies_df, tmdb_movies_df, tmdb_credits_df)
    except DatasetNotFoundError as exc:
        logger.error("Preprocessing failed: %s", exc)
        raise SystemExit(1) from exc
