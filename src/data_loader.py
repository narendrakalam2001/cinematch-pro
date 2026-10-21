"""
Dataset loading utilities for CineMatch Pro.

Responsible for reading the raw MovieLens 25M, TMDB 5000, poster-genre,
and IMDB 50k review datasets off disk into pandas DataFrames, with
defensive checks (missing file errors, dtype coercion, basic schema
validation) so downstream modules can assume clean, typed input.

Expected raw file layout (see README "Dataset sources" section for
download links)::

    data/raw/
        ml-25m/
            ratings.csv
            movies.csv
            tags.csv
        tmdb_5000_movies.csv
        tmdb_5000_credits.csv
        imdb_reviews_50k.csv
        poster_genre_metadata.csv
    data/posters/
        <various poster .jpg files>
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    IMDB_REVIEWS_CSV,
    MOVIELENS_LINKS_CSV,
    MOVIELENS_MOVIES_CSV,
    MOVIELENS_RATINGS_CSV,
    POSTERS_METADATA_CSV,
    TMDB_CREDITS_CSV,
    TMDB_FULL_METADATA_CSV,
    TMDB_METADATA_CSV,
    logger,
)


# Columns of the combined TMDB/IMDB export that CineMatch Pro actually uses.
_FULL_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "id", "title", "overview", "release_date", "release_year", "adult",
        "vote_average", "vote_count", "popularity", "runtime", "budget", "revenue",
        "original_language", "imdb_id", "poster_path", "backdrop_path",
        "genres_list", "keywords", "Cast_list", "Director", "Writer",
        "Music_Composer", "Director_of_Photography", "IMDB_Rating", "Meta_score",
        "Certificate", "overview_sentiment",
    }
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a required raw dataset file is missing from disk."""


class UnsupportedFileFormatError(ValueError):
    """Raised when a dataset file's extension isn't one we know how to read."""


def _require_file(path: Path, hint: str) -> Path:
    """Ensure a required dataset file exists before attempting to read it.

    Args:
        path: Expected location of the dataset file.
        hint: Short human-readable instruction on how to obtain the file,
            surfaced in the raised error to speed up debugging.

    Returns:
        Path: The same path, unchanged, if it exists.

    Raises:
        DatasetNotFoundError: If the file does not exist on disk.
    """
    if not path.exists():
        raise DatasetNotFoundError(
            f"Expected dataset file not found at '{path}'. {hint}"
        )
    return path


def _read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a tabular data file, auto-dispatching on its extension.

    Supports ``.csv``/``.tsv`` (via `pandas.read_csv`) and ``.xls``/``.xlsx``
    (via `pandas.read_excel`, which needs the `xlrd` package for legacy
    ``.xls`` and `openpyxl` for ``.xlsx`` — both are in requirements.txt).
    This lets every loader below accept whichever format a dataset was
    actually downloaded in (Kaggle exports are sometimes CSV, sometimes
    Excel) without the caller having to know which reader to call.

    CSV/TSV reads default to UTF-8 and automatically retry with Latin-1
    (``cp1252``-compatible) on a `UnicodeDecodeError` — several popular
    Kaggle movie datasets (e.g. the "Movie Genre from its Poster" poster
    metadata CSV) ship with non-UTF-8 characters in title fields, and
    failing loudly there would block the whole pipeline over a handful of
    accented characters.

    Args:
        path: Path to the data file. Its suffix determines the reader used.
        **kwargs: Forwarded to `pandas.read_csv` / `pandas.read_excel`
            (e.g. ``dtype=``, ``nrows=``). Excel reads silently ignore a
            ``sep`` kwarg if one is accidentally passed.

    Returns:
        pd.DataFrame: The loaded table.

    Raises:
        UnsupportedFileFormatError: If the file extension isn't one of
            ``.csv``, ``.tsv``, ``.xls``, ``.xlsx``.
    """
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        try:
            return pd.read_csv(path, sep=sep, encoding="utf-8", **kwargs)
        except UnicodeDecodeError:
            logger.warning(
                "%s is not valid UTF-8 — retrying with Latin-1 encoding "
                "(common for older Kaggle exports with accented characters).",
                path.name,
            )
            return pd.read_csv(path, sep=sep, encoding="latin-1", **kwargs)
    if suffix in (".xls", ".xlsx"):
        kwargs.pop("sep", None)
        return pd.read_excel(path, **kwargs)
    raise UnsupportedFileFormatError(
        f"Don't know how to read '{path.name}' — unsupported extension '{suffix}'. "
        f"Supported: .csv, .tsv, .xls, .xlsx."
    )


def load_movielens_ratings(
    path: Path = MOVIELENS_RATINGS_CSV, nrows: int | None = None
) -> pd.DataFrame:
    """Load the MovieLens 25M ratings table.

    Args:
        path: Location of ``ratings.csv``. Defaults to the standard
            project location under ``data/raw/ml-25m/``.
        nrows: Optional cap on the number of rows to read, useful for
            fast local iteration before running on the full 25M rows.

    Returns:
        pd.DataFrame: Columns ``userId`` (int), ``movieId`` (int),
        ``rating`` (float, 0.5-5.0), ``timestamp`` (int, unix seconds).

    Raises:
        DatasetNotFoundError: If the file is missing.
    """
    _require_file(
        path,
        hint=(
            "Download the MovieLens 25M dataset from "
            "https://grouplens.org/datasets/movielens/25m/ and extract "
            "'ratings.csv' into data/raw/ml-25m/."
        ),
    )
    logger.info("Loading MovieLens ratings from %s (nrows=%s)", path, nrows)
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
    df = _read_table(path, dtype=dtypes, nrows=nrows)
    if df.empty:
        logger.warning("MovieLens ratings file loaded but contains 0 rows.")
    logger.info("Loaded %d ratings from %d unique users.", len(df), df["userId"].nunique())
    return df


def load_movielens_movies(path: Path = MOVIELENS_MOVIES_CSV) -> pd.DataFrame:
    """Load the MovieLens movies table (movieId, title, genres).

    Args:
        path: Location of ``movies.csv``.

    Returns:
        pd.DataFrame: Columns ``movieId`` (int), ``title`` (str),
        ``genres`` (str, pipe-separated, e.g. ``"Action|Adventure"``).

    Raises:
        DatasetNotFoundError: If the file is missing.
    """
    _require_file(
        path,
        hint="Extract 'movies.csv' from the MovieLens 25M zip into data/raw/ml-25m/.",
    )
    logger.info("Loading MovieLens movies metadata from %s", path)
    df = _read_table(path, dtype={"movieId": "int32", "title": "string", "genres": "string"})
    logger.info("Loaded metadata for %d movies.", len(df))
    return df


def load_movielens_links(path: Path = MOVIELENS_LINKS_CSV) -> pd.DataFrame:
    """Load the MovieLens ``links.csv`` (movieId <-> IMDB / TMDB ids).

    This is the bridge that lets poster metadata keyed by IMDB id (such as
    the "Movie Genre from its Poster" CSV) be attached to MovieLens
    ``movieId`` values.

    Args:
        path: Location of ``links.csv``.

    Returns:
        pd.DataFrame: Columns ``movieId`` (int), ``imdbId`` (int, no ``tt``
        prefix and no leading zeros) and ``tmdbId`` (float, may be NaN).

    Raises:
        DatasetNotFoundError: If the file is missing.
    """
    _require_file(
        path,
        hint="Extract 'links.csv' from the MovieLens 25M zip into data/raw/ml-25m/.",
    )
    logger.info("Loading MovieLens links from %s", path)
    df = _read_table(path)
    logger.info("Loaded %d movie id links.", len(df))
    return df


def load_tmdb_movies(path: Path = TMDB_METADATA_CSV) -> pd.DataFrame:
    """Load the TMDB 5000 movie metadata dataset.

    Args:
        path: Location of ``tmdb_5000_movies.csv``.

    Returns:
        pd.DataFrame: Raw TMDB movie metadata including JSON-encoded
        string columns (``genres``, ``keywords``, ``production_companies``,
        etc.) that should be parsed downstream in ``preprocessing.py``.

    Raises:
        DatasetNotFoundError: If the file is missing.
    """
    _require_file(
        path,
        hint=(
            "Download from "
            "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata "
            "and place 'tmdb_5000_movies.csv' in data/raw/."
        ),
    )
    logger.info("Loading TMDB movie metadata from %s", path)
    df = _read_table(path)
    logger.info("Loaded TMDB metadata for %d movies.", len(df))
    return df


def load_tmdb_credits(path: Path = TMDB_CREDITS_CSV) -> pd.DataFrame:
    """Load the TMDB 5000 credits (cast & crew) dataset.

    Args:
        path: Location of ``tmdb_5000_credits.csv``.

    Returns:
        pd.DataFrame: Columns ``movie_id``, ``title``, ``cast`` (JSON str),
        ``crew`` (JSON str).

    Raises:
        DatasetNotFoundError: If the file is missing.
    """
    _require_file(
        path,
        hint=(
            "Download from "
            "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata "
            "and place 'tmdb_5000_credits.csv' in data/raw/."
        ),
    )
    logger.info("Loading TMDB credits from %s", path)
    df = _read_table(path)
    logger.info("Loaded credits for %d movies.", len(df))
    return df


def load_tmdb_full_metadata(path: Path = TMDB_FULL_METADATA_CSV) -> pd.DataFrame:
    """Load a single pre-merged "full TMDB metadata" export (recommended).

    Unlike `load_tmdb_movies` + `load_tmdb_credits` (the original TMDB
    API's two-file, list-of-dicts JSON export), this covers combined
    Kaggle exports that already merge movies + credits into one CSV with
    flat string-list columns (``genres_list``, ``keywords``, ``Cast_list``)
    and separate ``Director``/``Music_Composer``/``Writer`` columns —
    richer and easier to work with for a portfolio-grade project, since
    it also typically includes ``IMDB_Rating``, ``Meta_score``, and a
    precomputed ``overview_sentiment``.

    Pair this with `preprocessing.prepare_full_tmdb_metadata` and
    `preprocessing.merge_movielens_with_full_tmdb` rather than
    `merge_tmdb_datasets` / `merge_with_movielens`.

    Args:
        path: Location of the full metadata CSV. Pass an explicit path
            (e.g. via a script's ``--tmdb-full-path`` argument) since
            filenames vary a lot between Kaggle re-uploads of this kind
            of combined dataset.

    Returns:
        pd.DataFrame: Raw full TMDB metadata, one row per movie.

    Raises:
        DatasetNotFoundError: If the file is missing.
        ValueError: If core expected columns are missing, which usually
            means this isn't actually a compatible "full metadata" export.
    """
    _require_file(
        path,
        hint=(
            "Pass the correct location with --tmdb-full-path, e.g. "
            r'"D:\Data_Science_Capstone_Datasets\cinematch-pro-complete\IMDB TMDB Movie Metadata Big Dataset (1M).csv".'
        ),
    )
    logger.info("Loading full TMDB metadata from %s", path)
    # The export has ~42 columns and can be close to 1 GB; reading only the
    # columns the pipeline uses cuts memory and load time substantially.
    # A callable `usecols` silently ignores wanted columns that are absent.
    df = _read_table(path, usecols=lambda col: col in _FULL_METADATA_COLUMNS)
    required = {"id", "title", "overview", "genres_list", "keywords", "Cast_list", "Director"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"'{path.name}' is missing expected full-metadata columns: {missing}. "
            "This loader expects the combined TMDB+credits+ratings schema "
            "(genres_list, keywords, Cast_list, Director, ...), not the "
            "original two-file TMDB 5000 API export — use load_tmdb_movies() "
            "+ load_tmdb_credits() for that instead."
        )
    logger.info("Loaded full TMDB metadata for %d movies.", len(df))
    return df


def load_imdb_reviews(path: Path = IMDB_REVIEWS_CSV) -> pd.DataFrame:
    """Load the IMDB 50k review sentiment dataset.

    Args:
        path: Location of ``imdb_reviews_50k.csv``.

    Returns:
        pd.DataFrame: Columns ``review`` (str) and ``sentiment``
        (str, one of ``"positive"`` / ``"negative"``).

    Raises:
        DatasetNotFoundError: If the file is missing.
    """
    _require_file(
        path,
        hint=(
            "Download from https://www.kaggle.com/datasets/lakshmi25npathi/"
            "imdb-dataset-of-50k-movie-reviews and place the CSV at "
            "data/raw/imdb_reviews_50k.csv (rename the Kaggle file if needed)."
        ),
    )
    logger.info("Loading IMDB reviews dataset from %s", path)
    df = _read_table(path)
    expected_cols = {"review", "sentiment"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"IMDB reviews file is missing expected columns: {missing}")
    # The raw reviews contain literal "<br />" HTML line breaks; they are noise
    # for the tokenizer, so replace them with spaces.
    df["review"] = df["review"].astype(str).str.replace(r"<br\s*/?>", " ", regex=True)
    logger.info(
        "Loaded %d reviews (%d positive, %d negative).",
        len(df),
        (df["sentiment"] == "positive").sum(),
        (df["sentiment"] == "negative").sum(),
    )
    return df


def load_poster_metadata(path: Path = POSTERS_METADATA_CSV) -> pd.DataFrame:
    """Load poster-to-genre metadata used to build the visual search index.

    Args:
        path: Location of the poster metadata CSV, expected to contain at
            least an image filename column and a genre column.

    Returns:
        pd.DataFrame: Raw poster metadata as provided by the Kaggle
        "Movie Genre from its Poster" dataset.

    Raises:
        DatasetNotFoundError: If the file is missing.
    """
    _require_file(
        path,
        hint=(
            "Download from https://www.kaggle.com/datasets/neha1703/"
            "movie-genre-from-its-poster and place the metadata CSV at "
            "data/raw/poster_genre_metadata.csv, with poster images in "
            "data/posters/."
        ),
    )
    logger.info("Loading poster metadata from %s", path)
    df = _read_table(path)
    logger.info("Loaded metadata for %d posters.", len(df))
    return df


def load_all_raw_datasets(
    ratings_nrows: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Convenience loader that pulls every raw dataset into memory at once.

    Any individual dataset that is missing on disk is skipped with a
    warning rather than raising, so partial pipelines (e.g. sentiment
    training without MovieLens) can still run.

    Args:
        ratings_nrows: Optional cap forwarded to `load_movielens_ratings`
            for fast local iteration.

    Returns:
        dict[str, pd.DataFrame]: Mapping of dataset name to DataFrame for
        every dataset that was successfully loaded.
    """
    loaders = {
        "movielens_ratings": lambda: load_movielens_ratings(nrows=ratings_nrows),
        "movielens_movies": load_movielens_movies,
        "tmdb_movies": load_tmdb_movies,
        "tmdb_credits": load_tmdb_credits,
        "imdb_reviews": load_imdb_reviews,
        "poster_metadata": load_poster_metadata,
    }
    loaded: dict[str, pd.DataFrame] = {}
    for name, loader_fn in loaders.items():
        try:
            loaded[name] = loader_fn()
        except DatasetNotFoundError as exc:
            logger.warning("Skipping '%s': %s", name, exc)
    return loaded


if __name__ == "__main__":
    datasets = load_all_raw_datasets(ratings_nrows=100_000)
    for name, frame in datasets.items():
        logger.info("%s -> shape=%s", name, frame.shape)
