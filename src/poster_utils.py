"""
Poster catalogue + downloader for the visual search index.

The "Movie Genre from its Poster" Kaggle dataset ships poster *URLs*
(Amazon thumbnails keyed by IMDB id), not image files. This module:

1. joins those URLs to MovieLens ``movieId`` values through MovieLens'
   ``links.csv`` (so FAISS results map straight back to real movies),
2. optionally adds TMDB poster URLs (``poster_path`` in the merged movie
   table) as a fallback for dead Amazon links and for movies the poster
   dataset does not cover, and
3. downloads everything with a resumable, multi-threaded fetcher that
   validates each image and stores it as ``<movieId>.jpg``.
"""

from __future__ import annotations

import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter

from src.config import logger

TMDB_POSTER_URL_TEMPLATE = "https://image.tmdb.org/t/p/w342{path}"

_thread_local = threading.local()


class PosterCatalogError(ValueError):
    """Raised when poster metadata cannot be turned into a catalogue."""


def build_poster_catalog(
    genre_df: pd.DataFrame,
    links_df: pd.DataFrame,
    movies_df: pd.DataFrame | None = None,
    imdb_col: str = "imdbId",
    url_col: str = "Poster",
) -> pd.DataFrame:
    """Build a ``movieId -> poster URL(s)`` table.

    Args:
        genre_df: Poster metadata with an IMDB id column and a poster URL
            column (``imdbId`` / ``Poster`` in the Kaggle dataset).
        links_df: MovieLens links with ``movieId`` and ``imdbId``.
        movies_df: Optional merged movie table with ``movieId`` and
            ``poster_path`` (TMDB). When given, TMDB URLs are added as a
            fallback and as the only source for movies the poster dataset
            lacks.
        imdb_col: Name of the IMDB id column in `genre_df`.
        url_col: Name of the poster URL column in `genre_df`.

    Returns:
        pd.DataFrame: Columns ``movieId``, ``primary_url``,
        ``fallback_url`` (either may be ``None``, never both), one row
        per ``movieId``, sorted by ``movieId``.

    Raises:
        PosterCatalogError: If required columns are missing, or no movie
            ends up with a poster URL.
    """
    for frame, name, cols in (
        (genre_df, "poster metadata", {imdb_col, url_col}),
        (links_df, "links", {"movieId", "imdbId"}),
    ):
        missing = cols - set(frame.columns)
        if missing:
            raise PosterCatalogError(
                f"{name} is missing columns {sorted(missing)}; found {list(frame.columns)}. "
                "Use --imdb-col / --url-col if the poster CSV names them differently."
            )

    genre = pd.DataFrame(
        {
            "imdbId": pd.to_numeric(genre_df[imdb_col], errors="coerce"),
            "primary_url": genre_df[url_col],
        }
    ).dropna(subset=["imdbId"])
    genre["imdbId"] = genre["imdbId"].astype("int64")
    genre = genre[genre["primary_url"].astype(str).str.startswith("http")]
    genre = genre.drop_duplicates(subset="imdbId")

    links = pd.DataFrame(
        {
            "movieId": pd.to_numeric(links_df["movieId"], errors="coerce"),
            "imdbId": pd.to_numeric(links_df["imdbId"], errors="coerce"),
        }
    ).dropna()
    links = links.astype("int64").drop_duplicates(subset="movieId")

    primary = links.merge(genre, on="imdbId", how="inner")[["movieId", "primary_url"]]

    catalog = primary
    if movies_df is not None and {"movieId", "poster_path"} <= set(movies_df.columns):
        tmdb = movies_df[["movieId", "poster_path"]].dropna()
        tmdb = tmdb[tmdb["poster_path"].astype(str).str.startswith("/")]
        tmdb = tmdb.assign(
            fallback_url=tmdb["poster_path"].map(
                lambda p: TMDB_POSTER_URL_TEMPLATE.format(path=p)
            )
        )[["movieId", "fallback_url"]].drop_duplicates(subset="movieId")
        catalog = primary.merge(tmdb, on="movieId", how="outer")
    else:
        catalog = primary.assign(fallback_url=None)

    catalog = catalog.sort_values("movieId").reset_index(drop=True)
    catalog["movieId"] = catalog["movieId"].astype("int64")
    catalog = catalog.astype({"primary_url": "object", "fallback_url": "object"})
    catalog = catalog.where(catalog.notna(), None)

    if catalog.empty:
        raise PosterCatalogError(
            "No movie ended up with a poster URL — check that the poster CSV's "
            "imdbId values match MovieLens links.csv imdbId values."
        )
    logger.info(
        "Poster catalogue: %d movies (%d with a primary URL, %d with a TMDB fallback).",
        len(catalog),
        int(catalog["primary_url"].notna().sum()),
        int(catalog["fallback_url"].notna().sum()),
    )
    return catalog


def _thread_session(pool_size: int) -> requests.Session:
    """Return this thread's own `requests.Session` (created on first use).

    Args:
        pool_size: Connection-pool size for a newly created session.

    Returns:
        requests.Session: A session private to the calling thread.
    """
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "CineMatchPro/1.0 (poster indexer)"})
        _thread_local.session = session
    return session


def download_poster(
    movie_id: int,
    urls: list[str | None],
    dest_dir: Path,
    timeout: float = 10.0,
    pool_size: int = 32,
) -> Path | None:
    """Download one poster to ``<dest_dir>/<movie_id>.jpg``.

    Resumable: an existing non-empty file is reused. URLs are tried in
    order; the first response that decodes as a valid image wins, and the
    image is re-saved as an RGB JPEG so every file is uniform.

    Args:
        movie_id: MovieLens movie id (becomes the filename).
        urls: Candidate URLs, best first; ``None`` entries are skipped.
        dest_dir: Directory to store posters in (must exist).
        timeout: Per-request timeout in seconds.
        pool_size: Connection-pool size for this thread's session.

    Returns:
        Path | None: The saved file, or ``None`` if every URL failed.
    """
    dest = dest_dir / f"{movie_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    session = _thread_session(pool_size)
    for url in urls:
        if not url:
            continue
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                image.convert("RGB").save(dest, format="JPEG", quality=90)
            return dest
        except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as exc:
            logger.debug("Poster download failed for movie %s (%s): %s", movie_id, url, exc)
    if dest.exists():  # a partially written file must not be mistaken for a cache hit
        dest.unlink()
    return None


def collect_cached_posters(catalog: pd.DataFrame, dest_dir: Path) -> list[tuple[int, Path]]:
    """List catalogue posters that already exist on disk (no network).

    Args:
        catalog: Output of `build_poster_catalog`.
        dest_dir: Directory holding ``<movieId>.jpg`` files.

    Returns:
        list[tuple[int, Path]]: ``(movieId, path)`` for every cached file,
        in catalogue order.
    """
    found: list[tuple[int, Path]] = []
    for movie_id in catalog["movieId"].tolist():
        path = dest_dir / f"{movie_id}.jpg"
        if path.exists() and path.stat().st_size > 0:
            found.append((int(movie_id), path))
    return found


def download_posters(
    catalog: pd.DataFrame,
    dest_dir: Path,
    workers: int = 32,
    timeout: float = 10.0,
) -> list[tuple[int, Path]]:
    """Download every poster in the catalogue with a thread pool.

    Args:
        catalog: Output of `build_poster_catalog`.
        dest_dir: Directory to write ``<movieId>.jpg`` files into
            (created if missing).
        workers: Number of concurrent download threads.
        timeout: Per-request timeout in seconds.

    Returns:
        list[tuple[int, Path]]: ``(movieId, path)`` for each poster that is
        now on disk, in catalogue order. Failures are logged in aggregate,
        not raised, so a few dead links never abort the run.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    jobs = list(
        zip(catalog["movieId"].tolist(), catalog["primary_url"].tolist(), catalog["fallback_url"].tolist())
    )
    results: dict[int, Path] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_poster, int(mid), [primary, fallback], dest_dir, timeout, workers): int(mid)
            for mid, primary, fallback in jobs
        }
        for future in as_completed(futures):
            done += 1
            path = future.result()
            if path is not None:
                results[futures[future]] = path
            if done % 1000 == 0:
                logger.info("Poster downloads: %d/%d processed, %d ok.", done, len(jobs), len(results))

    ordered = [(int(mid), results[int(mid)]) for mid, _, _ in jobs if int(mid) in results]
    logger.info(
        "Poster download finished: %d/%d available (%d failed).",
        len(ordered),
        len(jobs),
        len(jobs) - len(ordered),
    )
    return ordered
