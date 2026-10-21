"""Movie catalogue layer: ID → title/poster summaries, title search, genre browse.

The recommendation engines only speak in MovieLens ``movieId`` integers. The
UI needs names, years, genres, ratings and posters. `MovieCatalog` is the one
place that turns ids into display-ready dictionaries, searches the catalogue
by (fuzzy) title, browses by genre and maps TMDB ids to MovieLens ids.

It only needs the merged-movies DataFrame (and, optionally, a TMDB client for
posters the dataset does not already carry), so it is cheap to unit-test.
"""

from __future__ import annotations

import math
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import TMDB_IMAGE_BASE_URL, logger
from src.json_utils import to_jsonable

POSTER_SIZE = "w342"
_OVERVIEW_LIMIT = 220


def normalize_text(text: object) -> str:
    """Normalise a title or query for forgiving matching.

    Lower-cases, strips accents and punctuation, and collapses repeated
    letters so that spelling variants such as ``bahubali`` / ``baahubali`` or
    ``avengers`` / ``avengerss`` compare equal.

    Args:
        text: Raw title or search text.

    Returns:
        str: Normalised text made of letters, digits and single spaces.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()
    return re.sub(r"([a-z])\1+", r"\1", ascii_text)


def _clean(value: Any) -> Any:
    """Return ``None`` for NaN/NA/empty scalars, otherwise the value."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def as_str_list(value: Any) -> list[str]:
    """Coerce a list / ndarray / delimited string into a clean list of strings.

    Args:
        value: A list, numpy array, ``"A|B"`` / ``"A, B"`` string or NaN.

    Returns:
        list[str]: Non-empty stripped strings.
    """
    value = _clean(value)
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[|,]", value.strip("[]"))
        return [p.strip(" '\"") for p in parts if p.strip(" '\"")]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v).strip() for v in list(value) if _clean(v) is not None and str(v).strip()]
    return [str(value)]


def _to_int(value: Any) -> int | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def tmdb_image(path: str | None, size: str = POSTER_SIZE) -> str | None:
    """Build a TMDB image URL.

    Args:
        path: Relative TMDB image path (``"/abc.jpg"``) or ``None``.
        size: TMDB size bucket such as ``"w342"`` or ``"w1280"``.

    Returns:
        str | None: Absolute URL, or ``None`` when `path` is empty.
    """
    return f"{TMDB_IMAGE_BASE_URL}/{size}{path}" if path else None


class MovieCatalog:
    """Display-ready view over the merged movies DataFrame.

    Attributes:
        df: The merged movies frame (one row per MovieLens ``movieId``).
        tmdb: Optional TMDB client used only to fetch missing posters.
    """

    def __init__(self, movies: pd.DataFrame, tmdb: Any | None = None) -> None:
        self.df = movies.reset_index(drop=True)
        self.tmdb = tmdb
        self._poster_cache: dict[int, str | None] = {}

        ids = pd.to_numeric(self.df["movieId"], errors="coerce")
        self._id_to_pos: dict[int, int] = {int(i): pos for pos, i in enumerate(ids) if not pd.isna(i)}

        self._tmdb_to_ml: dict[int, int] = {}
        if "id" in self.df.columns:
            tmdb_ids = pd.to_numeric(self.df["id"], errors="coerce")
            for pos, tmdb_id in enumerate(tmdb_ids):
                if not pd.isna(tmdb_id):
                    self._tmdb_to_ml.setdefault(int(tmdb_id), int(ids.iloc[pos]))

        title_col = "clean_title" if "clean_title" in self.df.columns else "title"
        self._title_col = title_col
        self._title_norm = pd.Series([normalize_text(t) for t in self.df[title_col]], dtype="object")
        self._exact_index: dict[str, list[int]] = {}
        for pos, norm in enumerate(self._title_norm):
            if norm:
                self._exact_index.setdefault(norm, []).append(pos)
        self._genre_sets: list[frozenset[str]] = [frozenset(g.lower() for g in self._genres_at(i)) for i in range(len(self.df))]
        popularity = None
        for col in ("vote_count", "popularity"):
            if col in self.df.columns:
                popularity = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)
                break
        self._popularity = popularity if popularity is not None else pd.Series(np.zeros(len(self.df)))
        self._years = pd.to_numeric(self.df["year"], errors="coerce") if "year" in self.df.columns else pd.Series(np.nan, index=self.df.index)
        self._ratings = pd.to_numeric(self.df["vote_average"], errors="coerce") if "vote_average" in self.df.columns else pd.Series(np.nan, index=self.df.index)

    # ------------------------------------------------------------------
    # Row access
    # ------------------------------------------------------------------
    def has(self, movie_id: int) -> bool:
        """Whether `movie_id` exists in the catalogue."""
        return int(movie_id) in self._id_to_pos

    def row(self, movie_id: int) -> dict[str, Any] | None:
        """Raw catalogue row as a JSON-safe dict, or ``None`` if unknown."""
        pos = self._id_to_pos.get(int(movie_id))
        if pos is None:
            return None
        return to_jsonable(self.df.iloc[pos].to_dict())

    def movie_id_for_tmdb(self, tmdb_id: int) -> int | None:
        """Map a TMDB id to the MovieLens ``movieId`` (``None`` if not catalogued)."""
        return self._tmdb_to_ml.get(int(tmdb_id))

    def _genres_at(self, pos: int) -> list[str]:
        row = self.df.iloc[pos]
        for col in ("genre_names", "ml_genres_list"):
            if col in self.df.columns:
                names = as_str_list(row.get(col))
                if names:
                    return names
        names = as_str_list(row.get("genres")) if "genres" in self.df.columns else []
        return [] if names == ["(no genres listed)"] else names

    # ------------------------------------------------------------------
    # Summaries (what a poster card needs)
    # ------------------------------------------------------------------
    def _poster_for(self, pos: int) -> str | None:
        row = self.df.iloc[pos]
        path = _clean(row.get("poster_path")) if "poster_path" in self.df.columns else None
        if path:
            return tmdb_image(str(path))
        tmdb_id = _to_int(row.get("id")) if "id" in self.df.columns else None
        if tmdb_id is not None:
            return self._poster_cache.get(tmdb_id)
        return None

    def prefetch_posters(self, movie_ids: Iterable[int], max_fetch: int = 24) -> None:
        """Fetch missing poster paths from TMDB in parallel (best effort).

        Args:
            movie_ids: Movies about to be displayed.
            max_fetch: Upper bound on TMDB calls per invocation so one page
                can never stall on dozens of slow requests.
        """
        if self.tmdb is None:
            return
        wanted: list[int] = []
        for movie_id in movie_ids:
            pos = self._id_to_pos.get(int(movie_id))
            if pos is None or self._poster_for(pos):
                continue
            tmdb_id = _to_int(self.df.iloc[pos].get("id")) if "id" in self.df.columns else None
            if tmdb_id is not None and tmdb_id not in self._poster_cache:
                wanted.append(tmdb_id)
        wanted = list(dict.fromkeys(wanted))[:max_fetch]
        if not wanted:
            return

        def fetch(tmdb_id: int) -> tuple[int, str | None]:
            try:
                details = self.tmdb._get(f"/movie/{tmdb_id}")  # noqa: SLF001 - cached, rate-limited helper
                return tmdb_id, tmdb_image(details.get("poster_path"))
            except Exception as exc:  # noqa: BLE001 - posters are cosmetic, never fatal
                logger.debug("Poster fetch failed for tmdb_id=%s: %s", tmdb_id, exc)
                return tmdb_id, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            for tmdb_id, url in pool.map(fetch, wanted):
                if url:
                    self._poster_cache[tmdb_id] = url

    def summary(self, movie_id: int) -> dict[str, Any] | None:
        """Display summary for one movie (no network calls).

        Args:
            movie_id: MovieLens ``movieId``.

        Returns:
            dict[str, Any] | None: ``movie_id``, ``tmdb_id``, ``title``, ``year``,
            ``genres``, ``overview``, ``rating``, ``vote_count``,
            ``poster_url``, ``director``, ``cast``, ``language`` — or ``None``
            if the movie is unknown.
        """
        pos = self._id_to_pos.get(int(movie_id))
        if pos is None:
            return None
        row = self.df.iloc[pos]
        overview = _clean(row.get("overview")) or ""
        overview = str(overview)
        if len(overview) > _OVERVIEW_LIMIT:
            overview = overview[: _OVERVIEW_LIMIT - 1].rsplit(" ", 1)[0] + "…"
        year = _to_int(row.get("year"))
        return to_jsonable(
            {
                "movie_id": int(movie_id),
                "tmdb_id": _to_int(row.get("id")) if "id" in self.df.columns else None,
                "title": str(row.get(self._title_col) or "Unknown"),
                "year": year,
                "genres": self._genres_at(pos),
                "overview": overview,
                "rating": _to_float(row.get("vote_average")) if "vote_average" in self.df.columns else None,
                "vote_count": _to_int(row.get("vote_count")) if "vote_count" in self.df.columns else None,
                "poster_url": self._poster_for(pos),
                "director": _clean(row.get("director")) if "director" in self.df.columns else None,
                "cast": as_str_list(row.get("cast_names"))[:3] if "cast_names" in self.df.columns else [],
                "language": _clean(row.get("original_language")) if "original_language" in self.df.columns else None,
                "source": "catalog",
            }
        )

    def summaries(self, movie_ids: Iterable[int], fetch_posters: bool = True) -> list[dict[str, Any]]:
        """Summaries for many movies, in order, skipping unknown ids.

        Args:
            movie_ids: MovieLens ids.
            fetch_posters: Fill in posters missing from the dataset via TMDB.

        Returns:
            list[dict[str, Any]]: One summary per known id.
        """
        ids = [int(i) for i in movie_ids]
        if fetch_posters:
            self.prefetch_posters(ids)
        return [s for s in (self.summary(i) for i in ids) if s is not None]

    def find_exact(self, title: str, year: int | None = None, tolerance: int = 1) -> int | None:
        """Find the movie whose title equals `title` (after normalisation).

        Used to turn a title written by CineBot ("The Dark Knight (2008)")
        back into a catalogue movie so it can be opened.

        Args:
            title: Candidate title text.
            year: Release year to match; ``None`` accepts any year.
            tolerance: Allowed difference between `year` and the catalogue year.

        Returns:
            int | None: MovieLens id of the most popular match, else ``None``.
        """
        positions = self._exact_index.get(normalize_text(title), [])
        best: tuple[float, int] | None = None
        for pos in positions:
            movie_year = self._years.iloc[pos]
            if year is not None and not pd.isna(movie_year) and abs(int(movie_year) - year) > tolerance:
                continue
            if year is not None and pd.isna(movie_year):
                continue
            popularity = float(self._popularity.iloc[pos])
            if best is None or popularity > best[0]:
                best = (popularity, int(self.df.iloc[pos]["movieId"]))
        return best[1] if best else None

    # ------------------------------------------------------------------
    # Search and browse
    # ------------------------------------------------------------------
    def search_titles(self, query: str, limit: int = 24) -> list[int]:
        """Find movies whose title matches `query` (forgiving, popularity-ranked).

        Args:
            query: User text, e.g. ``"iron man"`` or ``"bahubali"``.
            limit: Maximum number of ids to return.

        Returns:
            list[int]: MovieLens ids; exact titles first, then titles starting
            with the query, then titles containing it (or all its words).
        """
        q = normalize_text(query)
        if not q:
            return []
        titles = self._title_norm
        exact = titles == q
        starts = titles.str.startswith(q)
        contains = titles.str.contains(re.escape(q), regex=True)
        tokens = q.split()
        all_tokens = pd.Series(True, index=titles.index)
        if len(tokens) > 1:
            for token in tokens:
                all_tokens &= titles.str.contains(re.escape(token), regex=True)
        else:
            all_tokens = contains
        tier = np.select([exact, starts, contains, all_tokens], [4, 3, 2, 1], default=0)
        mask = tier > 0
        if not mask.any():
            return []
        frame = pd.DataFrame({"pos": np.flatnonzero(mask), "tier": tier[mask], "pop": self._popularity.to_numpy()[mask]})
        frame = frame.sort_values(["tier", "pop"], ascending=[False, False]).head(limit)
        return [int(self.df.iloc[p]["movieId"]) for p in frame["pos"]]

    def browse(
        self,
        genres: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        limit: int = 24,
        exclude: Iterable[int] = (),
        match_all: bool = False,
    ) -> list[int]:
        """Most popular movies matching genre/year filters.

        Args:
            genres: Keep movies having ANY (or ALL, see `match_all`) of these
                genres (case-insensitive).
            year_min: Earliest release year, inclusive.
            year_max: Latest release year, inclusive.
            limit: Maximum number of ids.
            exclude: Movie ids to leave out (e.g. the movie being viewed).
            match_all: Require every listed genre instead of any one of them.

        Returns:
            list[int]: MovieLens ids ordered by popularity.
        """
        mask = pd.Series(True, index=self.df.index)
        if genres:
            wanted = {g.lower() for g in genres}
            check = (lambda s: wanted <= s) if match_all else (lambda s: bool(s & wanted))
            mask &= pd.Series([check(s) for s in self._genre_sets], index=self.df.index)
        if year_min is not None:
            mask &= self._years >= year_min
        if year_max is not None:
            mask &= self._years <= year_max
        skip = {int(i) for i in exclude}
        order = self._popularity[mask].sort_values(ascending=False).index
        ids: list[int] = []
        for pos in order:
            movie_id = int(self.df.iloc[pos]["movieId"])
            if movie_id in skip:
                continue
            ids.append(movie_id)
            if len(ids) >= limit:
                break
        return ids

    def passes_filters(
        self,
        summary: dict[str, Any],
        genres: list[str] | None,
        year_min: int | None,
        year_max: int | None,
        min_rating: float | None,
    ) -> bool:
        """Whether a summary satisfies the user's search filters.

        Args:
            summary: A dict from `summary`.
            genres: Required-any genres (case-insensitive) or ``None``.
            year_min: Earliest year or ``None``.
            year_max: Latest year or ``None``.
            min_rating: Minimum TMDB rating; unrated movies fail when set.

        Returns:
            bool: ``True`` if the movie should be shown.
        """
        if genres and not ({g.lower() for g in genres} & {g.lower() for g in summary.get("genres", [])}):
            return False
        year = summary.get("year")
        if year_min is not None and (year is None or year < year_min):
            return False
        if year_max is not None and (year is None or year > year_max):
            return False
        if min_rating:
            rating = summary.get("rating")
            if rating is None or rating < min_rating:
                return False
        return True
