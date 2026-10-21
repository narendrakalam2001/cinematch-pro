"""
TMDB API client (part of Module 4) for CineMatch Pro.

Thin, defensive wrapper around The Movie Database's REST API: movie
details, credits, images, trailers, and trending endpoints, with
in-memory caching (`functools.lru_cache`-style TTL cache), basic rate
limiting, and graceful fallback on API failures.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.exceptions import RequestException

from src.cache_utils import TTLCache
from src.json_utils import redact_secrets
from src.config import (
    TMDB_API_KEY,
    TMDB_BASE_URL,
    TMDB_IMAGE_BASE_URL,
    logger,
    require_api_key,
    settings,
)


class TMDBClientError(RuntimeError):
    """Raised when the TMDB API cannot fulfil a request after retries."""


class TMDBClient:
    """Rate-limited, caching client for the TMDB REST API.

    Attributes:
        api_key: TMDB API key, validated at construction time.
        session: Shared `requests.Session` for connection reuse.
        cache: TTL cache reducing redundant calls for hot endpoints.
    """

    def __init__(self, api_key: str | None = TMDB_API_KEY) -> None:
        self.api_key = require_api_key("TMDB_API_KEY", api_key)
        self.session = requests.Session()
        self.cache = TTLCache(settings.cache.ttl_seconds, settings.cache.max_entries)
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        """Sleep just long enough to respect the configured min request gap."""
        min_gap = settings.rate_limit.min_seconds_between_requests
        elapsed = time.time() - self._last_request_time
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a cached, rate-limited GET request against TMDB.

        Args:
            endpoint: API path, e.g. ``"/movie/550"``.
            params: Additional query parameters (the API key and
                language are added automatically).

        Returns:
            dict[str, Any]: Parsed JSON response body.

        Raises:
            TMDBClientError: On network failure, non-2xx response, or
                invalid JSON, after logging the underlying error.
        """
        params = dict(params or {})
        params["api_key"] = self.api_key
        params.setdefault("language", "en-US")

        cache_key = f"{endpoint}?{sorted(params.items())}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self._throttle()
        url = f"{TMDB_BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except RequestException as exc:
            safe = redact_secrets(exc)
            logger.error("TMDB request failed for %s: %s", endpoint, safe)
            raise TMDBClientError(f"TMDB request failed for {endpoint}: {safe}") from exc
        except ValueError as exc:
            logger.error("TMDB returned invalid JSON for %s: %s", endpoint, exc)
            raise TMDBClientError(f"TMDB returned invalid JSON for {endpoint}") from exc

        self.cache.set(cache_key, data)
        return data

    def get_movie_details(self, movie_id: int) -> dict[str, Any]:
        """Fetch full details for a single movie.

        Args:
            movie_id: TMDB movie ID.

        Returns:
            dict[str, Any]: Movie details payload (title, overview,
            runtime, budget, revenue, genres, ratings, etc.).
        """
        return self._get(f"/movie/{movie_id}", {"append_to_response": "videos,keywords"})

    def get_movie_full(self, movie_id: int) -> dict[str, Any]:
        """Fetch details, credits, videos, keywords and certificates in ONE call.

        Using ``append_to_response`` keeps a Movie Card to a single TMDB
        request instead of three or four.

        Args:
            movie_id: TMDB movie ID.

        Returns:
            dict[str, Any]: Details payload with ``credits``, ``videos``,
            ``keywords`` and ``release_dates`` sections.
        """
        return self._get(
            f"/movie/{movie_id}",
            {"append_to_response": "videos,keywords,credits,release_dates"},
        )

    def get_person(self, person_id: int) -> dict[str, Any]:
        """Fetch an actor/director's profile (name, photo, birthday, biography).

        Args:
            person_id: TMDB person ID.

        Returns:
            dict[str, Any]: Person details payload.
        """
        return self._get(f"/person/{person_id}")

    def get_movie_recommendations(self, movie_id: int, kind: str = "recommendations") -> list[dict[str, Any]]:
        """Fetch TMDB's own related-movie lists for a movie.

        Args:
            movie_id: TMDB movie ID.
            kind: ``"recommendations"`` or ``"similar"``.

        Returns:
            list[dict[str, Any]]: Related movie summaries.

        Raises:
            TMDBClientError: If `kind` is not supported.
        """
        if kind not in ("recommendations", "similar"):
            raise TMDBClientError("kind must be 'recommendations' or 'similar'.")
        return self._get(f"/movie/{movie_id}/{kind}").get("results", [])

    def get_movie_credits(self, movie_id: int) -> dict[str, Any]:
        """Fetch cast and crew for a single movie.

        Args:
            movie_id: TMDB movie ID.

        Returns:
            dict[str, Any]: Payload with ``cast`` and ``crew`` lists.
        """
        return self._get(f"/movie/{movie_id}/credits")

    def get_top_cast(self, movie_id: int, limit: int = 6) -> list[dict[str, Any]]:
        """Fetch the top-billed cast with photo paths for the Movie Card.

        Args:
            movie_id: TMDB movie ID.
            limit: Maximum number of cast members to return.

        Returns:
            list[dict[str, Any]]: Each dict has ``name``, ``character``,
            and ``profile_url`` (full image URL, or ``None`` if TMDB has
            no photo on file).
        """
        credits = self.get_movie_credits(movie_id)
        cast = credits.get("cast", [])[:limit]
        return [
            {
                "name": member.get("name"),
                "character": member.get("character"),
                "profile_url": self.build_image_url(member.get("profile_path"), size="w185"),
            }
            for member in cast
        ]

    def get_person_filmography(self, person_id: int) -> list[dict[str, Any]]:
        """Fetch an actor/director's filmography, sorted by popularity.

        Powers the "Other movies by [Actor/Director]" profile pages.

        Args:
            person_id: TMDB person ID.

        Returns:
            list[dict[str, Any]]: Movie credit entries, most popular first.
        """
        data = self._get(f"/person/{person_id}/movie_credits")
        credits = data.get("cast", []) + data.get("crew", [])
        credits.sort(key=lambda c: c.get("popularity", 0), reverse=True)
        return credits

    def get_movie_reviews(self, movie_id: int, max_pages: int = 2) -> list[dict[str, Any]]:
        """Fetch user reviews for a movie from TMDB.

        The IMDB 50k dataset used to train the sentiment model has no
        movie ids, so per-movie sentiment has to be computed on reviews
        fetched at runtime — this is where they come from. TMDB usually
        has only a handful of reviews per movie (many have none).

        Args:
            movie_id: TMDB movie ID.
            max_pages: Maximum number of result pages to fetch (TMDB
                returns up to 20 reviews per page).

        Returns:
            list[dict[str, Any]]: One dict per non-empty review with keys
            ``author``, ``content``, ``created_at`` and ``rating`` (the
            author's 1-10 score, or ``None``). Empty if there are none.
        """
        reviews: list[dict[str, Any]] = []
        for page in range(1, max(1, max_pages) + 1):
            data = self._get(f"/movie/{movie_id}/reviews", {"page": page})
            for item in data.get("results", []):
                content = (item.get("content") or "").strip()
                if not content:
                    continue
                reviews.append(
                    {
                        "author": item.get("author"),
                        "content": content,
                        "created_at": item.get("created_at"),
                        "rating": (item.get("author_details") or {}).get("rating"),
                    }
                )
            if page >= int(data.get("total_pages", 1) or 1):
                break
        return reviews

    def get_trending(self, time_window: str = "week") -> list[dict[str, Any]]:
        """Fetch currently trending movies.

        Args:
            time_window: ``"day"`` or ``"week"``.

        Returns:
            list[dict[str, Any]]: Trending movie summaries.

        Raises:
            TMDBClientError: If `time_window` is not ``"day"`` or
                ``"week"``, or on API failure.
        """
        if time_window not in ("day", "week"):
            raise TMDBClientError("time_window must be 'day' or 'week'.")
        data = self._get(f"/trending/movie/{time_window}")
        return data.get("results", [])

    def search_movies(self, query: str, page: int = 1) -> list[dict[str, Any]]:
        """Search TMDB movies by title text.

        Args:
            query: Free-text search query.
            page: Result page (TMDB paginates at 20 results/page).

        Returns:
            list[dict[str, Any]]: Matching movie summaries.
        """
        data = self._get("/search/movie", {"query": query, "page": page})
        return data.get("results", [])

    @staticmethod
    def build_image_url(path: str | None, size: str = "w500") -> str | None:
        """Build a full TMDB image URL from a relative image path.

        Args:
            path: Relative path as returned by the TMDB API (e.g.
                ``"/abc123.jpg"``), or `None`.
            size: TMDB image size bucket, e.g. ``"w185"``, ``"w500"``,
                ``"original"``.

        Returns:
            str | None: Full image URL, or `None` if `path` was `None`.
        """
        if not path:
            return None
        return f"{TMDB_IMAGE_BASE_URL}/{size}{path}"


if __name__ == "__main__":
    client = TMDBClient()
    logger.info("Fetching this week's trending movies as a smoke test...")
    trending = client.get_trending("week")
    for movie in trending[:5]:
        logger.info("  - %s (%s)", movie.get("title"), movie.get("release_date"))
