"""
OMDB API client (part of Module 4) for CineMatch Pro.

Thin wrapper around the OMDB API, used to enrich the Movie Intelligence
Card with IMDB rating/ID and awards text that TMDB doesn't provide
directly, with the same TTL caching and rate-limiting approach as
`tmdb_client.py`.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.exceptions import RequestException

from src.cache_utils import TTLCache
from src.json_utils import redact_secrets
from src.config import OMDB_API_KEY, OMDB_BASE_URL, logger, require_api_key, settings


class OMDBClientError(RuntimeError):
    """Raised when the OMDB API cannot fulfil a request."""


class OMDBClient:
    """Rate-limited, caching client for the OMDB REST API.

    Attributes:
        api_key: OMDB API key, validated at construction time.
        session: Shared `requests.Session` for connection reuse.
        cache: TTL cache reducing redundant calls for repeat lookups.
    """

    def __init__(self, api_key: str | None = OMDB_API_KEY) -> None:
        self.api_key = require_api_key("OMDB_API_KEY", api_key)
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

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Perform a cached, rate-limited GET request against OMDB.

        Args:
            params: Query parameters (the API key is added automatically).

        Returns:
            dict[str, Any]: Parsed JSON response body.

        Raises:
            OMDBClientError: On network failure, non-2xx response,
                invalid JSON, or an explicit OMDB-reported error
                (``Response: "False"``).
        """
        params = dict(params)
        params["apikey"] = self.api_key

        cache_key = str(sorted(params.items()))
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self._throttle()
        try:
            response = self.session.get(OMDB_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except RequestException as exc:
            safe = redact_secrets(exc)
            logger.error("OMDB request failed: %s", safe)
            raise OMDBClientError(f"OMDB request failed: {safe}") from exc
        except ValueError as exc:
            logger.error("OMDB returned invalid JSON: %s", exc)
            raise OMDBClientError("OMDB returned invalid JSON.") from exc

        if data.get("Response") == "False":
            error_msg = data.get("Error", "Unknown OMDB error")
            logger.warning(
                "OMDB reported an error for %s: %s",
                {k: v for k, v in params.items() if k != "apikey"},
                error_msg,
            )
            raise OMDBClientError(error_msg)

        self.cache.set(cache_key, data)
        return data

    def get_by_imdb_id(self, imdb_id: str) -> dict[str, Any]:
        """Fetch full movie details by IMDB ID.

        Args:
            imdb_id: IMDB ID, e.g. ``"tt1345836"``.

        Returns:
            dict[str, Any]: OMDB payload including ``imdbRating``,
            ``Awards``, ``Ratings`` (per-source ratings list), etc.
        """
        return self._get({"i": imdb_id, "plot": "full"})

    def get_by_title(self, title: str, year: int | None = None) -> dict[str, Any]:
        """Fetch full movie details by exact title match.

        Args:
            title: Movie title to search for.
            year: Optional release year to disambiguate common titles.

        Returns:
            dict[str, Any]: OMDB payload for the best-matching movie.
        """
        params: dict[str, Any] = {"t": title, "plot": "full"}
        if year:
            params["y"] = year
        return self._get(params)

    def get_ratings_summary(self, imdb_id: str) -> dict[str, Any]:
        """Extract a compact ratings + awards summary for the Movie Card.

        Args:
            imdb_id: IMDB ID, e.g. ``"tt1345836"``.

        Returns:
            dict[str, Any]: ``{"imdb_rating": float | None, "imdb_votes":
            str | None, "awards": str | None, "metascore": int | None}``.
        """
        data = self.get_by_imdb_id(imdb_id)
        imdb_rating = data.get("imdbRating")
        metascore = data.get("Metascore")
        return {
            "imdb_rating": float(imdb_rating) if imdb_rating not in (None, "N/A") else None,
            "imdb_votes": data.get("imdbVotes") if data.get("imdbVotes") != "N/A" else None,
            "awards": data.get("Awards") if data.get("Awards") != "N/A" else None,
            "metascore": int(metascore) if metascore not in (None, "N/A") else None,
        }


if __name__ == "__main__":
    client = OMDBClient()
    logger.info("Fetching a sample title as a smoke test...")
    result = client.get_by_title("Inception", year=2010)
    logger.info("  - %s | IMDB rating: %s", result.get("Title"), result.get("imdbRating"))
