"""Tests for src/tmdb_client.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.tmdb_client import TMDBClient, TMDBClientError


@pytest.fixture
def client() -> TMDBClient:
    """A TMDBClient constructed with a fake API key (no real network calls)."""
    return TMDBClient(api_key="fake_key_123")


def _mock_response(payload: dict) -> MagicMock:
    """Build a mock requests.Response-like object returning the given JSON payload."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_get_trending_returns_results(client: TMDBClient):
    """get_trending parses the 'results' list out of the TMDB response."""
    payload = {"results": [{"title": "Bahubali"}, {"title": "RRR"}]}
    with patch.object(client.session, "get", return_value=_mock_response(payload)):
        trending = client.get_trending("week")
    assert len(trending) == 2


def test_get_trending_invalid_window_raises(client: TMDBClient):
    """An invalid time_window raises before any HTTP call is made."""
    with pytest.raises(TMDBClientError):
        client.get_trending("month")


def test_repeated_call_is_cached(client: TMDBClient):
    """Identical requests within the TTL window hit the cache, not the network."""
    payload = {"results": [{"title": "Bahubali"}]}
    mock_get = MagicMock(return_value=_mock_response(payload))
    with patch.object(client.session, "get", mock_get):
        client.get_trending("week")
        client.get_trending("week")
    assert mock_get.call_count == 1


def test_network_failure_raises_tmdb_client_error(client: TMDBClient):
    """A connection error is wrapped in TMDBClientError rather than propagating raw."""
    with patch.object(client.session, "get", side_effect=requests.exceptions.ConnectionError("down")):
        with pytest.raises(TMDBClientError):
            client.search_movies("some query")


def test_build_image_url_returns_none_for_missing_path():
    """build_image_url returns None when TMDB provided no image path."""
    assert TMDBClient.build_image_url(None) is None


def test_build_image_url_builds_full_url():
    """build_image_url constructs the correct TMDB CDN URL for a given size."""
    url = TMDBClient.build_image_url("/abc.jpg", size="w185")
    assert url == "https://image.tmdb.org/t/p/w185/abc.jpg"


def test_get_top_cast_builds_profile_urls(client: TMDBClient):
    """get_top_cast returns name/character/profile_url for each cast member."""
    credits_payload = {
        "cast": [
            {"name": "Prabhas", "character": "Amarendra Baahubali", "profile_path": "/prabhas.jpg"},
            {"name": "Rana Daggubati", "character": "Bhallaladeva", "profile_path": None},
        ]
    }
    with patch.object(client.session, "get", return_value=_mock_response(credits_payload)):
        cast = client.get_top_cast(101, limit=2)
    assert cast[0]["name"] == "Prabhas"
    assert cast[0]["profile_url"] == "https://image.tmdb.org/t/p/w185/prabhas.jpg"
    assert cast[1]["profile_url"] is None


def test_get_movie_reviews_parses_and_skips_empty_content(client: TMDBClient):
    """Reviews are normalised to author/content/created_at/rating; blank ones are dropped."""
    payload = {
        "total_pages": 1,
        "results": [
            {"author": "a", "content": "  Brilliant film  ", "created_at": "2024-01-01",
             "author_details": {"rating": 9.0}},
            {"author": "b", "content": "   ", "author_details": {"rating": None}},
            {"author": "c", "content": "Meh", "author_details": None},
        ],
    }
    with patch.object(client.session, "get", return_value=_mock_response(payload)):
        reviews = client.get_movie_reviews(550)
    assert [r["content"] for r in reviews] == ["Brilliant film", "Meh"]
    assert reviews[0]["rating"] == 9.0
    assert reviews[1]["rating"] is None


def test_get_movie_reviews_follows_pages_up_to_total_pages(client: TMDBClient):
    """Pagination stops at total_pages even when max_pages is larger."""
    pages = [
        _mock_response({"total_pages": 2, "results": [{"author": "a", "content": "one"}]}),
        _mock_response({"total_pages": 2, "results": [{"author": "b", "content": "two"}]}),
    ]
    mock_get = MagicMock(side_effect=pages)
    with patch.object(client.session, "get", mock_get):
        reviews = client.get_movie_reviews(550, max_pages=5)
    assert [r["content"] for r in reviews] == ["one", "two"]
    assert mock_get.call_count == 2


def test_get_movie_reviews_empty_result(client: TMDBClient):
    """A movie with no reviews yields an empty list."""
    with patch.object(client.session, "get", return_value=_mock_response({"results": []})):
        assert client.get_movie_reviews(1) == []
