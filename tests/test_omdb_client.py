"""Tests for src/omdb_client.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.omdb_client import OMDBClient, OMDBClientError


@pytest.fixture
def client() -> OMDBClient:
    """An OMDBClient constructed with a fake API key (no real network calls)."""
    return OMDBClient(api_key="fake_key")


def _mock_response(payload: dict) -> MagicMock:
    """Build a mock requests.Response-like object returning the given JSON payload."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_get_ratings_summary_parses_fields(client: OMDBClient):
    """get_ratings_summary extracts and type-converts imdbRating/Metascore/Awards."""
    payload = {
        "Response": "True",
        "Title": "Inception",
        "imdbRating": "8.8",
        "imdbVotes": "2,400,000",
        "Awards": "Won 4 Oscars.",
        "Metascore": "74",
    }
    with patch.object(client.session, "get", return_value=_mock_response(payload)):
        summary = client.get_ratings_summary("tt1375666")
    assert summary["imdb_rating"] == 8.8
    assert summary["metascore"] == 74
    assert summary["awards"] == "Won 4 Oscars."


def test_get_ratings_summary_handles_na_values(client: OMDBClient):
    """OMDB's 'N/A' sentinel values are converted to Python None, not left as strings."""
    payload = {"Response": "True", "imdbRating": "N/A", "Metascore": "N/A", "Awards": "N/A", "imdbVotes": "N/A"}
    with patch.object(client.session, "get", return_value=_mock_response(payload)):
        summary = client.get_ratings_summary("tt0000000")
    assert summary["imdb_rating"] is None
    assert summary["metascore"] is None
    assert summary["awards"] is None


def test_omdb_reported_error_raises(client: OMDBClient):
    """An OMDB 'Response: False' payload raises OMDBClientError with its Error message."""
    payload = {"Response": "False", "Error": "Movie not found!"}
    with patch.object(client.session, "get", return_value=_mock_response(payload)):
        with pytest.raises(OMDBClientError, match="Movie not found"):
            client.get_by_title("a movie that does not exist")


def test_repeated_call_is_cached(client: OMDBClient):
    """Identical requests within the TTL window hit the cache, not the network."""
    payload = {"Response": "True", "Title": "Inception", "imdbRating": "8.8"}
    mock_get = MagicMock(return_value=_mock_response(payload))
    with patch.object(client.session, "get", mock_get):
        client.get_by_title("Inception", year=2010)
        client.get_by_title("Inception", year=2010)
    assert mock_get.call_count == 1
