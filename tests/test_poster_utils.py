"""Tests for src/poster_utils.py."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
import requests
from PIL import Image

import src.poster_utils as poster_utils
from src.poster_utils import (
    PosterCatalogError,
    build_poster_catalog,
    collect_cached_posters,
    download_poster,
    download_posters,
)


def _png_bytes(color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 18), color).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self._status = status

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise requests.HTTPError(f"HTTP {self._status}")


class _FakeSession:
    """Serves canned responses per URL and records which URLs were requested."""

    def __init__(self, responses: dict[str, _FakeResponse]):
        self.responses = responses
        self.requested: list[str] = []

    def get(self, url: str, timeout: float = 10.0) -> _FakeResponse:
        self.requested.append(url)
        if url not in self.responses:
            raise requests.ConnectionError("unreachable")
        return self.responses[url]


@pytest.fixture
def genre_df() -> pd.DataFrame:
    """Poster metadata in the Kaggle 'MovieGenre.csv' shape."""
    return pd.DataFrame(
        {
            "imdbId": [114709, 113497, 999999],
            "Title": ["Toy Story (1995)", "Jumanji (1995)", "Not In MovieLens (2000)"],
            "Poster": ["http://img/toy.jpg", "http://img/jumanji.jpg", "http://img/none.jpg"],
        }
    )


@pytest.fixture
def links_df() -> pd.DataFrame:
    """MovieLens links.csv shape: imdbId has no 'tt' prefix or leading zeros."""
    return pd.DataFrame(
        {"movieId": [1, 2, 3], "imdbId": [114709, 113497, 113228], "tmdbId": [862.0, 8844.0, 15602.0]}
    )


def test_catalog_maps_imdb_id_to_movielens_movie_id(genre_df, links_df):
    """Poster URLs are attached to the right MovieLens movieId via links.csv."""
    catalog = build_poster_catalog(genre_df, links_df)
    assert catalog["movieId"].tolist() == [1, 2]
    assert catalog.loc[catalog["movieId"] == 1, "primary_url"].iloc[0] == "http://img/toy.jpg"


def test_catalog_adds_tmdb_fallback_and_tmdb_only_movies(genre_df, links_df):
    """TMDB poster_path becomes a fallback URL, and also covers movies the poster CSV lacks."""
    movies = pd.DataFrame({"movieId": [1, 3], "poster_path": ["/toy.jpg", "/gumpy.jpg"]})
    catalog = build_poster_catalog(genre_df, links_df, movies_df=movies).set_index("movieId")

    assert catalog.loc[1, "fallback_url"] == "https://image.tmdb.org/t/p/w342/toy.jpg"
    assert catalog.loc[1, "primary_url"] == "http://img/toy.jpg"
    assert catalog.loc[3, "primary_url"] is None  # only a TMDB URL exists
    assert catalog.loc[3, "fallback_url"].endswith("/gumpy.jpg")


def test_catalog_missing_column_raises_helpful_error(links_df):
    """A poster CSV without the expected columns fails loudly, listing what was found."""
    bad = pd.DataFrame({"id": [1], "url": ["http://x"]})
    with pytest.raises(PosterCatalogError, match="Use --imdb-col"):
        build_poster_catalog(bad, links_df)


def test_catalog_with_no_overlap_raises(links_df):
    """If no imdbId matches MovieLens, the user is told rather than getting an empty index."""
    genre = pd.DataFrame({"imdbId": [1], "Poster": ["http://x/y.jpg"]})
    with pytest.raises(PosterCatalogError, match="No movie ended up"):
        build_poster_catalog(genre, links_df)


def test_download_poster_saves_valid_image_as_jpeg(tmp_path: Path, monkeypatch):
    """A decodable image is stored as <movieId>.jpg."""
    session = _FakeSession({"http://img/a.png": _FakeResponse(_png_bytes())})
    monkeypatch.setattr(poster_utils, "_thread_session", lambda pool_size: session)

    path = download_poster(7, ["http://img/a.png"], tmp_path)
    assert path == tmp_path / "7.jpg"
    with Image.open(path) as image:
        assert image.format == "JPEG"


def test_download_poster_falls_back_when_primary_is_dead(tmp_path: Path, monkeypatch):
    """A failing primary URL falls through to the fallback URL."""
    session = _FakeSession({"http://tmdb/b.png": _FakeResponse(_png_bytes())})
    monkeypatch.setattr(poster_utils, "_thread_session", lambda pool_size: session)

    path = download_poster(8, ["http://dead/a.png", "http://tmdb/b.png"], tmp_path)
    assert path is not None
    assert session.requested == ["http://dead/a.png", "http://tmdb/b.png"]


def test_download_poster_rejects_non_image_payload(tmp_path: Path, monkeypatch):
    """An HTML error page served with HTTP 200 is not saved as a poster."""
    session = _FakeSession({"http://img/a.png": _FakeResponse(b"<html>not an image</html>")})
    monkeypatch.setattr(poster_utils, "_thread_session", lambda pool_size: session)

    assert download_poster(9, ["http://img/a.png"], tmp_path) is None
    assert not (tmp_path / "9.jpg").exists()


def test_download_poster_is_resumable(tmp_path: Path, monkeypatch):
    """An already-downloaded poster is reused without touching the network."""
    (tmp_path / "5.jpg").write_bytes(b"already here")
    session = _FakeSession({})
    monkeypatch.setattr(poster_utils, "_thread_session", lambda pool_size: session)

    assert download_poster(5, ["http://img/a.png"], tmp_path) == tmp_path / "5.jpg"
    assert session.requested == []


def test_download_posters_reports_partial_success(tmp_path: Path, monkeypatch):
    """Dead links are skipped; the rest are returned in catalogue order."""
    session = _FakeSession(
        {"http://img/1.png": _FakeResponse(_png_bytes()), "http://img/3.png": _FakeResponse(_png_bytes())}
    )
    monkeypatch.setattr(poster_utils, "_thread_session", lambda pool_size: session)
    catalog = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "primary_url": ["http://img/1.png", "http://img/2.png", "http://img/3.png"],
            "fallback_url": [None, None, None],
        }
    )
    results = download_posters(catalog, tmp_path / "posters", workers=2)
    assert [movie_id for movie_id, _ in results] == [1, 3]


def test_collect_cached_posters_only_lists_existing_files(tmp_path: Path):
    """--no-download mode only sees posters that are really on disk."""
    (tmp_path / "1.jpg").write_bytes(b"x")
    catalog = pd.DataFrame({"movieId": [1, 2], "primary_url": ["a", "b"], "fallback_url": [None, None]})
    assert collect_cached_posters(catalog, tmp_path) == [(1, tmp_path / "1.jpg")]
