"""Tests for src/data_loader.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    DatasetNotFoundError,
    UnsupportedFileFormatError,
    _read_table,
    load_imdb_reviews,
    load_movielens_links,
    load_tmdb_full_metadata,
)


def test_read_table_falls_back_to_latin1_for_non_utf8_csv(tmp_path: Path):
    """The poster CSV's accented titles (invalid UTF-8) load instead of crashing."""
    path = tmp_path / "MovieGenre.csv"
    path.write_bytes("Id,Title\n1,Am\xe9lie\n".encode("latin-1"))
    df = _read_table(path)
    assert df.loc[0, "Title"] == "Am\u00e9lie"


def test_read_table_rejects_unknown_extension(tmp_path: Path):
    """Unsupported file types fail with a clear message."""
    path = tmp_path / "data.json"
    path.write_text("{}")
    with pytest.raises(UnsupportedFileFormatError):
        _read_table(path)


def test_load_imdb_reviews_strips_html_line_breaks(tmp_path: Path):
    """'<br />' tags in IMDB reviews are replaced by spaces."""
    path = tmp_path / "IMDB Dataset.csv"
    pd.DataFrame(
        {"review": ["Great film.<br /><br />Loved it.", "Bad<br/>movie"], "sentiment": ["positive", "negative"]}
    ).to_csv(path, index=False)
    reviews = load_imdb_reviews(path=path)
    assert "<br" not in " ".join(reviews["review"])
    assert reviews.loc[0, "review"].startswith("Great film.")


def test_load_imdb_reviews_requires_expected_columns(tmp_path: Path):
    """A file without review/sentiment columns is rejected."""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_imdb_reviews(path=path)


def test_load_movielens_links_missing_file_raises(tmp_path: Path):
    """A missing links.csv raises DatasetNotFoundError with guidance."""
    with pytest.raises(DatasetNotFoundError):
        load_movielens_links(path=tmp_path / "links.csv")


def test_load_tmdb_full_metadata_reads_only_needed_columns(tmp_path: Path):
    """Unused columns of the 42-column export are not loaded; required ones are validated."""
    path = tmp_path / "big.csv"
    pd.DataFrame(
        {
            "id": [1], "title": ["Inception"], "overview": ["x"], "genres_list": ["['Action']"],
            "keywords": ["['dream']"], "Cast_list": ["['A']"], "Director": ["Nolan"],
            "homepage": ["http://unused"], "tagline": ["unused"],
        }
    ).to_csv(path, index=False)
    df = load_tmdb_full_metadata(path=path)
    assert "homepage" not in df.columns and "tagline" not in df.columns
    assert df.loc[0, "Director"] == "Nolan"


def test_load_tmdb_full_metadata_rejects_legacy_schema(tmp_path: Path):
    """The original two-file TMDB export is rejected with a pointer to the right loader."""
    path = tmp_path / "tmdb_5000_movies.csv"
    pd.DataFrame({"id": [1], "title": ["x"], "overview": ["y"], "genres": ["[]"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="load_tmdb_movies"):
        load_tmdb_full_metadata(path=path)
