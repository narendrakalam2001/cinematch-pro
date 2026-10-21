"""Tests for services/movie_catalog.py (ids -> titles/posters, search, browse)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from services.movie_catalog import MovieCatalog, as_str_list, normalize_text


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "clean_title": ["Baahubali: The Beginning", "Iron Man", "Iron Man 2", "The Iron Giant", "Heat"],
            "year": [2015, 2008, 2010, 1999, 1995],
            "id": [256040, 1726, 10138, 10386, 949],
            "overview": ["A man raised in a village discovers his royal destiny.", "Billionaire builds a suit.", "Sequel.", "A boy befriends a robot.", None],
            "genre_names": [np.array(["Action", "Drama"]), np.array(["Action", "Sci-Fi"]), np.array(["Action", "Sci-Fi"]), np.array(["Animation"]), np.array([])],
            "genres": ["Action|Drama", "Action|Sci-Fi", "Action|Sci-Fi", "Animation", "Crime|Thriller"],
            "vote_count": [9000, 22000, 15000, 3000, 6500],
            "vote_average": [7.2, 7.6, 6.8, 8.0, 7.9],
            "poster_path": ["/bb.jpg", None, "/im2.jpg", None, np.nan],
            "director": ["S. S. Rajamouli", "Jon Favreau", "Jon Favreau", "", ""],
            "cast_names": [np.array(["Prabhas", "Rana"]), np.array(["RDJ"]), np.array([]), np.array([]), np.array([])],
        }
    )


def test_normalize_text_collapses_spelling_variants():
    assert normalize_text("Baahubali: The Beginning") == normalize_text("bahubali the begining")
    assert normalize_text(None) == "" and normalize_text(float("nan")) == ""


def test_as_str_list_handles_arrays_strings_and_nan():
    assert as_str_list(np.array(["a", "b"])) == ["a", "b"]
    assert as_str_list("Action|Drama") == ["Action", "Drama"]
    assert as_str_list(float("nan")) == [] and as_str_list(None) == []


def test_search_titles_is_forgiving_and_popularity_ranked(frame):
    catalog = MovieCatalog(frame)
    assert catalog.search_titles("bahubali") == [1]  # spelling variant (baahubali)
    assert catalog.search_titles("iron man") == [2, 3]  # exact first, then sequel; no 'Iron Giant'
    assert catalog.search_titles("man iron") == [2, 3]  # word order does not matter
    assert catalog.search_titles("zzzz nothing") == []


def test_browse_filters_by_genre_year_and_excludes(frame):
    catalog = MovieCatalog(frame)
    assert catalog.browse(["Action"]) == [2, 3, 1]  # by popularity
    assert catalog.browse(["sci-fi"], year_min=2009) == [3]
    assert catalog.browse(["Action", "Sci-Fi"], match_all=True, exclude=[2]) == [3]


def test_summary_is_display_ready_and_nan_safe(frame):
    catalog = MovieCatalog(frame)
    first = catalog.summary(1)
    assert first["title"] == "Baahubali: The Beginning" and first["year"] == 2015
    assert first["genres"] == ["Action", "Drama"] and first["cast"] == ["Prabhas", "Rana"]
    assert first["poster_url"].endswith("/w342/bb.jpg")
    heat = catalog.summary(5)  # NaN poster/overview, ML-genre fallback
    assert heat["poster_url"] is None and heat["overview"] == "" and heat["genres"] == ["Crime", "Thriller"]
    assert catalog.summary(999) is None
    assert catalog.movie_id_for_tmdb(1726) == 2 and catalog.movie_id_for_tmdb(1) is None


def test_prefetch_posters_uses_tmdb_only_for_missing_and_caches(frame):
    tmdb = MagicMock()
    tmdb._get.return_value = {"poster_path": "/fetched.jpg"}
    catalog = MovieCatalog(frame, tmdb=tmdb)
    result = catalog.summaries([1, 2])  # movie 1 has a poster in the dataset, movie 2 does not
    assert tmdb._get.call_count == 1
    assert result[1]["poster_url"].endswith("/fetched.jpg")
    catalog.summaries([2])
    assert tmdb._get.call_count == 1  # cached

    tmdb._get.side_effect = RuntimeError("TMDB down")
    assert MovieCatalog(frame, tmdb=tmdb).summaries([2])[0]["poster_url"] is None  # never raises


def test_passes_filters(frame):
    catalog = MovieCatalog(frame)
    movie = catalog.summary(3)
    assert catalog.passes_filters(movie, ["Action"], 2000, 2026, 6.0)
    assert not catalog.passes_filters(movie, ["Action"], 2000, 2026, 7.0)
    assert not catalog.passes_filters(movie, ["Horror"], None, None, None)
    assert not catalog.passes_filters(movie, None, 2011, None, None)
