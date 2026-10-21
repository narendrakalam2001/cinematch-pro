"""Tests for src/content_filter.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.content_filter import ContentBasedFilter, ContentFilterError


@pytest.fixture
def small_movies() -> pd.DataFrame:
    """A tiny movie corpus with clearly distinguishable content clusters."""
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "combined_features": [
                "epic historical action kingdom war Rajamouli Prabhas",
                "epic historical war kingdom RRR Rajamouli",
                "romantic comedy light hearted date night",
                "sci-fi mind-bending thriller Nolan twist ending",
            ],
        }
    )


@pytest.fixture
def fitted_filter(small_movies: pd.DataFrame) -> ContentBasedFilter:
    """A ContentBasedFilter fitted on the small synthetic corpus."""
    cf = ContentBasedFilter()
    cf.vectorizer.min_df = 1  # allow rare terms given the tiny corpus
    return cf.fit(small_movies)


def test_fit_builds_matrix_with_correct_shape(fitted_filter: ContentBasedFilter):
    """After fit(), the TF-IDF matrix has one row per movie."""
    assert fitted_filter.tfidf_matrix.shape[0] == 4


def test_similar_movies_ranks_thematically_close_movie_first(fitted_filter: ContentBasedFilter):
    """The most similar movie to an epic-war movie should be the other epic-war movie."""
    results = fitted_filter.similar_movies(1, top_k=2)
    assert results[0][0] == 2


def test_similar_movies_excludes_query_movie(fitted_filter: ContentBasedFilter):
    """The query movie itself never appears in its own similarity results."""
    results = fitted_filter.similar_movies(1, top_k=10)
    ids = [mid for mid, _ in results]
    assert 1 not in ids


def test_similar_movies_unknown_id_raises(fitted_filter: ContentBasedFilter):
    """An unknown movie_id raises ContentFilterError, not a silent failure."""
    with pytest.raises(ContentFilterError):
        fitted_filter.similar_movies(999)


def test_fit_on_empty_dataframe_raises():
    """Fitting on an empty DataFrame raises rather than producing a degenerate model."""
    with pytest.raises(ContentFilterError):
        ContentBasedFilter().fit(pd.DataFrame(columns=["movieId", "combined_features"]))


def test_unfitted_filter_raises_on_query():
    """Calling similar_movies before fit()/load() raises ContentFilterError."""
    with pytest.raises(ContentFilterError):
        ContentBasedFilter().similar_movies(1)


def test_save_and_load_round_trip(fitted_filter: ContentBasedFilter, tmp_path: Path):
    """A saved filter, when reloaded, produces identical results."""
    matrix_path = tmp_path / "matrix.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    fitted_filter.save(matrix_path, vectorizer_path)

    reloaded = ContentBasedFilter().load(matrix_path, vectorizer_path)
    assert reloaded.similar_movies(1, top_k=2) == fitted_filter.similar_movies(1, top_k=2)
