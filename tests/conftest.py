"""
Shared pytest fixtures for the CineMatch Pro test suite.

Provides small, hand-crafted synthetic datasets (movies, ratings) that
exercise the same code paths as the real MovieLens/TMDB data without
requiring multi-gigabyte downloads, plus mocked versions of the heavy
ML engines (CF, content filter, semantic search) for tests that only
need to verify orchestration logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def synthetic_movies() -> pd.DataFrame:
    """A small, hand-crafted movie catalogue modeled on the Bahubali example
    from the product spec, exercising director/cast/music/genre/keyword
    matching used throughout the fusion engine and discovery rows.
    """
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "clean_title": ["Bahubali", "RRR", "Magadheera", "Pushpa", "The Notebook", "Saaho"],
            "director": ["Rajamouli", "Rajamouli", "Rajamouli", "Sukumar", "Cassavetes", "Sujeeth"],
            "cast_names": [
                ["Prabhas", "Rana"],
                ["Ramcharan", "NTR"],
                ["Ramcharan"],
                ["Allu Arjun"],
                ["Gosling"],
                ["Prabhas", "Shraddha"],
            ],
            "music_director": ["Keeravani", "Keeravani", "Keeravani", "DSP", "N/A", "Shankar-Ehsaan-Loy"],
            "original_language": ["te", "te", "te", "te", "en", "te"],
            "budget": [25_000_000, 72_000_000, 12_000_000, 15_000_000, 29_000_000, 55_000_000],
            "keyword_names": [
                ["kingdom", "war"],
                ["revolution", "war"],
                ["kingdom"],
                ["smuggling"],
                ["romance"],
                ["heist"],
            ],
            "genre_names": [
                ["Action", "Drama"],
                ["Action", "Drama"],
                ["Action"],
                ["Action", "Crime"],
                ["Romance"],
                ["Action"],
            ],
            "combined_features": [
                "epic kingdom war",
                "epic revolution war",
                "epic kingdom",
                "crime action",
                "romantic drama",
                "heist action",
            ],
            "sentiment_score": [91.0, 94.0, 70.0, 60.0, 85.0, 58.0],
        }
    )


@pytest.fixture
def synthetic_ratings() -> pd.DataFrame:
    """A small synthetic MovieLens-style ratings table for CF tests."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
            "movieId": [1, 2, 3, 1, 3, 2, 4, 5, 1, 6],
            "rating": [4.5, 4.0, 3.5, 5.0, 4.5, 3.0, 4.0, 2.5, 4.5, 3.5],
        }
    )


@pytest.fixture
def mock_content_filter() -> MagicMock:
    """A mocked `ContentBasedFilter` returning fixed similarity results."""
    mock = MagicMock()
    mock.similar_movies.return_value = [(2, 0.8), (3, 0.6)]
    return mock


@pytest.fixture
def mock_collaborative_filter() -> MagicMock:
    """A mocked `CollaborativeFilter` returning fixed recommendation scores."""
    mock = MagicMock()
    mock.recommend_for_user.return_value = [(4, 4.2), (2, 4.0)]
    return mock


@pytest.fixture
def mock_semantic_engine() -> MagicMock:
    """A mocked `SemanticSearchEngine` returning fixed search results."""
    mock = MagicMock()
    mock.search.return_value = [
        {"movie_id": 2, "title": "RRR", "similarity_pct": 88.0},
        {"movie_id": 5, "title": "The Notebook", "similarity_pct": 20.0},
    ]
    return mock
