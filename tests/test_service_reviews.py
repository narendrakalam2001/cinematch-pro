"""Tests for RecommendationService.fetch_reviews / get_sentiment (TMDB review flow)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from services.recommendation_service import RecommendationService, RecommendationServiceError


@pytest.fixture
def service() -> RecommendationService:
    """A RecommendationService with only the attributes the review flow touches."""
    svc = object.__new__(RecommendationService)  # skip loading real models
    svc.movies = pd.DataFrame({"movieId": [1, 2, 3], "id": [862.0, 603.0, float("nan")]})
    svc.tmdb = MagicMock()
    svc.tmdb.get_movie_reviews.return_value = [
        {"author": "a", "content": "Loved the story", "created_at": None, "rating": 9},
        {"author": "b", "content": "Weak acting", "created_at": None, "rating": 3},
    ]
    svc.sentiment = MagicMock()
    svc.sentiment.aggregate_movie_sentiment.return_value = {"positive_pct": 50.0, "review_count": 2}
    svc.aspect = None
    return svc


def test_fetch_reviews_maps_movielens_id_to_tmdb_id(service: RecommendationService):
    """movieId 1 is looked up on TMDB by its tmdb id (862), as an int."""
    texts = service.fetch_reviews(1)
    service.tmdb.get_movie_reviews.assert_called_once_with(862)
    assert texts == ["Loved the story", "Weak acting"]


def test_fetch_reviews_unknown_movie_raises(service: RecommendationService):
    """An id missing from the catalogue is reported as unknown."""
    with pytest.raises(RecommendationServiceError, match="Unknown"):
        service.fetch_reviews(999)


def test_fetch_reviews_movie_without_tmdb_id_raises(service: RecommendationService):
    """A catalogue movie with no TMDB match cannot have reviews fetched."""
    with pytest.raises(RecommendationServiceError, match="No reviews available"):
        service.fetch_reviews(3)
    service.tmdb.get_movie_reviews.assert_not_called()


def test_fetch_reviews_with_no_reviews_raises(service: RecommendationService):
    """TMDB returning nothing is surfaced as 'No reviews available'."""
    service.tmdb.get_movie_reviews.return_value = []
    with pytest.raises(RecommendationServiceError, match="No reviews available"):
        service.fetch_reviews(1)


def test_fetch_reviews_without_tmdb_client_raises(service: RecommendationService):
    """Without a TMDB key the feature degrades with a clear message."""
    service.tmdb = None
    with pytest.raises(RecommendationServiceError, match="TMDB_API_KEY"):
        service.fetch_reviews(1)


def test_fetch_reviews_wraps_tmdb_failures(service: RecommendationService):
    """A TMDB outage becomes a RecommendationServiceError, not an unhandled 500."""
    service.tmdb.get_movie_reviews.side_effect = RuntimeError("TMDB down")
    with pytest.raises(RecommendationServiceError, match="Could not fetch reviews"):
        service.fetch_reviews(1)


def test_get_sentiment_fetches_reviews_when_none_supplied(service: RecommendationService):
    """With no reviews supplied, TMDB reviews are fetched and scored."""
    result = service.get_sentiment(None, movie_id=1)
    service.tmdb.get_movie_reviews.assert_called_once()
    assert result["overall_positive_pct"] == 50.0
    assert result["review_count"] == 2
    assert result["aspects"] == {}


def test_get_sentiment_uses_supplied_reviews_without_calling_tmdb(service: RecommendationService):
    """Caller-supplied reviews are analysed directly; TMDB is not consulted."""
    service.get_sentiment(["great"], movie_id=1)
    service.tmdb.get_movie_reviews.assert_not_called()


def test_get_sentiment_without_engine_raises(service: RecommendationService):
    """No fine-tuned model loaded -> a clear 'unavailable' error."""
    service.sentiment = None
    with pytest.raises(RecommendationServiceError, match="unavailable"):
        service.get_sentiment(["great"])
