"""Tests for serving/api.py (FastAPI backend), using TestClient with a
fully mocked RecommendationService so no real models need to be loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import serving.api as api_module
from services.recommendation_service import RecommendationServiceError
from src.fusion_engine import ScoredMovie


@pytest.fixture
def mock_service() -> MagicMock:
    """A MagicMock standing in for the full RecommendationService."""
    service = MagicMock()
    service.health.return_value = {
        "movies_loaded": True,
        "collaborative_filter": True,
        "content_filter": True,
        "semantic_search": True,
        "visual_search": False,
        "sentiment_engine": False,
        "fusion_engine": True,
        "tmdb_client": False,
        "omdb_client": False,
        "chatbot": False,
    }
    return service


@pytest.fixture
def client(mock_service: MagicMock, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient wired to the FastAPI app with get_service() patched out."""
    monkeypatch.setattr(api_module, "get_service", lambda: mock_service)
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_endpoint_returns_ok(client: TestClient):
    """GET /health returns status 'ok' when at least one engine is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_endpoint_returns_ranked_list(client: TestClient, mock_service: MagicMock):
    """POST /recommend returns the fusion engine's scored recommendations."""
    mock_service.recommend.return_value = [
        ScoredMovie(movie_id=2, score=0.9, source_scores={"collaborative": 0.5, "content": 1.0, "semantic": 0.8}),
        ScoredMovie(movie_id=3, score=0.4, source_scores={"collaborative": 0.1, "content": 0.5, "semantic": 0.2}),
    ]
    response = client.post("/recommend", json={"user_id": 1, "seed_movie_id": 1, "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 2
    assert body["recommendations"][0]["movie_id"] == 2


def test_recommend_endpoint_unknown_movie_returns_404(client: TestClient, mock_service: MagicMock):
    """POST /recommend returns 404 when the seed movie is unknown."""
    mock_service.recommend.side_effect = RecommendationServiceError("Unknown movie_id: 999")
    response = client.post("/recommend", json={"user_id": 1, "seed_movie_id": 999, "top_k": 5})
    assert response.status_code == 404


def test_visual_search_endpoint_returns_results(client: TestClient, mock_service: MagicMock):
    """POST /visual-search returns similarity-scored movie matches."""
    mock_service.visual_search_text_cards.return_value = [
        {"movie_id": 5, "similarity_pct": 91.2, "title": "Se7en", "year": 1995, "genres": ["Thriller"], "poster_url": "http://x/p.jpg"},
        {"movie_id": 7, "similarity_pct": 80.5, "title": "Zodiac"},
    ]
    response = client.post("/visual-search", json={"query": "dark moody thriller poster", "top_k": 5})
    assert response.status_code == 200
    first = response.json()["results"][0]
    assert first["movie_id"] == 5 and first["title"] == "Se7en" and first["poster_url"] == "http://x/p.jpg"


def test_sentiment_endpoint_returns_aspect_breakdown(client: TestClient, mock_service: MagicMock):
    """POST /sentiment/{movie} returns overall and per-aspect sentiment scores."""
    mock_service.get_sentiment.return_value = {
        "overall_positive_pct": 87.0,
        "aspects": {"story": {"positive_pct": 94.0, "mention_count": 12, "label": "Excellent", "stars": 5}},
    }
    response = client.post("/sentiment/1", json={"reviews": ["great story", "ok acting"]})
    assert response.status_code == 200
    assert response.json()["overall_positive_pct"] == 87.0


def test_sentiment_endpoint_no_reviews_returns_400(client: TestClient, mock_service: MagicMock):
    """POST /sentiment/{movie} returns 400 when no reviews are provided."""
    mock_service.get_sentiment.side_effect = RecommendationServiceError("No reviews provided.")
    response = client.post("/sentiment/1", json={"reviews": []})
    assert response.status_code == 400


def test_get_movie_endpoint_returns_full_card(client: TestClient, mock_service: MagicMock):
    """GET /movie/{id} returns the display-ready card, including numpy values and related rows."""
    import numpy as np

    mock_service.get_movie_card.return_value = {
        "movie_id": 1,
        "title": "Bahubali",
        "genres": np.array(["Action", "Drama"]),  # the original 500: numpy arrays are not JSON-serialisable
        "cast": [{"id": 5, "name": "Prabhas", "photo_url": None}],
        "rows": [{"title": "More Like Bahubali", "reason": "similar", "movies": [{"movie_id": 2, "title": "RRR"}]}],
    }
    response = client.get("/movie/1")
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Bahubali" and body["genres"] == ["Action", "Drama"]
    assert body["rows"][0]["movies"][0]["title"] == "RRR"


def test_get_movie_endpoint_unknown_id_returns_404(client: TestClient, mock_service: MagicMock):
    """GET /movie/{id} returns 404 for an unknown movie ID."""
    mock_service.get_movie_card.side_effect = RecommendationServiceError("Unknown movie_id: 999")
    response = client.get("/movie/999")
    assert response.status_code == 404


def test_search_endpoint_passes_filters_and_returns_cards(client: TestClient, mock_service: MagicMock):
    """GET /search forwards query + filters to the service and returns display cards."""
    mock_service.search_movies.return_value = [{"movie_id": 3, "title": "Iron Man", "year": 2008, "match": "Title match"}]
    response = client.get("/search", params={"q": "iron man", "genres": ["Action", "Sci-Fi"], "year_min": 2000, "min_rating": 6.5})
    assert response.status_code == 200
    assert response.json() == {"query": "iron man", "count": 1, "results": mock_service.search_movies.return_value}
    kwargs = mock_service.search_movies.call_args.kwargs
    assert kwargs["query"] == "iron man" and kwargs["genres"] == ["Action", "Sci-Fi"]
    assert kwargs["year_min"] == 2000 and kwargs["min_rating"] == 6.5


def test_search_endpoint_catalogue_unavailable_returns_503(client: TestClient, mock_service: MagicMock):
    """GET /search maps a missing catalogue to 503."""
    mock_service.search_movies.side_effect = RecommendationServiceError("Movie catalogue not loaded.")
    assert client.get("/search", params={"q": "x"}).status_code == 503


def test_tmdb_movie_and_person_endpoints(client: TestClient, mock_service: MagicMock):
    """GET /tmdb-movie/{id} and /person/{id} return what the service builds."""
    mock_service.get_tmdb_movie_card.return_value = {"title": "New Release", "rows": []}
    mock_service.get_person.return_value = {"person": {"name": "Prabhas"}, "movies": [{"title": "Salaar"}]}
    assert client.get("/tmdb-movie/123").json()["title"] == "New Release"
    assert client.get("/person/9").json()["movies"][0]["title"] == "Salaar"
    mock_service.get_person.side_effect = RecommendationServiceError("TMDB client unavailable")
    assert client.get("/person/9").status_code == 503


def test_visual_search_upload_endpoint(client: TestClient, mock_service: MagicMock):
    """POST /visual-search/upload accepts multipart images, rejects empty and unreadable files."""
    mock_service.visual_search_image.return_value = [{"movie_id": 4, "similarity_pct": 77.7, "title": "Alien"}]
    ok = client.post("/visual-search/upload", files={"file": ("poster.jpg", b"fake-bytes", "image/jpeg")})
    assert ok.status_code == 200 and ok.json()["results"][0]["title"] == "Alien"
    mock_service.visual_search_image.assert_called_once_with(b"fake-bytes", top_k=10)

    empty = client.post("/visual-search/upload", files={"file": ("poster.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400

    mock_service.visual_search_image.side_effect = RecommendationServiceError("Could not read the uploaded image: bad")
    assert client.post("/visual-search/upload", files={"file": ("x.jpg", b"zzz", "image/jpeg")}).status_code == 400


def test_chat_endpoint_returns_reply(client: TestClient, mock_service: MagicMock):
    """POST /chat returns CineBot's reply text."""
    mock_service.chat_with_movies.return_value = {
        "reply": "Try **Shutter Island (2010)** — DiCaprio at his best!",
        "movies": [{"movie_id": 7, "title": "Shutter Island", "year": 2010}],
    }
    response = client.post("/chat", json={"message": "recommend a thriller"})
    assert response.status_code == 200
    body = response.json()
    assert "Shutter Island" in body["reply"]
    assert body["movies"][0]["movie_id"] == 7  # clickable card data travels with the reply


def test_chat_endpoint_unavailable_returns_503(client: TestClient, mock_service: MagicMock):
    """POST /chat returns 503 when CineBot isn't configured."""
    mock_service.chat_with_movies.side_effect = RecommendationServiceError("CineBot unavailable")
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 503


def test_trending_endpoint_returns_movies(client: TestClient, mock_service: MagicMock):
    """GET /trending returns the trending movie list from TMDB."""
    mock_service.get_trending.return_value = [{"title": "RRR"}, {"title": "Pushpa 2"}]
    response = client.get("/trending?time_window=week")
    assert response.status_code == 200
    assert len(response.json()["movies"]) == 2


def test_recommend_request_rejects_out_of_range_top_k(client: TestClient):
    """POST /recommend returns 422 for a top_k outside the allowed [1, 50] range."""
    response = client.post("/recommend", json={"user_id": 1, "seed_movie_id": 1, "top_k": 999})
    assert response.status_code == 422


def test_sentiment_endpoint_without_reviews_asks_service_to_fetch_them(
    client: TestClient, mock_service: MagicMock
):
    """Omitting reviews makes the API pass the movie id so the service can fetch from TMDB."""
    mock_service.get_sentiment.return_value = {
        "overall_positive_pct": 72.0,
        "review_count": 3,
        "aspects": {},
    }
    response = client.post("/sentiment/42", json={})
    assert response.status_code == 200
    mock_service.get_sentiment.assert_called_once_with(None, movie_id=42)
    assert response.json()["review_count"] == 3


def test_sentiment_endpoint_unknown_movie_returns_404(client: TestClient, mock_service: MagicMock):
    """An unknown movie id maps to 404, not a generic 503."""
    mock_service.get_sentiment.side_effect = RecommendationServiceError("Unknown movie_id: 5")
    assert client.post("/sentiment/5", json={}).status_code == 404


def test_sentiment_endpoint_tmdb_unavailable_returns_503(client: TestClient, mock_service: MagicMock):
    """A missing TMDB key (service unavailable) maps to 503."""
    mock_service.get_sentiment.side_effect = RecommendationServiceError(
        "TMDB client unavailable — TMDB_API_KEY not set."
    )
    assert client.post("/sentiment/5", json={}).status_code == 503
