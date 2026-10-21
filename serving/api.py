"""
FastAPI backend (serving layer) for CineMatch Pro.

Exposes the recommendation, visual search, sentiment, movie detail,
chat, and trending functionality over HTTP for the Streamlit frontend
(and any other client) to consume.

Endpoints:
    GET  /search               — title / natural-language / genre search (display-ready cards)
    POST /recommend           — hybrid recommendation for a user + movie
    POST /visual-search        — text-to-poster visual search
    POST /visual-search/upload — poster image upload visual search
    GET  /tmdb-movie/{id}      — Movie Card for any TMDB id
    GET  /person/{id}          — actor / director profile + filmography
    POST /sentiment/{movie}    — aspect-based sentiment for a movie's reviews
    GET  /movie/{id}           — full Movie Intelligence Card data
    POST /chat                 — CineBot conversational endpoint
    GET  /trending              — currently trending movies
    GET  /health                — service + engine health check

Run locally with:
    uvicorn serving.api:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.recommendation_service import RecommendationServiceError, get_service
from src.artifact_store import ArtifactError, ensure_artifacts
from src.config import logger
from src.json_utils import to_jsonable


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    """Request body for POST /recommend."""

    user_id: int = Field(default=0, description="MovieLens user ID; 0 for a guest/cold-start user.")
    seed_movie_id: int = Field(description="Movie ID the recommendation is anchored to.")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return.")


class RecommendedMovie(BaseModel):
    """A single scored recommendation in the API response."""

    movie_id: int
    score: float
    source_scores: dict[str, float]


class RecommendResponse(BaseModel):
    """Response body for POST /recommend."""

    seed_movie_id: int
    recommendations: list[RecommendedMovie]


class VisualSearchRequest(BaseModel):
    """Request body for POST /visual-search (text-to-poster mode).

    Note: image-upload visual search is accepted as multipart form data
    in a real deployment; this JSON body covers the text-query mode
    described in the product spec ("dark moody thriller poster").
    """

    query: str = Field(description="Free-text description of the desired poster style.")
    top_k: int = Field(default=10, ge=1, le=50)


class VisualSearchResult(BaseModel):
    """A single visually-matched movie (with display fields for the UI)."""

    movie_id: int
    similarity_pct: float
    title: str | None = None
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    rating: float | None = None
    poster_url: str | None = None


class VisualSearchResponse(BaseModel):
    """Response body for POST /visual-search."""

    results: list[VisualSearchResult]


class SentimentRequest(BaseModel):
    """Request body for POST /sentiment/{movie}."""

    reviews: list[str] | None = Field(
        default=None,
        description="Review texts to analyze. Omit (or send []) to fetch the movie's reviews from TMDB.",
    )


class SentimentResponse(BaseModel):
    """Response body for POST /sentiment/{movie}."""

    movie_id: int
    overall_positive_pct: float
    review_count: int = 0
    aspects: dict[str, dict[str, Any]]


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    message: str = Field(description="The user's message to CineBot.")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    reply: str
    movies: list[dict[str, Any]] = Field(
        default_factory=list, description="Movies CineBot mentioned, as clickable display cards."
    )


class TrendingResponse(BaseModel):
    """Response body for GET /trending."""

    time_window: str
    movies: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    engines: dict[str, bool]


# ---------------------------------------------------------------------------
# App lifecycle — load models once at startup, not per-request
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the `RecommendationService` singleton on startup.

    Loading models during request handling would add multi-second
    latency to whichever request happens to be first; doing it once at
    startup keeps every request fast and predictable.

    Args:
        app: The FastAPI application instance (unused, required by the
            lifespan context-manager protocol).

    Yields:
        None: Control returns to FastAPI once startup work is done.
    """
    try:
        ensure_artifacts()  # no-op locally; downloads from the Hugging Face Hub in a fresh container
    except ArtifactError as exc:
        logger.error("Artifact download failed: %s", exc)
    logger.info("Starting CineMatch Pro API — warming RecommendationService...")
    get_service()
    yield
    logger.info("Shutting down CineMatch Pro API.")


app = FastAPI(
    title="CineMatch Pro API",
    description="AI-powered movie recommendation and review intelligence platform.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _visual_result(card: dict[str, Any]) -> VisualSearchResult:
    """Convert a display summary carrying ``similarity_pct`` into the API model."""
    return VisualSearchResult(
        movie_id=card["movie_id"],
        similarity_pct=card.get("similarity_pct", 0.0),
        title=card.get("title"),
        year=card.get("year"),
        genres=card.get("genres", []),
        rating=card.get("rating"),
        poster_url=card.get("poster_url"),
    )


def _http_error(exc: RecommendationServiceError) -> HTTPException:
    """Map a service error to the matching HTTP status (404 unknown, else 503)."""
    status_code = 404 if "Unknown" in str(exc) else 503
    return HTTPException(status_code=status_code, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Report which engines are loaded and the service is ready to serve.

    Returns:
        HealthResponse: Overall status plus a per-engine availability map.
    """
    engines = get_service().health()
    overall = "ok" if any(engines.values()) else "degraded"
    return HealthResponse(status=overall, engines=engines)


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Get hybrid (CF + content + semantic) recommendations for a movie.

    Args:
        request: `RecommendRequest` with `user_id`, `seed_movie_id`, `top_k`.

    Returns:
        RecommendResponse: Ranked list of recommended movies.

    Raises:
        HTTPException: 503 if the recommendation engine isn't loaded;
            404 if `seed_movie_id` is unknown.
    """
    service = get_service()
    try:
        scored = service.recommend(request.user_id, request.seed_movie_id, top_k=request.top_k)
    except RecommendationServiceError as exc:
        status_code = 404 if "Unknown" in str(exc) else 503
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return RecommendResponse(
        seed_movie_id=request.seed_movie_id,
        recommendations=[
            RecommendedMovie(movie_id=m.movie_id, score=m.score, source_scores=m.source_scores)
            for m in scored
        ],
    )


@app.post("/visual-search", response_model=VisualSearchResponse, tags=["Visual Search"])
def visual_search(request: VisualSearchRequest) -> VisualSearchResponse:
    """Find posters visually matching a free-text style description.

    Args:
        request: `VisualSearchRequest` with `query` and `top_k`.

    Returns:
        VisualSearchResponse: Visually similar movies with similarity %.

    Raises:
        HTTPException: 503 if the visual search engine isn't loaded.
    """
    service = get_service()
    try:
        cards = service.visual_search_text_cards(request.query, top_k=request.top_k)
    except RecommendationServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return VisualSearchResponse(results=[_visual_result(card) for card in cards])


@app.post("/visual-search/upload", response_model=VisualSearchResponse, tags=["Visual Search"])
async def visual_search_upload(file: UploadFile = File(...), top_k: int = Query(10, ge=1, le=50)) -> VisualSearchResponse:
    """Find posters visually similar to an uploaded poster / image.

    Args:
        file: Image file (JPEG, PNG or WebP) sent as multipart form data.
        top_k: Number of similar movies to return.

    Returns:
        VisualSearchResponse: Visually similar movies with cosine similarity %.

    Raises:
        HTTPException: 400 for an empty or unreadable image; 503 if visual
            search isn't loaded.
    """
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    service = get_service()
    try:
        cards = service.visual_search_image(payload, top_k=top_k)
    except RecommendationServiceError as exc:
        status_code = 400 if "image" in str(exc).lower() and "read" in str(exc).lower() else 503
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return VisualSearchResponse(results=[_visual_result(card) for card in cards])


@app.post("/sentiment/{movie_id}", response_model=SentimentResponse, tags=["Sentiment"])
def sentiment(movie_id: int, request: SentimentRequest) -> SentimentResponse:
    """Get an aspect-based sentiment breakdown for a movie's reviews.

    Args:
        movie_id: The movie's unique identifier (path parameter).
        request: `SentimentRequest`; when it carries no reviews they are
            fetched from TMDB for `movie_id`.

    Returns:
        SentimentResponse: Overall positive %, review count and the
        per-aspect breakdown.

    Raises:
        HTTPException: 503 if the sentiment engine (or TMDB) isn't
            available; 400 if there are no reviews to analyze; 404 if the
            movie is unknown.
    """
    service = get_service()
    try:
        result = service.get_sentiment(request.reviews, movie_id=movie_id)
    except RecommendationServiceError as exc:
        message = str(exc)
        if "Unknown" in message:
            status_code = 404
        elif "No reviews" in message:
            status_code = 400
        else:
            status_code = 503
        raise HTTPException(status_code=status_code, detail=message) from exc

    return SentimentResponse(
        movie_id=movie_id,
        overall_positive_pct=result["overall_positive_pct"],
        review_count=result.get("review_count", 0),
        aspects=result["aspects"],
    )


@app.get("/search", tags=["Search"])
def search(
    q: str = Query("", description="Title, or a description such as 'thriller with a twist ending'."),
    genres: list[str] | None = Query(None, description="Keep movies having any of these genres."),
    year_min: int | None = Query(None, ge=1880, le=2100),
    year_max: int | None = Query(None, ge=1880, le=2100),
    min_rating: float | None = Query(None, ge=0, le=10),
    top_k: int = Query(24, ge=1, le=60),
) -> dict[str, Any]:
    """Search movies by title / description, or browse by genre.

    Args:
        q: Free text. Leave empty to browse using the filters only.
        genres: Genre names to keep (any match).
        year_min: Earliest release year.
        year_max: Latest release year.
        min_rating: Minimum TMDB rating.
        top_k: Maximum number of results.

    Returns:
        dict[str, Any]: ``{"query", "count", "results"}`` where each result
        has title, year, genres, rating, poster URL, plot snippet and a
        ``match`` label explaining the hit.

    Raises:
        HTTPException: 503 if the movie catalogue isn't loaded.
    """
    service = get_service()
    try:
        results = service.search_movies(
            query=q, genres=genres, year_min=year_min, year_max=year_max, min_rating=min_rating, top_k=top_k
        )
    except RecommendationServiceError as exc:
        raise _http_error(exc) from exc
    return to_jsonable({"query": q, "count": len(results), "results": results})


@app.get("/movie/{movie_id}", tags=["Movies"])
def get_movie(movie_id: int) -> dict[str, Any]:
    """Get the full Movie Card for one movie.

    Args:
        movie_id: MovieLens ``movieId`` (path parameter).

    Returns:
        dict[str, Any]: Title, year, runtime, certificate, genres, plot,
        poster/backdrop, ratings, cast with photos, crew, trailer, budget,
        awards-ready IMDB data and related-movie ``rows``.

    Raises:
        HTTPException: 404 if the movie is unknown; 503 if the catalogue
            isn't loaded.
    """
    service = get_service()
    try:
        card = service.get_movie_card(movie_id)
    except RecommendationServiceError as exc:
        raise _http_error(exc) from exc
    return to_jsonable(card)


@app.get("/tmdb-movie/{tmdb_id}", tags=["Movies"])
def get_tmdb_movie(tmdb_id: int) -> dict[str, Any]:
    """Get a Movie Card for any TMDB id (also covers brand-new releases).

    Args:
        tmdb_id: TMDB movie id.

    Returns:
        dict[str, Any]: Same shape as ``GET /movie/{id}``.

    Raises:
        HTTPException: 503 if TMDB isn't available.
    """
    service = get_service()
    try:
        card = service.get_tmdb_movie_card(tmdb_id)
    except RecommendationServiceError as exc:
        raise _http_error(exc) from exc
    return to_jsonable(card)


@app.get("/person/{person_id}", tags=["Movies"])
def get_person(person_id: int) -> dict[str, Any]:
    """Get an actor / director profile with their filmography.

    Args:
        person_id: TMDB person id.

    Returns:
        dict[str, Any]: ``{"person": {...}, "movies": [...]}``.

    Raises:
        HTTPException: 503 if TMDB isn't available.
    """
    service = get_service()
    try:
        payload = service.get_person(person_id)
    except RecommendationServiceError as exc:
        raise _http_error(exc) from exc
    return to_jsonable(payload)


@app.post("/chat", response_model=ChatResponse, tags=["CineBot"])
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to CineBot and get its reply.

    Args:
        request: `ChatRequest` with the user's message.

    Returns:
        ChatResponse: CineBot's reply text and the movies it mentioned.

    Raises:
        HTTPException: 503 if CineBot isn't configured/loaded; 400 if
            the message was empty.
    """
    service = get_service()
    try:
        result = service.chat_with_movies(request.message)
    except RecommendationServiceError as exc:
        status_code = 400 if "empty" in str(exc).lower() else 503
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return ChatResponse(reply=result["reply"], movies=to_jsonable(result["movies"]))


@app.get("/trending", response_model=TrendingResponse, tags=["Movies"])
def trending(time_window: str = "week") -> TrendingResponse:
    """Get currently trending movies from TMDB.

    Args:
        time_window: ``"day"`` or ``"week"`` (query parameter).

    Returns:
        TrendingResponse: Trending movie summaries.

    Raises:
        HTTPException: 503 if TMDB isn't configured; 400 if
            `time_window` is invalid.
    """
    service = get_service()
    try:
        movies = service.get_trending(time_window)
    except RecommendationServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surfaces TMDBClientError's "must be 'day' or 'week'"
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TrendingResponse(time_window=time_window, movies=movies)
