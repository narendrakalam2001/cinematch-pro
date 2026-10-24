"""
Recommendation service — orchestration layer for CineMatch Pro.

Lazily loads and wires together every trained model / engine (SVD, TF-IDF,
FAISS, ChromaDB, RoBERTa sentiment, TMDB/OMDB clients, CineBot) behind a
single `RecommendationService` facade, so `serving/api.py` and
`frontend/app.py` never have to know about model file paths, engine
constructors, or load order.

Models are loaded once at process startup (via `get_service()`) and
reused for the lifetime of the process — this is intentionally simple
and appropriate for a single-process demo deployment (see README for
scaling notes).
"""

from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.chatbot import CineBot, ChatbotError
from src.config import (
    FAISS_INDEX_PATH,
    GEMINI_API_KEY,
    LITE_MODE,
    MERGED_MOVIES_PARQUET,
    OMDB_API_KEY,
    ROBERTA_MODEL_DIR,
    SVD_MODEL_PATH,
    TFIDF_MATRIX_PATH,
    TMDB_API_KEY,
    logger,
)
from src.content_filter import ContentBasedFilter, ContentFilterError
from src.errors import (
    AspectSentimentError,
    CollaborativeFilterError,
    SemanticSearchError,
    VisualSearchError,
)
from src.fusion_engine import FusionEngine, FusionEngineError, RecommendationRow, ScoredMovie
from src.omdb_client import OMDBClient
from src.tmdb_client import TMDBClient
from services.movie_catalog import MovieCatalog, as_str_list, normalize_text, tmdb_image
from src.json_utils import to_jsonable

# The heavy engines (PyTorch, ChromaDB, scikit-surprise, spaCy, google.generativeai) are imported lazily
# inside their `_safe_load_*` loader, so lite mode never pays for them.
if TYPE_CHECKING:
    from src.aspect_sentiment import AspectSentimentAnalyzer
    from src.collaborative_filter import CollaborativeFilter
    from src.explanation_engine import ExplanationEngine
    from src.semantic_search import SemanticSearchEngine
    from src.sentiment_engine import SentimentEngine
    from src.visual_search import VisualSearchEngine

_REMOTE_TIMEOUT_SECONDS = 20
_ROW_SIZE = 12


class RecommendationServiceError(RuntimeError):
    """Raised for orchestration-level failures not owned by a single engine."""


@dataclass
class MovieCardData:
    """Everything the Movie Intelligence Card needs, assembled in one call.

    Attributes:
        movie_id: The movie's unique identifier.
        title: Display title.
        metadata: Raw metadata row as a dict (genres, cast, overview, etc.).
        tmdb_details: Live TMDB details payload, if TMDB is configured.
        omdb_ratings: Live OMDB ratings/awards summary, if OMDB is configured.
        discovery_rows: The 10 "infinite discovery loop" recommendation rows.
    """

    movie_id: int
    title: str
    metadata: dict[str, Any]
    tmdb_details: dict[str, Any] | None
    omdb_ratings: dict[str, Any] | None
    discovery_rows: list[RecommendationRow]


class RecommendationService:
    """Facade coordinating every CineMatch Pro engine behind one interface.

    Attributes:
        movies: Merged movies metadata DataFrame, indexed by movie ID.
        cf: Trained `CollaborativeFilter`, or `None` if unavailable.
        content: Fitted `ContentBasedFilter`, or `None` if unavailable.
        semantic: `SemanticSearchEngine`, or `None` if unavailable.
        visual: `VisualSearchEngine`, or `None` if unavailable.
        sentiment: `SentimentEngine`, or `None` if unavailable.
        aspect: `AspectSentimentAnalyzer`, or `None` if unavailable.
        fusion: `FusionEngine`, or `None` if CF/content/semantic/movies
            are not all available.
        tmdb: `TMDBClient`, or `None` if `TMDB_API_KEY` is not set.
        omdb: `OMDBClient`, or `None` if `OMDB_API_KEY` is not set.
        explanation: `ExplanationEngine`, or `None` if `GEMINI_API_KEY`
            is not set.
    """

    def __init__(self) -> None:
        logger.info("Initializing RecommendationService — loading available models...")

        self.movies: pd.DataFrame | None = self._safe_load_movies()

        self.content = self._safe_load_content()
        if LITE_MODE:
            logger.info("Lite mode: skipping collaborative, semantic, visual and sentiment engines.")
            self.cf = self.semantic = self.visual = self.sentiment = self.aspect = None
        else:
            self.cf = self._safe_load_cf()
            self.semantic = self._safe_load_semantic()
            self.visual = self._safe_load_visual()
            self.sentiment = self._safe_load_sentiment()
            self.aspect = self._safe_load_aspect()

        # Fusion needs the content filter and the catalogue; CF and semantic search are optional
        # extras (without them recommendations are content-based and some rows are skipped).
        self.fusion: FusionEngine | None = None
        if self.content and self.movies is not None:
            self.fusion = FusionEngine(self.cf, self.content, self.semantic, self.movies)
        else:
            logger.warning("FusionEngine unavailable — one or more dependencies missing.")

        self.tmdb = self._safe_load_tmdb()
        self.omdb = self._safe_load_omdb()
        self.catalog: MovieCatalog | None = (
            MovieCatalog(self.movies, tmdb=self.tmdb) if self.movies is not None else None
        )
        self.explanation = self._safe_load_explanation()
        self.chatbot = self._safe_load_chatbot()

        logger.info(
            "RecommendationService ready. Loaded: cf=%s content=%s semantic=%s "
            "visual=%s sentiment=%s fusion=%s tmdb=%s omdb=%s chatbot=%s",
            bool(self.cf), bool(self.content), bool(self.semantic), bool(self.visual),
            bool(self.sentiment), bool(self.fusion), bool(self.tmdb), bool(self.omdb),
            bool(self.chatbot),
        )

    # ------------------------------------------------------------------
    # Lazy / defensive loaders — each swallows its own missing-artifact
    # error so one absent model doesn't take down the whole service.
    # ------------------------------------------------------------------
    def _safe_load_movies(self) -> pd.DataFrame | None:
        if not MERGED_MOVIES_PARQUET.exists():
            logger.warning("Merged movies parquet not found at %s.", MERGED_MOVIES_PARQUET)
            return None
        return pd.read_parquet(MERGED_MOVIES_PARQUET)

    def _safe_load_cf(self) -> CollaborativeFilter | None:
        if not SVD_MODEL_PATH.exists():
            logger.warning("SVD model not found at %s.", SVD_MODEL_PATH)
            return None
        try:
            from src.collaborative_filter import CollaborativeFilter

            return CollaborativeFilter().load(SVD_MODEL_PATH)
        except CollaborativeFilterError as exc:
            logger.error("Failed to load collaborative filter: %s", exc)
            return None

    def _safe_load_content(self) -> ContentBasedFilter | None:
        if not TFIDF_MATRIX_PATH.exists():
            logger.warning("TF-IDF matrix not found at %s.", TFIDF_MATRIX_PATH)
            return None
        try:
            return ContentBasedFilter().load()
        except ContentFilterError as exc:
            logger.error("Failed to load content filter: %s", exc)
            return None

    def _safe_load_semantic(self) -> SemanticSearchEngine | None:
        try:
            from src.semantic_search import SemanticSearchEngine

            engine = SemanticSearchEngine()
            if engine.count() == 0:
                logger.warning("ChromaDB collection is empty — semantic search will return no results.")
            return engine
        except Exception as exc:  # noqa: BLE001 - ChromaDB init failures vary by backend
            logger.error("Failed to initialize semantic search engine: %s", exc)
            return None

    def _safe_load_visual(self) -> VisualSearchEngine | None:
        if not FAISS_INDEX_PATH.exists():
            logger.warning("FAISS poster index not found at %s.", FAISS_INDEX_PATH)
            return None
        try:
            from src.visual_search import VisualSearchEngine

            engine = VisualSearchEngine()
            engine.load()
            return engine
        except VisualSearchError as exc:
            logger.error("Failed to load visual search engine: %s", exc)
            return None

    def _safe_load_sentiment(self) -> SentimentEngine | None:
        if not ROBERTA_MODEL_DIR.exists() or not any(ROBERTA_MODEL_DIR.iterdir()):
            logger.warning("Fine-tuned sentiment model not found at %s.", ROBERTA_MODEL_DIR)
            return None
        try:
            from src.sentiment_engine import SentimentEngine

            return SentimentEngine(model_path=ROBERTA_MODEL_DIR)
        except Exception as exc:  # noqa: BLE001 - transformers raises many error types
            logger.error("Failed to load sentiment engine: %s", exc)
            return None

    def _safe_load_aspect(self) -> AspectSentimentAnalyzer | None:
        if self.sentiment is None:
            return None
        try:
            from src.aspect_sentiment import AspectSentimentAnalyzer

            return AspectSentimentAnalyzer(self.sentiment)
        except AspectSentimentError as exc:
            logger.error("Failed to load aspect sentiment analyzer: %s", exc)
            return None

    def _safe_load_tmdb(self) -> TMDBClient | None:
        if not TMDB_API_KEY:
            logger.warning("TMDB_API_KEY not set — TMDB features disabled.")
            return None
        return TMDBClient()

    def _safe_load_omdb(self) -> OMDBClient | None:
        if not OMDB_API_KEY:
            logger.warning("OMDB_API_KEY not set — OMDB features disabled.")
            return None
        return OMDBClient()

    def _safe_load_explanation(self) -> ExplanationEngine | None:
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — LLM explanations disabled.")
            return None
        if LITE_MODE or self.fusion is None:
            return None
        from src.explanation_engine import ExplanationEngine

        return ExplanationEngine(fusion_engine=self.fusion)

    def _safe_load_chatbot(self) -> CineBot | None:
        if not (GEMINI_API_KEY and self.tmdb and self.catalog):
            logger.warning("CineBot disabled — missing GEMINI_API_KEY, TMDB client or movie catalogue.")
            return None
        # Tools whose engine is absent (lite mode) say so themselves; movie_search falls back to
        # title/TMDB search so the bot can still ground its picks in the catalogue.
        return CineBot(
            semantic=self.semantic,
            tmdb=self.tmdb,
            fusion=self.fusion,
            sentiment=self.sentiment,
            search_fn=lambda query: self.search_movies(query, top_k=5),
        )

    # ------------------------------------------------------------------
    # Public API used by serving/api.py and frontend/app.py
    # ------------------------------------------------------------------
    def health(self) -> dict[str, bool]:
        """Report which engines are currently loaded and ready.

        Returns:
            dict[str, bool]: Engine name -> availability flag.
        """
        return {
            "movies_loaded": self.movies is not None,
            "collaborative_filter": self.cf is not None,
            "content_filter": self.content is not None,
            "semantic_search": self.semantic is not None,
            "visual_search": self.visual is not None,
            "sentiment_engine": self.sentiment is not None,
            "fusion_engine": self.fusion is not None,
            "tmdb_client": self.tmdb is not None,
            "omdb_client": self.omdb is not None,
            "chatbot": self.chatbot is not None,
            "lite_mode": LITE_MODE,
        }

    def get_movie(self, movie_id: int) -> MovieCardData:
        """Assemble full Movie Intelligence Card data for one movie.

        Args:
            movie_id: The movie's unique identifier (MovieLens ``movieId``).

        Returns:
            MovieCardData: Metadata, live TMDB/OMDB enrichment (if
            configured), and the 10 discovery rows.

        Raises:
            RecommendationServiceError: If the movie catalogue is not
                loaded or `movie_id` is unknown.
        """
        if self.movies is None:
            raise RecommendationServiceError("Movie catalogue not loaded.")
        matches = self.movies[self.movies["movieId"] == movie_id]
        if matches.empty:
            raise RecommendationServiceError(f"Unknown movie_id: {movie_id}")
        row = matches.iloc[0].to_dict()
        title = row.get("clean_title") or row.get("title") or "Unknown"

        tmdb_details = None
        if self.tmdb and row.get("id"):
            try:
                tmdb_details = self.tmdb.get_movie_details(int(row["id"]))
            except Exception as exc:  # noqa: BLE001 - degrade gracefully on API failure
                logger.warning("TMDB lookup failed for movie_id=%s: %s", movie_id, exc)

        omdb_ratings = None
        if self.omdb and row.get("imdb_id"):
            try:
                omdb_ratings = self.omdb.get_ratings_summary(row["imdb_id"])
            except Exception as exc:  # noqa: BLE001 - degrade gracefully on API failure
                logger.warning("OMDB lookup failed for movie_id=%s: %s", movie_id, exc)

        discovery_rows: list[RecommendationRow] = []
        if self.fusion:
            try:
                discovery_rows = self.fusion.build_discovery_rows(user_id=0, seed_movie_id=movie_id)
            except FusionEngineError as exc:
                logger.warning("Discovery rows unavailable for movie_id=%s: %s", movie_id, exc)

        return MovieCardData(
            movie_id=movie_id,
            title=title,
            metadata=row,
            tmdb_details=tmdb_details,
            omdb_ratings=omdb_ratings,
            discovery_rows=discovery_rows,
        )

    def recommend(self, user_id: int, seed_movie_id: int, top_k: int = 10) -> list[ScoredMovie]:
        """Get the main hybrid recommendation list for a user + movie.

        Args:
            user_id: Raw MovieLens ``userId`` (0 for guest/cold-start).
            seed_movie_id: The movie currently being viewed.
            top_k: Number of recommendations to return.

        Returns:
            list[ScoredMovie]: Fused, ranked recommendations.

        Raises:
            RecommendationServiceError: If the fusion engine is unavailable.
        """
        if self.fusion is None:
            raise RecommendationServiceError(
                "Recommendation engine unavailable — required models not loaded."
            )
        try:
            return self.fusion.recommend(user_id, seed_movie_id, top_k=top_k)
        except FusionEngineError as exc:
            raise RecommendationServiceError(str(exc)) from exc

    def semantic_recommend(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Natural-language recommendation, e.g. "thriller with twist ending".

        Args:
            query: Free-text description of the desired movie.
            top_k: Number of results to return.

        Returns:
            list[dict[str, Any]]: Semantic search results.

        Raises:
            RecommendationServiceError: If semantic search is unavailable.
        """
        if self.semantic is None:
            raise RecommendationServiceError("Semantic search engine unavailable.")
        try:
            return self.semantic.search(query, top_k=top_k)
        except SemanticSearchError as exc:
            raise RecommendationServiceError(str(exc)) from exc

    def visual_search_by_text(self, text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Find posters visually matching a text description.

        Args:
            text: Free-text style description, e.g. "dark moody thriller poster".
            top_k: Number of results to return.

        Returns:
            list[tuple[int, float]]: ``(movie_id, similarity_pct)`` pairs.

        Raises:
            RecommendationServiceError: If visual search is unavailable.
        """
        if self.visual is None:
            raise RecommendationServiceError("Visual search engine unavailable.")
        try:
            return self.visual.search_by_text(text, top_k=top_k)
        except VisualSearchError as exc:
            raise RecommendationServiceError(str(exc)) from exc

    def fetch_reviews(self, movie_id: int) -> list[str]:
        """Fetch review texts for a movie from TMDB (runtime source of reviews).

        Args:
            movie_id: MovieLens ``movieId``; mapped to the TMDB id through
                the merged movie catalogue.

        Returns:
            list[str]: Review texts, newest pages first as returned by TMDB.

        Raises:
            RecommendationServiceError: If TMDB or the catalogue is
                unavailable, the movie is unknown / has no TMDB id, or
                TMDB has no reviews for it.
        """
        if self.tmdb is None:
            raise RecommendationServiceError("TMDB client unavailable — TMDB_API_KEY not set.")
        if self.movies is None:
            raise RecommendationServiceError("Movie catalogue not loaded.")
        matches = self.movies[self.movies["movieId"] == movie_id]
        if matches.empty:
            raise RecommendationServiceError(f"Unknown movie_id: {movie_id}")
        tmdb_id = matches.iloc[0].get("id")
        if tmdb_id is None or pd.isna(tmdb_id):
            raise RecommendationServiceError(f"No reviews available: movie_id {movie_id} has no TMDB id.")

        try:
            reviews = self.tmdb.get_movie_reviews(int(tmdb_id))
        except Exception as exc:  # noqa: BLE001 - TMDB outages must not become 500s
            raise RecommendationServiceError(f"Could not fetch reviews from TMDB: {exc}") from exc
        texts = [review["content"] for review in reviews]
        if not texts:
            raise RecommendationServiceError(
                f"No reviews available on TMDB for movie_id {movie_id}."
            )
        return texts

    def get_sentiment(
        self, reviews: list[str] | None = None, movie_id: int | None = None
    ) -> dict[str, Any]:
        """Get an aspect-based sentiment breakdown for a movie's reviews.

        Args:
            reviews: Review texts to analyze. If empty/``None`` and
                `movie_id` is given, reviews are fetched from TMDB.
            movie_id: MovieLens ``movieId``, used only to fetch reviews
                when none are supplied.

        Returns:
            dict[str, Any]: Overall positive percentage, the number of
            reviews analyzed, plus a per-aspect breakdown (story, acting,
            direction, music, VFX).

        Raises:
            RecommendationServiceError: If the sentiment engine is
                unavailable, or there are no reviews to analyze.
        """
        if self.sentiment is None:
            raise RecommendationServiceError("Sentiment engine unavailable.")
        if not reviews and movie_id is not None:
            reviews = self.fetch_reviews(movie_id)
        if not reviews:
            raise RecommendationServiceError("No reviews provided.")

        if self.aspect is not None:
            try:
                report = self.aspect.analyze(reviews)
                return {
                    "overall_positive_pct": report.overall_positive_pct,
                    "review_count": len(reviews),
                    "aspects": {
                        name: {
                            "positive_pct": score.positive_pct,
                            "mention_count": score.mention_count,
                            "label": score.label,
                            "stars": score.stars,
                        }
                        for name, score in report.aspects.items()
                    },
                }
            except AspectSentimentError:
                pass  # fall through to overall-only sentiment below

        overall = self.sentiment.aggregate_movie_sentiment(reviews)
        return {
            "overall_positive_pct": overall["positive_pct"],
            "review_count": len(reviews),
            "aspects": {},
        }

    def chat(self, message: str) -> str:
        """Send a message to CineBot and get its reply.

        Args:
            message: The user's chat message.

        Returns:
            str: CineBot's response text.

        Raises:
            RecommendationServiceError: If CineBot is unavailable.
        """
        if self.chatbot is None:
            raise RecommendationServiceError(
                "CineBot unavailable — GEMINI_API_KEY or required engines missing."
            )
        try:
            return self.chatbot.chat(message)
        except ChatbotError as exc:
            raise RecommendationServiceError(str(exc)) from exc

    def chat_with_movies(self, message: str) -> dict[str, Any]:
        """CineBot reply plus the movies it mentioned, as clickable cards.

        Args:
            message: The user's chat message.

        Returns:
            dict[str, Any]: ``{"reply": str, "movies": [summary, ...]}``. The
            movies list is best-effort: resolution problems never fail the chat.

        Raises:
            RecommendationServiceError: If CineBot is unavailable.
        """
        reply = self.chat(message)
        try:
            movies = self.resolve_movies_in_text(reply)
        except Exception as exc:  # noqa: BLE001 - cards are an enhancement of the reply
            logger.warning("Could not resolve movies mentioned by CineBot: %s", exc)
            movies = []
        return {"reply": reply, "movies": movies}

    def resolve_movies_in_text(self, text: str, limit: int = 8) -> list[dict[str, Any]]:
        """Find "Title (Year)" mentions in free text and resolve them to movies.

        Each mention is looked up in our catalogue first (so it opens with full
        recommendations) and then on TMDB (so recent releases not in
        MovieLens still open).

        Args:
            text: Text such as a CineBot reply.
            limit: Maximum number of movies to return.

        Returns:
            list[dict[str, Any]]: Display summaries in order of mention.
        """
        catalog = self._require_catalog()
        found: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        tmdb_calls = 0
        for mention in extract_movie_mentions(text):
            if len(found) >= limit:
                break
            year = mention["year"]
            summary: dict[str, Any] | None = None
            for candidate in mention["candidates"]:
                movie_id = catalog.find_exact(candidate, year)
                if movie_id is not None:
                    summary = catalog.summary(movie_id)
                    break
            if summary is None and self.tmdb is not None:
                # Longest capitalised guesses first: "Also try Mission: Impossible" -> "Mission: Impossible".
                guesses = [c for c in mention["candidates"] if c[:1].isupper() or c[:1].isdigit()][:3]
                for guess in guesses:
                    if tmdb_calls >= 8:
                        break
                    tmdb_calls += 1
                    summary = self._tmdb_match(guess, year, catalog)
                    if summary is not None:
                        break
            if summary is None:
                continue
            key = ("ml", summary["movie_id"]) if summary.get("movie_id") is not None else ("tm", summary.get("tmdb_id") or 0)
            if key in seen:
                continue
            seen.add(key)
            found.append(summary)
        catalog.prefetch_posters([m["movie_id"] for m in found if m.get("movie_id") is not None])
        for movie in found:
            if movie.get("movie_id") is not None and not movie.get("poster_url"):
                movie["poster_url"] = (catalog.summary(movie["movie_id"]) or {}).get("poster_url")
        return found

    def _tmdb_match(self, title: str, year: int | None, catalog: MovieCatalog) -> dict[str, Any] | None:
        """Best TMDB search hit for a title+year, mapped to our catalogue when possible."""
        try:
            hits = self.tmdb.search_movies(title)[:8]
        except Exception as exc:  # noqa: BLE001
            logger.debug("TMDB lookup for chat title %r failed: %s", title, exc)
            return None
        wanted = normalize_text(title)
        for hit in hits:
            release = hit.get("release_date") or ""
            hit_year = int(release[:4]) if release[:4].isdigit() else None
            if year is not None and (hit_year is None or abs(hit_year - year) > 1):
                continue
            found = normalize_text(hit.get("title"))
            if found != wanted and wanted not in found and found not in wanted:
                continue
            ml_id = catalog.movie_id_for_tmdb(hit.get("id", -1))
            if ml_id is not None:
                return catalog.summary(ml_id)
            return self._tmdb_summary(hit)
        return None

    def get_trending(self, time_window: str = "week") -> list[dict[str, Any]]:
        """Get currently trending movies from TMDB.

        Args:
            time_window: ``"day"`` or ``"week"``.

        Returns:
            list[dict[str, Any]]: Trending movie summaries.

        Raises:
            RecommendationServiceError: If TMDB is not configured.
        """
        if self.tmdb is None:
            raise RecommendationServiceError("TMDB client unavailable — TMDB_API_KEY not set.")
        return self.tmdb.get_trending(time_window)

    # ------------------------------------------------------------------
    # Display-ready API: search, movie cards, people, visual search
    # ------------------------------------------------------------------
    def _require_catalog(self) -> MovieCatalog:
        if self.catalog is None:
            raise RecommendationServiceError("Movie catalogue not loaded.")
        return self.catalog

    @staticmethod
    def _tmdb_summary(item: dict[str, Any]) -> dict[str, Any]:
        """Summary for a movie that exists on TMDB but not in our catalogue."""
        release = item.get("release_date") or ""
        overview = (item.get("overview") or "")[:220]
        return {
            "movie_id": None,
            "tmdb_id": item.get("id"),
            "title": item.get("title") or item.get("name") or "Unknown",
            "year": int(release[:4]) if release[:4].isdigit() else None,
            "genres": [],
            "overview": overview,
            "rating": item.get("vote_average") or None,
            "vote_count": item.get("vote_count"),
            "poster_url": tmdb_image(item.get("poster_path")),
            "director": None,
            "cast": [],
            "language": item.get("original_language"),
            "source": "tmdb",
        }

    def search_movies(
        self,
        query: str = "",
        genres: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        min_rating: float | None = None,
        top_k: int = 24,
    ) -> list[dict[str, Any]]:
        """Search by title, natural-language description, or just browse a genre.

        Title matches come first (our catalogue, then TMDB's own search so
        misspellings and brand-new releases are found), followed by semantic
        "plot" matches for descriptive queries such as "thriller with a twist".

        Args:
            query: Free text. Empty means "browse" using the filters only.
            genres: Keep movies having any of these genres.
            year_min: Earliest release year.
            year_max: Latest release year.
            min_rating: Minimum TMDB rating (unrated movies are dropped).
            top_k: Maximum number of results.

        Returns:
            list[dict[str, Any]]: Display summaries, each with a ``match``
            label explaining why it was returned.

        Raises:
            RecommendationServiceError: If the catalogue is not loaded.
        """
        catalog = self._require_catalog()
        query = (query or "").strip()
        results: list[dict[str, Any]] = []
        seen_ml: set[int] = set()
        seen_tmdb: set[int] = set()

        def add(summary: dict[str, Any] | None, match: str, score: float | None = None) -> None:
            if summary is None:
                return
            movie_id, tmdb_id = summary.get("movie_id"), summary.get("tmdb_id")
            if movie_id is not None and movie_id in seen_ml:
                return
            if movie_id is None and tmdb_id in seen_tmdb:
                return
            if movie_id is not None:
                seen_ml.add(movie_id)
            if tmdb_id is not None:
                seen_tmdb.add(tmdb_id)
            if not catalog.passes_filters(summary, genres, year_min, year_max, min_rating):
                return
            entry = dict(summary, match=match)
            if score is not None:
                entry["score"] = score
            results.append(entry)

        if not query:
            ids = catalog.browse(genres, year_min, year_max, limit=top_k * 3)
            label = f"Popular in {', '.join(genres)}" if genres else "Popular"
            for summary in catalog.summaries(ids[: top_k * 2], fetch_posters=False):
                add(summary, label)
            catalog.prefetch_posters([r["movie_id"] for r in results[:top_k]])
            return [dict(r, poster_url=catalog.summary(r["movie_id"])["poster_url"]) for r in results[:top_k]]

        for summary in catalog.summaries(catalog.search_titles(query, limit=top_k), fetch_posters=False):
            add(summary, "Title match")

        if self.tmdb is not None and len(results) < top_k:
            try:
                found = self.tmdb.search_movies(query)[:10]
            except Exception as exc:  # noqa: BLE001 - TMDB being down must not break search
                logger.warning("TMDB title search failed for %r: %s", query, exc)
                found = []
            for item in found:
                ml_id = catalog.movie_id_for_tmdb(item.get("id", -1))
                if ml_id is not None:
                    summary = catalog.summary(ml_id)
                    if summary and not summary.get("poster_url"):
                        summary["poster_url"] = tmdb_image(item.get("poster_path"))
                    add(summary, "Title match")
                else:
                    add(self._tmdb_summary(item), "On TMDB")

        if self.semantic is not None and (len(query.split()) >= 3 or len(results) < 3):
            try:
                semantic = self.semantic.search(query, top_k=top_k)
            except Exception as exc:  # noqa: BLE001 - semantic search is an enhancement here
                logger.warning("Semantic search failed for %r: %s", query, exc)
                semantic = []
            catalog.prefetch_posters([r["movie_id"] for r in semantic])
            for hit in semantic:
                add(catalog.summary(hit["movie_id"]), f"Plot match {hit['similarity_pct']}%", hit["similarity_pct"])

        results = results[:top_k]
        missing = [r["movie_id"] for r in results if r.get("movie_id") is not None and not r.get("poster_url")]
        if missing:
            catalog.prefetch_posters(missing)
            for r in results:
                if r.get("movie_id") is not None and not r.get("poster_url"):
                    r["poster_url"] = (catalog.summary(r["movie_id"]) or {}).get("poster_url")
        return results

    def _fetch_remote(self, tmdb_id: int | None, imdb_id: str | None) -> tuple[dict | None, dict | None, list[str]]:
        """Call TMDB (full details + credits) and OMDB in parallel, never raising."""
        warnings: list[str] = []
        full: dict | None = None
        omdb: dict | None = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            tmdb_future = pool.submit(self.tmdb.get_movie_full, int(tmdb_id)) if (self.tmdb and tmdb_id) else None
            omdb_future = pool.submit(self.omdb.get_ratings_summary, imdb_id) if (self.omdb and imdb_id) else None
            if tmdb_future is not None:
                try:
                    full = tmdb_future.result(timeout=_REMOTE_TIMEOUT_SECONDS)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("TMDB lookup failed for tmdb_id=%s: %s", tmdb_id, exc)
                    warnings.append("TMDB is unreachable right now — showing offline catalogue data.")
            if omdb_future is not None:
                try:
                    omdb = omdb_future.result(timeout=_REMOTE_TIMEOUT_SECONDS)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("OMDB lookup failed for imdb_id=%s: %s", imdb_id, exc)
                    warnings.append("IMDB rating unavailable (OMDB error).")
        return full, omdb, warnings

    def _rows_to_payload(self, rows: list[RecommendationRow]) -> list[dict[str, Any]]:
        catalog = self._require_catalog()
        catalog.prefetch_posters([i for row in rows for i in row.movie_ids], max_fetch=36)
        payload = []
        for row in rows:
            movies = catalog.summaries(row.movie_ids, fetch_posters=False)
            if movies:
                payload.append({"title": row.title, "reason": row.reason, "movies": movies})
        return payload

    def _credit_rows(self, full: dict | None, seed_movie_id: int | None, have_titles: str) -> list[RecommendationRow]:
        """Fallback "More with <actor>" / "More from <director>" rows via TMDB filmographies."""
        if not (self.tmdb and full):
            return []
        catalog = self._require_catalog()
        credits = full.get("credits") or {}
        people: list[tuple[str, int, str]] = []
        director = next((c for c in credits.get("crew", []) if c.get("job") == "Director"), None)
        if director and "Same Director" not in have_titles:
            people.append(("director", director["id"], director["name"]))
        if "Starring" not in have_titles:
            for member in (credits.get("cast") or [])[:2]:
                people.append(("actor", member["id"], member["name"]))
        rows: list[RecommendationRow] = []
        for kind, person_id, name in people:
            try:
                films = self.tmdb.get_person_filmography(int(person_id))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Filmography lookup failed for %s: %s", name, exc)
                continue
            ids: list[int] = []
            for film in films:
                ml_id = catalog.movie_id_for_tmdb(film.get("id", -1))
                if ml_id is not None and ml_id != seed_movie_id and ml_id not in ids:
                    ids.append(ml_id)
                if len(ids) >= _ROW_SIZE:
                    break
            if ids:
                title = f"By {name}" if kind == "director" else f"More with {name}"
                rows.append(RecommendationRow(title=title, reason=f"{name}'s other films", movie_ids=ids))
        return rows

    def get_movie_card(self, movie_id: int) -> dict[str, Any]:
        """Everything the Movie Card page shows, ready for JSON.

        Combines the offline catalogue, live TMDB (details, cast photos,
        crew, trailer, budget), OMDB (IMDB rating, awards) and related-movie
        rows (genre, plot, director, cast, music, collaborative, ...), each
        resolved to titles + posters so every related movie is clickable.

        Args:
            movie_id: MovieLens ``movieId``.

        Returns:
            dict[str, Any]: JSON-safe card (see `compose_card`) plus ``rows``.

        Raises:
            RecommendationServiceError: If the catalogue is not loaded or
                `movie_id` is unknown.
        """
        catalog = self._require_catalog()
        row = catalog.row(movie_id)
        if row is None:
            raise RecommendationServiceError(f"Unknown movie_id: {movie_id}")
        summary = catalog.summary(movie_id) or {}
        if not summary.get("poster_url"):
            catalog.prefetch_posters([movie_id])
            summary = catalog.summary(movie_id) or summary
        full, omdb, warnings = self._fetch_remote(summary.get("tmdb_id"), row.get("imdb_id"))

        rows: list[RecommendationRow] = []
        genres = summary.get("genres") or []
        if genres:
            ids = catalog.browse(genres[:2], limit=_ROW_SIZE, exclude=[movie_id], match_all=len(genres) >= 2)
            if len(ids) < 6 and len(genres) >= 2:
                ids = catalog.browse(genres[:1], limit=_ROW_SIZE, exclude=[movie_id])
            if ids:
                rows.append(RecommendationRow(title=f"Popular {' · '.join(genres[:2])} movies", reason="Same genres", movie_ids=ids))
        if self.fusion:
            try:
                rows.extend(self.fusion.build_discovery_rows(user_id=0, seed_movie_id=movie_id, per_row_limit=_ROW_SIZE))
            except FusionEngineError as exc:
                logger.warning("Discovery rows unavailable for movie_id=%s: %s", movie_id, exc)
        rows.extend(self._credit_rows(full, movie_id, " ".join(r.title for r in rows)))

        card = compose_card(summary, row, full, omdb)
        card["rows"] = self._rows_to_payload(rows)
        card["warnings"] = warnings
        return to_jsonable(card)

    def get_tmdb_movie_card(self, tmdb_id: int) -> dict[str, Any]:
        """Movie Card for any TMDB id — including releases absent from MovieLens.

        If the movie is in our catalogue this is identical to
        `get_movie_card`. Otherwise the card is built from TMDB alone and its
        related rows come from TMDB's recommendations plus our semantic index
        searched with the movie's plot.

        Args:
            tmdb_id: TMDB movie id.

        Returns:
            dict[str, Any]: JSON-safe card with ``rows``.

        Raises:
            RecommendationServiceError: If TMDB is unavailable or fails.
        """
        catalog = self._require_catalog()
        ml_id = catalog.movie_id_for_tmdb(tmdb_id)
        if ml_id is not None:
            return self.get_movie_card(ml_id)
        if self.tmdb is None:
            raise RecommendationServiceError("TMDB client unavailable — TMDB_API_KEY not set.")
        try:
            full = self.tmdb.get_movie_full(int(tmdb_id))
        except Exception as exc:  # noqa: BLE001
            raise RecommendationServiceError(f"Could not load movie from TMDB: {exc}") from exc

        summary = self._tmdb_summary(full)
        summary["genres"] = [g["name"] for g in full.get("genres", [])]
        omdb = None
        if self.omdb and full.get("imdb_id"):
            try:
                omdb = self.omdb.get_ratings_summary(full["imdb_id"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("OMDB lookup failed for %s: %s", full.get("imdb_id"), exc)

        payload_rows: list[dict[str, Any]] = []
        for kind, label in (("recommendations", "TMDB recommends"), ("similar", "Similar on TMDB")):
            try:
                items = self.tmdb.get_movie_recommendations(int(tmdb_id), kind)[:_ROW_SIZE]
            except Exception as exc:  # noqa: BLE001
                logger.debug("TMDB %s failed: %s", kind, exc)
                continue
            movies = []
            for item in items:
                ml = catalog.movie_id_for_tmdb(item.get("id", -1))
                movies.append(catalog.summary(ml) if ml is not None else self._tmdb_summary(item))
            movies = [m for m in movies if m]
            if movies:
                payload_rows.append({"title": label, "reason": "Based on TMDB viewing patterns", "movies": movies})
        if self.semantic and full.get("overview"):
            try:
                hits = self.semantic.search(full["overview"][:500], top_k=_ROW_SIZE)
                ids = [h["movie_id"] for h in hits]
                movies = catalog.summaries(ids)
                if movies:
                    payload_rows.append({"title": "Similar plots in our catalogue (AI)", "reason": "Semantic plot similarity", "movies": movies})
            except Exception as exc:  # noqa: BLE001
                logger.debug("Semantic rows failed: %s", exc)

        card = compose_card(summary, None, full, omdb)
        card["rows"] = payload_rows
        card["warnings"] = []
        return to_jsonable(card)

    def get_person(self, person_id: int) -> dict[str, Any]:
        """Actor / director profile with their filmography.

        Args:
            person_id: TMDB person id.

        Returns:
            dict[str, Any]: ``person`` (name, photo, birthday, biography, ...)
            and ``movies`` (display summaries, most popular first; movies in
            our catalogue are openable with full recommendations).

        Raises:
            RecommendationServiceError: If TMDB is unavailable or fails.
        """
        catalog = self._require_catalog()
        if self.tmdb is None:
            raise RecommendationServiceError("TMDB client unavailable — TMDB_API_KEY not set.")
        try:
            details = self.tmdb.get_person(int(person_id))
            films = self.tmdb.get_person_filmography(int(person_id))
        except Exception as exc:  # noqa: BLE001
            raise RecommendationServiceError(f"Could not load person from TMDB: {exc}") from exc

        movies: list[dict[str, Any]] = []
        seen: set[int] = set()
        for film in films:
            film_id = film.get("id")
            if film_id in seen or not film.get("title"):
                continue
            seen.add(film_id)
            ml_id = catalog.movie_id_for_tmdb(film_id)
            summary = catalog.summary(ml_id) if ml_id is not None else None
            if summary is None:
                summary = self._tmdb_summary(film)
            elif not summary.get("poster_url"):
                summary["poster_url"] = tmdb_image(film.get("poster_path"))
            summary["role"] = film.get("character") or film.get("job")
            movies.append(summary)
            if len(movies) >= 60:
                break
        biography = (details.get("biography") or "").strip()
        person = {
            "id": details.get("id"),
            "name": details.get("name"),
            "photo_url": tmdb_image(details.get("profile_path"), "w342"),
            "known_for": details.get("known_for_department"),
            "birthday": details.get("birthday"),
            "place_of_birth": details.get("place_of_birth"),
            "biography": biography[:700] + ("…" if len(biography) > 700 else ""),
        }
        return to_jsonable({"person": person, "movies": movies})

    def visual_search_image(self, image_bytes: bytes, top_k: int = 10) -> list[dict[str, Any]]:
        """Find posters visually similar to an uploaded image.

        Args:
            image_bytes: Raw bytes of a JPEG/PNG/WebP image.
            top_k: Number of results.

        Returns:
            list[dict[str, Any]]: Display summaries with ``similarity_pct``.

        Raises:
            RecommendationServiceError: If visual search is unavailable or
                the bytes are not a readable image.
        """
        if self.visual is None:
            raise RecommendationServiceError("Visual search engine unavailable.")
        from PIL import Image, UnidentifiedImageError  # local import: only needed on this path

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RecommendationServiceError(f"Could not read the uploaded image: {exc}") from exc
        try:
            pairs = self.visual.search_by_image(image, top_k=top_k)
        except VisualSearchError as exc:
            raise RecommendationServiceError(str(exc)) from exc
        return self._attach_similarity(pairs)

    def _attach_similarity(self, pairs: list[tuple[int, float]]) -> list[dict[str, Any]]:
        catalog = self._require_catalog()
        catalog.prefetch_posters([m for m, _ in pairs])
        out = []
        for movie_id, pct in pairs:
            summary = catalog.summary(movie_id)
            if summary:
                out.append(dict(summary, similarity_pct=pct))
        return out

    def visual_search_text_cards(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Text-to-poster visual search returning display summaries.

        Args:
            text: Style description, e.g. "dark moody thriller poster".
            top_k: Number of results.

        Returns:
            list[dict[str, Any]]: Summaries with ``similarity_pct``.
        """
        return self._attach_similarity(self.visual_search_by_text(text, top_k=top_k))


_MENTION_YEAR = re.compile(r"\((19\d{2}|20\d{2})\)")
_TITLE_BOUNDARY = re.compile(r"[,;!?]|\.\s|\s[—–]\s")


def extract_movie_mentions(text: str) -> list[dict[str, Any]]:
    """Find "Title (Year)" patterns in text, e.g. ``**The Batman (2022)**``.

    Args:
        text: Free text (markdown allowed), typically a CineBot reply.

    Returns:
        list[dict[str, Any]]: One dict per mention with ``year``; ``clean`` (the
        best single title guess, used for TMDB lookups) and ``candidates``
        (progressively shorter title guesses, tried against the catalogue).
    """
    mentions: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        for match in _MENTION_YEAR.finditer(line):
            before = line[: match.start()]
            bold_open = before.rfind("**") if before.count("**") % 2 == 1 else -1
            if bold_open >= 0:
                clean = before[bold_open + 2 :]
            else:
                clean = _TITLE_BOUNDARY.split(before)[-1] if _TITLE_BOUNDARY.search(before) else before
            clean = re.sub(r"^\s*(?:\d+[.)]|[-•*])\s*", "", clean)
            clean = clean.strip(" *_`\"'“”‘’:-")
            if not clean:
                continue
            tokens = clean.split()[-8:]
            candidates = [" ".join(tokens[i:]) for i in range(len(tokens))]
            mentions.append({"year": int(match.group(1)), "clean": " ".join(tokens), "candidates": candidates})
    return mentions


def _first_certification(full: dict[str, Any]) -> str | None:
    """Pick a certificate (India, then US, then any) from TMDB ``release_dates``."""
    results = (full.get("release_dates") or {}).get("results", [])
    by_country = {r.get("iso_3166_1"): r.get("release_dates", []) for r in results}
    for country in ("IN", "US", *by_country):
        for entry in by_country.get(country, []) or []:
            if entry.get("certification"):
                return f"{entry['certification']} ({country})"
    return None


def _people(crew: list[dict[str, Any]], jobs: set[str], limit: int = 4) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for member in crew:
        if member.get("job") in jobs and member.get("id") not in seen:
            seen.add(member["id"])
            out.append({"id": member["id"], "name": member.get("name"), "job": member.get("job")})
        if len(out) >= limit:
            break
    return out


def compose_card(
    summary: dict[str, Any],
    row: dict[str, Any] | None,
    full: dict[str, Any] | None,
    omdb: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge catalogue, TMDB and OMDB data into one Movie Card payload.

    Every section degrades gracefully: if TMDB (``full``) is missing, the card
    falls back to the offline catalogue's genres, plot, director and cast
    names (without photos).

    Args:
        summary: Display summary (catalogue or TMDB-derived).
        row: Raw catalogue row, or ``None`` for TMDB-only movies.
        full: TMDB ``/movie/{id}`` payload with credits/videos/release_dates.
        omdb: OMDB ratings summary (``imdb_rating``, ``awards``, ...).

    Returns:
        dict[str, Any]: Card fields: title, year, runtime, certificate,
        genres, overview, poster/backdrop URLs, ratings, cast (with photos),
        crew, trailer URL, budget/revenue, companies, keywords, language.
    """
    row = row or {}
    full = full or {}
    credits = full.get("credits") or {}
    crew = credits.get("crew") or []
    cast_raw = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 999))[:12]

    if cast_raw:
        cast = [
            {
                "id": c.get("id"),
                "name": c.get("name"),
                "character": c.get("character"),
                "photo_url": tmdb_image(c.get("profile_path"), "w185"),
            }
            for c in cast_raw
        ]
    else:
        cast = [{"id": None, "name": n, "character": None, "photo_url": None} for n in as_str_list(row.get("cast_names"))]

    if crew:
        crew_block = {
            "directors": _people(crew, {"Director"}),
            "writers": _people(crew, {"Writer", "Screenplay", "Story", "Novel"}),
            "producers": _people(crew, {"Producer", "Executive Producer"}),
            "composers": _people(crew, {"Original Music Composer", "Composer", "Music"}),
            "cinematographers": _people(crew, {"Director of Photography"}),
        }
    else:
        def named(value: Any) -> list[dict[str, Any]]:
            return [{"id": None, "name": value, "job": None}] if value else []

        crew_block = {
            "directors": named(row.get("director")),
            "writers": [],
            "producers": [],
            "composers": named(row.get("music_director")),
            "cinematographers": [],
        }

    videos = (full.get("videos") or {}).get("results", [])
    youtube = [v for v in videos if v.get("site") == "YouTube"]
    youtube.sort(key=lambda v: (v.get("type") != "Trailer", not v.get("official", False)))
    trailer_url = f"https://www.youtube.com/watch?v={youtube[0]['key']}" if youtube else None

    release_date = full.get("release_date") or row.get("release_date")
    genres = [g["name"] for g in full.get("genres", [])] or summary.get("genres") or []
    keywords = [k["name"] for k in (full.get("keywords") or {}).get("keywords", [])][:12] or as_str_list(row.get("keyword_names"))[:12]
    languages = [l.get("english_name") for l in full.get("spoken_languages", []) if l.get("english_name")]

    return {
        "movie_id": summary.get("movie_id"),
        "tmdb_id": summary.get("tmdb_id") or full.get("id"),
        "imdb_id": full.get("imdb_id") or row.get("imdb_id"),
        "title": full.get("title") or summary.get("title"),
        "tagline": full.get("tagline"),
        "year": int(release_date[:4]) if release_date and str(release_date)[:4].isdigit() else summary.get("year"),
        "release_date": release_date,
        "runtime": full.get("runtime") or row.get("runtime"),
        "certificate": _first_certification(full) if full else None,
        "status": full.get("status"),
        "genres": genres,
        "languages": languages or ([summary["language"]] if summary.get("language") else []),
        "overview": full.get("overview") or row.get("overview") or summary.get("overview") or "",
        "poster_url": tmdb_image(full.get("poster_path"), "w500") or summary.get("poster_url"),
        "backdrop_url": tmdb_image(full.get("backdrop_path"), "w1280"),
        "ratings": {
            "tmdb": {"score": full.get("vote_average") or summary.get("rating"), "votes": full.get("vote_count") or summary.get("vote_count")},
            "imdb": omdb,
        },
        "cast": cast,
        "crew": crew_block,
        "trailer_url": trailer_url,
        "budget": full.get("budget") or None,
        "revenue": full.get("revenue") or None,
        "companies": [c["name"] for c in full.get("production_companies", [])][:6],
        "keywords": keywords,
        "countries": [c.get("name") for c in full.get("production_countries", []) if c.get("name")],
        "homepage": full.get("homepage") or None,
    }


@lru_cache(maxsize=1)
def get_service() -> RecommendationService:
    """Get the process-wide singleton `RecommendationService` instance.

    Models are loaded once, on first call, and reused for every
    subsequent request — use this instead of constructing
    `RecommendationService()` directly in request handlers.

    Returns:
        RecommendationService: The shared service instance.
    """
    return RecommendationService()


if __name__ == "__main__":
    service = get_service()
    logger.info("Service health: %s", service.health())
