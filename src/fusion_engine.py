"""
Hybrid recommendation fusion engine (Module 2D) for CineMatch Pro.

Combines collaborative filtering, content-based filtering, and semantic
search into a single weighted, re-ranked recommendation list, and
generates the 10 "infinite discovery loop" rows shown on every Movie
Intelligence Card (More Like X, By Same Director, Starring X, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import logger, settings
from src.content_filter import ContentBasedFilter, ContentFilterError
from src.errors import CollaborativeFilterError, SemanticSearchError

if TYPE_CHECKING:  # type hints only — importing these at runtime would load PyTorch / ChromaDB / surprise
    from src.collaborative_filter import CollaborativeFilter
    from src.semantic_search import SemanticSearchEngine


# Columns holding Python lists in the merged movie table. Parquet stores them
# as arrays and pandas hands them back as numpy arrays, which would silently
# defeat every `isinstance(value, list)` check below.
_LIST_COLUMNS = ("genre_names", "keyword_names", "cast_names", "ml_genres_list")


def normalize_list_columns(movies: pd.DataFrame) -> pd.DataFrame:
    """Convert array-valued list columns (e.g. after a parquet round-trip) to lists.

    Args:
        movies: Merged movie table.

    Returns:
        pd.DataFrame: A copy in which ``genre_names``, ``keyword_names``,
        ``cast_names`` and ``ml_genres_list`` cells that are numpy arrays
        or tuples are plain Python lists; everything else is untouched.
    """
    movies = movies.copy()
    for column in _LIST_COLUMNS:
        if column in movies.columns:
            movies[column] = movies[column].map(
                lambda value: list(value) if isinstance(value, (np.ndarray, tuple)) else value
            )
    return movies


class FusionEngineError(RuntimeError):
    """Raised for fusion-engine-specific failures."""


@dataclass
class RecommendationRow:
    """A single labeled row of movie recommendations for the UI.

    Attributes:
        title: Row heading shown in the UI, e.g. ``"More Like Bahubali"``.
        reason: Short machine-readable explanation of why these movies
            were chosen, shown in the "Why Recommended" tooltip.
        movie_ids: Ordered list of recommended movie IDs.
    """

    title: str
    reason: str
    movie_ids: list[int] = field(default_factory=list)


@dataclass
class ScoredMovie:
    """A single movie with its fused recommendation score.

    Attributes:
        movie_id: The movie's unique identifier.
        score: Final fused score after weighting and re-ranking.
        source_scores: Per-source raw scores that contributed, useful for
            building "Why Recommended" explanations.
    """

    movie_id: int
    score: float
    source_scores: dict[str, float] = field(default_factory=dict)


def _normalize_scores(pairs: list[tuple[int, float]]) -> dict[int, float]:
    """Min-max normalize a list of (id, score) pairs into [0, 1].

    Args:
        pairs: List of ``(movie_id, raw_score)`` tuples.

    Returns:
        dict[int, float]: ``movie_id -> normalized_score``. Returns an
        empty dict for empty input; if all scores are equal, every
        movie is assigned 1.0.
    """
    if not pairs:
        return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return {mid: 1.0 for mid, _ in pairs}
    return {mid: (s - lo) / (hi - lo) for mid, s in pairs}


class FusionEngine:
    """Combines CF + content + semantic signals into one ranked list.

    Attributes:
        cf: Trained `CollaborativeFilter`.
        content: Fitted `ContentBasedFilter`.
        semantic: `SemanticSearchEngine` backed by a populated ChromaDB
            collection.
        movies: Merged movies metadata DataFrame (see
            `preprocessing.run_preprocessing_pipeline`), indexed by
            movie ID for row-generation lookups.
    """

    def __init__(
        self,
        cf: CollaborativeFilter | None,
        content: ContentBasedFilter,
        semantic: SemanticSearchEngine | None,
        movies: pd.DataFrame,
        id_col: str = "movieId",
    ) -> None:
        self.cf = cf
        self.content = content
        self.semantic = semantic
        self.id_col = id_col
        self.movies = normalize_list_columns(movies).set_index(id_col, drop=False)

    def recommend(
        self,
        user_id: int,
        seed_movie_id: int,
        candidate_pool_size: int = 50,
        top_k: int = 10,
    ) -> list[ScoredMovie]:
        """Produce the main hybrid recommendation list for a user + movie.

        Weighted combination: CF (default 40%) + content-based (30%) +
        semantic (30%), per `settings.fusion`, re-ranked by the fused
        score.

        Args:
            user_id: Raw MovieLens ``userId`` (cold-start users still
                work — CF falls back to the global-mean baseline).
            seed_movie_id: The movie currently being viewed, used as the
                anchor for content and semantic similarity.
            candidate_pool_size: How many content-similar movies to pull
                as the candidate pool before scoring with all 3 signals.
            top_k: Number of final recommendations to return.

        Returns:
            list[ScoredMovie]: Top-K movies sorted by fused score
            descending.

        Raises:
            FusionEngineError: If the seed movie is unknown to the
                content filter.
        """
        try:
            content_candidates = self.content.similar_movies(
                seed_movie_id, top_k=candidate_pool_size
            )
        except ContentFilterError as exc:
            raise FusionEngineError(f"Cannot generate recommendations: {exc}") from exc

        candidate_ids = [mid for mid, _ in content_candidates]
        if not candidate_ids:
            return []

        content_norm = _normalize_scores(content_candidates)

        cf_scores: dict[int, float] = {}
        if self.cf is not None:
            try:
                cf_ranked = self.cf.recommend_for_user(user_id, candidate_ids, top_k=len(candidate_ids))
                cf_scores = _normalize_scores(cf_ranked)
            except CollaborativeFilterError as exc:
                logger.warning("CF unavailable, falling back to content+semantic only: %s", exc)

        semantic_scores: dict[int, float] = {}
        seed_row = self.movies.loc[seed_movie_id] if seed_movie_id in self.movies.index else None
        if self.semantic is not None and seed_row is not None and isinstance(seed_row.get("combined_features"), str):
            try:
                semantic_results = self.semantic.search(
                    seed_row["combined_features"][:500], top_k=candidate_pool_size
                )
                semantic_pairs = [(r["movie_id"], r["similarity_pct"]) for r in semantic_results]
                semantic_scores = _normalize_scores(semantic_pairs)
            except SemanticSearchError as exc:
                logger.warning("Semantic search unavailable for fusion: %s", exc)

        weights = settings.fusion
        # Signals that are unavailable (lite mode, or a failed engine) are dropped and the remaining
        # weights re-normalised so scores stay on the same 0..1 scale.
        active = {
            "collaborative": weights.collaborative_weight if cf_scores else 0.0,
            "content": weights.content_weight,
            "semantic": weights.semantic_weight if semantic_scores else 0.0,
        }
        total_weight = sum(active.values()) or 1.0
        fused: list[ScoredMovie] = []
        for movie_id in candidate_ids:
            if movie_id == seed_movie_id:
                continue
            c = content_norm.get(movie_id, 0.0)
            cf_s = cf_scores.get(movie_id, 0.0)
            sem = semantic_scores.get(movie_id, 0.0)
            fused_score = (
                active["collaborative"] * cf_s + active["content"] * c + active["semantic"] * sem
            ) / total_weight
            fused.append(
                ScoredMovie(
                    movie_id=movie_id,
                    score=round(fused_score, 4),
                    source_scores={"collaborative": cf_s, "content": c, "semantic": sem},
                )
            )

        fused.sort(key=lambda m: m.score, reverse=True)
        return fused[:top_k]

    def explain(self, scored_movie: ScoredMovie, seed_title: str) -> str:
        """Generate a short human-readable "why recommended" string.

        Args:
            scored_movie: A `ScoredMovie` from `recommend()`.
            seed_title: Title of the movie the recommendation is based on.

        Returns:
            str: A short explanation naming the strongest contributing
            signal, e.g. ``"Because you viewed Bahubali (similar plot &
            genre)"``.
        """
        sources = scored_movie.source_scores
        top_source = max(sources, key=sources.get) if sources else "content"
        explanations = {
            "collaborative": f"Popular among viewers of {seed_title}",
            "content": f"Because you viewed {seed_title} (similar plot & genre)",
            "semantic": f"Thematically similar to {seed_title}",
        }
        return explanations.get(top_source, f"Similar to {seed_title}")

    # ------------------------------------------------------------------
    # The 10 "Infinite Discovery Loop" rows
    # ------------------------------------------------------------------
    def _filter_by_field(
        self, field_name: str, value: str | None, exclude_id: int, limit: int
    ) -> list[int]:
        """Filter the movie catalogue by an exact-match field value.

        Args:
            field_name: Column to match on, e.g. ``"director"``.
            value: Value the column must equal. If falsy, returns `[]`.
            exclude_id: Movie ID to exclude from results (the seed movie).
            limit: Max number of results.

        Returns:
            list[int]: Matching movie IDs, up to `limit`.
        """
        if not value or field_name not in self.movies.columns:
            return []
        matches = self.movies[
            (self.movies[field_name] == value) & (self.movies[self.id_col] != exclude_id)
        ]
        if "vote_average" in matches.columns:
            matches = matches.sort_values("vote_average", ascending=False)
        return matches[self.id_col].head(limit).tolist()

    def build_discovery_rows(
        self, user_id: int, seed_movie_id: int, per_row_limit: int = 6
    ) -> list[RecommendationRow]:
        """Build the full 10-row "infinite discovery loop" for a movie page.

        Mirrors the product spec's Bahubali walkthrough: More Like X, By
        Same Director, Starring X, Similar Production Scale, Same
        Theme/Setting, Same Music Director, Trending in Language,
        International Similar Films, Users Also Watched, Sentiment
        Champions.

        Args:
            user_id: Raw MovieLens ``userId`` viewing this page.
            seed_movie_id: The movie currently being viewed.
            per_row_limit: Max movies shown per row.

        Returns:
            list[RecommendationRow]: Up to 10 rows, each already
            deduplicated internally. Rows with no matches are omitted so
            the UI never renders an empty strip.

        Raises:
            FusionEngineError: If `seed_movie_id` is not in the movie
                catalogue.
        """
        if seed_movie_id not in self.movies.index:
            raise FusionEngineError(f"Unknown seed_movie_id: {seed_movie_id}")

        seed = self.movies.loc[seed_movie_id]
        seed_title = seed.get("clean_title", seed.get("title", "this movie"))
        director = seed.get("director")
        cast_list = seed.get("cast_names")
        lead_actor = cast_list[0] if isinstance(cast_list, list) and cast_list else None
        music_director = seed.get("music_director")
        language = seed.get("original_language")
        genre_names = seed.get("genre_names") if isinstance(seed.get("genre_names"), list) else []

        rows: list[RecommendationRow] = []

        def add_row(title: str, reason: str, ids: list[int]) -> None:
            if ids:
                rows.append(RecommendationRow(title=title, reason=reason, movie_ids=ids[:per_row_limit]))

        # Row 1 — More Like [Movie] (content-based)
        try:
            similar = self.content.similar_movies(seed_movie_id, top_k=per_row_limit)
            add_row(f"More Like {seed_title}", "Similar genre, plot, and tone", [m for m, _ in similar])
        except ContentFilterError:
            pass

        # Row 2 — By Same Director
        add_row(
            f"By Same Director ({director})" if director else "By Same Director",
            f"Also directed by {director}",
            self._filter_by_field("director", director, seed_movie_id, per_row_limit),
        )

        # Row 3 — Starring [Lead Actor]
        if lead_actor and "cast_names" in self.movies.columns:
            mask = self.movies["cast_names"].apply(
                lambda names: isinstance(names, list) and lead_actor in names
            )
            candidates = self.movies[mask & (self.movies[self.id_col] != seed_movie_id)]
            add_row(
                f"Starring {lead_actor}",
                f"Also features {lead_actor}",
                candidates[self.id_col].head(per_row_limit).tolist(),
            )

        # Row 4 — Similar Production Scale (budget +/- 30%)
        budget = seed.get("budget")
        if budget and budget > 0 and "budget" in self.movies.columns:
            low, high = budget * 0.7, budget * 1.3
            candidates = self.movies[
                (self.movies["budget"].between(low, high))
                & (self.movies[self.id_col] != seed_movie_id)
            ]
            add_row(
                "Similar Production Scale",
                "Comparable budget and visual spectacle",
                candidates[self.id_col].head(per_row_limit).tolist(),
            )

        # Row 5 — Same Theme/Setting (keyword overlap)
        keywords = seed.get("keyword_names") if isinstance(seed.get("keyword_names"), list) else []
        if keywords and "keyword_names" in self.movies.columns:
            keyword_set = set(keywords)
            overlap = self.movies["keyword_names"].apply(
                lambda kws: len(keyword_set & set(kws)) if isinstance(kws, list) else 0
            )
            candidates = self.movies[(overlap > 0) & (self.movies[self.id_col] != seed_movie_id)]
            candidates = candidates.assign(_overlap=overlap[candidates.index]).sort_values(
                "_overlap", ascending=False
            )
            add_row(
                "Same Theme & Setting",
                "Shares key plot elements and setting",
                candidates[self.id_col].head(per_row_limit).tolist(),
            )

        # Row 6 — Same Music Director
        add_row(
            f"If You Liked {seed_title}'s Music" if music_director else "Same Music Director",
            f"Scored by {music_director}",
            self._filter_by_field("music_director", music_director, seed_movie_id, per_row_limit),
        )

        # Row 7 — Trending in [Language] Cinema
        if language and "original_language" in self.movies.columns:
            candidates = self.movies[
                (self.movies["original_language"] == language)
                & (self.movies[self.id_col] != seed_movie_id)
            ]
            if "popularity" in candidates.columns:
                candidates = candidates.sort_values("popularity", ascending=False)
            add_row(
                f"Trending in {str(language).upper()} Cinema",
                "Popular right now in the same language/industry",
                candidates[self.id_col].head(per_row_limit).tolist(),
            )

        # Row 8 — International Similar Films (semantic, cross-language)
        combined_text = seed.get("combined_features")
        if self.semantic is not None and isinstance(combined_text, str) and combined_text.strip():
            try:
                semantic_results = self.semantic.search(combined_text[:500], top_k=per_row_limit + 5)
                ids = [r["movie_id"] for r in semantic_results if r["movie_id"] != seed_movie_id]
                add_row(
                    "International Films With a Similar Theme",
                    "Cross-language semantic match on plot and tone",
                    ids,
                )
            except SemanticSearchError:
                pass

        # Row 9 — Users Also Watched (pure collaborative filtering)
        if self.cf is not None:
            try:
                candidate_pool = [mid for mid in self.movies[self.id_col].tolist() if mid != seed_movie_id]
                cf_ranked = self.cf.recommend_for_user(user_id, candidate_pool[:200], top_k=per_row_limit)
                add_row(
                    "Users Who Watched This Also Watched",
                    "Based on collaborative filtering across similar viewers",
                    [mid for mid, _ in cf_ranked],
                )
            except CollaborativeFilterError:
                pass

        # Row 10 — Sentiment Champions (same genre, highest sentiment)
        if genre_names and "genre_names" in self.movies.columns and "sentiment_score" in self.movies.columns:
            genre_set = set(genre_names)
            genre_mask = self.movies["genre_names"].apply(
                lambda gs: isinstance(gs, list) and bool(genre_set & set(gs))
            )
            same_genre = self.movies[genre_mask & (self.movies[self.id_col] != seed_movie_id)]
            same_genre = same_genre.sort_values("sentiment_score", ascending=False)
            add_row(
                "Highly Rated on Our Sentiment Engine",
                "Same genre, highest positive-sentiment scores",
                same_genre[self.id_col].head(per_row_limit).tolist(),
            )

        logger.info("Built %d discovery rows for seed_movie_id=%s", len(rows), seed_movie_id)
        return rows


if __name__ == "__main__":
    logger.info(
        "FusionEngine requires trained CF, ContentBasedFilter, and a populated "
        "SemanticSearchEngine. Wire these up in serving/api.py."
    )
