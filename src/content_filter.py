"""
Content-based filtering module (Module 2B) for CineMatch Pro.

Builds a TF-IDF representation of each movie's `combined_features` text
(overview + genres + keywords + cast + director) and serves
"Because you watched X..." style recommendations via cosine similarity.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from src.config import TFIDF_MATRIX_PATH, TFIDF_VECTORIZER_PATH, logger


class ContentFilterError(RuntimeError):
    """Raised for content-filter-specific failures."""


class ContentBasedFilter:
    """TF-IDF + cosine-similarity content-based movie recommender.

    Attributes:
        vectorizer: Fitted `TfidfVectorizer`, or `None` before `fit`.
        tfidf_matrix: Sparse TF-IDF matrix, one row per movie.
        movie_ids: Array mapping matrix row index -> movie ID, so callers
            can look up recommendations by their domain ID.
    """

    def __init__(self, max_features: int = 20_000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=2,
        )
        self.tfidf_matrix = None
        self.movie_ids: np.ndarray | None = None
        self._id_to_row: dict[int, int] = {}

    def fit(self, movies: pd.DataFrame, id_col: str = "movieId", text_col: str = "combined_features") -> "ContentBasedFilter":
        """Fit the TF-IDF vectorizer on the movie corpus.

        Args:
            movies: Merged movies DataFrame containing at least `id_col`
                and `text_col`.
            id_col: Column holding each movie's unique identifier.
            text_col: Column holding the combined free-text features.

        Returns:
            ContentBasedFilter: ``self``, for chained construction.

        Raises:
            ContentFilterError: If `movies` is empty or missing required
                columns.
        """
        if movies.empty:
            raise ContentFilterError("Cannot fit content filter on empty movies DataFrame.")
        for col in (id_col, text_col):
            if col not in movies.columns:
                raise ContentFilterError(f"Missing required column '{col}' in movies DataFrame.")

        texts = movies[text_col].fillna("").astype(str)
        logger.info("Fitting TF-IDF vectorizer on %d movies.", len(movies))
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.movie_ids = movies[id_col].to_numpy()
        self._id_to_row = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        logger.info(
            "TF-IDF matrix built: shape=%s, vocabulary_size=%d",
            self.tfidf_matrix.shape,
            len(self.vectorizer.vocabulary_),
        )
        return self

    def similar_movies(self, movie_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Find the most similar movies to a given movie by content.

        This is the engine behind "More Like [Movie]" (Row 1) in the UI.

        Args:
            movie_id: The reference movie's ID.
            top_k: Number of similar movies to return.

        Returns:
            list[tuple[int, float]]: ``(movie_id, similarity_score)``
            pairs sorted by similarity descending, excluding the query
            movie itself.

        Raises:
            ContentFilterError: If the filter hasn't been fit, or
                `movie_id` is unknown.
        """
        self._ensure_fitted()
        if movie_id not in self._id_to_row:
            raise ContentFilterError(f"Unknown movie_id: {movie_id}")

        row_idx = self._id_to_row[movie_id]
        sims = linear_kernel(self.tfidf_matrix[row_idx], self.tfidf_matrix).flatten()
        # Exclude the movie itself (always similarity 1.0 to itself).
        sims[row_idx] = -1.0
        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        return [(int(self.movie_ids[i]), float(sims[i])) for i in top_indices if sims[i] > 0]

    def similar_to_query_text(self, query_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Score all movies against an ad-hoc free-text query.

        Useful for simple keyword-style search that doesn't need full
        semantic (SBERT) matching.

        Args:
            query_text: Free-text query, e.g. genre keywords.
            top_k: Number of results to return.

        Returns:
            list[tuple[int, float]]: ``(movie_id, similarity_score)``
            pairs sorted by similarity descending.

        Raises:
            ContentFilterError: If the filter hasn't been fit.
        """
        self._ensure_fitted()
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        return [(int(self.movie_ids[i]), float(sims[i])) for i in top_indices if sims[i] > 0]

    def _ensure_fitted(self) -> None:
        """Raise if the vectorizer / matrix have not been built yet.

        Raises:
            ContentFilterError: If `fit()` has not been called and no
                model has been loaded.
        """
        if self.tfidf_matrix is None or self.movie_ids is None:
            raise ContentFilterError("Content filter not fitted. Call fit() or load() first.")

    def save(
        self,
        matrix_path: Path = TFIDF_MATRIX_PATH,
        vectorizer_path: Path = TFIDF_VECTORIZER_PATH,
    ) -> None:
        """Persist the fitted vectorizer, matrix, and ID mapping to disk.

        Args:
            matrix_path: Destination for the pickled TF-IDF matrix + IDs.
            vectorizer_path: Destination for the pickled vectorizer.

        Raises:
            ContentFilterError: If the filter has not been fitted.
        """
        self._ensure_fitted()
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        with open(matrix_path, "wb") as f:
            pickle.dump({"tfidf_matrix": self.tfidf_matrix, "movie_ids": self.movie_ids}, f)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info("Saved content filter artifacts to %s and %s", matrix_path, vectorizer_path)

    def load(
        self,
        matrix_path: Path = TFIDF_MATRIX_PATH,
        vectorizer_path: Path = TFIDF_VECTORIZER_PATH,
    ) -> "ContentBasedFilter":
        """Load a previously fitted vectorizer, matrix, and ID mapping.

        Args:
            matrix_path: Source path for the pickled TF-IDF matrix + IDs.
            vectorizer_path: Source path for the pickled vectorizer.

        Returns:
            ContentBasedFilter: ``self``, for chained construction.

        Raises:
            ContentFilterError: If either file is missing or corrupt.
        """
        if not matrix_path.exists() or not vectorizer_path.exists():
            raise ContentFilterError(
                f"Saved content filter not found at {matrix_path} / {vectorizer_path}"
            )
        try:
            with open(matrix_path, "rb") as f:
                payload = pickle.load(f)
            self.tfidf_matrix = payload["tfidf_matrix"]
            self.movie_ids = payload["movie_ids"]
            self._id_to_row = {mid: idx for idx, mid in enumerate(self.movie_ids)}
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        except (pickle.UnpicklingError, KeyError, EOFError) as exc:
            raise ContentFilterError(f"Failed to load content filter: {exc}") from exc
        logger.info("Loaded content filter with %d movies.", len(self.movie_ids))
        return self


if __name__ == "__main__":
    import pandas as pd

    from src.config import MERGED_MOVIES_PARQUET

    movies_df = pd.read_parquet(MERGED_MOVIES_PARQUET)
    cf = ContentBasedFilter().fit(movies_df)
    cf.save()
