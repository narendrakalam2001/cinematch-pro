"""
Collaborative filtering module (Module 2A) for CineMatch Pro.

Trains a matrix-factorization (SVD) model on MovieLens ratings using the
`surprise` library, with grid search over factors/epochs/learning-rate,
and exposes prediction + "users also watched" style retrieval on top of
the trained model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.accuracy import rmse as surprise_rmse
from surprise.model_selection import GridSearchCV, train_test_split

from src.config import SVD_MODEL_PATH, logger, settings


from src.errors import CollaborativeFilterError  # noqa: E402,F401  (re-exported: defined in src/errors.py)


class CollaborativeFilter:
    """SVD-based collaborative filtering recommender.

    Wraps a `surprise.SVD` model with training, evaluation, persistence,
    and inference helpers tailored to CineMatch Pro's needs (predicting a
    user's rating for a movie, and finding similar users' top picks).

    Attributes:
        model: The underlying trained `surprise.SVD` instance, or `None`
            before training / loading.
        trainset: The full `surprise.Trainset` built during `fit`, kept
            around so we can map raw <-> inner ids for lookups.
    """

    def __init__(self) -> None:
        self.model: SVD | None = None
        self.trainset = None

    def _build_dataset(self, ratings: pd.DataFrame) -> Dataset:
        """Convert a ratings DataFrame into a `surprise.Dataset`.

        Args:
            ratings: DataFrame with ``userId``, ``movieId``, ``rating``
                columns (MovieLens scale 0.5-5.0).

        Returns:
            surprise.Dataset: Dataset ready for `surprise` model training.
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        return Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

    def grid_search(self, ratings: pd.DataFrame) -> dict[str, object]:
        """Run grid search over SVD hyperparameters to find the best config.

        Args:
            ratings: Cleaned ratings DataFrame (see `preprocessing.clean_ratings`).

        Returns:
            dict[str, object]: The best hyperparameter combination found,
            with keys ``n_factors``, ``n_epochs``, ``lr_all``.
        """
        cfg = settings.collaborative
        param_grid = {
            "n_factors": list(cfg.n_factors_grid),
            "n_epochs": list(cfg.n_epochs_grid),
            "lr_all": list(cfg.lr_grid),
        }
        logger.info("Starting SVD grid search over: %s", param_grid)
        data = self._build_dataset(ratings)
        gs = GridSearchCV(
            SVD, param_grid, measures=["rmse"], cv=cfg.cv_folds, n_jobs=-1
        )
        gs.fit(data)
        best_params = gs.best_params["rmse"]
        logger.info(
            "Grid search complete. Best RMSE=%.4f with params=%s",
            gs.best_score["rmse"],
            best_params,
        )
        return best_params

    def fit(
        self,
        ratings: pd.DataFrame,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        test_size: float = 0.2,
    ) -> float:
        """Train the SVD model and report held-out RMSE.

        Args:
            ratings: Cleaned ratings DataFrame.
            n_factors: Number of latent factors.
            n_epochs: Number of SGD training epochs.
            lr_all: Learning rate for all trained parameters.
            test_size: Fraction of ratings held out for RMSE evaluation.

        Returns:
            float: RMSE on the held-out test split.

        Raises:
            CollaborativeFilterError: If `ratings` is empty.
        """
        if ratings.empty:
            raise CollaborativeFilterError("Cannot fit collaborative filter on empty ratings.")

        data = self._build_dataset(ratings)
        trainset, testset = train_test_split(
            data, test_size=test_size, random_state=settings.collaborative.random_state
        )

        logger.info(
            "Training SVD: n_factors=%d, n_epochs=%d, lr_all=%.4f on %d ratings.",
            n_factors,
            n_epochs,
            lr_all,
            trainset.n_ratings,
        )
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            random_state=settings.collaborative.random_state,
        )
        self.model.fit(trainset)
        self.trainset = trainset

        predictions = self.model.test(testset)
        score = surprise_rmse(predictions, verbose=False)
        logger.info("SVD training complete. Test RMSE=%.4f (target <%.2f)", score, settings.collaborative.target_rmse)
        if score > settings.collaborative.target_rmse:
            logger.warning(
                "RMSE %.4f exceeds target %.4f — consider grid search or more data.",
                score,
                settings.collaborative.target_rmse,
            )
        return score

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict a single user's rating for a single movie.

        Args:
            user_id: Raw MovieLens ``userId``.
            movie_id: Raw MovieLens ``movieId``.

        Returns:
            float: Predicted rating, clipped to [0.5, 5.0] by `surprise`.

        Raises:
            CollaborativeFilterError: If the model has not been trained
                or loaded yet.
        """
        if self.model is None:
            raise CollaborativeFilterError("Model not trained/loaded. Call fit() or load().")
        return self.model.predict(user_id, movie_id).est

    def recommend_for_user(
        self, user_id: int, candidate_movie_ids: list[int], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Rank a set of candidate movies for a user by predicted rating.

        Used for "Users like you also liked..." and cold-start-safe
        re-ranking within the fusion layer.

        Args:
            user_id: Raw MovieLens ``userId``. Unseen users fall back to
                `surprise`'s global-mean baseline, so this degrades
                gracefully rather than erroring (cold start).
            candidate_movie_ids: Pool of movie IDs to score and rank.
            top_k: Number of top recommendations to return.

        Returns:
            list[tuple[int, float]]: ``(movie_id, predicted_rating)``
            pairs, sorted by predicted rating descending.

        Raises:
            CollaborativeFilterError: If the model has not been trained
                or loaded yet.
        """
        if self.model is None:
            raise CollaborativeFilterError("Model not trained/loaded. Call fit() or load().")
        scored = [
            (movie_id, self.predict_rating(user_id, movie_id))
            for movie_id in candidate_movie_ids
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def save(self, path: Path = SVD_MODEL_PATH) -> None:
        """Persist the trained model to disk via pickle.

        Args:
            path: Destination path for the pickled model.

        Raises:
            CollaborativeFilterError: If the model has not been trained.
        """
        if self.model is None:
            raise CollaborativeFilterError("Nothing to save — model has not been trained.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model}, f)
        logger.info("Saved SVD model to %s", path)

    def load(self, path: Path = SVD_MODEL_PATH) -> "CollaborativeFilter":
        """Load a previously trained model from disk.

        Args:
            path: Source path of the pickled model.

        Returns:
            CollaborativeFilter: ``self``, for chained construction, e.g.
            ``cf = CollaborativeFilter().load(path)``.

        Raises:
            CollaborativeFilterError: If the file does not exist or is
                corrupt/unreadable.
        """
        if not path.exists():
            raise CollaborativeFilterError(f"No saved model found at {path}")
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            self.model = payload["model"]
        except (pickle.UnpicklingError, KeyError, EOFError) as exc:
            raise CollaborativeFilterError(f"Failed to load model from {path}: {exc}") from exc
        logger.info("Loaded SVD model from %s", path)
        return self


if __name__ == "__main__":
    from src.data_loader import load_movielens_ratings
    from src.preprocessing import clean_ratings

    raw_ratings = load_movielens_ratings(nrows=500_000)
    ratings_df = clean_ratings(raw_ratings)

    cf = CollaborativeFilter()
    cf.fit(ratings_df)
    cf.save()
