"""Tests for src/collaborative_filter.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.collaborative_filter import CollaborativeFilter, CollaborativeFilterError


@pytest.fixture
def dense_ratings() -> pd.DataFrame:
    """A denser synthetic ratings table (enough interactions for a stable SVD fit)."""
    rng = np.random.default_rng(42)
    rows = []
    for user in range(1, 31):
        for movie in range(1, 16):
            if rng.random() < 0.4:
                rows.append((user, movie, float(rng.choice([1, 2, 3, 3.5, 4, 4.5, 5]))))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def test_fit_returns_positive_rmse(dense_ratings: pd.DataFrame):
    """Training returns a finite, positive RMSE on the held-out split."""
    cf = CollaborativeFilter()
    rmse = cf.fit(dense_ratings, n_factors=10, n_epochs=10, lr_all=0.01)
    assert rmse > 0


def test_fit_on_empty_ratings_raises():
    """Fitting on an empty ratings DataFrame raises rather than crashing deep in surprise."""
    with pytest.raises(CollaborativeFilterError):
        CollaborativeFilter().fit(pd.DataFrame(columns=["userId", "movieId", "rating"]))


def test_predict_rating_within_valid_range(dense_ratings: pd.DataFrame):
    """Predicted ratings stay within MovieLens's valid [0.5, 5.0] scale."""
    cf = CollaborativeFilter()
    cf.fit(dense_ratings, n_factors=10, n_epochs=10, lr_all=0.01)
    pred = cf.predict_rating(1, 1)
    assert 0.5 <= pred <= 5.0


def test_predict_before_fit_raises():
    """Calling predict_rating before fit()/load() raises CollaborativeFilterError."""
    with pytest.raises(CollaborativeFilterError):
        CollaborativeFilter().predict_rating(1, 1)


def test_recommend_for_user_returns_sorted_top_k(dense_ratings: pd.DataFrame):
    """recommend_for_user returns exactly top_k results, sorted by predicted rating."""
    cf = CollaborativeFilter()
    cf.fit(dense_ratings, n_factors=10, n_epochs=10, lr_all=0.01)
    recs = cf.recommend_for_user(1, list(range(1, 16)), top_k=5)
    assert len(recs) == 5
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


def test_save_and_load_round_trip(dense_ratings: pd.DataFrame, tmp_path: Path):
    """A saved model, when reloaded, produces identical predictions."""
    cf = CollaborativeFilter()
    cf.fit(dense_ratings, n_factors=10, n_epochs=10, lr_all=0.01)
    pred_before = cf.predict_rating(1, 1)

    path = tmp_path / "svd.pkl"
    cf.save(path)
    reloaded = CollaborativeFilter().load(path)
    pred_after = reloaded.predict_rating(1, 1)
    assert pred_before == pytest.approx(pred_after)


def test_load_missing_file_raises(tmp_path: Path):
    """Loading from a path with no saved model raises CollaborativeFilterError."""
    with pytest.raises(CollaborativeFilterError):
        CollaborativeFilter().load(tmp_path / "does_not_exist.pkl")
