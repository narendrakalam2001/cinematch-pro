"""
CLI entry point: train (and optionally grid-search) the SVD collaborative
filtering model on MovieLens ratings.

Usage:
    python -m scripts.build_cf_model
    python -m scripts.build_cf_model --grid-search
    python -m scripts.build_cf_model --nrows 500000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.collaborative_filter import CollaborativeFilter, CollaborativeFilterError
from src.config import SVD_MODEL_PATH, logger
from src.data_loader import DatasetNotFoundError, load_movielens_ratings
from src.model_card import update_model_card
from src.preprocessing import clean_ratings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CF training script.

    Returns:
        argparse.Namespace: Parsed arguments controlling row count,
        whether to run hyperparameter grid search, and an optional
        override for the ratings file location.
    """
    parser = argparse.ArgumentParser(description="Train CineMatch Pro's collaborative filter.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional: cap number of ratings rows loaded, for a quick local run.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run hyperparameter grid search before training the final model.",
    )
    parser.add_argument(
        "--ratings-path",
        type=str,
        default=None,
        help=(
            "Path to the MovieLens ratings file (.csv/.xls/.xlsx), e.g. "
            r'"D:\Data_Science_Capstone_Datasets\cinematch-pro-complete\ml-25m\ratings.csv". '
            "Defaults to data/raw/ml-25m/ratings.csv if not given."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the collaborative filter training pipeline.

    Returns:
        int: Process exit code (0 on success, 1 on failure).
    """
    args = parse_args()

    try:
        if args.ratings_path:
            raw_ratings = load_movielens_ratings(path=Path(args.ratings_path), nrows=args.nrows)
        else:
            raw_ratings = load_movielens_ratings(nrows=args.nrows)
    except DatasetNotFoundError as exc:
        logger.error("Cannot train collaborative filter: %s", exc)
        return 1

    ratings_df = clean_ratings(raw_ratings)
    cf = CollaborativeFilter()

    n_factors, n_epochs, lr_all = 100, 20, 0.005
    if args.grid_search:
        best_params = cf.grid_search(ratings_df)
        n_factors = best_params["n_factors"]
        n_epochs = best_params["n_epochs"]
        lr_all = best_params["lr_all"]

    try:
        rmse = cf.fit(ratings_df, n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all)
    except CollaborativeFilterError as exc:
        logger.error("Collaborative filter training failed: %s", exc)
        return 1

    cf.save(SVD_MODEL_PATH)
    update_model_card(
        "collaborative_filter",
        {
            "rmse": round(float(rmse), 4),
            "n_factors": n_factors,
            "n_epochs": n_epochs,
            "lr_all": lr_all,
            "grid_search": bool(args.grid_search),
            "ratings_used": int(len(ratings_df)),
            "users": int(ratings_df["userId"].nunique()),
            "movies": int(ratings_df["movieId"].nunique()),
        },
    )
    logger.info("Collaborative filter trained. RMSE=%.4f. Saved to %s", rmse, SVD_MODEL_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
