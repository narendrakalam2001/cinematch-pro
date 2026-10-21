"""
CLI entry point: fine-tune the RoBERTa sentiment model on IMDB reviews.

Usage:
    python -m scripts.train_sentiment
    python -m scripts.train_sentiment --sample 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ROBERTA_MODEL_DIR, logger
from src.config import settings
from src.data_loader import DatasetNotFoundError, load_imdb_reviews
from src.model_card import update_model_card
from src.sentiment_engine import SentimentEngine, SentimentEngineError


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sentiment training script.

    Returns:
        argparse.Namespace: Parsed arguments, with ``sample`` optionally
        set to cap the training set size for fast local runs, and
        ``reviews_path`` optionally overriding where the IMDB reviews
        file is read from (accepts .csv, .tsv, .xls, .xlsx).
    """
    parser = argparse.ArgumentParser(description="Fine-tune CineMatch Pro's sentiment model.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: cap number of reviews used, for a quick smoke-test run.",
    )
    parser.add_argument(
        "--reviews-path",
        type=str,
        default=None,
        help=(
            "Path to the IMDB reviews file (.csv/.tsv/.xls/.xlsx), e.g. "
            r'"D:\Data_Science_Capstone_Datasets\cinematch-pro-complete\IMDB Dataset.xls". '
            "Defaults to data/raw/imdb_reviews_50k.csv if not given. The file must have "
            "'review' and 'sentiment' columns."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the sentiment fine-tuning pipeline.

    Returns:
        int: Process exit code (0 on success, 1 on failure).
    """
    args = parse_args()

    try:
        if args.reviews_path:
            reviews_df = load_imdb_reviews(path=Path(args.reviews_path))
        else:
            reviews_df = load_imdb_reviews()
    except DatasetNotFoundError as exc:
        logger.error("Cannot train sentiment model: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Reviews file has an unexpected format: %s", exc)
        return 1

    if args.sample:
        reviews_df = reviews_df.sample(n=min(args.sample, len(reviews_df)), random_state=42)
        logger.info("Sampled down to %d reviews for this run.", len(reviews_df))

    engine = SentimentEngine()
    try:
        metrics = engine.fine_tune(reviews_df, output_dir=ROBERTA_MODEL_DIR)
    except SentimentEngineError as exc:
        logger.error("Sentiment fine-tuning failed: %s", exc)
        return 1

    update_model_card(
        "sentiment",
        {
            **metrics,
            "base_model": settings.sentiment.base_model,
            "epochs": settings.sentiment.epochs,
            "batch_size": settings.sentiment.batch_size,
            "learning_rate": settings.sentiment.learning_rate,
            "reviews_used": int(len(reviews_df)),
            "target_accuracy": settings.sentiment.target_accuracy,
        },
    )
    logger.info("Training complete. Metrics: %s", metrics)
    logger.info("Model saved to: %s", ROBERTA_MODEL_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
