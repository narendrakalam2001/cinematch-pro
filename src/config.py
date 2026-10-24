"""
Central configuration module for CineMatch Pro.

Loads environment variables (via python-dotenv), defines project-wide
paths, model hyperparameters, and the shared logger used across all
modules. Every other module in `src/` should import settings from here
rather than reading `os.environ` directly, so that configuration stays
in one place and is easy to test / override.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
# Loads a local .env file if present (see .env.example). In production
# (Docker / Render) real environment variables are injected directly and
# this call is a harmless no-op if no .env file exists.
load_dotenv()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _build_logger() -> logging.Logger:
    """Configure and return the shared application logger.

    Returns:
        logging.Logger: A logger named ``cinematch`` writing to stdout
        with a timestamped, leveled format. Safe to call multiple times;
        handlers are only attached once.
    """
    logger = logging.getLogger("cinematch")
    if logger.handlers:
        # Avoid duplicate handlers if _build_logger() is called more than
        # once (e.g. in tests or interactive notebooks).
        return logger

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger: Final[logging.Logger] = _build_logger()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
POSTERS_DIR: Final[Path] = DATA_DIR / "posters"

MODELS_DIR: Final[Path] = BASE_DIR / "models"
CHROMADB_DIR: Final[Path] = BASE_DIR / "chromadb_store"

SVD_MODEL_PATH: Final[Path] = MODELS_DIR / "svd_model.pkl"
TFIDF_MATRIX_PATH: Final[Path] = MODELS_DIR / "tfidf_matrix.pkl"
TFIDF_VECTORIZER_PATH: Final[Path] = MODELS_DIR / "tfidf_vectorizer.pkl"
ROBERTA_MODEL_DIR: Final[Path] = MODELS_DIR / "roberta_sentiment"
FAISS_INDEX_PATH: Final[Path] = MODELS_DIR / "faiss_poster.index"
FAISS_METADATA_PATH: Final[Path] = MODELS_DIR / "faiss_poster_metadata.pkl"
MODEL_CARD_PATH: Final[Path] = MODELS_DIR / "model_card.json"

MOVIELENS_RATINGS_CSV: Final[Path] = RAW_DATA_DIR / "ml-25m" / "ratings.csv"
MOVIELENS_MOVIES_CSV: Final[Path] = RAW_DATA_DIR / "ml-25m" / "movies.csv"
MOVIELENS_LINKS_CSV: Final[Path] = RAW_DATA_DIR / "ml-25m" / "links.csv"
TMDB_METADATA_CSV: Final[Path] = RAW_DATA_DIR / "tmdb_5000_movies.csv"
TMDB_CREDITS_CSV: Final[Path] = RAW_DATA_DIR / "tmdb_5000_credits.csv"
TMDB_FULL_METADATA_CSV: Final[Path] = RAW_DATA_DIR / "tmdb_full_metadata.csv"
IMDB_REVIEWS_CSV: Final[Path] = RAW_DATA_DIR / "imdb_reviews_50k.csv"
POSTERS_METADATA_CSV: Final[Path] = RAW_DATA_DIR / "poster_genre_metadata.csv"

MERGED_MOVIES_PARQUET: Final[Path] = PROCESSED_DATA_DIR / "merged_movies.parquet"
CLEANED_RATINGS_PARQUET: Final[Path] = PROCESSED_DATA_DIR / "cleaned_ratings.parquet"

for _dir in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    POSTERS_DIR,
    MODELS_DIR,
    CHROMADB_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API keys / secrets (read from environment; never hardcode secrets)
# ---------------------------------------------------------------------------
TMDB_API_KEY: Final[str | None] = os.getenv("TMDB_API_KEY")
OMDB_API_KEY: Final[str | None] = os.getenv("OMDB_API_KEY")
GEMINI_API_KEY: Final[str | None] = os.getenv("GEMINI_API_KEY")
# Gemini model ids are retired regularly; override in .env without touching code.
# Lite mode: skip the heavy engines (PyTorch / CLIP / RoBERTa / SBERT / ChromaDB / SVD) so the app fits
# in ~1 GB of RAM (free Streamlit Community Cloud). Visual search, plot-semantic search, collaborative
# filtering and review sentiment are unavailable; search, movie cards, content-based rows and CineBot work.
LITE_MODE: Final[bool] = os.getenv("CINEMATCH_LITE", "").strip().lower() in {"1", "true", "yes", "on"}
GEMINI_MODEL: Final[str] = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

TMDB_BASE_URL: Final[str] = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL: Final[str] = "https://image.tmdb.org/t/p"
OMDB_BASE_URL: Final[str] = "https://www.omdbapi.com/"


def require_api_key(name: str, value: str | None) -> str:
    """Validate that a required API key is present.

    Args:
        name: Human-readable name of the key, used in the error message.
        value: The key value as read from the environment.

    Returns:
        str: The validated, non-empty API key.

    Raises:
        RuntimeError: If the key is missing, so failures happen loudly at
            startup instead of causing confusing downstream HTTP errors.
    """
    if not value:
        raise RuntimeError(
            f"Missing required API key: {name}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value


# ---------------------------------------------------------------------------
# Model / algorithm hyperparameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SentimentConfig:
    """Hyperparameters for the RoBERTa sentiment fine-tuning pipeline."""

    base_model: str = "cardiffnlp/twitter-roberta-base-sentiment"
    max_length: int = 256
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    test_size: float = 0.2
    random_state: int = 42
    target_accuracy: float = 0.93


@dataclass(frozen=True)
class CollaborativeFilterConfig:
    """Hyperparameters and grid-search space for the SVD collaborative
    filtering model (trained via the `surprise` library)."""

    n_factors_grid: tuple[int, ...] = (50, 100, 200)
    n_epochs_grid: tuple[int, ...] = (20, 50)
    lr_grid: tuple[float, ...] = (0.005, 0.01)
    cv_folds: int = 3
    random_state: int = 42
    target_rmse: float = 0.87


@dataclass(frozen=True)
class VisualSearchConfig:
    """Configuration for the CLIP + FAISS visual search module."""

    clip_model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    top_k: int = 10


@dataclass(frozen=True)
class SemanticSearchConfig:
    """Configuration for the SBERT + ChromaDB semantic search module."""

    sbert_model_name: str = "all-MiniLM-L6-v2"
    collection_name: str = "movies"
    top_k: int = 20


@dataclass(frozen=True)
class FusionConfig:
    """Weights used by the hybrid recommendation fusion layer.

    Weights must sum to 1.0; this is validated in `__post_init__`.
    """

    collaborative_weight: float = 0.4
    content_weight: float = 0.3
    semantic_weight: float = 0.3

    def __post_init__(self) -> None:
        total = self.collaborative_weight + self.content_weight + self.semantic_weight
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Fusion weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class CacheConfig:
    """Simple in-memory caching configuration used as a Redis stand-in."""

    ttl_seconds: int = 60 * 30  # 30 minutes
    max_entries: int = 2048


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limits respected when calling third-party APIs."""

    tmdb_requests_per_day: int = 1000
    omdb_requests_per_day: int = 1000
    min_seconds_between_requests: float = 0.05


@dataclass(frozen=True)
class AppSettings:
    """Top-level settings bundle, composed of the sub-configs above."""

    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    collaborative: CollaborativeFilterConfig = field(
        default_factory=CollaborativeFilterConfig
    )
    visual_search: VisualSearchConfig = field(default_factory=VisualSearchConfig)
    semantic_search: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))


settings: Final[AppSettings] = AppSettings()

ASPECT_KEYWORDS: Final[dict[str, list[str]]] = {
    "story": ["story", "plot", "narrative", "script", "writing", "screenplay"],
    "acting": ["acting", "performance", "cast", "actor", "actress", "portrayal"],
    "direction": ["direction", "director", "directing", "vision", "pacing"],
    "music": ["music", "score", "soundtrack", "songs", "background score", "bgm"],
    "vfx": ["vfx", "visual effects", "cgi", "graphics", "special effects"],
}

if __name__ == "__main__":
    logger.info("CineMatch Pro configuration loaded successfully.")
    logger.info("Environment: %s", settings.environment)
    logger.info("Base directory: %s", BASE_DIR)
