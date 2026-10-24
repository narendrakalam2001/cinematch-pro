"""Exception types shared across engines.

They live here (not in the engine modules) so code can catch them without importing the
engines themselves — the engines pull in PyTorch / ChromaDB / scikit-surprise, which the
memory-light "lite" deployment must never load. The engine modules re-export these names,
so ``from src.semantic_search import SemanticSearchError`` keeps working.
"""

from __future__ import annotations


class CollaborativeFilterError(RuntimeError):
    """Raised when the collaborative filter fails to train, load, or predict."""


class SemanticSearchError(RuntimeError):
    """Raised when semantic search (SBERT + ChromaDB) fails."""


class VisualSearchError(RuntimeError):
    """Raised when visual search (CLIP + FAISS) fails."""


class AspectSentimentError(RuntimeError):
    """Raised when aspect-based sentiment analysis fails."""


class SentimentEngineError(RuntimeError):
    """Raised when the sentiment model fails to load or predict."""
