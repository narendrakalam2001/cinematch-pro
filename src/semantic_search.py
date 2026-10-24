"""
Semantic search module (Module 2C) for CineMatch Pro.

Embeds every movie's plot overview with Sentence-BERT
(`all-MiniLM-L6-v2`) and stores the vectors in a persistent ChromaDB
collection, enabling natural-language queries like "Thriller with
unreliable narrator and plot twist" to match movies by meaning rather
than exact keyword overlap.
"""

from __future__ import annotations

import chromadb
import pandas as pd
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from src.config import CHROMADB_DIR, logger, settings


from src.errors import SemanticSearchError  # noqa: E402,F401  (re-exported: defined in src/errors.py)


class SemanticSearchEngine:
    """SBERT + ChromaDB powered natural-language movie search engine.

    Attributes:
        client: Persistent ChromaDB client backed by `CHROMADB_DIR`.
        collection: The ``"movies"`` ChromaDB collection, created lazily.
        embedding_fn: Sentence-Transformers embedding function used both
            to ingest movie descriptions and to embed queries.
    """

    def __init__(
        self,
        persist_dir: str = str(CHROMADB_DIR),
        model_name: str = settings.semantic_search.sbert_model_name,
        collection_name: str = settings.semantic_search.collection_name,
        device: str | None = None,
    ) -> None:
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        logger.info("Initializing ChromaDB client at %s (embedding device=%s)", persist_dir, device)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name, device=device
        )
        self.collection_name = collection_name
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )

    def ingest_movies(
        self,
        movies: pd.DataFrame,
        id_col: str = "movieId",
        text_col: str = "combined_features",
        title_col: str = "clean_title",
        batch_size: int = 256,
    ) -> int:
        """Embed and upsert movie descriptions into the ChromaDB collection.

        Args:
            movies: DataFrame with at least `id_col` and `text_col`
                columns. `title_col`, when present, is stored as metadata
                for quick display without a separate lookup.
            id_col: Column holding each movie's unique identifier.
            text_col: Column holding the text to embed (plot overview /
                combined features).
            title_col: Optional column with the display title, stored as
                Chroma metadata.
            batch_size: Number of movies to upsert per Chroma call.

        Returns:
            int: Number of movies successfully ingested.

        Raises:
            SemanticSearchError: If `movies` is empty or missing required
                columns.
        """
        if movies.empty:
            raise SemanticSearchError("Cannot ingest an empty movies DataFrame.")
        for col in (id_col, text_col):
            if col not in movies.columns:
                raise SemanticSearchError(f"Missing required column '{col}'.")

        df = movies[movies[text_col].fillna("").astype(str).str.strip() != ""].copy()
        logger.info("Ingesting %d movie descriptions into ChromaDB collection '%s'.", len(df), self.collection_name)

        total = 0
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]
            ids = [str(x) for x in chunk[id_col].tolist()]
            documents = chunk[text_col].astype(str).tolist()
            metadatas = [
                {"movie_id": int(mid), "title": str(title) if title_col in chunk.columns else ""}
                for mid, title in zip(
                    chunk[id_col].tolist(),
                    chunk[title_col].tolist() if title_col in chunk.columns else [""] * len(chunk),
                )
            ]
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            total += len(chunk)

        logger.info("Ingested %d movies into ChromaDB.", total)
        return total

    def search(
        self, query: str, top_k: int = settings.semantic_search.top_k
    ) -> list[dict[str, object]]:
        """Search for movies matching a natural-language query by meaning.

        This is the engine behind Module 2C's "Natural language
        recommendation" (e.g. "Thriller with twist ending like Memento").

        Args:
            query: Free-text natural-language description of what the
                user wants to watch.
            top_k: Number of results to return.

        Returns:
            list[dict[str, object]]: Each dict has keys ``movie_id``,
            ``title``, ``distance`` (lower = more similar), and
            ``similarity_pct`` (0-100, higher = more similar).

        Raises:
            SemanticSearchError: If `query` is empty or the collection is
                empty.
        """
        if not query or not query.strip():
            raise SemanticSearchError("Cannot search with an empty query.")
        if self.collection.count() == 0:
            raise SemanticSearchError(
                "ChromaDB collection is empty. Run scripts/ingest_descriptions.py first."
            )

        k = min(top_k, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=k)

        formatted: list[dict[str, object]] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for doc_id, distance, meta in zip(ids, distances, metadatas):
            # Chroma's default distance is squared L2 on normalized
            # embeddings, roughly in [0, 4]; convert to an intuitive
            # 0-100% similarity score for the UI.
            similarity_pct = round(max(0.0, 1 - (distance / 2)) * 100, 1)
            formatted.append(
                {
                    "movie_id": int(meta.get("movie_id", doc_id)),
                    "title": meta.get("title", ""),
                    "distance": round(float(distance), 4),
                    "similarity_pct": similarity_pct,
                }
            )
        return formatted

    def count(self) -> int:
        """Return the number of movies currently indexed.

        Returns:
            int: Document count in the ChromaDB collection.
        """
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (used before a full re-ingest)."""
        logger.warning("Resetting ChromaDB collection '%s'.", self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=self.embedding_fn
        )


if __name__ == "__main__":
    logger.info("Run scripts/ingest_descriptions.py to populate the ChromaDB collection.")
