"""
Visual search module (Module 1) for CineMatch Pro.

Uses a pretrained CLIP model (zero-shot, no fine-tuning needed) to embed
movie posters and free-text queries into a shared 512-dim space, indexed
with FAISS for fast cosine-similarity search. Supports both:

* Image -> similar posters ("upload a poster, find similar movies")
* Text -> similar posters ("dark moody thriller poster" -> matching posters)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

from src.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    logger,
    settings,
)



def _to_feature_tensor(output: Any) -> "torch.Tensor":
    """Normalize CLIP feature output to a plain tensor.

    Newer ``transformers`` releases may return a ``BaseModelOutputWithPooling``
    from ``get_image_features`` / ``get_text_features`` instead of a tensor.

    Args:
        output: Tensor or model-output object.

    Returns:
        torch.Tensor: Projected embeddings of shape ``(batch, dim)``.
    """
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    return output[0]


from src.errors import VisualSearchError  # noqa: E402,F401  (re-exported: defined in src/errors.py)


class VisualSearchEngine:
    """CLIP + FAISS powered poster similarity search engine.

    Attributes:
        model: Pretrained `CLIPModel` used to embed images and text.
        processor: Matching `CLIPProcessor` for preprocessing inputs.
        index: FAISS `IndexFlatIP` (inner product == cosine similarity on
            normalized vectors), or `None` before `build_index`/`load`.
        movie_ids: Array mapping FAISS index position -> movie ID.
        device: ``"cuda"`` if available, else ``"cpu"``.
    """

    def __init__(self, model_name: str = settings.visual_search.clip_model_name) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CLIP model '%s' onto device=%s", model_name, self.device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.index: faiss.Index | None = None
        self.movie_ids: np.ndarray | None = None

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image into CLIP's shared embedding space.

        Args:
            image: A PIL Image (any mode; will be converted to RGB).

        Returns:
            np.ndarray: L2-normalized embedding of shape
            ``(embedding_dim,)``.
        """
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = _to_feature_tensor(self.model.get_image_features(**inputs))
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a free-text query into CLIP's shared embedding space.

        Enables text-to-poster search, e.g. "dark moody thriller poster".

        Args:
            text: Free-text description of the desired visual style.

        Returns:
            np.ndarray: L2-normalized embedding of shape
            ``(embedding_dim,)``.

        Raises:
            VisualSearchError: If `text` is empty or whitespace only.
        """
        if not text or not text.strip():
            raise VisualSearchError("Cannot embed an empty text query.")
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        features = _to_feature_tensor(self.model.get_text_features(**inputs))
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype("float32")

    def _embed_image_batch(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of poster images efficiently.

        Args:
            images: List of PIL images (already loaded).
            batch_size: Number of images per forward pass.

        Returns:
            np.ndarray: L2-normalized embeddings of shape ``(n, dim)``.
        """
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            chunk = [img.convert("RGB") for img in images[start : start + batch_size]]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = _to_feature_tensor(self.model.get_image_features(**inputs))
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy().astype("float32"))
        return np.vstack(all_embeddings)

    def build_index(
        self, poster_paths: list[Path], movie_ids: list[int], batch_size: int = 64
    ) -> None:
        """Build the FAISS index from poster image files, streaming in batches.

        Images are opened, converted and closed batch by batch, so memory
        and open-file-handle usage stay flat no matter how many posters
        are indexed (40k+ is routine).

        Args:
            poster_paths: Filesystem paths to poster image files, one per
                movie, aligned with `movie_ids`.
            movie_ids: Movie IDs aligned 1:1 with `poster_paths`.
            batch_size: Number of posters embedded per forward pass.

        Raises:
            VisualSearchError: If `poster_paths` and `movie_ids` have
                mismatched lengths, or if no valid images could be loaded.
        """
        if len(poster_paths) != len(movie_ids):
            raise VisualSearchError(
                f"poster_paths ({len(poster_paths)}) and movie_ids "
                f"({len(movie_ids)}) must be the same length."
            )

        total = len(poster_paths)
        logger.info("Building FAISS index from %d poster images.", total)
        index: faiss.Index | None = None
        kept_ids: list[int] = []
        skipped = 0

        for batch_no, start in enumerate(range(0, total, batch_size), start=1):
            images: list[Image.Image] = []
            ids: list[int] = []
            for path, movie_id in zip(
                poster_paths[start : start + batch_size], movie_ids[start : start + batch_size]
            ):
                try:
                    with Image.open(path) as handle:
                        images.append(handle.convert("RGB"))
                    ids.append(int(movie_id))
                except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                    logger.warning("Skipping unreadable poster %s: %s", path, exc)
                    skipped += 1
            if not images:
                continue

            embeddings = self._embed_image_batch(images, batch_size=len(images))
            if index is None:
                index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            kept_ids.extend(ids)
            if batch_no % 50 == 0:
                logger.info("Embedded %d/%d posters.", min(start + batch_size, total), total)

        if index is None:
            raise VisualSearchError("No valid poster images could be loaded to build the index.")
        if skipped:
            logger.warning("Skipped %d unreadable posters out of %d.", skipped, total)

        self.index = index
        self.movie_ids = np.array(kept_ids)
        logger.info("FAISS index built: %d vectors, dim=%d.", index.ntotal, index.d)

    def search_by_image(
        self, image: Image.Image, top_k: int = settings.visual_search.top_k
    ) -> list[tuple[int, float]]:
        """Find the most visually similar posters to an uploaded image.

        Args:
            image: The uploaded poster / reference image.
            top_k: Number of similar movies to return.

        Returns:
            list[tuple[int, float]]: ``(movie_id, similarity_pct)`` pairs
            sorted by similarity descending. ``similarity_pct`` is cosine
            similarity expressed as a 0-100 percentage.

        Raises:
            VisualSearchError: If the index has not been built/loaded.
        """
        self._ensure_index()
        query_vec = self.embed_image(image).reshape(1, -1)
        return self._search(query_vec, top_k)

    def search_by_text(
        self, text: str, top_k: int = settings.visual_search.top_k
    ) -> list[tuple[int, float]]:
        """Find posters visually matching a free-text style description.

        Args:
            text: Free-text query, e.g. "dark moody thriller poster".
            top_k: Number of similar movies to return.

        Returns:
            list[tuple[int, float]]: ``(movie_id, similarity_pct)`` pairs
            sorted by similarity descending.

        Raises:
            VisualSearchError: If the index has not been built/loaded.
        """
        self._ensure_index()
        query_vec = self.embed_text(text).reshape(1, -1)
        return self._search(query_vec, top_k)

    def _search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Run the actual FAISS search and format results.

        Args:
            query_vec: Normalized query embedding of shape ``(1, dim)``.
            top_k: Number of results to return.

        Returns:
            list[tuple[int, float]]: ``(movie_id, similarity_pct)`` pairs.
        """
        k = min(top_k, self.index.ntotal)
        similarities, indices = self.index.search(query_vec, k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            movie_id = int(self.movie_ids[idx])
            # Cosine similarity in [-1, 1] on normalized vectors -> 0-100%.
            similarity_pct = round(float(max(sim, 0.0)) * 100, 1)
            results.append((movie_id, similarity_pct))
        return results

    def _ensure_index(self) -> None:
        """Raise if the FAISS index has not been built or loaded yet.

        Raises:
            VisualSearchError: If `index` is `None`.
        """
        if self.index is None or self.movie_ids is None:
            raise VisualSearchError("FAISS index not built/loaded. Call build_index() or load().")

    def save(
        self, index_path: Path = FAISS_INDEX_PATH, metadata_path: Path = FAISS_METADATA_PATH
    ) -> None:
        """Persist the FAISS index and movie-ID mapping to disk.

        Args:
            index_path: Destination for the FAISS binary index file.
            metadata_path: Destination for the pickled movie-ID mapping.

        Raises:
            VisualSearchError: If the index has not been built.
        """
        self._ensure_index()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump({"movie_ids": self.movie_ids}, f)
        logger.info("Saved FAISS index (%d vectors) to %s", self.index.ntotal, index_path)

    def load(
        self, index_path: Path = FAISS_INDEX_PATH, metadata_path: Path = FAISS_METADATA_PATH
    ) -> "VisualSearchEngine":
        """Load a previously built FAISS index and movie-ID mapping.

        Args:
            index_path: Source path of the FAISS binary index file.
            metadata_path: Source path of the pickled movie-ID mapping.

        Returns:
            VisualSearchEngine: ``self``, for chained construction.

        Raises:
            VisualSearchError: If either file is missing or unreadable.
        """
        if not index_path.exists() or not metadata_path.exists():
            raise VisualSearchError(
                f"Saved FAISS index not found at {index_path} / {metadata_path}"
            )
        try:
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, "rb") as f:
                self.movie_ids = pickle.load(f)["movie_ids"]
        except (RuntimeError, pickle.UnpicklingError, KeyError) as exc:
            raise VisualSearchError(f"Failed to load FAISS index: {exc}") from exc
        logger.info("Loaded FAISS index with %d vectors from %s", self.index.ntotal, index_path)
        return self


if __name__ == "__main__":
    logger.info(
        "Run scripts/build_faiss_index.py to build the poster index from data/posters/."
    )
