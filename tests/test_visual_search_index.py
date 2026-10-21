"""Tests for the streaming FAISS index build in src/visual_search.py.

CLIP itself is replaced by a tiny colour-based embedder so the index
mechanics (batching, skipping bad files, id alignment, search) can be
verified without downloading any model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("faiss")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.visual_search import VisualSearchEngine, VisualSearchError  # noqa: E402


class _ColourEngine(VisualSearchEngine):
    """VisualSearchEngine whose 'embedding' is the normalized mean colour."""

    def __init__(self) -> None:  # deliberately skips loading CLIP
        self.index = None
        self.movie_ids = None
        self.device = "cpu"

    @staticmethod
    def _vector(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB").resize((1, 1)), dtype="float32").reshape(3)
        norm = float(np.linalg.norm(rgb))
        return rgb / norm if norm else rgb

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self._vector(image)

    def _embed_image_batch(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        return np.vstack([self._vector(image) for image in images])


def _solid(path: Path, colour: tuple[int, int, int]) -> Path:
    Image.new("RGB", (8, 8), colour).save(path, format="JPEG", quality=100)
    return path


@pytest.fixture
def posters(tmp_path: Path) -> dict[str, Path]:
    return {
        "red": _solid(tmp_path / "red.jpg", (255, 0, 0)),
        "green": _solid(tmp_path / "green.jpg", (0, 255, 0)),
        "blue": _solid(tmp_path / "blue.jpg", (0, 0, 255)),
    }


@pytest.mark.parametrize("batch_size", [1, 2, 64])
def test_build_index_aligns_ids_and_skips_unreadable_files(
    posters: dict[str, Path], tmp_path: Path, batch_size: int
):
    """Corrupt and missing files are skipped without shifting later movie ids."""
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("definitely not an image")
    paths = [posters["red"], corrupt, posters["green"], tmp_path / "missing.jpg", posters["blue"]]
    ids = [11, 12, 13, 14, 15]

    engine = _ColourEngine()
    engine.build_index(paths, ids, batch_size=batch_size)

    assert engine.index.ntotal == 3
    assert engine.movie_ids.tolist() == [11, 13, 15]


def test_search_by_image_returns_matching_poster_first(posters: dict[str, Path]):
    """Searching with a red image returns the red poster's movie id as the best match."""
    engine = _ColourEngine()
    engine.build_index(list(posters.values()), [1, 2, 3], batch_size=2)

    with Image.open(posters["red"]) as query:
        results = engine.search_by_image(query, top_k=2)

    assert results[0][0] == 1
    assert results[0][1] > 99.0


def test_build_index_with_only_unreadable_files_raises(tmp_path: Path):
    """If nothing can be loaded the caller gets a clear error, not an empty index."""
    with pytest.raises(VisualSearchError, match="No valid poster"):
        _ColourEngine().build_index([tmp_path / "nope.jpg"], [1])


def test_build_index_length_mismatch_raises(posters: dict[str, Path]):
    """poster_paths and movie_ids must line up one-to-one."""
    with pytest.raises(VisualSearchError, match="same length"):
        _ColourEngine().build_index(list(posters.values()), [1, 2])
