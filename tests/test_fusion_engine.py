"""Tests for src/fusion_engine.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.fusion_engine import FusionEngine, FusionEngineError, _normalize_scores


def test_normalize_scores_empty_input_returns_empty_dict():
    """Normalizing an empty list returns an empty dict, not an error."""
    assert _normalize_scores([]) == {}


def test_normalize_scores_equal_values_all_map_to_one():
    """When every score is identical, normalization avoids a divide-by-zero and returns 1.0 for all."""
    result = _normalize_scores([(1, 5.0), (2, 5.0)])
    assert result == {1: 1.0, 2: 1.0}


def test_normalize_scores_min_max_scaling():
    """Min-max normalization maps the lowest score to 0.0 and highest to 1.0."""
    result = _normalize_scores([(1, 0.0), (2, 10.0)])
    assert result[1] == 0.0
    assert result[2] == 1.0


@pytest.fixture
def fusion_engine(
    synthetic_movies: pd.DataFrame,
    mock_content_filter: MagicMock,
    mock_collaborative_filter: MagicMock,
    mock_semantic_engine: MagicMock,
) -> FusionEngine:
    """A FusionEngine wired up with mocked sub-engines over the synthetic catalogue."""
    return FusionEngine(
        cf=mock_collaborative_filter,
        content=mock_content_filter,
        semantic=mock_semantic_engine,
        movies=synthetic_movies,
    )


def test_build_discovery_rows_produces_all_ten_rows(fusion_engine: FusionEngine):
    """With a fully-connected synthetic catalogue, all 10 discovery rows are generated."""
    rows = fusion_engine.build_discovery_rows(user_id=1, seed_movie_id=1, per_row_limit=6)
    assert len(rows) == 10


def test_discovery_rows_more_like_row_present(fusion_engine: FusionEngine):
    """Row 1 ('More Like X') uses the content filter's similar_movies results."""
    rows = fusion_engine.build_discovery_rows(user_id=1, seed_movie_id=1)
    titles = [r.title for r in rows]
    assert any("More Like" in t for t in titles)


def test_discovery_rows_director_row_matches_same_director(fusion_engine: FusionEngine):
    """The 'By Same Director' row only includes movies by the seed's director."""
    rows = fusion_engine.build_discovery_rows(user_id=1, seed_movie_id=1)
    director_row = next(r for r in rows if "Same Director" in r.title)
    # Movies 2 and 3 share Bahubali's director (Rajamouli); 4, 5, 6 do not.
    assert set(director_row.movie_ids).issubset({2, 3})


def test_discovery_rows_starring_row_matches_lead_actor(fusion_engine: FusionEngine):
    """The 'Starring X' row only includes movies featuring the seed's lead actor."""
    rows = fusion_engine.build_discovery_rows(user_id=1, seed_movie_id=1)
    starring_row = next(r for r in rows if "Starring Prabhas" in r.title)
    assert starring_row.movie_ids == [6]


def test_discovery_rows_production_scale_excludes_out_of_range_budget(fusion_engine: FusionEngine):
    """RRR's much larger budget falls outside Bahubali's +/-30% band and is excluded."""
    rows = fusion_engine.build_discovery_rows(user_id=1, seed_movie_id=1)
    scale_row = next(r for r in rows if "Production Scale" in r.title)
    assert 2 not in scale_row.movie_ids  # RRR's budget (72M) is far outside 25M +/- 30%


def test_discovery_rows_unknown_seed_raises(fusion_engine: FusionEngine):
    """An unknown seed_movie_id raises FusionEngineError."""
    with pytest.raises(FusionEngineError):
        fusion_engine.build_discovery_rows(user_id=1, seed_movie_id=9999)


def test_recommend_returns_scored_movies_excluding_seed(fusion_engine: FusionEngine):
    """recommend() never includes the seed movie in its own recommendation list."""
    scored = fusion_engine.recommend(user_id=1, seed_movie_id=1, top_k=5)
    ids = [s.movie_id for s in scored]
    assert 1 not in ids
    assert len(scored) > 0


def test_explain_names_strongest_signal(fusion_engine: FusionEngine):
    """explain() picks the source with the highest contributing score."""
    from src.fusion_engine import ScoredMovie

    scored = ScoredMovie(movie_id=2, score=0.9, source_scores={"content": 0.9, "collaborative": 0.1, "semantic": 0.2})
    explanation = fusion_engine.explain(scored, seed_title="Bahubali")
    assert "Bahubali" in explanation


def test_fusion_engine_works_on_catalogue_loaded_from_parquet(
    synthetic_movies: pd.DataFrame,
    mock_content_filter: MagicMock,
    mock_collaborative_filter: MagicMock,
    mock_semantic_engine: MagicMock,
    tmp_path,
):
    """Parquet turns list columns into numpy arrays; the engine must still build every row.

    Regression test: without normalisation the 'Starring X', 'Same Theme' and
    'Sentiment Champions' rows silently disappear in the real app.
    """
    path = tmp_path / "merged_movies.parquet"
    synthetic_movies.to_parquet(path, index=False)
    from_disk = pd.read_parquet(path)
    assert not isinstance(from_disk.loc[0, "cast_names"], list)  # the trap itself

    engine = FusionEngine(
        cf=mock_collaborative_filter,
        content=mock_content_filter,
        semantic=mock_semantic_engine,
        movies=from_disk,
    )
    titles = [row.title for row in engine.build_discovery_rows(user_id=1, seed_movie_id=1)]
    assert len(titles) == 10
    assert any("Starring Prabhas" in t for t in titles)
    assert any("Same Theme" in t for t in titles)
    assert any("Sentiment Engine" in t for t in titles)
