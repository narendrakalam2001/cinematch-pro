"""Tests for the memory-light "lite" deployment (free Streamlit Community Cloud).

Lite mode never imports PyTorch / ChromaDB / scikit-surprise / spaCy, runs the UI and the API in one
process, and degrades the features that need those engines.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.artifact_store import LITE_ALLOW_PATTERNS, LITE_REQUIRED_PATHS, ensure_artifacts, missing_artifacts
from src.chatbot import _build_tools
from src.content_filter import ContentBasedFilter
from src.fusion_engine import FusionEngine

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "frontend" / "app.py")


@pytest.fixture
def movies() -> pd.DataFrame:
    rows = [
        (1, "Iron Man", 2008, 1726, "A billionaire builds a suit of armour and fights terrorists.", ["Action", "Sci-Fi"], "Jon Favreau", ["Robert Downey Jr."], 25000),
        (2, "Iron Man 2", 2010, 10138, "Tony Stark faces a Russian rival and government pressure over his suit.", ["Action", "Sci-Fi"], "Jon Favreau", ["Robert Downey Jr."], 18000),
        (3, "Thor", 2011, 10195, "A god is exiled to Earth and must prove himself worthy of his hammer.", ["Action", "Fantasy"], "Kenneth Branagh", ["Chris Hemsworth"], 17000),
        (4, "Heat", 1995, 949, "Detective hunts a professional thief in Los Angeles.", ["Crime"], "Michael Mann", ["Al Pacino"], 7000),
    ]
    frame = pd.DataFrame(
        rows, columns=["movieId", "clean_title", "year", "id", "overview", "genre_names", "director", "cast_names", "vote_count"]
    )
    frame["imdb_id"] = ["tt1", "tt2", "tt3", "tt4"]
    frame["genres"] = ["|".join(g) for g in frame["genre_names"]]
    frame["vote_average"] = [7.6, 6.8, 6.8, 7.9]
    frame["poster_path"] = ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"]
    frame["original_language"] = "en"
    frame["keyword_names"] = [["superhero"], ["superhero"], ["norse"], ["heist"]]
    frame["music_director"] = ""
    frame["combined_features"] = frame["overview"] + " " + frame["genres"] + " " + frame["director"]
    return frame


@pytest.fixture
def content(movies) -> ContentBasedFilter:
    return ContentBasedFilter(max_features=500).fit(movies)


# ---------------------------------------------------------------------------
# Heavy libraries must not be imported
# ---------------------------------------------------------------------------
def test_the_api_imports_and_works_when_the_heavy_libraries_are_not_installed():
    """Lite mode's promise: with PyTorch/ChromaDB/surprise/spaCy absent, the API still imports.

    The heavy modules are *blocked* by an import hook that raises ModuleNotFoundError, exactly as on a
    machine without them. Merely checking "nothing got loaded" would be wrong on a full install, because
    LangChain itself loads ``transformers`` (and so PyTorch) whenever it is installed.

    ``google.generativeai`` is deliberately NOT blocked: it is a small package (not a memory problem) and some
    ``langchain-google-genai`` 2.0.x releases import it at import time, so blocking it would test pip's
    version resolution rather than lite mode.
    """
    code = (
        "import importlib.abc, sys\n"
        "BLOCKED = ('torch','transformers','chromadb','sentence_transformers','faiss','surprise','spacy','datasets')\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if any(name == b or name.startswith(b + '.') for b in BLOCKED):\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import serving.api\n"
        "import services.recommendation_service as svc\n"
        "assert svc.RecommendationService and svc.LITE_MODE\n"
        "loaded = [n for n in BLOCKED if n in sys.modules]\n"
        "print('LOADED:' + ','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(ROOT), "CINEMATCH_LITE": "1"},
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert result.stdout.strip().splitlines()[-1] == "LOADED:"


# ---------------------------------------------------------------------------
# Recommendation engine without CF / semantic search
# ---------------------------------------------------------------------------
def test_fusion_without_cf_or_semantic_uses_content_only_and_stays_in_range(movies, content):
    engine = FusionEngine(None, content, None, movies)
    scored = engine.recommend(user_id=0, seed_movie_id=1, top_k=3)
    assert scored and scored[0].movie_id == 2  # the sequel is the closest by content
    assert all(0.0 <= m.score <= 1.0 for m in scored)
    assert all(m.source_scores["collaborative"] == 0.0 and m.source_scores["semantic"] == 0.0 for m in scored)

    rows = engine.build_discovery_rows(user_id=0, seed_movie_id=1)
    titles = [r.title for r in rows]
    assert titles and not any("Also Watched" in t or "International" in t for t in titles)  # CF / semantic rows skipped


# ---------------------------------------------------------------------------
# Service in lite mode
# ---------------------------------------------------------------------------
@pytest.fixture
def lite_service(tmp_path, movies, content):
    import services.recommendation_service as svc_module

    parquet = tmp_path / "merged_movies.parquet"
    movies.to_parquet(parquet)
    tfidf = tmp_path / "tfidf_matrix.pkl"
    tfidf.write_bytes(b"x")
    content_cls = MagicMock()
    content_cls.return_value.load.return_value = content
    with patch.object(svc_module, "LITE_MODE", True), patch.object(svc_module, "MERGED_MOVIES_PARQUET", parquet), \
            patch.object(svc_module, "TFIDF_MATRIX_PATH", tfidf), patch.object(svc_module, "ContentBasedFilter", content_cls), \
            patch.object(svc_module, "TMDB_API_KEY", None), patch.object(svc_module, "OMDB_API_KEY", None), \
            patch.object(svc_module, "GEMINI_API_KEY", None):
        yield svc_module.RecommendationService()


def test_lite_service_skips_heavy_engines_but_serves_search_and_cards(lite_service):
    health = lite_service.health()
    assert health["lite_mode"] is True and health["movies_loaded"] and health["content_filter"] and health["fusion_engine"]
    assert not any(health[k] for k in ("collaborative_filter", "semantic_search", "visual_search", "sentiment_engine"))

    assert [r["title"] for r in lite_service.search_movies("iron man")][:2] == ["Iron Man", "Iron Man 2"]
    card = lite_service.get_movie_card(1)  # TMDB/OMDB are off: the card still renders from the catalogue
    assert card["title"] == "Iron Man" and card["crew"]["directors"][0]["name"] == "Jon Favreau"
    related = {m["title"] for row in card["rows"] for m in row["movies"]}
    assert "Iron Man 2" in related and "Iron Man" not in related


def test_chatbot_tools_degrade_gracefully_without_engines():
    tools = {t.name: t for t in _build_tools(None, MagicMock(), None, None, search_fn=lambda q: [{"title": "Heat", "movie_id": 4}])}
    assert tools["movie_search"].invoke({"query": "crime"}) == "Heat [movie_id=4]"  # falls back to catalogue search
    assert "unavailable" in tools["recommendation_tool"].invoke({"seed_movie_id": 1})
    assert "unavailable" in tools["sentiment_tool"].invoke({"reviews": ["great"]})
    no_search = {t.name: t for t in _build_tools(None, MagicMock(), None, None)}
    assert "unavailable" in no_search["movie_search"].invoke({"query": "crime"})


# ---------------------------------------------------------------------------
# Small artifact subset
# ---------------------------------------------------------------------------
def test_lite_downloads_only_the_catalogue_and_tfidf(tmp_path):
    assert set(missing_artifacts(tmp_path, lite=True)) == set(LITE_REQUIRED_PATHS)

    def fake_download(**kwargs):
        for relative in LITE_REQUIRED_PATHS:
            target = Path(kwargs["local_dir"]) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")

    with patch("huggingface_hub.snapshot_download", side_effect=fake_download) as download:
        assert ensure_artifacts(repo_id="u/r", base_dir=tmp_path, lite=True) is True
    assert download.call_args.kwargs["allow_patterns"] == list(LITE_ALLOW_PATTERNS)
    assert "models/**" not in download.call_args.kwargs["allow_patterns"]  # no 500 MB RoBERTa download
    assert missing_artifacts(tmp_path, lite=True) == []


# ---------------------------------------------------------------------------
# UI in embedded mode (no HTTP server at all)
# ---------------------------------------------------------------------------
def test_embedded_ui_runs_the_api_in_process_and_hides_unavailable_features(monkeypatch):
    import serving.api as api_module

    monkeypatch.setenv("CINEMATCH_EMBEDDED", "1")
    service = MagicMock()
    service.health.return_value = {"movies_loaded": True, "visual_search": False, "sentiment_engine": False, "lite_mode": True}
    service.get_trending.return_value = []
    service.search_movies.return_value = [{"movie_id": 1, "title": "Iron Man", "year": 2008, "genres": ["Action"], "match": "Title match"}]
    import streamlit as st

    st.cache_resource.clear()
    try:
        with patch.object(api_module, "get_service", return_value=service), patch.object(api_module, "ensure_artifacts"), \
                patch("requests.get", side_effect=AssertionError("embedded mode must not use HTTP")), \
                patch("requests.post", side_effect=AssertionError("embedded mode must not use HTTP")):
            app = AppTest.from_file(APP_PATH, default_timeout=60).run()
            assert not app.exception
            assert any("Lite demo" in i.value for i in app.info)  # honest banner on Home

            app.session_state["_nav_target"] = "Visual Search"
            app.run()
            assert not app.exception
            assert any("lite deployment" in i.value for i in app.info) and len(app.tabs) == 0  # no uploader offered

            app.session_state["_nav_target"] = "Smart Search"
            app.run()
            app.text_input(key="sq_query").set_value("iron man")
            [b for b in app.button if b.label == "Search"][0].click().run()
            assert not app.exception
            assert any("Iron Man" in md.value for md in app.markdown)  # real results via the in-process API
    finally:
        st.cache_resource.clear()


def test_lite_entrypoint_is_valid_python_and_turns_on_lite_and_embedded_modes():
    source = (ROOT / "lite" / "streamlit_app.py").read_text(encoding="utf-8")
    compile(source, "lite/streamlit_app.py", "exec")
    assert 'setdefault("CINEMATCH_LITE", "1")' in source and 'setdefault("CINEMATCH_EMBEDDED", "1")' in source
    lines = (ROOT / "lite" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.split("#")[0].strip().lower() for line in lines if line.split("#")[0].strip()]
    for heavy in ("torch", "transformers", "chromadb", "sentence-transformers", "surprise", "spacy", "faiss"):
        assert not any(p.startswith(heavy) for p in packages)  # the lite install must stay small