"""Tests for scripts/check_artifacts.py and scripts/build_content_model.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from scripts import build_content_model
from scripts.check_artifacts import REQUIRED_ARTIFACTS, check_presence, compare_versions


def test_check_presence_reports_everything_missing_in_empty_project(tmp_path: Path):
    """An empty directory yields one failed entry per required artifact."""
    results = check_presence(tmp_path)
    assert len(results) == len(REQUIRED_ARTIFACTS)
    assert not any(ok for _, ok, _ in results)


def test_check_presence_detects_present_files_and_ignores_empty_ones(tmp_path: Path):
    """A non-empty file counts as present; a zero-byte file (failed copy) does not."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "svd_model.pkl").write_bytes(b"data")
    (tmp_path / "models" / "faiss_poster.index").write_bytes(b"")

    status = {relative: ok for _, ok, relative in check_presence(tmp_path)}
    assert status["models/svd_model.pkl"] is True
    assert status["models/faiss_poster.index"] is False


def test_compare_versions_flags_major_minor_differences_only():
    """Patch-level differences are fine; minor-version differences are warned about."""
    trained = {"python": "3.11.9", "numpy": "1.26.4", "scikit-learn": "1.5.0", "faiss-cpu": "1.8.0"}
    local = {"python": "3.10.21", "numpy": "1.26.2", "scikit-learn": "1.4.2", "faiss-cpu": "not installed"}
    warnings = compare_versions(trained, local)

    joined = " | ".join(warnings)
    assert "python" in joined and "scikit-learn" in joined and "faiss-cpu" in joined
    assert "numpy" not in joined  # 1.26.4 vs 1.26.2 is fine


def test_compare_versions_ignores_packages_not_used_in_training():
    """A package that wasn't installed on the training machine cannot mismatch."""
    assert compare_versions({"chromadb": "not installed"}, {"chromadb": "0.5.3"}) == []


def test_build_content_model_fails_cleanly_without_parquet(tmp_path: Path, monkeypatch):
    """A missing merged_movies.parquet is a clear error exit, not a traceback."""
    monkeypatch.setattr(sys, "argv", ["x", "--movies-parquet", str(tmp_path / "nope.parquet")])
    assert build_content_model.main() == 1


def test_build_content_model_fits_and_saves(tmp_path: Path, monkeypatch):
    """With a valid parquet the filter is fitted, saved and recorded in the model card."""
    parquet = tmp_path / "merged.parquet"
    pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "combined_features": ["epic war kingdom", "epic war revolution", "romantic comedy"],
        }
    ).to_parquet(parquet, index=False)

    saved, recorded = [], []
    monkeypatch.setattr(build_content_model.ContentBasedFilter, "save", lambda self: saved.append(True))
    monkeypatch.setattr(build_content_model, "update_model_card", lambda section, data: recorded.append((section, data)))
    monkeypatch.setattr(sys, "argv", ["x", "--movies-parquet", str(parquet)])

    assert build_content_model.main() == 0
    assert saved == [True]
    assert recorded[0][0] == "content_filter" and recorded[0][1]["movies"] == 3
