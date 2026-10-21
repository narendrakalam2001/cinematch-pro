"""Tests for src/model_card.py."""

from __future__ import annotations

import json
from pathlib import Path

from src.model_card import environment_versions, update_model_card


def test_environment_versions_reports_python_and_marks_missing_packages():
    """Python is always reported; absent packages are labelled, never raise."""
    versions = environment_versions()
    assert versions["python"].count(".") >= 1
    assert all(isinstance(value, str) for value in versions.values())


def test_update_model_card_creates_file_with_section_and_metadata(tmp_path: Path):
    """The first update creates the file, stamping time and environment."""
    path = tmp_path / "models" / "model_card.json"
    update_model_card("sentiment", {"accuracy": 0.94}, path=path)

    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["sentiment"]["accuracy"] == 0.94
    assert "updated_at" in card["sentiment"]
    assert "python" in card["sentiment"]["environment"]


def test_update_model_card_preserves_other_sections(tmp_path: Path):
    """Updating one section leaves previously recorded sections untouched."""
    path = tmp_path / "model_card.json"
    update_model_card("sentiment", {"accuracy": 0.94}, path=path)
    update_model_card("collaborative_filter", {"rmse": 0.81}, path=path)
    update_model_card("sentiment", {"accuracy": 0.95}, path=path)

    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["sentiment"]["accuracy"] == 0.95
    assert card["collaborative_filter"]["rmse"] == 0.81


def test_update_model_card_recovers_from_corrupt_file(tmp_path: Path):
    """A half-written/corrupt card is replaced rather than crashing a training run."""
    path = tmp_path / "model_card.json"
    path.write_text("{not json", encoding="utf-8")
    update_model_card("sentiment", {"accuracy": 0.9}, path=path)
    assert json.loads(path.read_text(encoding="utf-8"))["sentiment"]["accuracy"] == 0.9
