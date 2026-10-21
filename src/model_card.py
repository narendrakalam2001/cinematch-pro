"""
Model card writer for CineMatch Pro.

Each training / indexing script records its real results (accuracy,
RMSE, index size, ...) plus the library versions it ran with into
``models/model_card.json``. That gives the README's performance table a
single source of truth, and makes version mismatches between the machine
that trained a model (e.g. a Kaggle notebook) and the machine that serves
it easy to spot.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from src.config import MODEL_CARD_PATH, logger

_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scikit-surprise",
    "torch",
    "transformers",
    "sentence-transformers",
    "faiss-cpu",
    "chromadb",
)


def environment_versions() -> dict[str, str]:
    """Collect the Python and key library versions of the running environment.

    Returns:
        dict[str, str]: ``python`` plus one entry per tracked package;
        packages that are not installed are reported as ``"not installed"``.
    """
    versions = {"python": platform.python_version()}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def update_model_card(
    section: str, data: dict[str, Any], path: Path = MODEL_CARD_PATH
) -> dict[str, Any]:
    """Write one section of the model card, preserving all other sections.

    Args:
        section: Section name, e.g. ``"sentiment"`` or ``"collaborative_filter"``.
        data: JSON-serializable results for that section. A UTC
            ``updated_at`` timestamp and the current ``environment``
            versions are added automatically.
        path: Model card location (defaults to ``models/model_card.json``).

    Returns:
        dict[str, Any]: The full model card after the update.
    """
    card: dict[str, Any] = {}
    if path.exists():
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Model card at %s was unreadable; starting a fresh one.", path)

    card[section] = {
        **data,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_versions(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    logger.info("Updated model card section '%s' in %s", section, path)
    return card
