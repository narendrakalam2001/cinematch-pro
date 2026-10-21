"""
CLI entry point: verify that all trained artifacts are present and compatible.

Run this on the serving machine after copying the files produced by a
training run (e.g. a Kaggle notebook) into the project:

    python -m scripts.check_artifacts          # presence + version comparison
    python -m scripts.check_artifacts --deep   # also actually load every artifact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import BASE_DIR, logger
from src.model_card import environment_versions

# (label, path relative to the project root, required for the full app?)
REQUIRED_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("Merged movie catalogue", "data/processed/merged_movies.parquet"),
    ("SVD collaborative filter", "models/svd_model.pkl"),
    ("TF-IDF matrix", "models/tfidf_matrix.pkl"),
    ("TF-IDF vectorizer", "models/tfidf_vectorizer.pkl"),
    ("Sentiment model config", "models/roberta_sentiment/config.json"),
    ("FAISS poster index", "models/faiss_poster.index"),
    ("FAISS poster id map", "models/faiss_poster_metadata.pkl"),
    ("ChromaDB store", "chromadb_store/chroma.sqlite3"),
    ("Model card", "models/model_card.json"),
)

# Pickles and indexes are sensitive to these; a different major.minor is a warning.
_VERSION_SENSITIVE = ("python", "numpy", "pandas", "scikit-learn", "scikit-surprise", "faiss-cpu", "chromadb")


def check_presence(base_dir: Path = BASE_DIR) -> list[tuple[str, bool, str]]:
    """Check that every required artifact file exists and is non-empty.

    Args:
        base_dir: Project root to look in.

    Returns:
        list[tuple[str, bool, str]]: ``(label, ok, relative_path)`` per artifact.
    """
    results = []
    for label, relative in REQUIRED_ARTIFACTS:
        path = base_dir / relative
        results.append((label, path.exists() and path.stat().st_size > 0, relative))
    return results


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def compare_versions(card_environment: dict[str, str], local_environment: dict[str, str]) -> list[str]:
    """List version-sensitive packages whose major.minor differs between machines.

    Args:
        card_environment: Versions recorded in model_card.json at training time.
        local_environment: Versions on this machine.

    Returns:
        list[str]: One human-readable warning per mismatched package
        (empty if everything lines up).
    """
    warnings = []
    for package in _VERSION_SENSITIVE:
        trained = card_environment.get(package)
        local = local_environment.get(package)
        if not trained or trained == "not installed":
            continue
        if local is None or local == "not installed":
            warnings.append(f"{package}: trained with {trained} but it is NOT installed here")
        elif _major_minor(trained) != _major_minor(local):
            warnings.append(f"{package}: trained with {trained}, running {local}")
    return warnings


def deep_load_checks() -> list[tuple[str, bool, str]]:
    """Actually load each artifact with the project's own classes.

    Returns:
        list[tuple[str, bool, str]]: ``(label, ok, detail)`` per artifact;
        a failure carries the exception message instead of raising.
    """
    results: list[tuple[str, bool, str]] = []

    def attempt(label: str, loader) -> None:
        try:
            results.append((label, True, str(loader())))
        except Exception as exc:  # noqa: BLE001 - report every kind of load failure
            results.append((label, False, f"{type(exc).__name__}: {exc}"))

    def load_svd() -> str:
        from src.collaborative_filter import CollaborativeFilter

        return f"prediction(1,1)={CollaborativeFilter().load().predict_rating(1, 1):.2f}"

    def load_tfidf() -> str:
        from src.content_filter import ContentBasedFilter

        return f"{len(ContentBasedFilter().load().movie_ids)} movies"

    def load_faiss() -> str:
        from src.visual_search import VisualSearchEngine

        engine = VisualSearchEngine.__new__(VisualSearchEngine)  # skip loading CLIP weights
        engine.index = engine.movie_ids = None
        return f"{engine.load().index.ntotal} posters"

    def load_chroma() -> str:
        from src.semantic_search import SemanticSearchEngine

        return f"{SemanticSearchEngine().count()} documents"

    def load_sentiment() -> str:
        from src.config import ROBERTA_MODEL_DIR
        from src.sentiment_engine import SentimentEngine

        result = SentimentEngine(model_path=ROBERTA_MODEL_DIR).predict("A wonderful, moving film.")
        return f"sample review -> {result.label} ({result.confidence:.2f})"

    attempt("SVD collaborative filter", load_svd)
    attempt("TF-IDF content filter", load_tfidf)
    attempt("FAISS poster index", load_faiss)
    attempt("ChromaDB semantic store", load_chroma)
    attempt("RoBERTa sentiment model", load_sentiment)
    return results


def main() -> int:
    """Print a pass/fail report for the trained artifacts.

    Returns:
        int: 0 if every artifact is present (and, with ``--deep``, loads), else 1.
    """
    parser = argparse.ArgumentParser(description="Verify CineMatch Pro's trained artifacts.")
    parser.add_argument("--deep", action="store_true", help="Also load every artifact.")
    args = parser.parse_args()

    failed = False
    print("\nArtifact presence")
    for label, ok, relative in check_presence():
        print(f"  {'OK     ' if ok else 'MISSING'}  {label:<26} {relative}")
        failed |= not ok

    card_path = BASE_DIR / "models" / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        trained_env = next((section["environment"] for section in card.values() if "environment" in section), {})
        warnings = compare_versions(trained_env, environment_versions())
        print("\nLibrary versions (training machine vs this machine)")
        if warnings:
            for warning in warnings:
                print(f"  WARN  {warning}")
            print("  -> Pickles/indexes can fail to load across versions. If --deep fails, re-run the cheap "
                  "ones locally (scripts.build_content_model, scripts.ingest_descriptions).")
        else:
            print("  OK    no version-sensitive differences")

    if args.deep:
        print("\nDeep load check")
        for label, ok, detail in deep_load_checks():
            print(f"  {'OK     ' if ok else 'FAILED '}  {label:<26} {detail}")
            failed |= not ok

    print("\nResult:", "PROBLEMS FOUND" if failed else "all good")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
