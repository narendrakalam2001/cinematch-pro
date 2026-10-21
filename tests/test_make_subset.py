"""Tests for scripts/make_subset.py (shrinking downloaded datasets for a Kaggle run)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import make_subset
from src.data_loader import load_movielens_ratings, load_tmdb_full_metadata
from src.preprocessing import merge_movielens_with_full_tmdb, prepare_full_tmdb_metadata

N_MOVIES = 60
WORDS = ["Dark", "Night", "Love", "War", "Home", "King"]


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """A miniature copy of the user's download folder (same names, same column shapes)."""
    src = tmp_path / "downloads"
    (src / "ml-25m").mkdir(parents=True)
    (src / "ml-25m" / "genome-scores.csv").write_text("unused,file\n")  # must not be copied

    base = [(f"{WORDS[i % 6]} {WORDS[(i // 6) % 6]} {i}", 1990 + i % 25) for i in range(N_MOVIES)]
    pd.DataFrame(
        {
            "movieId": range(1, N_MOVIES + 1),
            "title": [f"{t}, The ({y})" if i % 3 == 0 else f"{t} ({y})" for i, (t, y) in enumerate(base)],
            "genres": "Action|Drama",
        }
    ).to_csv(src / "ml-25m" / "movies.csv", index=False)
    pd.DataFrame(
        {"movieId": range(1, N_MOVIES + 1), "imdbId": 100000 + np.arange(N_MOVIES), "tmdbId": np.arange(N_MOVIES)}
    ).to_csv(src / "ml-25m" / "links.csv", index=False)

    rng = np.random.default_rng(0)
    pd.DataFrame(
        {
            "userId": np.repeat(np.arange(1, 41), 10),
            "movieId": rng.integers(1, N_MOVIES + 1, 400),
            "rating": rng.choice([3.0, 4.0, 5.0], 400),
            "timestamp": 1,
        }
    ).to_csv(src / "ml-25m" / "ratings.csv", index=False)

    rows = []
    for i, (t, y) in enumerate(base):
        rows.append(
            dict(id=1000 + i, title=f"The {t}" if i % 3 == 0 else t, overview="story", release_year=y,
                 vote_count=10 + i, adult=False, genres_list="['Action']", keywords="['war']",
                 Cast_list="['A', 'B']", Director=f"Dir {i % 4}", Music_Composer="Comp", budget=1e6,
                 homepage="http://unused", tagline="unused")
        )
    for j in range(300):  # rows that match nothing in MovieLens
        rows.append({**rows[0], "id": 50000 + j, "title": f"Unrelated Film {j}", "release_year": 2005})
    pd.DataFrame(rows).to_csv(src / "IMDB TMDB Movie Metadata Big Dataset (1M).csv", index=False)

    pd.DataFrame(
        {"review": ["good"] * 100 + ["bad"] * 100, "sentiment": ["positive"] * 100 + ["negative"] * 100}
    ).to_csv(src / "IMDB Dataset.csv", index=False)

    genre = pd.DataFrame(
        {"imdbId": 100000 + np.arange(N_MOVIES), "Title": [t for t, _ in base], "Poster": "http://x/y.jpg"}
    )
    genre.loc[0, "Title"] = "Amélie"
    genre.to_csv(src / "MovieGenre.csv", index=False, encoding="latin-1")  # not valid UTF-8
    return src


def _run(monkeypatch, source: Path, out: Path, *extra: str) -> int:
    monkeypatch.setattr(sys, "argv", ["x", "--source-dir", str(source), "--out-dir", str(out), *extra])
    return make_subset.main()


def test_subset_creates_same_named_smaller_files(source_dir: Path, tmp_path: Path, monkeypatch):
    """The output uses the original file names so the Kaggle notebook finds them unchanged."""
    out = tmp_path / "subset"
    assert _run(monkeypatch, source_dir, out, "--users", "10") == 0

    assert (out / "ml-25m" / "ratings.csv").exists()
    assert (out / "ml-25m" / "movies.csv").exists() and (out / "ml-25m" / "links.csv").exists()
    assert (out / "IMDB Dataset.csv").exists() and (out / "MovieGenre.csv").exists()
    assert (out / "IMDB TMDB Movie Metadata Big Dataset (1M).csv").exists()
    assert not (out / "ml-25m" / "genome-scores.csv").exists()  # unused files are not copied


def test_subset_ratings_keeps_only_requested_users(source_dir: Path, tmp_path: Path, monkeypatch):
    """Only ratings of users with userId <= --users survive; movies/links are kept in full."""
    out = tmp_path / "subset"
    _run(monkeypatch, source_dir, out, "--users", "10")

    ratings = load_movielens_ratings(path=out / "ml-25m" / "ratings.csv")
    assert ratings["userId"].max() <= 10 and ratings["userId"].nunique() == 10
    assert len(ratings) == 100
    assert len(pd.read_csv(out / "ml-25m" / "movies.csv")) == N_MOVIES


def test_subset_tmdb_drops_unmatched_rows_and_unused_columns(source_dir: Path, tmp_path: Path, monkeypatch):
    """Only MovieLens-matching TMDB rows and only the columns the project reads are written."""
    out = tmp_path / "subset"
    _run(monkeypatch, source_dir, out)

    tmdb = load_tmdb_full_metadata(path=out / "IMDB TMDB Movie Metadata Big Dataset (1M).csv")
    assert len(tmdb) == N_MOVIES  # 60 matching rows, the 300 unrelated ones are gone
    assert "homepage" not in tmdb.columns and "tagline" not in tmdb.columns


def test_subset_does_not_change_what_the_pipeline_produces(source_dir: Path, tmp_path: Path, monkeypatch):
    """Merging from the subset gives exactly the same movieId -> TMDB id mapping as from the full file."""
    out = tmp_path / "subset"
    _run(monkeypatch, source_dir, out)
    movies = pd.read_csv(source_dir / "ml-25m" / "movies.csv")

    def mapping(tmdb_path: Path) -> pd.DataFrame:
        from src.preprocessing import restrict_tmdb_to_movielens

        full = load_tmdb_full_metadata(path=tmdb_path)
        merged = merge_movielens_with_full_tmdb(
            prepare_full_tmdb_metadata(restrict_tmdb_to_movielens(full, movies)), movies
        )
        return merged[["movieId", "id"]].reset_index(drop=True)

    name = "IMDB TMDB Movie Metadata Big Dataset (1M).csv"
    pd.testing.assert_frame_equal(mapping(source_dir / name), mapping(out / name))


def test_subset_reviews_balanced_sample(source_dir: Path, tmp_path: Path, monkeypatch):
    """--reviews N keeps N reviews, half per class."""
    out = tmp_path / "subset"
    _run(monkeypatch, source_dir, out, "--reviews", "40")
    reviews = pd.read_csv(out / "IMDB Dataset.csv")
    assert len(reviews) == 40
    assert reviews["sentiment"].value_counts().to_dict() == {"positive": 20, "negative": 20}


def test_subset_posters_are_resaved_as_utf8(source_dir: Path, tmp_path: Path, monkeypatch):
    """The Latin-1 poster CSV becomes strict UTF-8 and keeps its accented title."""
    out = tmp_path / "subset"
    _run(monkeypatch, source_dir, out, "--posters", "5")
    posters = pd.read_csv(out / "MovieGenre.csv", encoding="utf-8")  # would raise if not UTF-8
    assert len(posters) == 5 and posters.loc[0, "Title"] == "Amélie"


def test_subset_fails_cleanly_when_a_dataset_is_missing(source_dir: Path, tmp_path: Path, monkeypatch):
    """A missing file is a clear error exit, not a traceback."""
    (source_dir / "MovieGenre.csv").unlink()
    assert _run(monkeypatch, source_dir, tmp_path / "subset") == 1


def test_subset_refuses_to_write_into_the_source_folder(source_dir: Path, monkeypatch):
    """--out-dir equal to --source-dir would overwrite the originals, so it is rejected."""
    assert _run(monkeypatch, source_dir, source_dir) == 1
