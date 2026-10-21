"""Tests for src/preprocessing.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    clean_movielens_title,
    clean_ratings,
    extract_director,
    extract_music_director,
    extract_names,
    merge_movielens_with_full_tmdb,
    merge_tmdb_datasets,
    merge_with_movielens,
    normalize_title_key,
    parse_string_list,
    prepare_full_tmdb_metadata,
    restrict_tmdb_to_movielens,
)

_GENRES_JSON = "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
_CREW_JSON = (
    "[{'job': 'Director', 'name': 'S.S. Rajamouli'}, "
    "{'job': 'Original Music Composer', 'name': 'M.M. Keeravani'}]"
)


def test_extract_names_parses_tmdb_json_list():
    """extract_names pulls out the 'name' field from a TMDB JSON-list column."""
    assert extract_names(_GENRES_JSON) == ["Action", "Adventure"]


def test_extract_names_handles_null():
    """extract_names returns an empty list for NaN/null input."""
    assert extract_names(float("nan")) == []


def test_extract_director_finds_director_job():
    """extract_director pulls the crew member whose job is 'Director'."""
    assert extract_director(_CREW_JSON) == "S.S. Rajamouli"


def test_extract_music_director_finds_composer_job():
    """extract_music_director pulls the 'Original Music Composer' crew member."""
    assert extract_music_director(_CREW_JSON) == "M.M. Keeravani"


def test_extract_director_returns_empty_when_absent():
    """extract_director returns '' when no Director entry exists."""
    crew = "[{'job': 'Producer', 'name': 'Someone'}]"
    assert extract_director(crew) == ""


def test_clean_movielens_title_splits_year():
    """clean_movielens_title separates the trailing (YYYY) into a year field."""
    title, year = clean_movielens_title("Bahubali: The Beginning (2015)")
    assert title == "Bahubali: The Beginning"
    assert year == 2015


def test_clean_movielens_title_handles_no_year():
    """A title without a trailing year returns None for the year."""
    title, year = clean_movielens_title("Untitled Project")
    assert title == "Untitled Project"
    assert year is None


def test_merge_tmdb_datasets_builds_combined_features():
    """merge_tmdb_datasets produces a non-empty combined_features column."""
    tmdb_movies = pd.DataFrame(
        {
            "id": [1],
            "title": ["Bahubali: The Beginning"],
            "overview": ["A young man learns of his royal heritage."],
            "genres": [_GENRES_JSON],
            "keywords": ["[{'id': 1, 'name': 'kingdom'}]"],
            "release_date": ["2015-07-10"],
        }
    )
    tmdb_credits = pd.DataFrame(
        {
            "movie_id": [1],
            "cast": ["[{'name': 'Prabhas'}, {'name': 'Rana Daggubati'}]"],
            "crew": [_CREW_JSON],
        }
    )
    merged = merge_tmdb_datasets(tmdb_movies, tmdb_credits)
    assert merged.loc[0, "director"] == "S.S. Rajamouli"
    assert "Prabhas" in merged.loc[0, "cast_names"]
    assert merged.loc[0, "combined_features"].strip() != ""


def test_merge_with_movielens_matches_by_title_and_year():
    """merge_with_movielens attaches TMDB detail to the matching MovieLens row."""
    tmdb_movies = pd.DataFrame(
        {
            "id": [1],
            "title": ["Bahubali: The Beginning"],
            "overview": ["desc"],
            "genres": [_GENRES_JSON],
            "keywords": ["[]"],
            "release_date": ["2015-07-10"],
        }
    )
    tmdb_credits = pd.DataFrame({"movie_id": [1], "cast": ["[]"], "crew": [_CREW_JSON]})
    ml_movies = pd.DataFrame(
        {"movieId": [101], "title": ["Bahubali: The Beginning (2015)"], "genres": ["Action|Adventure"]}
    )
    tmdb_merged = merge_tmdb_datasets(tmdb_movies, tmdb_credits)
    final = merge_with_movielens(tmdb_merged, ml_movies)
    assert final.loc[0, "director"] == "S.S. Rajamouli"


def test_clean_ratings_dedups_and_filters_invalid():
    """clean_ratings removes duplicate (user, movie) pairs and out-of-range ratings."""
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [101, 101, 101, 102],
            "rating": [4.5, 4.5, 6.0, 3.0],  # duplicate + one invalid (6.0 > 5.0)
        }
    )
    cleaned = clean_ratings(ratings)
    assert len(cleaned) == 2
    assert cleaned["rating"].max() <= 5.0


# ---------------------------------------------------------------------------
# Full-metadata pipeline (pre-merged TMDB export, e.g. the "IMDB TMDB Movie
# Metadata" Kaggle dataset with flat string-list genres/keywords/cast)
# ---------------------------------------------------------------------------
_REAL_INCEPTION_ROW = {
    "id": 27205,
    "title": "Inception",
    "overview": "A thief who steals corporate secrets through dream-sharing technology.",
    "release_date": "2010-07-15",
    "release_year": 2010,
    "budget": 160_000_000,
    "original_language": "en",
    "genres_list": "['Action', 'Science Fiction', 'Adventure']",
    "keywords": "['rescue', 'mission', 'dream', 'philosophy', 'spy']",
    "Cast_list": "['Leonardo DiCaprio', 'Joseph Gordon-Levitt', 'Elliot Page', 'Tom Hardy']",
    "Director": "Christopher Nolan",
    "Music_Composer": "Hans Zimmer",
    "Writer": "Christopher Nolan",
    "overview_sentiment": -0.011111,
}


def test_parse_string_list_handles_flat_string_lists():
    """parse_string_list correctly parses the 'Big Dataset' schema's flat lists."""
    assert parse_string_list("['Action', 'Science Fiction', 'Adventure']") == [
        "Action",
        "Science Fiction",
        "Adventure",
    ]


def test_parse_string_list_handles_null():
    """parse_string_list returns an empty list for NaN/null input."""
    assert parse_string_list(float("nan")) == []


def test_parse_string_list_handles_malformed_input():
    """parse_string_list returns an empty list rather than raising on garbage input."""
    assert parse_string_list("not a list at all {{{") == []


def test_prepare_full_tmdb_metadata_extracts_correct_fields():
    """prepare_full_tmdb_metadata correctly parses the real dataset's column schema."""
    tmdb_full = pd.DataFrame([_REAL_INCEPTION_ROW])
    prepared = prepare_full_tmdb_metadata(tmdb_full)
    row = prepared.iloc[0]

    assert row["director"] == "Christopher Nolan"
    assert row["music_director"] == "Hans Zimmer"
    assert row["genre_names"] == ["Action", "Science Fiction", "Adventure"]
    assert "Leonardo DiCaprio" in row["cast_names"]
    assert row["year"] == 2010
    assert row["clean_title"] == "Inception"
    assert row["combined_features"].strip() != ""


def test_prepare_full_tmdb_metadata_seeds_sentiment_score_from_overview_polarity():
    """A near-zero overview_sentiment polarity scales to roughly 50 (neutral) on 0-100."""
    tmdb_full = pd.DataFrame([_REAL_INCEPTION_ROW])
    prepared = prepare_full_tmdb_metadata(tmdb_full)
    assert 45.0 <= prepared.iloc[0]["sentiment_score"] <= 55.0


def test_prepare_full_tmdb_metadata_missing_required_column_raises():
    """Missing a required column (e.g. Director) raises ValueError, not a silent KeyError later."""
    bad_df = pd.DataFrame([{k: v for k, v in _REAL_INCEPTION_ROW.items() if k != "Director"}])
    with pytest.raises(ValueError, match="Director"):
        prepare_full_tmdb_metadata(bad_df)


def test_merge_movielens_with_full_tmdb_matches_real_inception_row():
    """The full-metadata merge correctly joins a MovieLens row to its real TMDB match."""
    tmdb_full = pd.DataFrame([_REAL_INCEPTION_ROW])
    prepared = prepare_full_tmdb_metadata(tmdb_full)
    ml_movies = pd.DataFrame(
        {
            "movieId": [79132],
            "title": ["Inception (2010)"],
            "genres": ["Action|Crime|Drama|Mystery|Sci-Fi|Thriller"],
        }
    )
    merged = merge_movielens_with_full_tmdb(prepared, ml_movies)
    assert merged.loc[0, "movieId"] == 79132
    assert merged.loc[0, "director"] == "Christopher Nolan"


def test_merge_movielens_with_full_tmdb_unmatched_falls_back_to_ml_genres():
    """A MovieLens movie with no TMDB match falls back to MovieLens genres for combined_features."""
    tmdb_full = pd.DataFrame([_REAL_INCEPTION_ROW])
    prepared = prepare_full_tmdb_metadata(tmdb_full)
    ml_movies = pd.DataFrame(
        {
            "movieId": [999],
            "title": ["Some Totally Unrelated Movie (1987)"],
            "genres": ["Comedy|Romance"],
        }
    )
    merged = merge_movielens_with_full_tmdb(prepared, ml_movies)
    assert merged.loc[0, "combined_features"].strip() == "Comedy Romance"


# ---------------------------------------------------------------------------
# Title matching, de-duplication and filtering (real-world MovieLens/TMDB quirks)
# ---------------------------------------------------------------------------
def test_clean_movielens_title_moves_trailing_article_to_front():
    """MovieLens 'Matrix, The (1999)' becomes TMDB-style 'The Matrix'."""
    assert clean_movielens_title("Matrix, The (1999)") == ("The Matrix", 1999)
    assert clean_movielens_title("Beautiful Mind, A (2001)") == ("A Beautiful Mind", 2001)


def test_clean_movielens_title_drops_alternate_title_parenthetical():
    """The '(Alt Title)' MovieLens appends before the year is discarded."""
    title, year = clean_movielens_title("City of Lost Children, The (Cite des enfants perdus, La) (1995)")
    assert (title, year) == ("The City of Lost Children", 1995)


def test_clean_movielens_title_keeps_leading_parenthesis_titles():
    """A title that merely starts with parentheses is not mangled."""
    assert clean_movielens_title("(500) Days of Summer (2009)") == ("(500) Days of Summer", 2009)


def test_normalize_title_key_ignores_case_punctuation_and_accents():
    """Keys match across case, punctuation, accents and '&' vs 'and'."""
    assert normalize_title_key("Spider-Man: Homecoming") == normalize_title_key("spider man homecoming")
    assert normalize_title_key("Amélie") == normalize_title_key("Amelie")
    assert normalize_title_key("Mr. & Mrs. Smith") == normalize_title_key("Mr. and Mrs. Smith")


def test_normalize_title_key_non_latin_title_yields_empty_key():
    """An all-CJK title has no usable key, so it can never produce a false match."""
    assert normalize_title_key("千と千尋の神隠し") == ""
    assert normalize_title_key(None) == ""


def _tmdb_row(**overrides):
    row = dict(_REAL_INCEPTION_ROW)
    row.update(overrides)
    return row


def test_restrict_tmdb_to_movielens_keeps_only_matching_rows():
    """Only TMDB rows whose normalized title+year exist in MovieLens survive."""
    tmdb_full = pd.DataFrame(
        [
            _tmdb_row(id=1, title="The Matrix", release_year=1999),
            _tmdb_row(id=2, title="Some Obscure Film", release_year=1999),
            _tmdb_row(id=3, title="The Matrix", release_year=2003),  # wrong year
        ]
    )
    ml_movies = pd.DataFrame(
        {"movieId": [2571], "title": ["Matrix, The (1999)"], "genres": ["Action|Sci-Fi"]}
    )
    restricted = restrict_tmdb_to_movielens(tmdb_full, ml_movies)
    assert restricted["id"].tolist() == [1]


def test_merge_matches_movielens_article_style_title():
    """'Matrix, The (1999)' in MovieLens matches 'The Matrix' in TMDB."""
    tmdb_full = pd.DataFrame([_tmdb_row(id=603, title="The Matrix", release_year=1999)])
    ml_movies = pd.DataFrame(
        {"movieId": [2571], "title": ["Matrix, The (1999)"], "genres": ["Action|Sci-Fi"]}
    )
    merged = merge_movielens_with_full_tmdb(prepare_full_tmdb_metadata(tmdb_full), ml_movies)
    assert merged.loc[0, "id"] == 603
    assert merged.loc[0, "clean_title"] == "The Matrix"


def test_merge_duplicate_tmdb_titles_yield_one_row_per_movie_with_most_voted():
    """Duplicate title+year rows in TMDB must not duplicate MovieLens rows."""
    tmdb_full = pd.DataFrame(
        [
            _tmdb_row(id=10, title="Home", release_year=2015, vote_count=5, Director="Unknown A"),
            _tmdb_row(id=11, title="Home", release_year=2015, vote_count=9000, Director="Tim Johnson"),
            _tmdb_row(id=12, title="Home", release_year=2015, vote_count=1, Director="Unknown B"),
        ]
    )
    ml_movies = pd.DataFrame(
        {"movieId": [100], "title": ["Home (2015)"], "genres": ["Animation|Comedy"]}
    )
    merged = merge_movielens_with_full_tmdb(prepare_full_tmdb_metadata(tmdb_full), ml_movies)
    assert len(merged) == 1
    assert merged.loc[0, "id"] == 11
    assert merged.loc[0, "director"] == "Tim Johnson"
    assert merged["movieId"].is_unique


def test_merge_unknown_year_never_matches():
    """A MovieLens movie without a year must not join to a TMDB row without a year."""
    tmdb_full = pd.DataFrame(
        [_tmdb_row(id=7, title="Mystery Film", release_year=None, release_date=None)]
    )
    ml_movies = pd.DataFrame(
        {"movieId": [55], "title": ["Mystery Film"], "genres": ["Drama"]}  # no (YYYY)
    )
    merged = merge_movielens_with_full_tmdb(prepare_full_tmdb_metadata(tmdb_full), ml_movies)
    assert len(merged) == 1
    assert pd.isna(merged.loc[0, "id"])
    assert merged.loc[0, "combined_features"].strip() == "Drama"


def test_prepare_full_tmdb_metadata_drops_adult_titles():
    """Rows flagged adult are excluded from the catalogue."""
    tmdb_full = pd.DataFrame(
        [_tmdb_row(id=1, adult=False), _tmdb_row(id=2, title="Adult Film", adult=True)]
    )
    prepared = prepare_full_tmdb_metadata(tmdb_full)
    assert prepared["id"].tolist() == [1]
