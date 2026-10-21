"""Tests for the display-ready methods of services/recommendation_service.py."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from services.movie_catalog import MovieCatalog
from services.recommendation_service import RecommendationService, RecommendationServiceError, compose_card
from src.fusion_engine import RecommendationRow

_FULL = {
    "id": 1726, "title": "Iron Man", "tagline": "Heroes are made.", "runtime": 126,
    "release_date": "2008-04-30", "status": "Released", "overview": "A billionaire builds a suit.",
    "poster_path": "/p.jpg", "backdrop_path": "/b.jpg", "vote_average": 7.6, "vote_count": 25000,
    "budget": 140000000, "revenue": 585000000, "imdb_id": "tt0371746",
    "genres": [{"name": "Action"}, {"name": "Sci-Fi"}],
    "production_companies": [{"name": "Marvel Studios"}],
    "production_countries": [{"name": "United States of America"}],
    "spoken_languages": [{"english_name": "English"}],
    "keywords": {"keywords": [{"name": "superhero"}]},
    "videos": {"results": [
        {"site": "YouTube", "type": "Teaser", "key": "TEASER", "official": True},
        {"site": "YouTube", "type": "Trailer", "key": "TRAILER", "official": True},
        {"site": "Vimeo", "type": "Trailer", "key": "NOPE"},
    ]},
    "release_dates": {"results": [{"iso_3166_1": "US", "release_dates": [{"certification": "PG-13"}]},
                                  {"iso_3166_1": "IN", "release_dates": [{"certification": "UA"}]}]},
    "credits": {
        "cast": [{"id": 3223, "name": "Robert Downey Jr.", "character": "Tony Stark", "profile_path": "/rdj.jpg", "order": 0},
                 {"id": 12052, "name": "Gwyneth Paltrow", "character": "Pepper", "profile_path": None, "order": 1}],
        "crew": [{"id": 15277, "name": "Jon Favreau", "job": "Director"},
                 {"id": 1, "name": "Mark Fergus", "job": "Screenplay"},
                 {"id": 2, "name": "Avi Arad", "job": "Producer"},
                 {"id": 3, "name": "Ramin Djawadi", "job": "Original Music Composer"}],
    },
}


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "clean_title": ["Iron Man", "Iron Man 2", "Thor", "Heat"],
            "year": [2008, 2010, 2011, 1995],
            "id": [1726, 10138, 10195, 949],
            "imdb_id": ["tt0371746", "tt1228705", "tt0800369", "tt0113277"],
            "overview": ["A billionaire builds a suit of armour.", "Tony Stark returns.", "A god is exiled to Earth.", "Cops and robbers."],
            "genre_names": [np.array(["Action", "Sci-Fi"]), np.array(["Action", "Sci-Fi"]), np.array(["Action", "Fantasy"]), np.array(["Crime"])],
            "genres": ["Action|Sci-Fi", "Action|Sci-Fi", "Action|Fantasy", "Crime"],
            "vote_count": [25000, 18000, 17000, 7000], "vote_average": [7.6, 6.8, 6.8, 7.9],
            "poster_path": ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"],
            "director": ["Jon Favreau", "Jon Favreau", "Kenneth Branagh", "Michael Mann"],
            "cast_names": [np.array(["Robert Downey Jr."]), np.array(["Robert Downey Jr."]), np.array(["Chris Hemsworth"]), np.array(["Al Pacino"])],
        }
    )


@pytest.fixture
def service(frame) -> RecommendationService:
    """A RecommendationService whose engines are mocks (no models are loaded)."""
    svc = RecommendationService.__new__(RecommendationService)
    svc.movies = frame
    svc.tmdb = MagicMock()
    svc.omdb = MagicMock()
    svc.semantic = MagicMock()
    svc.visual = MagicMock()
    svc.fusion = MagicMock()
    svc.catalog = MovieCatalog(frame, tmdb=svc.tmdb)
    svc.tmdb.get_movie_full.return_value = _FULL
    svc.tmdb.search_movies.return_value = []
    svc.omdb.get_ratings_summary.return_value = {"imdb_rating": 7.9, "awards": "Nominated for 2 Oscars."}
    svc.fusion.build_discovery_rows.return_value = [RecommendationRow(title="More Like Iron Man", reason="similar", movie_ids=[2, 3])]
    return svc


def test_compose_card_merges_tmdb_omdb_and_picks_trailer_and_certificate():
    card = compose_card({"movie_id": 1, "tmdb_id": 1726, "title": "Iron Man"}, None, _FULL, {"imdb_rating": 7.9})
    assert card["trailer_url"].endswith("v=TRAILER")  # official YouTube trailer beats teaser and Vimeo
    assert card["certificate"] == "UA (IN)"  # India preferred
    assert [c["name"] for c in card["cast"]] == ["Robert Downey Jr.", "Gwyneth Paltrow"]
    assert card["cast"][0]["photo_url"].endswith("/w185/rdj.jpg") and card["cast"][1]["photo_url"] is None
    assert card["crew"]["directors"][0]["name"] == "Jon Favreau"
    assert card["crew"]["writers"][0]["name"] == "Mark Fergus" and card["crew"]["composers"][0]["name"] == "Ramin Djawadi"
    assert card["genres"] == ["Action", "Sci-Fi"] and card["year"] == 2008 and card["runtime"] == 126
    assert card["backdrop_url"].endswith("/w1280/b.jpg") and card["ratings"]["imdb"]["imdb_rating"] == 7.9


def test_compose_card_falls_back_to_catalogue_when_tmdb_is_down():
    row = {"director": "Michael Mann", "cast_names": ["Al Pacino"], "overview": "Cops and robbers.", "music_director": "Elliot Goldenthal"}
    summary = {"movie_id": 4, "title": "Heat", "year": 1995, "genres": ["Crime"], "rating": 7.9, "poster_url": "http://p"}
    card = compose_card(summary, row, None, None)
    assert card["title"] == "Heat" and card["overview"] == "Cops and robbers." and card["genres"] == ["Crime"]
    assert card["cast"][0]["name"] == "Al Pacino" and card["cast"][0]["photo_url"] is None
    assert card["crew"]["directors"][0]["name"] == "Michael Mann" and card["trailer_url"] is None


def test_search_by_title_returns_named_cards_with_posters(service):
    results = service.search_movies("iron man")
    assert [r["title"] for r in results][:2] == ["Iron Man", "Iron Man 2"]
    assert all(r["poster_url"] and r["match"] == "Title match" for r in results[:2])


def test_search_adds_tmdb_hits_mapped_to_catalogue_and_tmdb_only_titles(service):
    service.tmdb.search_movies.return_value = [
        {"id": 10195, "title": "Thor", "release_date": "2011-04-21", "poster_path": "/t.jpg"},
        {"id": 999999, "title": "Brand New 2026 Film", "release_date": "2026-01-01", "overview": "Fresh.", "poster_path": "/n.jpg", "vote_average": 7.1},
    ]
    results = service.search_movies("brand new")
    by_title = {r["title"]: r for r in results}
    assert by_title["Brand New 2026 Film"]["movie_id"] is None and by_title["Brand New 2026 Film"]["tmdb_id"] == 999999
    assert by_title["Brand New 2026 Film"]["match"] == "On TMDB"
    assert by_title["Thor"]["movie_id"] == 3  # mapped back to our catalogue so it opens with recommendations


def test_search_survives_tmdb_and_semantic_failures(service):
    service.tmdb.search_movies.side_effect = RuntimeError("TMDB down")
    service.semantic.search.side_effect = RuntimeError("chroma down")
    assert [r["title"] for r in service.search_movies("heat")] == ["Heat"]


def test_descriptive_queries_use_semantic_search_with_plot_match_label(service):
    service.semantic.search.return_value = [{"movie_id": 4, "title": "Heat", "similarity_pct": 83.0}]
    results = service.search_movies("crime drama about cops and thieves")
    assert results[0]["movie_id"] == 4 and results[0]["match"] == "Plot match 83.0%" and results[0]["score"] == 83.0


def test_empty_query_browses_genre_and_applies_filters(service):
    results = service.search_movies("", genres=["Action"], year_min=2009)
    assert [r["title"] for r in results] == ["Iron Man 2", "Thor"]
    assert all(r["match"] == "Popular in Action" for r in results)
    assert service.search_movies("iron man", min_rating=7.0)[0]["title"] == "Iron Man"


def test_get_movie_card_builds_full_page_with_related_rows(service):
    card = service.get_movie_card(1)
    assert card["title"] == "Iron Man" and card["ratings"]["imdb"]["awards"] == "Nominated for 2 Oscars."
    titles = [row["title"] for row in card["rows"]]
    assert titles[0].startswith("Popular Action") and "More Like Iron Man" in titles  # genre row + engine rows
    related = {m["title"] for row in card["rows"] for m in row["movies"]}
    assert {"Iron Man 2", "Thor"} <= related and "Iron Man" not in related  # never recommends itself
    assert card["warnings"] == []


def test_get_movie_card_still_works_when_tmdb_and_omdb_fail(service):
    service.tmdb.get_movie_full.side_effect = RuntimeError("timed out")
    service.omdb.get_ratings_summary.side_effect = RuntimeError("401")
    card = service.get_movie_card(4)
    assert card["title"] == "Heat" and card["crew"]["directors"][0]["name"] == "Michael Mann"
    assert len(card["warnings"]) == 2 and card["rows"]


def test_get_movie_card_unknown_id(service):
    with pytest.raises(RecommendationServiceError, match="Unknown"):
        service.get_movie_card(12345)


def test_get_movie_card_adds_tmdb_credit_rows_when_engine_has_none(service):
    service.fusion.build_discovery_rows.return_value = []
    service.tmdb.get_person_filmography.return_value = [{"id": 10138}, {"id": 1726}, {"id": 10195}]
    rows = service.get_movie_card(1)["rows"]
    titles = [r["title"] for r in rows]
    assert "By Jon Favreau" in titles and "More with Robert Downey Jr." in titles
    by_director = next(r for r in rows if r["title"] == "By Jon Favreau")
    assert [m["title"] for m in by_director["movies"]] == ["Iron Man 2", "Thor"]  # the seed itself is skipped


def test_tmdb_card_for_known_id_delegates_and_unknown_id_uses_tmdb_rows(service):
    assert service.get_tmdb_movie_card(1726)["title"] == "Iron Man"

    new_release = dict(_FULL, id=999999, title="Brand New", imdb_id=None)
    service.tmdb.get_movie_full.return_value = new_release
    service.tmdb.get_movie_recommendations.return_value = [{"id": 10138, "title": "Iron Man 2"}, {"id": 5, "title": "Unknown To Us", "release_date": "2020-01-01"}]
    service.semantic.search.return_value = [{"movie_id": 3, "title": "Thor", "similarity_pct": 80.0}]
    card = service.get_tmdb_movie_card(999999)
    assert card["title"] == "Brand New" and card["movie_id"] is None
    movies = {m["title"]: m for row in card["rows"] for m in row["movies"]}
    assert movies["Iron Man 2"]["movie_id"] == 2 and movies["Unknown To Us"]["movie_id"] is None
    assert "Thor" in movies  # semantic "similar plots in our catalogue" row


def test_get_person_maps_filmography_to_catalogue(service):
    service.tmdb.get_person.return_value = {"id": 3223, "name": "Robert Downey Jr.", "profile_path": "/p.jpg", "birthday": "1965-04-04", "biography": "x" * 900}
    service.tmdb.get_person_filmography.return_value = [
        {"id": 1726, "title": "Iron Man", "character": "Tony Stark"},
        {"id": 424242, "title": "Not In Catalogue", "release_date": "2019-01-01", "poster_path": "/z.jpg"},
        {"id": 1726, "title": "Iron Man"},  # duplicate credit is dropped
    ]
    data = service.get_person(3223)
    assert data["person"]["name"] == "Robert Downey Jr." and len(data["person"]["biography"]) <= 701
    assert [m["title"] for m in data["movies"]] == ["Iron Man", "Not In Catalogue"]
    assert data["movies"][0]["movie_id"] == 1 and data["movies"][0]["role"] == "Tony Stark"
    assert data["movies"][1]["movie_id"] is None


def test_visual_search_image_returns_cards_with_similarity(service):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    service.visual.search_by_image.return_value = [(4, 91.5), (2, 70.0), (999, 60.0)]
    cards = service.visual_search_image(buffer.getvalue(), top_k=3)
    assert [c["movie_id"] for c in cards] == [4, 2]  # unknown id 999 is skipped
    assert cards[0]["title"] == "Heat" and cards[0]["similarity_pct"] == 91.5

    with pytest.raises(RecommendationServiceError, match="read the uploaded image"):
        service.visual_search_image(b"this is not an image")


# ---------------------------------------------------------------------------
# CineBot replies -> clickable movie cards
# ---------------------------------------------------------------------------
def test_extract_movie_mentions_handles_bold_numbered_and_plain_titles():
    from services.recommendation_service import extract_movie_mentions

    text = (
        "Arre waah, you loved Nolan!\n"
        "1. **The Batman (2022)**\n"
        "   - Why it fits: dark noir.\n"
        "2. **Heat (1995)** — a classic\n"
        "Also try Mission: Impossible - Fallout (2018), it rocks.\n"
        "No year here: **Inception**"
    )
    mentions = extract_movie_mentions(text)
    assert [(m["clean"], m["year"]) for m in mentions[:2]] == [("The Batman", 2022), ("Heat", 1995)]
    assert [m["year"] for m in mentions] == [2022, 1995, 2018]
    assert "Mission: Impossible - Fallout" in mentions[2]["candidates"]  # lead-in words are trimmed off


def test_resolve_movies_in_text_prefers_catalogue_then_tmdb_and_dedupes(service):
    service.tmdb.search_movies.side_effect = lambda title: (
        [{"id": 414906, "title": "The Batman", "release_date": "2022-03-01", "poster_path": "/b.jpg", "vote_average": 7.7}]
        if "batman" in title.lower() else []
    )
    reply = (
        "Aap ye dekho:\n1. **Heat (1995)** - gritty\n2. **The Batman (2022)** - noir\n"
        "3. **Heat (1995)** again\n4. **Totally Made Up Film (2001)**"
    )
    movies = service.resolve_movies_in_text(reply)
    assert [m["title"] for m in movies] == ["Heat", "The Batman"]  # dedupe + unknown film dropped
    assert movies[0]["movie_id"] == 4 and movies[0]["source"] == "catalog"
    assert movies[1]["movie_id"] is None and movies[1]["tmdb_id"] == 414906  # recent release: opens via TMDB


def test_resolve_movies_trims_lead_in_words_for_plain_text_titles(service):
    service.tmdb.search_movies.side_effect = lambda title: (
        [{"id": 56292, "title": "Mission: Impossible - Ghost Protocol", "release_date": "2011-12-07"}]
        if title.startswith("Mission") else []
    )
    movies = service.resolve_movies_in_text("Also try Mission: Impossible - Ghost Protocol (2011), it rocks.")
    assert [m["tmdb_id"] for m in movies] == [56292]


def test_resolve_movies_year_mismatch_is_rejected(service):
    service.tmdb.search_movies.return_value = [{"id": 1, "title": "Heat", "release_date": "1986-01-01"}]
    assert service.resolve_movies_in_text("**Heat (2011)**") == []


def test_chat_with_movies_returns_reply_and_cards_and_never_fails_on_resolution(service):
    service.chatbot = MagicMock()
    service.chatbot.chat.return_value = "Dekho **Heat (1995)**!"
    result = service.chat_with_movies("crime movie")
    assert result["reply"] == "Dekho **Heat (1995)**!" and result["movies"][0]["movie_id"] == 4

    service.catalog = None  # resolution blows up -> the reply must still come back
    result = service.chat_with_movies("crime movie")
    assert result == {"reply": "Dekho **Heat (1995)**!", "movies": []}
