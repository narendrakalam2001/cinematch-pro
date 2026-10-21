"""UI tests for frontend/app.py using Streamlit's AppTest harness (API calls are mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")

_TILE = {"movie_id": 2, "title": "Interstellar", "year": 2014, "genres": ["Sci-Fi"], "rating": 8.4, "poster_url": None, "overview": "Space.", "match": "Title match"}
_CARD = {
    "movie_id": 1,
    "title": "Inception",
    "year": 2010,
    "runtime": 148,
    "certificate": "UA (IN)",
    "genres": ["Action", "Sci-Fi"],
    "overview": "Dreams within dreams.",
    "poster_url": None,
    "backdrop_url": None,
    "ratings": {"tmdb": {"score": 8.4, "votes": 1000}, "imdb": {"imdb_rating": 8.8, "awards": "Won 4 Oscars."}},
    "cast": [{"id": 6193, "name": "Leonardo DiCaprio", "character": "Cobb", "photo_url": None}],
    "crew": {"directors": [{"id": 525, "name": "Christopher Nolan"}], "writers": [], "producers": [], "composers": [], "cinematographers": []},
    "trailer_url": None,
    "budget": 160000000,
    "revenue": 836800000,
    "rows": [{"title": "More Like Inception", "reason": "Similar plot", "movies": [_TILE]}],
    "warnings": [],
}
_INTERSTELLAR = dict(_CARD, movie_id=2, title="Interstellar", year=2014, rows=[])
_PERSON = {"person": {"id": 6193, "name": "Leonardo DiCaprio", "photo_url": None, "biography": "Actor."}, "movies": [_TILE]}
_SENTIMENT = {
    "movie_id": 1,
    "overall_positive_pct": 82.0,
    "review_count": 5,
    "aspects": {"story": {"positive_pct": 90.0, "mention_count": 4, "label": "Excellent", "stars": 5}},
}


def _response(payload: dict, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    if status >= 400:
        error = requests.HTTPError(f"{status}")
        error.response = response
        response.raise_for_status.side_effect = error
    else:
        response.raise_for_status.return_value = None
    return response


def _fake_get(url: str, **kwargs):
    if url.endswith("/health"):
        return _response({"status": "ok", "engines": {"movies_loaded": True}})
    if url.endswith("/movie/2"):
        return _response(_INTERSTELLAR)
    if "/movie/" in url:
        return _response(_CARD)
    if url.endswith("/search"):
        kwargs_params = kwargs.get("params") or {}
        return _response({"query": kwargs_params.get("q", ""), "count": 1, "results": [dict(_TILE, movie_id=1, title="Inception")]})
    if "/person/" in url:
        return _response(_PERSON)
    return _response({"movies": []})


def _movie_card_app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["_nav_target"] = "Movie Card"
    app.session_state["current_movie_id"] = 1
    return app


def _button(app: AppTest, label: str):
    return next(b for b in app.button if b.label == label)


def test_movie_card_shows_sentiment_dashboard_after_click():
    """Clicking 'Analyze review sentiment' calls POST /sentiment/{id} and renders the results."""
    with patch("requests.get", side_effect=_fake_get), patch(
        "requests.post", return_value=_response(_SENTIMENT)
    ) as post:
        app = _movie_card_app().run()
        assert not app.exception
        _button(app, "Analyze review sentiment").click().run()

    assert post.call_args.args[0].endswith("/sentiment/1")
    assert post.call_args.kwargs["json"] == {"reviews": []}
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Overall sentiment"] == "🟢 82%"
    assert metrics["Reviews analyzed"] == "5"
    assert any("Story" in md.value and "Excellent" in md.value for md in app.markdown)


def test_sentiment_http_error_shows_the_apis_own_message_not_a_connectivity_error():
    """A 400 'no reviews' answer must be shown as such, not as 'Couldn't reach the API'."""
    detail = {"detail": "No reviews available on TMDB for movie_id 1."}
    with patch("requests.get", side_effect=_fake_get), patch(
        "requests.post", return_value=_response(detail, status=400)
    ):
        app = _movie_card_app().run()
        _button(app, "Analyze review sentiment").click().run()

    warnings = " ".join(w.value for w in app.warning)
    assert "No reviews available on TMDB" in warnings
    assert "Couldn't reach" not in warnings


def test_movie_card_stays_on_screen_after_programmatic_navigation_and_clicks():
    """Regression: after arriving via _nav_target, later clicks must not bounce the user to Home."""
    with patch("requests.get", side_effect=_fake_get), patch("requests.post", return_value=_response(_SENTIMENT)):
        app = _movie_card_app().run()
        for _ in range(2):  # two further interactions
            app = _button(app, "➕ Watchlist").click().run()
            assert not app.exception
            assert any("Movie Intelligence Card" in md.value for md in app.markdown)
            assert app.sidebar.radio[0].value == "Movie Card"
    assert app.session_state["watchlist"][1]["status"] == "want_to_watch"


def _search_app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["_nav_target"] = "Smart Search"
    return app


def test_search_shows_named_results_and_keeps_them_after_clicking_view():
    """Regression: results used to vanish (and nothing opened) because View sat inside `if st.button`."""
    with patch("requests.get", side_effect=_fake_get) as get:
        app = _search_app().run()
        app.text_input(key="sq_query").set_value("inception")
        _button(app, "Search").click().run()
        assert not app.exception
        searches = [c for c in get.call_args_list if str(c.args[0]).endswith("/search")]
        assert searches and searches[-1].kwargs["params"]["q"] == "inception"
        assert any("1 results for" in md.value for md in app.markdown)
        assert any("Inception" in md.value for md in app.markdown)  # a real title, not "Movie ID: 1"

        app = _button(app, "View ▶").click().run()
        assert not app.exception
        assert app.sidebar.radio[0].value == "Movie Card"
        assert any("Inception" in md.value for md in app.markdown)  # the opened movie's own page


def test_genre_chip_on_home_opens_search_prefilled_with_that_genre():
    """Clicking a genre chip jumps to Smart Search with the genre selected and runs the search."""
    with patch("requests.get", side_effect=_fake_get) as get:
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
        _button(app, "Thriller").click().run()
        assert not app.exception
        assert app.sidebar.radio[0].value == "Smart Search"
        assert app.multiselect(key="sq_genres").value == ["Thriller"]
        searched = [c for c in get.call_args_list if str(c.args[0]).endswith("/search")]
        assert searched and searched[-1].kwargs["params"]["genres"] == ["Thriller"]


def test_movie_card_shows_details_cast_crew_and_related_rows():
    """The detail page lists cast, crew, box office, awards and clickable related movies."""
    with patch("requests.get", side_effect=_fake_get):
        app = _movie_card_app().run()
    assert not app.exception
    text = " ".join(md.value for md in app.markdown)
    for expected in ("Inception", "Leonardo DiCaprio", "Christopher Nolan", "More Like Inception", "Interstellar", "Won 4 Oscars."):
        assert expected in text
    assert any(b.label == "Filmography" for b in app.button)
    assert any(b.label == "View ▶" for b in app.button)


def test_clicking_a_related_movie_opens_it_and_back_returns():
    """The infinite discovery loop: related movie opens its own page; Back goes to the previous movie."""
    with patch("requests.get", side_effect=_fake_get):
        app = _movie_card_app().run()
        app = _button(app, "View ▶").click().run()
        assert not app.exception
        assert app.session_state["current_movie_id"] == 2
        assert any("Interstellar" in md.value for md in app.markdown)
        assert [t["title"] for t in app.session_state["trail"]] == ["Interstellar"]

        app = _button(app, "← Back").click().run()
        assert not app.exception
        assert app.sidebar.radio[0].value == "Home"


def test_actor_filmography_page_and_back_to_movie():
    """Clicking an actor opens their filmography; 'Back to movie' returns to the card."""
    with patch("requests.get", side_effect=_fake_get):
        app = _movie_card_app().run()
        app = _button(app, "Filmography").click().run()
        assert not app.exception
        assert app.session_state["view_person_id"] == 6193
        assert any("Leonardo DiCaprio" in md.value for md in app.markdown)
        app = _button(app, "← Back to movie").click().run()
        assert app.session_state["view_person_id"] is None
        assert any("Christopher Nolan" in md.value for md in app.markdown)


def test_tmdb_only_movie_opens_via_tmdb_endpoint():
    """A trending movie not in MovieLens opens through /tmdb-movie/{id}."""
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "/tmdb-movie/" in url:
            return _response(dict(_CARD, movie_id=None, title="Brand New Release", rows=[]))
        return _fake_get(url, **kwargs)

    with patch("requests.get", side_effect=fake_get):
        app = AppTest.from_file(APP_PATH, default_timeout=30)
        app.session_state["_nav_target"] = "Movie Card"
        app.session_state["current_tmdb_id"] = 999999
        app.run()
    assert not app.exception
    assert any(u.endswith("/tmdb-movie/999999") for u in calls)
    assert any("Brand New Release" in md.value for md in app.markdown)
    assert any("Sentiment analysis is available for movies in our catalogue." in c.value for c in app.caption)


_CHAT_REPLY = {
    "reply": "Try **Interstellar (2014)** — mind-bending sci-fi!",
    "movies": [_TILE],
}


def _cinebot_app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["_nav_target"] = "CineBot"
    return app


def test_chat_box_stays_visible_after_clicking_a_suggestion_chip():
    """Regression: `pending or st.chat_input()` skipped the input after a chip click, hiding the chat box."""
    with patch("requests.get", side_effect=_fake_get), patch("requests.post", return_value=_response(_CHAT_REPLY)):
        app = _cinebot_app().run()
        assert len(app.chat_input) == 1  # visible before any chat
        app = _button(app, "Movies like The Dark Knight").click().run()
        assert not app.exception
        assert len(app.chat_input) == 1  # still visible right after the chip click
        assert [m.name for m in app.chat_message] == ["user", "assistant"]


def test_chat_suggested_movie_is_clickable_and_opens_its_page():
    """A movie CineBot mentions shows up as a card under the reply; clicking it opens that movie."""
    with patch("requests.get", side_effect=_fake_get), patch("requests.post", return_value=_response(_CHAT_REPLY)):
        app = _cinebot_app().run()
        app.chat_input[0].set_value("suggest a sci-fi").run()
        assert not app.exception
        assert any(b.label == "View ▶" for b in app.button)
        app = _button(app, "View ▶").click().run()
        assert not app.exception
        assert app.sidebar.radio[0].value == "Movie Card"
        assert app.session_state["current_movie_id"] == 2
        # coming back to CineBot keeps the conversation and the cards
        app.session_state["_nav_target"] = "CineBot"
        app.run()
        assert len(app.session_state["chat_history"]) == 2 and len(app.chat_input) == 1


def test_sidebar_hides_api_details_and_is_branded():
    """The API URL / engine count are not shown to users; the sidebar has the brand block."""
    with patch("requests.get", side_effect=_fake_get):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    sidebar_text = " ".join(m.value for m in app.sidebar.markdown) + " ".join(c.value for c in app.sidebar.caption)
    assert "localhost" not in sidebar_text and "API:" not in sidebar_text and "Engines ready" not in sidebar_text
    assert "CineMatch" in sidebar_text and "AI Movie Intelligence" in sidebar_text
    assert not app.sidebar.error  # backend is up
    assert [o for o in app.sidebar.radio[0].options] == ["🏠  Home", "🖼️  Visual Search", "🎥  Movie Card", "🔍  Smart Search", "💬  CineBot", "📌  My Watchlist"]


def test_offline_backend_shows_friendly_notice_without_url():
    """If /health fails the sidebar says so plainly, without leaking the API address."""
    import requests as real_requests

    def down(url, **kwargs):
        raise real_requests.exceptions.ConnectionError("refused")

    with patch("requests.get", side_effect=down):
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert any("offline" in e.value.lower() for e in app.sidebar.error)
    assert not any("localhost" in e.value for e in app.sidebar.error)


def test_backend_url_comes_from_the_api_base_url_variable(monkeypatch):
    """On Streamlit Cloud the secret API_BASE_URL points the UI at the deployed backend (no trailing slash)."""
    monkeypatch.setenv("API_BASE_URL", "https://user-cinematch-pro-api.hf.space/")
    with patch("requests.get", side_effect=_fake_get) as get:
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    urls = [str(c.args[0]) for c in get.call_args_list]
    assert "https://user-cinematch-pro-api.hf.space/health" in urls
    assert not any("localhost" in u for u in urls)
