"""
CineMatch Pro — Streamlit frontend.

A six-screen, Netflix-styled portal (dark theme, red primary, gold
accent) that talks to the FastAPI backend in `serving/api.py` over
HTTP. Screens: Home/Discovery, Visual Search, Movie Intelligence Card,
Smart Search Results, CineBot Chat, and My Watchlist.

Run with:
    streamlit run frontend/app.py
or:
    python -m scripts.run_app
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st



def _resolve_api_base_url() -> str:
    """Backend URL from ``API_BASE_URL`` (env var or Streamlit Cloud secret), else localhost.

    Returns:
        str: The API base URL without a trailing slash.
    """
    url = os.getenv("API_BASE_URL")
    if not url:
        try:
            url = st.secrets.get("API_BASE_URL")
        except Exception:  # noqa: BLE001 - no secrets file locally; that's fine
            url = None
    return (url or "http://localhost:8000").rstrip("/")


API_BASE_URL = _resolve_api_base_url()

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --bg-primary: #0B0B0D; --bg-card: #16161A; --bg-card-2: #1C1C21; --line: #26262C;
    --accent-red: #E50914; --accent-red-2: #FF4D5A; --accent-gold: #F5C518;
    --text-primary: #FFFFFF; --text-muted: #A1A1AA;
}
html, body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}
.stApp {
    background:
        radial-gradient(1100px 520px at 88% -8%, rgba(229,9,20,0.15), transparent 60%),
        radial-gradient(900px 520px at -8% 6%, rgba(99,102,241,0.10), transparent 55%),
        var(--bg-primary);
    color: var(--text-primary);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, .stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] {
    display: none !important; visibility: hidden !important;
}
.block-container { padding-top: 2rem; max-width: 1480px; }
h1, h2, h3 { color: var(--text-primary) !important; letter-spacing: -0.01em; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E0E11 0%, #121216 100%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
.cm-brand { display: flex; align-items: center; gap: 0.75rem; padding: 0.4rem 0.3rem 1.2rem 0.3rem; }
.cm-logo {
    width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; background: linear-gradient(135deg, #FF4D5A, #B0060F); box-shadow: 0 6px 20px rgba(229,9,20,0.45);
}
.cm-brand-name { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.cm-brand-name span {
    font-size: 0.62rem; font-weight: 800; letter-spacing: 0.12em; padding: 2px 6px; margin-left: 4px; border-radius: 5px;
    background: var(--accent-gold); color: #111; vertical-align: middle;
}
.cm-brand-tag { color: var(--text-muted); font-size: 0.78rem; margin-top: 2px; }
.cm-nav-label { color: #6B6B75; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; margin: 0.2rem 0 0.4rem 0.4rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 6px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    width: 100%; padding: 0.7rem 0.9rem; border-radius: 12px; cursor: pointer; margin: 0;
    border: 1px solid transparent; transition: background 0.15s, border-color 0.15s, transform 0.1s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child:not([data-testid]) { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-size: 1rem; font-weight: 600; color: #D4D4D8; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #1C1C21; transform: translateX(2px); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(229,9,20,0.32), rgba(229,9,20,0.06));
    border-color: rgba(229,9,20,0.55); box-shadow: inset 3px 0 0 var(--accent-red);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: #FFFFFF; }
.cm-side-foot { color: #5F5F69; font-size: 0.72rem; line-height: 1.5; margin-top: 1.4rem; padding: 0 0.4rem; }
.cm-pill-status {
    display: inline-block; font-size: 0.74rem; padding: 3px 10px; border-radius: 999px; margin: 0.8rem 0 0 0.3rem;
    background: rgba(255,255,255,0.05); border: 1px solid var(--line); color: #C9C9D1;
}

/* ---------- Page headers ---------- */
.cm-page-title { display: flex; align-items: center; gap: 0.7rem; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
.cm-page-title .cm-ico {
    width: 42px; height: 42px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; font-size: 1.3rem;
    background: linear-gradient(135deg, rgba(229,9,20,0.35), rgba(229,9,20,0.08)); border: 1px solid rgba(229,9,20,0.4);
}
.cm-page-sub { color: var(--text-muted); margin: 0 0 1.2rem 0; font-size: 0.98rem; }

/* ---------- Hero ---------- */
.cm-hero {
    position: relative; overflow: hidden; border-radius: 22px; padding: 3.2rem 2.8rem; margin-bottom: 1.6rem;
    border: 1px solid #2B2B33; box-shadow: 0 24px 60px rgba(0,0,0,0.5);
    background: linear-gradient(120deg, #1B0609 0%, #13131B 48%, #0D0D12 100%);
}
.cm-hero::before {
    content: ""; position: absolute; top: -45%; right: -8%; width: 640px; height: 640px; border-radius: 50%;
    background: radial-gradient(circle, rgba(229,9,20,0.50), transparent 62%); animation: cmfloat 9s ease-in-out infinite alternate;
}
.cm-hero::after {
    content: ""; position: absolute; bottom: -50%; left: -8%; width: 540px; height: 540px; border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.32), transparent 62%); animation: cmfloat 11s ease-in-out infinite alternate-reverse;
}
.cm-hero-inner { position: relative; z-index: 1; }
.cm-eyebrow { color: var(--accent-gold); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.22em; }
.cm-hero-title { font-size: 3.6rem; font-weight: 800; line-height: 1.04; letter-spacing: -0.03em; margin: 0.5rem 0 0.7rem 0; }
.cm-grad { background: linear-gradient(90deg, #FF6A74, #E50914 45%, #FF9966); -webkit-background-clip: text; background-clip: text; color: transparent; }
.cm-tagline { color: #D4D4DB; font-size: 1.2rem; font-weight: 500; margin: 0 0 0.4rem 0; }
.cm-hero-sub { color: #A8A8B3; max-width: 640px; font-size: 1rem; line-height: 1.55; margin: 0; }
.cm-stats span {
    display: inline-block; margin: 1.1rem 0.5rem 0 0; padding: 0.38rem 0.85rem; border-radius: 999px; font-size: 0.84rem; color: #ECECEF;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.13); backdrop-filter: blur(6px);
}
@keyframes cmfloat { from { transform: translate(0, 0) scale(1); } to { transform: translate(-36px, 24px) scale(1.08); } }

/* ---------- Shared components ---------- */
.cm-card { background-color: var(--bg-card); border-radius: 12px; padding: 1rem; border: 1px solid var(--line); height: 100%; }
.cm-badge-green { color: #2ECC71; font-weight: 600; }
.cm-badge-yellow { color: var(--accent-gold); font-weight: 600; }
.cm-badge-red { color: var(--accent-red); font-weight: 600; }
.cm-row-title {
    font-size: 1.35rem; font-weight: 800; letter-spacing: -0.01em; margin: 1.8rem 0 0.15rem 0; padding-left: 0.75rem;
    border-left: 4px solid var(--accent-red);
}
.cm-reason { color: var(--text-muted); font-size: 0.85rem; margin: 0 0 0.6rem 0.9rem; }
.cm-chip {
    display: inline-block; padding: 3px 12px; margin: 0 6px 6px 0; border-radius: 999px; font-size: 0.8rem; font-weight: 500;
    background: #1F1F25; color: #E7E7EA; border: 1px solid #34343C;
}
.cm-tile-title { font-weight: 700; font-size: 0.93rem; margin-top: 0.45rem; line-height: 1.25; }
.cm-tile-meta { color: var(--text-muted); font-size: 0.78rem; margin: 0.1rem 0 0.15rem 0; }
.cm-rating { color: var(--accent-gold); font-weight: 700; font-size: 0.8rem; }
.cm-match { color: #7FDBFF; font-size: 0.72rem; margin-bottom: 0.2rem; }
.cm-noposter {
    display: flex; align-items: center; justify-content: center; text-align: center; aspect-ratio: 2 / 3;
    background: linear-gradient(160deg, #202028, #16161A); border: 1px solid #2B2B33; border-radius: 12px;
    color: #8C8C95; padding: 0.6rem; font-size: 0.85rem; font-weight: 600;
}
.cm-backdrop {
    border-radius: 20px; padding: 5rem 2.2rem 1.6rem 2.2rem; margin-bottom: 1.1rem; border: 1px solid #2B2B33;
    background-size: cover; background-position: center top; box-shadow: 0 20px 50px rgba(0,0,0,0.45);
}
.cm-backdrop-title { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; margin: 0; }
.cm-backdrop .cm-sub { color: #D6D6DC; margin-top: 0.4rem; font-weight: 500; }
.cm-crumbs { color: var(--text-muted); font-size: 0.85rem; padding-top: 0.45rem; }
.cm-person-photo-fallback {
    display: flex; align-items: center; justify-content: center; aspect-ratio: 2 / 3; background: #202028;
    border-radius: 12px; color: #8C8C95; font-size: 1.6rem; font-weight: 700;
}
.cm-welcome {
    border: 1px solid var(--line); border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    background: linear-gradient(120deg, rgba(229,9,20,0.10), rgba(99,102,241,0.07));
}
.cm-welcome b { font-size: 1.05rem; }
.cm-welcome p { color: #B8B8C2; margin: 0.3rem 0 0 0; }

/* ---------- Streamlit widgets ---------- */
.stButton > button, .stFormSubmitButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
    background: linear-gradient(180deg, #F0161F, #C20711); color: #FFFFFF; border: 0; border-radius: 10px; font-weight: 600;
    box-shadow: 0 3px 12px rgba(229,9,20,0.28); transition: transform 0.12s, box-shadow 0.12s, filter 0.12s;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    color: #FFFFFF; filter: brightness(1.12); transform: translateY(-1px); box-shadow: 0 6px 18px rgba(229,9,20,0.42); border: 0;
}
.stButton > button:active { transform: translateY(0); }
.stTextInput input, .stNumberInput input, [data-baseweb="select"] > div, .stTextArea textarea {
    border-radius: 12px !important; background: #17171C !important; border: 1px solid #2D2D35 !important;
}
.stTextInput input:focus { border-color: var(--accent-red) !important; box-shadow: 0 0 0 1px var(--accent-red) !important; }
[data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--line); border-radius: 14px; padding: 0.9rem 1.1rem; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,0.015); }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 0.5rem 1rem; }
.stTabs [aria-selected="true"] { color: #FFFFFF; }
[data-testid="stImage"] img { border-radius: 12px; transition: transform 0.2s ease, box-shadow 0.2s ease; }
[data-testid="stImage"] img:hover { transform: scale(1.035); box-shadow: 0 12px 32px rgba(0,0,0,0.6); }
[data-testid="stChatMessage"] { background: var(--bg-card); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.2rem; }
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stForm"] { border: 0; padding: 0; }
hr { border-color: var(--line); }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API client helpers — thin wrappers around the FastAPI backend with
# defensive error handling so a backend hiccup never crashes the UI.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Backend transport — remote API (default) or the same API running inside this process
# ---------------------------------------------------------------------------
# Embedded mode (CINEMATCH_EMBEDDED=1) is how the free "lite" deployment on Streamlit Community Cloud
# runs: there is no separate API server — the FastAPI app is called in-process, so every screen uses
# exactly the same endpoints and code as the full deployment.
_EMBEDDED = os.getenv("CINEMATCH_EMBEDDED", "").strip().lower() in {"1", "true", "yes", "on"}

try:  # httpx is present whenever the backend is (embedded mode / full install); a UI-only install may lack it
    import httpx

    _HTTP_ERRORS: tuple[type[BaseException], ...] = (requests.exceptions.RequestException, httpx.HTTPError)
except ImportError:  # pragma: no cover - depends on the install
    _HTTP_ERRORS = (requests.exceptions.RequestException,)


@st.cache_resource(show_spinner="Starting CineMatch engines — the first load takes a minute...")
def _embedded_client() -> Any:
    """Create (once per server process) the in-process client for the FastAPI app.

    Entering the client runs the API's startup: it downloads the trained artifacts from the Hugging Face
    Hub if they are missing and loads the engines.

    Returns:
        Any: A started ``fastapi.testclient.TestClient``.
    """
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from fastapi.testclient import TestClient

    from serving.api import app

    client = TestClient(app, raise_server_exceptions=False)  # server errors come back as HTTP 500, not crashes
    client.__enter__()
    return client


def _fetch_health() -> dict[str, Any] | None:
    """Ask the backend which engines are loaded (``None`` if it is unreachable).

    Returns:
        dict[str, Any] | None: The ``/health`` JSON body, or ``None``.
    """
    try:
        if _EMBEDDED:
            response = _embedded_client().get("/health")
        else:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except _HTTP_ERRORS:
        return None
    except Exception:  # noqa: BLE001 - embedded startup failures (missing artifacts, import errors) must not crash the page
        return None


def _engine_available(name: str) -> bool:
    """Whether a backend engine is loaded (unknown engines count as available).

    Args:
        name: Key from the ``/health`` engines map, e.g. ``"visual_search"``.

    Returns:
        bool: ``False`` only when the backend explicitly reported the engine as not loaded.
    """
    engines = st.session_state.get("_engines") or {}
    return bool(engines.get(name, True))


def _describe_request_error(path: str, exc: Exception) -> str:
    """Turn a `requests` failure into a message that says what really went wrong.

    Args:
        path: The API path that was called.
        exc: The exception raised by `requests`.

    Returns:
        str: For an HTTP error response, the API's own ``detail`` message
        (e.g. "No reviews available ..."); otherwise a connectivity message.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return f"{path}: {detail}"
    return f"Couldn't reach CineMatch Pro API ({path}): {exc}"


def _api_get(path: str, params: dict[str, Any] | None = None, timeout: int = 15) -> dict[str, Any] | None:
    """GET a JSON endpoint from the CineMatch Pro API, handling errors gracefully.

    Args:
        path: API path, e.g. ``"/trending"``.
        params: Optional query parameters.
        timeout: Request timeout in seconds.

    Returns:
        dict[str, Any] | None: Parsed JSON body, or `None` on any
        request failure (network error, timeout, non-2xx response). The
        caller is responsible for showing an appropriate UI fallback.
    """
    try:
        if _EMBEDDED:
            response = _embedded_client().get(path, params=params)
        else:
            response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except _HTTP_ERRORS as exc:
        st.warning(_describe_request_error(path, exc))
        return None


def _api_post(path: str, json_body: dict[str, Any], timeout: int = 30) -> dict[str, Any] | None:
    """POST a JSON body to the CineMatch Pro API, handling errors gracefully.

    Args:
        path: API path, e.g. ``"/recommend"``.
        json_body: JSON-serializable request body.
        timeout: Request timeout in seconds.

    Returns:
        dict[str, Any] | None: Parsed JSON body, or `None` on any
        request failure.
    """
    try:
        if _EMBEDDED:
            response = _embedded_client().post(path, json=json_body)
        else:
            response = requests.post(f"{API_BASE_URL}{path}", json=json_body, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except _HTTP_ERRORS as exc:
        st.warning(_describe_request_error(path, exc))
        return None


def _sentiment_badge(pct: float | None) -> str:
    """Render a colored sentiment badge string for a positive-sentiment %.

    Args:
        pct: Positive-sentiment percentage (0-100), or `None` if unknown.

    Returns:
        str: Markdown-formatted badge, e.g. ``"🟢 87%"``.
    """
    if pct is None:
        return "⚪ N/A"
    if pct >= 75:
        return f"🟢 {pct:.0f}%"
    if pct >= 50:
        return f"🟡 {pct:.0f}%"
    return f"🔴 {pct:.0f}%"


def _api_post_file(
    path: str, file_name: str, data: bytes, mime: str, params: dict[str, Any] | None = None, timeout: int = 120
) -> dict[str, Any] | None:
    """POST a file as multipart form data (used for poster-image visual search).

    Args:
        path: API path, e.g. ``"/visual-search/upload"``.
        file_name: Original file name.
        data: File bytes.
        mime: MIME type, e.g. ``"image/jpeg"``.
        params: Optional query parameters.
        timeout: Request timeout in seconds (CLIP on CPU can take a few seconds).

    Returns:
        dict[str, Any] | None: Parsed JSON body, or `None` on failure.
    """
    try:
        files = {"file": (file_name, data, mime)}
        if _EMBEDDED:
            response = _embedded_client().post(path, params=params, files=files)
        else:
            response = requests.post(f"{API_BASE_URL}{path}", params=params, files=files, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except _HTTP_ERRORS as exc:
        st.warning(_describe_request_error(path, exc))
        return None


# ---------------------------------------------------------------------------
# Navigation + shared building blocks
# ---------------------------------------------------------------------------
_GENRES = ["Action", "Comedy", "Thriller", "Drama", "Sci-Fi", "Horror", "Romance", "Animation"]
_YEAR_BOUNDS = (1900, 2026)


def _esc(value: Any) -> str:
    """HTML-escape any value for safe use inside ``unsafe_allow_html`` markup."""
    return html.escape("" if value is None else str(value))


def _page_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a consistent page title with an icon badge and optional subtitle.

    Args:
        icon: Emoji shown in the badge.
        title: Page title.
        subtitle: One-line description under the title.
    """
    st.markdown(
        f'<div class="cm-page-title"><span class="cm-ico">{icon}</span>{_esc(title)}</div>'
        + (f'<div class="cm-page-sub">{_esc(subtitle)}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def _open_movie(movie_id: int | None = None, tmdb_id: int | None = None, title: str = "") -> None:
    """Callback: open a movie's detail page (and extend the breadcrumb trail).

    Runs as a button ``on_click`` so the new page is chosen *before* the next
    script run — this is what keeps results/rows from vanishing on click.

    Args:
        movie_id: MovieLens id, for movies in our catalogue.
        tmdb_id: TMDB id, for movies we only know from TMDB.
        title: Display title for the breadcrumb.
    """
    trail = st.session_state["trail"]
    entry = {"movie_id": movie_id, "tmdb_id": tmdb_id, "title": title or "Movie"}
    if not trail or (trail[-1]["movie_id"], trail[-1]["tmdb_id"]) != (movie_id, tmdb_id):
        trail.append(entry)
    st.session_state["trail"] = trail[-12:]
    st.session_state["current_movie_id"] = movie_id
    st.session_state["current_tmdb_id"] = tmdb_id
    st.session_state["view_person_id"] = None
    st.session_state["_nav_target"] = "Movie Card"


def _go_back() -> None:
    """Callback: step back to the previous movie in the breadcrumb trail."""
    trail = st.session_state["trail"]
    if trail:
        trail.pop()
    if trail:
        previous = trail[-1]
        st.session_state["current_movie_id"] = previous["movie_id"]
        st.session_state["current_tmdb_id"] = previous["tmdb_id"]
    else:
        st.session_state["current_movie_id"] = None
        st.session_state["current_tmdb_id"] = None
    st.session_state["view_person_id"] = None
    st.session_state["_nav_target"] = "Movie Card" if trail else "Home"


def _open_person(person_id: int) -> None:
    """Callback: show an actor's / director's profile and filmography."""
    st.session_state["view_person_id"] = person_id
    st.session_state["_nav_target"] = "Movie Card"


def _close_person() -> None:
    """Callback: leave the person profile and return to the movie."""
    st.session_state["view_person_id"] = None


def _set_search(query: str = "", genres: list[str] | None = None) -> None:
    """Callback: jump to Smart Search with a pre-filled query / genre."""
    st.session_state["_search_prefill"] = {"query": query, "genres": genres or []}
    st.session_state["_nav_target"] = "Smart Search"


def _movie_tile(movie: dict[str, Any], key: str, show_match: bool = True) -> None:
    """Render one clickable poster card (poster, title, year, rating, genres, View).

    Args:
        movie: A movie summary from the API (``/search``, card rows, ...).
        key: Unique widget key for this tile's button.
        show_match: Show the "why this result" label (``match``/similarity).
    """
    title = movie.get("title") or "Unknown"
    year = movie.get("year")
    poster = movie.get("poster_url")
    if poster:
        st.image(poster, use_container_width=True)
    else:
        st.markdown(f'<div class="cm-noposter">{_esc(title)}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="cm-tile-title">{_esc(title)}</div>', unsafe_allow_html=True)
    meta_bits = [str(year)] if year else []
    meta_bits += (movie.get("genres") or [])[:2]
    meta = " · ".join(meta_bits)
    rating = movie.get("rating")
    rating_html = f'<span class="cm-rating">⭐ {float(rating):.1f}</span> ' if rating else ""
    st.markdown(f'<div class="cm-tile-meta">{rating_html}{_esc(meta)}</div>', unsafe_allow_html=True)
    if movie.get("similarity_pct") is not None:
        st.markdown(f'<div class="cm-match">🎯 {movie["similarity_pct"]}% visual match</div>', unsafe_allow_html=True)
    elif show_match and movie.get("match"):
        st.markdown(f'<div class="cm-match">{_esc(movie["match"])}</div>', unsafe_allow_html=True)

    st.button(
        "View ▶",
        key=key,
        use_container_width=True,
        help=movie.get("overview") or None,
        on_click=_open_movie,
        kwargs={"movie_id": movie.get("movie_id"), "tmdb_id": movie.get("tmdb_id"), "title": title},
    )


def _render_grid(movies: list[dict[str, Any]], key_prefix: str, per_row: int = 6, show_match: bool = True) -> None:
    """Render movies as rows of poster tiles.

    Args:
        movies: Movie summaries from the API.
        key_prefix: Prefix that keeps widget keys unique across grids.
        per_row: Tiles per line.
        show_match: Show the match label under each tile.
    """
    for start in range(0, len(movies), per_row):
        cols = st.columns(per_row)
        for offset, movie in enumerate(movies[start : start + per_row]):
            with cols[offset]:
                ident = movie.get("movie_id") or f"t{movie.get('tmdb_id')}"
                _movie_tile(movie, key=f"{key_prefix}_{ident}_{start + offset}", show_match=show_match)


def _render_trail() -> None:
    """Show the breadcrumb (Home → A → B) and a Back button."""
    trail = st.session_state["trail"]
    if not trail:
        return
    crumbs = " › ".join(["Home"] + [_esc(t["title"]) for t in trail])
    col_back, col_trail = st.columns([1, 8])
    col_back.button("← Back", key="trail_back", on_click=_go_back)
    col_trail.markdown(f'<div class="cm-crumbs">{crumbs}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state — watchlist, taste profile, view history (per-session demo)
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Initialize per-session state on first load (watchlist, history, chat, trail)."""
    defaults: dict[str, Any] = {
        "watchlist": {},  # movie_id -> {"status": ..., "title": ..., "poster_url": ..., "year": ...}
        "viewed_movies": [],  # last N movie_ids viewed, most recent first
        "viewed_meta": {},  # movie_id -> {"title", "genres", "poster_url", "year"}
        "chat_history": [],  # list of {"role": "user"|"bot", "text": str}
        "current_movie_id": None,
        "current_tmdb_id": None,
        "view_person_id": None,
        "trail": [],
        "search_results": None,
        "_card_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _record_view(movie_id: int | None, card: dict[str, Any] | None = None) -> None:
    """Add a movie to the session's view history (feeds the taste profile).

    Args:
        movie_id: The viewed movie's ID (``None`` for TMDB-only movies).
        card: The loaded card, used to remember title/genres/poster.
    """
    if movie_id is None:
        return
    history = st.session_state["viewed_movies"]
    if movie_id in history:
        history.remove(movie_id)
    history.insert(0, movie_id)
    st.session_state["viewed_movies"] = history[:10]
    if card:
        st.session_state["viewed_meta"][movie_id] = {
            "movie_id": movie_id,
            "title": card.get("title"),
            "genres": card.get("genres", []),
            "poster_url": card.get("poster_url"),
            "year": card.get("year"),
        }


# ---------------------------------------------------------------------------
# Screen 1 — Home / Discovery
# ---------------------------------------------------------------------------
def screen_home() -> None:
    """Render Screen 1: Home / Discovery — hero, search, trending, taste profile."""
    st.markdown(
        '<div class="cm-hero"><div class="cm-hero-inner">'
        '<div class="cm-eyebrow">🎬 AI-POWERED MOVIE INTELLIGENCE</div>'
        '<div class="cm-hero-title">Cine<span class="cm-grad">Match</span> Pro</div>'
        '<p class="cm-tagline">AI-Powered Movie Intelligence</p>'
        '<p class="cm-hero-sub">Search by title or mood, upload a poster, or chat with CineBot — '
        'every movie comes with cast, reviews sentiment and smart recommendations.</p>'
        '<div class="cm-stats"><span>🎞️ 62K movies</span><span>🖼️ 56K posters indexed</span>'
        '<span>🤖 RoBERTa sentiment</span><span>💬 CineBot assistant</span></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if (st.session_state.get("_engines") or {}).get("lite_mode"):
        st.info(
            "⚡ **Lite demo** — search, movie cards, cast & crew, related movies and CineBot are live. "
            "Poster visual search, plot-semantic search and RoBERTa review sentiment need the full models "
            "(see the README)."
        )

    with st.form("home_search", border=False):
        col1, col2 = st.columns([4, 1])
        search_query = col1.text_input(
            "Search movies or describe what you want to watch",
            placeholder='e.g. "Bahubali" or "Thriller with twist ending like Memento"',
        )
        submitted = col2.form_submit_button("🔎 Search", use_container_width=True)
    if submitted and search_query.strip():
        _set_search(search_query.strip())
        st.rerun()

    st.write("**Browse by genre:**")
    for col, genre in zip(st.columns(len(_GENRES)), _GENRES):
        with col:
            st.button(genre, key=f"genre_{genre}", use_container_width=True, on_click=_set_search, kwargs={"genres": [genre]})

    st.markdown('<div class="cm-row-title">🔥 Trending This Week</div>', unsafe_allow_html=True)
    trending = _api_get("/trending", params={"time_window": "week"})
    movies = (trending or {}).get("movies", [])[:12]
    tiles = [
        {
            "movie_id": None,
            "tmdb_id": m.get("id"),
            "title": m.get("title"),
            "year": int(m["release_date"][:4]) if (m.get("release_date") or "")[:4].isdigit() else None,
            "rating": m.get("vote_average") or None,
            "poster_url": f"https://image.tmdb.org/t/p/w342{m['poster_path']}" if m.get("poster_path") else None,
            "overview": m.get("overview"),
            "genres": [],
        }
        for m in movies
        if m.get("id")
    ]
    if tiles:
        _render_grid(tiles, "trend", per_row=6, show_match=False)
    else:
        st.caption("No trending movies to show right now.")

    _render_taste_profile()


def _render_taste_profile() -> None:
    """Show recently viewed movies and genre preferences from this session."""
    viewed = st.session_state["viewed_movies"]
    meta = st.session_state["viewed_meta"]
    if not viewed:
        return
    st.markdown('<div class="cm-row-title">📊 Your Taste Profile</div>', unsafe_allow_html=True)
    counts: dict[str, int] = {}
    for movie_id in viewed:
        for genre in meta.get(movie_id, {}).get("genres", []):
            counts[genre] = counts.get(genre, 0) + 1
    if counts:
        total = sum(counts.values())
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        st.write(" | ".join(f"**{g}** {round(100 * n / total)}%" for g, n in top))
    st.caption(f"You've viewed {len(viewed)} movies this session. Keep exploring to personalize your picks!")
    recent = [meta[i] for i in viewed[:6] if i in meta]
    if recent:
        _render_grid(recent, "recent", per_row=6, show_match=False)


# ---------------------------------------------------------------------------
# Screen 2 — Visual Search
# ---------------------------------------------------------------------------
def screen_visual_search() -> None:
    """Render Screen 2: Visual Search — poster upload + text-to-poster search."""
    _page_header("🖼️", "Visual Search", "Upload a movie poster, or describe the visual style you're looking for.")

    if not _engine_available("visual_search"):
        st.info(
            "🖼️ Visual search needs the CLIP model and a 56,000-poster index, which this lite deployment "
            "does not load. It is part of the full version — see the project README for the full setup."
        )
        return

    tab_upload, tab_text = st.tabs(["Upload Poster", "Describe the Style"])

    with tab_upload:
        uploaded = st.file_uploader("Upload movie poster or any cinematic image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            st.image(uploaded, caption="Uploaded poster", width=250)
            if st.button("Find Similar Posters", key="find_by_image"):
                with st.spinner("Comparing against 56,000 movie posters..."):
                    st.session_state["visual_results"] = _api_post_file(
                        "/visual-search/upload", uploaded.name, uploaded.getvalue(),
                        uploaded.type or "image/jpeg", params={"top_k": 12},
                    )

    with tab_text:
        query = st.text_input("Describe the poster style", placeholder="Dark moody thriller poster")
        if st.button("Find Similar Posters", key="find_by_text") and query:
            with st.spinner("Searching visually similar posters..."):
                st.session_state["visual_results"] = _api_post("/visual-search", {"query": query, "top_k": 12})

    result = st.session_state.get("visual_results")
    if result and result.get("results"):
        st.markdown(f"### {len(result['results'])} Visually Similar Movies Found")
        _render_grid(result["results"], "visual", per_row=6, show_match=False)
    elif result is not None and "visual_results" in st.session_state:
        st.caption("No visually similar posters found.")


# ---------------------------------------------------------------------------
# Screen 3 — Movie Intelligence Card
# ---------------------------------------------------------------------------
def _load_card(movie_id: int | None, tmdb_id: int | None) -> dict[str, Any] | None:
    """Fetch (and cache for this session) the Movie Card payload."""
    key = f"ml{movie_id}" if movie_id is not None else f"tm{tmdb_id}"
    cache = st.session_state["_card_cache"]
    if key not in cache:
        path = f"/movie/{movie_id}" if movie_id is not None else f"/tmdb-movie/{tmdb_id}"
        card = _api_get(path, timeout=60)
        if card is None:
            return None
        cache[key] = card
    return cache[key]


def _money(value: int | None) -> str:
    return f"${value:,.0f}" if value else "—"


def screen_movie_card(movie_id: int | None = None, tmdb_id: int | None = None) -> None:
    """Render Screen 3: full Movie Intelligence Card for one movie.

    Args:
        movie_id: MovieLens id of the movie to show.
        tmdb_id: TMDB id, for movies outside our catalogue.
    """
    _page_header("🎥", "Movie Intelligence Card")

    person_id = st.session_state.get("view_person_id")
    if person_id:
        _render_person(person_id)
        return

    if movie_id is None and tmdb_id is None:
        st.info("Pick a movie from Smart Search, Trending or Visual Search to see its full details here.")
        movie_id_input = st.number_input("…or enter a movie ID", min_value=1, step=1, value=1)
        st.button("View Movie", on_click=_open_movie, kwargs={"movie_id": int(movie_id_input), "title": f"#{int(movie_id_input)}"})
        return

    _render_trail()
    with st.spinner("Loading movie intelligence..."):
        card = _load_card(movie_id, tmdb_id)
    if card is None:
        st.error("Could not load this movie. Is the API running?")
        return

    movie_id = card.get("movie_id", movie_id)
    _record_view(movie_id, card)
    for note in card.get("warnings", []):
        st.caption(f"⚠️ {note}")

    _render_hero(card)
    _render_ratings(card)

    col_poster, col_info = st.columns([1, 3])
    with col_poster:
        if card.get("poster_url"):
            st.image(card["poster_url"], use_container_width=True)
    with col_info:
        if card.get("genres"):
            st.markdown("".join(f'<span class="cm-chip">{_esc(g)}</span>' for g in card["genres"]), unsafe_allow_html=True)
        with st.expander("Plot Overview", expanded=True):
            st.write(card.get("overview") or "No overview available.")
        facts = [
            ("Release date", card.get("release_date")),
            ("Language", ", ".join(card.get("languages") or []) or None),
            ("Country", ", ".join(card.get("countries") or []) or None),
            ("Studios", ", ".join(card.get("companies") or []) or None),
        ]
        for label, value in facts:
            if value:
                st.write(f"**{label}:** {value}")
        if card.get("keywords"):
            st.caption("Themes: " + " · ".join(card["keywords"][:8]))

    if card.get("trailer_url"):
        with st.expander("▶ Watch trailer"):
            st.video(card["trailer_url"])

    st.markdown("### 🎭 Cast")
    _render_cast(card.get("cast", []), movie_id or card.get("tmdb_id"))
    _render_crew(card.get("crew", {}), movie_id or card.get("tmdb_id"))

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 💰 Box Office")
        st.write(f"**Budget:** {_money(card.get('budget'))}  \n**Revenue:** {_money(card.get('revenue'))}")
        st.markdown("### 🏆 Awards")
        imdb = (card.get("ratings") or {}).get("imdb") or {}
        st.write(imdb.get("awards") or "No awards data available.")
    with col_right:
        st.markdown("### Rate this movie")
        st.slider("Your rating", 1, 5, 3, key=f"rate_{movie_id}")
        if movie_id is not None:
            _render_watchlist_buttons(movie_id, card)

    st.markdown("---")
    st.markdown("### 🤖 Review Sentiment Intelligence")
    if movie_id is None:
        st.caption("Sentiment analysis is available for movies in our catalogue.")
    elif not _engine_available("sentiment_engine"):
        st.caption("RoBERTa review sentiment is part of the full version and is not enabled in this lite deployment.")
    else:
        st.caption("Fetches this movie's reviews from TMDB and scores them with the fine-tuned RoBERTa model.")
        result_key = f"sentiment_result_{movie_id}"
        if st.button("Analyze review sentiment", key=f"sentiment_btn_{movie_id}"):
            with st.spinner("Fetching reviews and analyzing sentiment..."):
                st.session_state[result_key] = _api_post(f"/sentiment/{movie_id}", {"reviews": []}, timeout=120)
        if st.session_state.get(result_key):
            _render_sentiment_panel(st.session_state[result_key])

    st.markdown("---")
    st.markdown("### 🔁 More Movies You'll Love")
    st.caption("Click any movie to open it — it brings its own recommendations, so you never run out.")
    rows = card.get("rows", [])
    if not rows:
        st.caption("No related movies available for this title yet.")
    for r_index, row in enumerate(rows):
        st.markdown(f'<div class="cm-row-title">{_esc(row["title"])}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cm-reason">{_esc(row.get("reason", ""))}</div>', unsafe_allow_html=True)
        _render_grid(row.get("movies", []), f"row{r_index}_{movie_id or card.get('tmdb_id')}", per_row=6, show_match=False)


def _render_hero(card: dict[str, Any]) -> None:
    """Backdrop banner with title, year, runtime and certificate."""
    bits = [str(card["year"])] if card.get("year") else []
    if card.get("certificate"):
        bits.append(card["certificate"])
    if card.get("runtime"):
        bits.append(f"{card['runtime']} min")
    sub = " · ".join(bits)
    tagline = f'<div class="cm-sub"><i>{_esc(card["tagline"])}</i></div>' if card.get("tagline") else ""
    backdrop = card.get("backdrop_url")
    style = (
        f'background-image: linear-gradient(rgba(20,20,20,0.35), rgba(20,20,20,0.95)), url("{backdrop}");'
        if backdrop
        else "background: linear-gradient(135deg, #2A0A0E 0%, #141414 100%);"
    )
    st.markdown(
        f'<div class="cm-backdrop" style=\'{style}\'><div class="cm-backdrop-title">{_esc(card.get("title"))}</div>'
        f'<div class="cm-sub">{_esc(sub)}</div>{tagline}</div>',
        unsafe_allow_html=True,
    )


def _render_ratings(card: dict[str, Any]) -> None:
    """TMDB / IMDB / runtime metric row."""
    ratings = card.get("ratings") or {}
    tmdb = ratings.get("tmdb") or {}
    imdb = ratings.get("imdb") or {}
    col1, col2, col3 = st.columns(3)
    score = tmdb.get("score")
    col1.metric("TMDB Rating", f"{float(score):.1f}/10" if score else "N/A")
    imdb_score = imdb.get("imdb_rating")
    col2.metric("IMDB Rating", f"{imdb_score}/10" if imdb_score else "N/A")
    col3.metric("Runtime", f"{card['runtime']} min" if card.get("runtime") else "N/A")


def _render_cast(cast: list[dict[str, Any]], key_seed: Any) -> None:
    """Cast grid with photos; each actor opens their filmography."""
    if not cast:
        st.caption("Cast information isn't available for this title.")
        return
    for start in range(0, len(cast[:12]), 6):
        cols = st.columns(6)
        for offset, member in enumerate(cast[start : start + 6]):
            with cols[offset]:
                if member.get("photo_url"):
                    st.image(member["photo_url"], use_container_width=True)
                else:
                    initials = "".join(w[0] for w in (member.get("name") or "?").split()[:2]).upper()
                    st.markdown(f'<div class="cm-person-photo-fallback">{_esc(initials)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cm-tile-title">{_esc(member.get("name"))}</div>', unsafe_allow_html=True)
                if member.get("character"):
                    st.markdown(f'<div class="cm-tile-meta">as {_esc(member["character"])}</div>', unsafe_allow_html=True)
                if member.get("id"):
                    st.button(
                        "Filmography", key=f"cast_{key_seed}_{member['id']}_{start + offset}",
                        use_container_width=True, on_click=_open_person, args=(member["id"],),
                    )


def _render_crew(crew: dict[str, Any], key_seed: Any) -> None:
    """Director / writers / producers / composer / cinematographer lines."""
    labels = [
        ("directors", "Director"), ("writers", "Writer"), ("producers", "Producer"),
        ("composers", "Music"), ("cinematographers", "Cinematography"),
    ]
    st.markdown("### 🎬 Crew")
    shown = False
    for field, label in labels:
        people = crew.get(field) or []
        if not people:
            continue
        shown = True
        st.write(f"**{label}:** " + ", ".join(p["name"] for p in people if p.get("name")))
    if not shown:
        st.caption("Crew information isn't available for this title.")
    directors = [p for p in crew.get("directors", []) if p.get("id")]
    if directors:
        st.button(
            f"🎬 More by {directors[0]['name']}", key=f"dir_{key_seed}_{directors[0]['id']}",
            on_click=_open_person, args=(directors[0]["id"],),
        )


def _render_watchlist_buttons(movie_id: int, card: dict[str, Any]) -> None:
    """Watched / Watchlist toggles that remember title + poster for the watchlist page."""
    current_status = st.session_state["watchlist"].get(movie_id, {}).get("status")
    entry = {"movie_id": movie_id, "title": card.get("title"), "poster_url": card.get("poster_url"), "year": card.get("year")}
    col_a, col_b = st.columns(2)
    if col_a.button("👁️ Watched", key=f"watched_{movie_id}"):
        st.session_state["watchlist"][movie_id] = {**entry, "status": "watched"}
        st.rerun()
    if col_b.button("➕ Watchlist", key=f"watchlist_{movie_id}"):
        st.session_state["watchlist"][movie_id] = {**entry, "status": "want_to_watch"}
        st.rerun()
    if current_status:
        st.caption(f"Status: {current_status.replace('_', ' ').title()}")


def _render_person(person_id: int) -> None:
    """Actor / director profile page with a filmography grid."""
    st.button("← Back to movie", key="person_back", on_click=_close_person)
    with st.spinner("Loading filmography..."):
        data = _api_get(f"/person/{person_id}", timeout=60)
    if not data:
        st.error("Could not load this person.")
        return
    person = data.get("person", {})
    col_photo, col_bio = st.columns([1, 4])
    with col_photo:
        if person.get("photo_url"):
            st.image(person["photo_url"], use_container_width=True)
    with col_bio:
        st.markdown(f"# {person.get('name', 'Unknown')}")
        facts = [person.get("known_for"), f"Born {person['birthday']}" if person.get("birthday") else None, person.get("place_of_birth")]
        st.caption(" · ".join(f for f in facts if f))
        if person.get("biography"):
            st.write(person["biography"])

    movies = data.get("movies", [])
    st.markdown(f"### Filmography ({len(movies)})")
    sort = st.radio("Sort by", ["Popular", "Latest", "Highest rated"], horizontal=True, key=f"person_sort_{person_id}")
    if sort == "Latest":
        movies = sorted(movies, key=lambda m: m.get("year") or 0, reverse=True)
    elif sort == "Highest rated":
        movies = sorted(movies, key=lambda m: m.get("rating") or 0, reverse=True)
    _render_grid(movies, f"person{person_id}", per_row=6, show_match=False)


def _render_sentiment_panel(sentiment: dict[str, Any]) -> None:
    """Render overall and per-aspect sentiment results.

    Args:
        sentiment: JSON body returned by ``POST /sentiment/{movie_id}``
            (``overall_positive_pct``, ``review_count``, ``aspects``).
    """
    col1, col2 = st.columns(2)
    col1.metric("Overall sentiment", _sentiment_badge(sentiment.get("overall_positive_pct")))
    col2.metric("Reviews analyzed", sentiment.get("review_count", 0))
    for name, aspect in sentiment.get("aspects", {}).items():
        st.write(
            f"**{name.title()}** — {'⭐' * int(aspect['stars'])} {aspect['label']} "
            f"({aspect['positive_pct']:.0f}% positive, {aspect['mention_count']} mentions)"
        )
        st.progress(min(max(aspect["positive_pct"] / 100, 0.0), 1.0))
    if not sentiment.get("aspects"):
        st.caption("Too few reviews mention specific aspects for a breakdown.")


# ---------------------------------------------------------------------------
# Screen 4 — Smart Search Results
# ---------------------------------------------------------------------------
def _run_search() -> None:
    """Call ``GET /search`` with the current widget values and keep the results in session state."""
    params: dict[str, Any] = {"q": st.session_state.get("sq_query", "").strip(), "top_k": 24}
    genres = st.session_state.get("sq_genres") or []
    if genres:
        params["genres"] = genres
    years = st.session_state.get("sq_years", _YEAR_BOUNDS)
    if tuple(years) != _YEAR_BOUNDS:
        params["year_min"], params["year_max"] = years
    if st.session_state.get("sq_rating", 0.0) > 0:
        params["min_rating"] = st.session_state["sq_rating"]
    with st.spinner("Searching..."):
        st.session_state["search_results"] = _api_get("/search", params=params, timeout=60)


def screen_smart_search() -> None:
    """Render Screen 4: Smart Search — title, mood or genre search with working filters."""
    _page_header("🔍", "Smart Search", "Find movies by title, mood or description — then open any result for the full story.")

    auto_search = False
    prefill = st.session_state.pop("_search_prefill", None)
    if prefill is not None:
        st.session_state["sq_query"] = prefill["query"]
        st.session_state["sq_genres"] = prefill["genres"]
        auto_search = True

    with st.form("smart_search_form", border=False):
        st.text_input("Search by title, mood, or natural language", key="sq_query",
                      placeholder='e.g. "iron man", "bahubali", or "thriller with unreliable narrator"')
        with st.expander("Filters", expanded=bool(st.session_state.get("sq_genres"))):
            col1, col2 = st.columns(2)
            with col1:
                st.multiselect("Genre", _GENRES + ["Adventure", "Crime", "Fantasy", "Mystery", "Documentary", "Family"], key="sq_genres")
                st.slider("Year range", _YEAR_BOUNDS[0], _YEAR_BOUNDS[1], _YEAR_BOUNDS, key="sq_years")
            with col2:
                st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5, key="sq_rating")
        submitted = st.form_submit_button("Search")

    if submitted or auto_search:
        _run_search()

    data = st.session_state.get("search_results")
    if data is None:
        st.caption("Type a movie name or describe what you feel like watching, then press Search.")
        return
    results = data.get("results", [])
    if not results:
        st.caption("No results found. Try a different spelling, fewer filters, or describe the movie instead.")
        return
    label = f"“{data['query']}”" if data.get("query") else "your filters"
    st.markdown(f"**{len(results)} results for {label}**")
    _render_grid(results, "search", per_row=6)


# ---------------------------------------------------------------------------
# Screen 5 — CineBot Chat
# ---------------------------------------------------------------------------
def screen_cinebot() -> None:
    """Render Screen 5: CineBot Chat — conversational recommendation assistant.

    Movies CineBot mentions come back from the API as cards and are shown
    under its reply; clicking one opens that movie's page.
    """
    _page_header("💬", "CineBot", "Your knowledgeable movie critic friend — ask in English or Hinglish.")

    prompt_chips = [
        "Recommend a mind-bending thriller",
        "Best Bollywood movies of 2023",
        "Movies like The Dark Knight",
        "Something for a date night",
    ]
    chip_cols = st.columns(len(prompt_chips))
    for col, chip in zip(chip_cols, prompt_chips):
        with col:
            if st.button(chip, key=f"chip_{chip}", use_container_width=True):
                st.session_state["_pending_chat"] = chip

    history = st.session_state["chat_history"]
    if not history:
        st.markdown(
            '<div class="cm-welcome"><b>👋 Namaste! Main CineBot hoon.</b>'
            "<p>Mood, genre, actor ya koi favourite movie batao — main suggestions dunga, "
            "aur har suggested movie par click karke uski poori details dekh sakte ho.</p></div>",
            unsafe_allow_html=True,
        )

    for index, turn in enumerate(history):
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.write(turn["text"])
            if turn.get("movies"):
                st.caption("Tap a movie to open it")
                _render_grid(turn["movies"], f"chat{index}", per_row=6, show_match=False)

    # The input must be created on EVERY run. (`pending or st.chat_input(...)` skipped it
    # after a suggestion-chip click, so the chat box vanished until the next navigation.)
    typed = st.chat_input("Ask CineBot for a recommendation...")
    pending = st.session_state.pop("_pending_chat", None)
    user_message = pending or typed

    if user_message:
        history.append({"role": "user", "text": user_message})
        with st.chat_message("user"):
            st.write(user_message)
        with st.chat_message("assistant"):
            with st.spinner("CineBot is thinking..."):
                result = _api_post("/chat", {"message": user_message}, timeout=90)
            reply = result["reply"] if result else "Sorry, CineBot couldn't answer right now — see the message above for details."
            movies = (result or {}).get("movies", [])
            st.write(reply)
            if movies:
                st.caption("Tap a movie to open it")
                _render_grid(movies, f"chat{len(history)}", per_row=6, show_match=False)
        history.append({"role": "bot", "text": reply, "movies": movies})


# ---------------------------------------------------------------------------
# Screen 6 — My Watchlist
# ---------------------------------------------------------------------------
def screen_watchlist() -> None:
    """Render Screen 6: My Watchlist — session-based watched / want-to-watch tracking."""
    _page_header("📌", "My Watchlist", "Movies you've watched or want to watch this session.")

    watchlist = st.session_state["watchlist"]
    if not watchlist:
        st.info("Your watchlist is empty. Mark movies as 'Watched' or add them from a Movie Card.")
        return

    watched = [v for v in watchlist.values() if v["status"] == "watched"]
    want_to_watch = [v for v in watchlist.values() if v["status"] == "want_to_watch"]

    tab_watched, tab_want = st.tabs([f"Watched ({len(watched)})", f"Want to Watch ({len(want_to_watch)})"])
    with tab_watched:
        _render_watchlist_grid(watched, "wl_watched")
    with tab_want:
        _render_watchlist_grid(want_to_watch, "wl_want")

    st.markdown("---")
    if st.button("📄 Export Watchlist (list)"):
        lines = [f"- {v.get('title') or 'Movie ' + str(v.get('movie_id'))} ({v['status'].replace('_', ' ')})" for v in watchlist.values()]
        st.code("\n".join(lines), language="markdown")


def _render_watchlist_grid(entries: list[dict[str, Any]], key_prefix: str) -> None:
    """Render watchlist entries as poster tiles that open the Movie Card.

    Args:
        entries: Watchlist entries (title, poster, year, movie_id).
        key_prefix: Prefix keeping widget keys unique per tab.
    """
    if not entries:
        st.caption("Nothing here yet.")
        return
    _render_grid(entries, key_prefix, per_row=6, show_match=False)


# ---------------------------------------------------------------------------
# App shell — sidebar navigation
# ---------------------------------------------------------------------------
_NAV_ICONS = {
    "Home": "🏠", "Visual Search": "🖼️", "Movie Card": "🎥",
    "Smart Search": "🔍", "CineBot": "💬", "My Watchlist": "📌",
}


def main() -> None:
    """Render the sidebar navigation and dispatch to the selected screen."""
    _init_session_state()

    st.sidebar.markdown(
        '<div class="cm-brand"><div class="cm-logo">🎬</div><div>'
        '<div class="cm-brand-name">CineMatch <span>PRO</span></div>'
        '<div class="cm-brand-tag">AI Movie Intelligence</div></div></div>'
        '<div class="cm-nav-label">EXPLORE</div>',
        unsafe_allow_html=True,
    )
    screens = {
        "Home": screen_home,
        "Visual Search": screen_visual_search,
        "Movie Card": lambda: screen_movie_card(
            st.session_state.get("current_movie_id"), st.session_state.get("current_tmdb_id")
        ),
        "Smart Search": screen_smart_search,
        "CineBot": screen_cinebot,
        "My Watchlist": screen_watchlist,
    }

    # Programmatic navigation (button callbacks) sets `_nav_target`; it is applied to the
    # radio's *session-state value* before the widget is created. The radio needs a stable
    # `key`: a changing `index=` would make Streamlit treat it as a new widget and bounce the
    # user back to Home.
    nav_target = st.session_state.pop("_nav_target", None)
    if nav_target in screens:
        st.session_state["nav"] = nav_target

    choice = st.sidebar.radio(
        "Navigate", list(screens.keys()), key="nav", label_visibility="collapsed",
        format_func=lambda name: f"{_NAV_ICONS.get(name, '')}  {name}",
    )

    health = _fetch_health()
    st.session_state["_engines"] = (health or {}).get("engines", {})
    if health is None:
        st.sidebar.error("Backend is offline. Start the API server, then refresh this page.")
    watchlist_count = len(st.session_state["watchlist"])
    if watchlist_count:
        st.sidebar.markdown(f'<span class="cm-pill-status">📌 {watchlist_count} in your watchlist</span>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="cm-side-foot">Powered by CLIP · RoBERTa · SBERT · Gemini<br>Data: MovieLens · TMDB · IMDB</div>',
        unsafe_allow_html=True,
    )

    screens[choice]()


if __name__ == "__main__":
    main()
