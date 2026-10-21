"""Replace unreliable director/cast/composer/keywords with real TMDB API data.

Why this exists
---------------
The "IMDB & TMDB Movie Metadata Big Dataset (1M)" export has per-movie credit
columns (``Director``, ``Cast_list``, ``Music_Composer``) that are misaligned
with the movie rows (e.g. *Heat* (1995) lists Kenneth Branagh as director and
*Jumanji* lists Norman Spencer as composer). Titles, years, genres and
overviews are fine, but credits are not. This script re-fetches credits for
the most-voted movies straight from the TMDB API and blanks the credits of
movies it did not fetch (an honest "unknown" is better than a wrong name).

It then rebuilds ``combined_features`` so every downstream model (TF-IDF,
SBERT/ChromaDB) is trained on clean text. Run it AFTER ``src.preprocessing``
and BEFORE building the content filter / semantic index.

Usage:
    python -m scripts.enrich_credits --top 20000
    python -m scripts.enrich_credits --top 0      # every matched movie

The run is resumable: results are cached in ``credits_cache.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import MERGED_MOVIES_PARQUET, TMDB_BASE_URL, logger

CACHE_PATH = MERGED_MOVIES_PARQUET.parent / "credits_cache.jsonl"
MAX_RETRIES = 4


class RateLimiter:
    """Thread-safe limiter that spaces requests to at most ``rate`` per second."""

    def __init__(self, rate: float) -> None:
        """Create a limiter.

        Args:
            rate: Maximum number of calls per second.
        """
        self._interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self) -> None:
        """Block until the next call is allowed."""
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self._interval
        if delay > 0:
            time.sleep(delay)


def parse_credits(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields CineMatch needs from a TMDB movie payload.

    Args:
        payload: JSON from ``/movie/{id}?append_to_response=credits,keywords``.

    Returns:
        dict[str, Any]: ``director``, ``cast_names`` (top 6), ``music_director``
        and ``keyword_names``.
    """
    credits = payload.get("credits") or {}
    crew = credits.get("crew") or []
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 999))
    director = next((c["name"] for c in crew if c.get("job") == "Director"), "")
    composer = next(
        (c["name"] for c in crew if c.get("job") in {"Original Music Composer", "Composer", "Music"}),
        "",
    )
    kw_block = payload.get("keywords") or {}
    keywords = kw_block.get("keywords") or kw_block.get("results") or []
    return {
        "director": director,
        "cast_names": [c["name"] for c in cast[:6] if c.get("name")],
        "music_director": composer,
        "keyword_names": [k["name"] for k in keywords if k.get("name")][:15],
    }


def fetch_one(
    session: requests.Session, limiter: RateLimiter, api_key: str, tmdb_id: int
) -> dict[str, Any] | None:
    """Fetch and parse one movie, retrying on rate limits / transient errors.

    Args:
        session: Shared HTTP session.
        limiter: Shared rate limiter.
        api_key: TMDB v3 API key.
        tmdb_id: TMDB movie id.

    Returns:
        dict[str, Any] | None: Parsed credits (with ``id``), an empty-credits
        record for a 404, or ``None`` if every retry failed.
    """
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {"api_key": api_key, "append_to_response": "credits,keywords"}
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            resp = session.get(url, params=params, timeout=15)
        except requests.RequestException as exc:
            logger.debug("TMDB request error for %s: %s", tmdb_id, exc)
            time.sleep(1 + attempt)
            continue
        if resp.status_code == 200:
            return {"id": tmdb_id, **parse_credits(resp.json())}
        if resp.status_code == 404:
            return {"id": tmdb_id, "director": "", "cast_names": [], "music_director": "", "keyword_names": []}
        if resp.status_code == 401:
            raise PermissionError("TMDB rejected the API key (HTTP 401).")
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("Retry-After", 2)))
            continue
        time.sleep(1 + attempt)
    return None


def load_cache() -> dict[int, dict[str, Any]]:
    """Load previously fetched credits.

    Returns:
        dict[int, dict[str, Any]]: Mapping of TMDB id to parsed credits.
    """
    cache: dict[int, dict[str, Any]] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                cache[int(rec["id"])] = rec
            except (ValueError, KeyError):
                continue
    return cache


def rebuild_features(row: pd.Series) -> str:
    """Rebuild the TF-IDF/SBERT text for one movie from clean fields.

    Args:
        row: A merged-movies row.

    Returns:
        str: Combined text (overview + genres + keywords + cast + director),
        falling back to MovieLens genres when everything else is empty.
    """
    parts: list[str] = []
    overview = row.get("overview")
    if isinstance(overview, str):
        parts.append(overview)
    for col in ("genre_names", "keyword_names", "cast_names"):
        val = row.get(col)
        if val is not None and not isinstance(val, str):
            parts.append(" ".join(str(v) for v in list(val)))
    director = row.get("director")
    if isinstance(director, str):
        parts.append(director)
    text = " ".join(p for p in parts if p).strip()
    if not text:
        text = str(row.get("genres", "")).replace("|", " ").strip()
    return text


def apply_credits(df: pd.DataFrame, cache: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """Overwrite credit columns from the cache and blank everything unfetched.

    Args:
        df: Merged movies frame (needs an ``id`` column holding TMDB ids).
        cache: Fetched credits keyed by TMDB id.

    Returns:
        pd.DataFrame: Copy with clean ``director``, ``cast_names``,
        ``music_director``, ``keyword_names`` and rebuilt ``combined_features``.
    """
    out = df.copy()
    ids = pd.to_numeric(out["id"], errors="coerce")

    def pick(tmdb_id: Any, field: str, empty: Any) -> Any:
        if pd.isna(tmdb_id):
            return empty
        rec = cache.get(int(tmdb_id))
        return rec[field] if rec else empty

    out["director"] = [pick(i, "director", "") for i in ids]
    out["music_director"] = [pick(i, "music_director", "") for i in ids]
    out["cast_names"] = [pick(i, "cast_names", []) for i in ids]
    out["keyword_names"] = [pick(i, "keyword_names", []) for i in ids]
    for raw in ("Director", "Music_Composer"):
        if raw in out.columns:
            out[raw] = out["director"] if raw == "Director" else out["music_director"]
    if "Cast_list" in out.columns:
        out["Cast_list"] = out["cast_names"].apply(str)
    out["combined_features"] = out.apply(rebuild_features, axis=1)
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Fix credits with the TMDB API.")
    p.add_argument("--movies-parquet", default=str(MERGED_MOVIES_PARQUET))
    p.add_argument("--top", type=int, default=20000,
                   help="Fetch the N most-voted matched movies (0 = all).")
    p.add_argument("--rate", type=float, default=30.0, help="Requests per second.")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main() -> int:
    """Fetch credits, rewrite the parquet and report coverage.

    Returns:
        int: Exit code (0 success, 1 failure).
    """
    args = parse_args()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.error("TMDB_API_KEY is not set (add it as a Kaggle Secret / env var).")
        return 1
    path = Path(args.movies_parquet)
    if not path.exists():
        logger.error("%s not found. Run src.preprocessing first.", path)
        return 1

    df = pd.read_parquet(path)
    ids = pd.to_numeric(df["id"], errors="coerce")
    candidates = df.assign(_id=ids).dropna(subset=["_id"])
    rank = "vote_count" if "vote_count" in candidates.columns else None
    if rank:
        candidates = candidates.sort_values(rank, ascending=False)
    wanted = [int(i) for i in candidates["_id"].drop_duplicates()]
    if args.top > 0:
        wanted = wanted[: args.top]

    cache = load_cache()
    todo = [i for i in wanted if i not in cache]
    logger.info("%d movies wanted, %d cached, %d to fetch.", len(wanted), len(wanted) - len(todo), len(todo))

    if todo:
        limiter = RateLimiter(args.rate)
        failed = 0
        with requests.Session() as session, open(CACHE_PATH, "a", encoding="utf-8") as sink, \
                ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, session, limiter, api_key, i) for i in todo]
            for n, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                if rec is None:
                    failed += 1
                else:
                    cache[rec["id"]] = rec
                    sink.write(json.dumps(rec) + "\n")
                if n % 1000 == 0:
                    sink.flush()
                    logger.info("Fetched %d / %d (failed: %d)", n, len(todo), failed)
        logger.info("Fetch complete. Failed after retries: %d (re-run to retry them).", failed)

    fixed = apply_credits(df, {i: cache[i] for i in wanted if i in cache})
    fixed.to_parquet(path, index=False)
    with_dir = int((fixed["director"] != "").sum())
    logger.info("Saved %s. Movies with verified director: %d / %d.", path, with_dir, len(fixed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
