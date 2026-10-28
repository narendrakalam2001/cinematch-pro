"""Simulate real user sessions against a running CineMatch Pro API and report latencies.

Each simulated session behaves like a visitor: check health → search a title or a mood → open the top
result's Movie Card → follow a related-movie recommendation (the "infinite discovery" loop) → open an
actor's filmography → look at trending, and optionally ask CineBot / request review sentiment. Every
step is timed; the script prints a per-endpoint table (p50 / p95 / max) and can save a JSON report.

Usage:
    python -m scripts.simulate_user_journey --sessions 5
    python -m scripts.simulate_user_journey --base-url https://my-api.hf.space --sessions 20 --with-chat
    python -m scripts.simulate_user_journey --sessions 10 --with-sentiment --report logs/simulation.json

A step that cannot run because an earlier one returned nothing (e.g. no related movies) is reported as
"skipped", not as a failure; so is a sentiment request for a movie that has no reviews on TMDB (the API
answers HTTP 400 — a data condition, not an outage). The exit code is 1 when any step failed (HTTP error /
timeout), else 0, and the failed steps are listed with their reason after the table.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from src.config import logger

# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and the first request can stall ~2 s.
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SEARCH_QUERIES: tuple[str, ...] = (
    "iron man", "inception", "bahubali", "titanic", "the dark knight",
    "thriller with a twist ending", "feel good family comedy", "dark moody sci-fi",
)
CHAT_MESSAGES: tuple[str, ...] = (
    "Recommend a mind-bending thriller",
    "Movies like The Dark Knight",
    "Something for a date night",
)


@dataclass
class StepResult:
    """Outcome of one timed API call.

    Attributes:
        name: Endpoint label, e.g. ``"GET /search"``.
        status: ``"ok"``, ``"failed"`` or ``"skipped"``.
        latency_s: Wall-clock seconds (0 for skipped steps).
        detail: Short note — the HTTP status or the error text.
    """

    name: str
    status: str
    latency_s: float = 0.0
    detail: str = ""


class ApiClient:
    """Tiny timed HTTP client around the CineMatch Pro API."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        """Create a client.

        Args:
            base_url: API root, e.g. ``http://localhost:8000``.
            timeout: Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def call(self, method: str, path: str, **kwargs: Any) -> tuple[int, Any, float]:
        """Send one request and time it.

        Args:
            method: HTTP method.
            path: API path starting with ``/``.
            **kwargs: Passed to ``requests`` (``params``, ``json`` ...).

        Returns:
            tuple[int, Any, float]: ``(status_code, parsed_json_or_None, seconds)``.

        Raises:
            requests.exceptions.RequestException: On connection problems / timeouts.
        """
        started = time.perf_counter()
        response = self._session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        elapsed = time.perf_counter() - started
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, body, elapsed


def _step(
    client: ApiClient,
    results: list[StepResult],
    name: str,
    method: str,
    path: str,
    skip_statuses: dict[int, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Run one step, record its `StepResult` and return the JSON body (``None`` on failure).

    Args:
        client: API client.
        results: List the step outcome is appended to.
        name: Label shown in the report.
        method: HTTP method.
        path: API path.
        skip_statuses: HTTP statuses that mean "nothing to do here" (recorded as skipped, not failed),
            mapped to the explanation shown in the report.
        **kwargs: Passed to the request.
    """
    try:
        status, body, elapsed = client.call(method, path, **kwargs)
    except requests.exceptions.RequestException as exc:
        results.append(StepResult(name, "failed", 0.0, f"{type(exc).__name__}: {exc}"[:120]))
        return None
    if skip_statuses and status in skip_statuses:
        results.append(StepResult(name, "skipped", elapsed, f"HTTP {status}: {skip_statuses[status]}"))
        return None
    if status >= 400:
        results.append(StepResult(name, "failed", elapsed, f"HTTP {status}"))
        return None
    results.append(StepResult(name, "ok", elapsed, f"HTTP {status}"))
    return body


def _skip(results: list[StepResult], name: str, why: str) -> None:
    results.append(StepResult(name, "skipped", 0.0, why))


def _card_path(item: dict[str, Any]) -> str | None:
    """API path that opens a result/related-movie card (catalogue id first, else TMDB id)."""
    if item.get("movie_id") is not None:
        return f"/movie/{item['movie_id']}"
    if item.get("tmdb_id") is not None:
        return f"/tmdb-movie/{item['tmdb_id']}"
    return None


def run_session(
    client: ApiClient,
    rng: random.Random,
    with_chat: bool = False,
    with_sentiment: bool = False,
) -> list[StepResult]:
    """Play one visitor session and return every timed step.

    Args:
        client: API client.
        rng: Random source (seeded for reproducible runs).
        with_chat: Also ask CineBot a question (needs ``GEMINI_API_KEY`` on the server).
        with_sentiment: Also request review sentiment for the opened movie (needs the RoBERTa engine).

    Returns:
        list[StepResult]: One entry per step, in order.
    """
    results: list[StepResult] = []
    _step(client, results, "GET /health", "GET", "/health")

    query = rng.choice(SEARCH_QUERIES)
    found = _step(client, results, "GET /search", "GET", "/search", params={"q": query, "top_k": 12})
    hits = (found or {}).get("results") or []
    first_path = _card_path(hits[0]) if hits else None

    card = None
    if first_path:
        card = _step(client, results, "GET /movie (card)", "GET", first_path)
    else:
        _skip(results, "GET /movie (card)", f"no search results for {query!r}")

    related = [m for row in (card or {}).get("rows", []) for m in row.get("movies", [])]
    related_path = _card_path(rng.choice(related)) if related else None
    if related_path:
        _step(client, results, "GET /movie (related)", "GET", related_path)
    else:
        _skip(results, "GET /movie (related)", "no related movies on the first card")

    cast = (card or {}).get("cast") or []
    person_id = next((c.get("id") for c in cast if c.get("id")), None)
    if person_id:
        _step(client, results, "GET /person", "GET", f"/person/{person_id}")
    else:
        _skip(results, "GET /person", "no cast ids on the card")

    _step(client, results, "GET /trending", "GET", "/trending", params={"time_window": "week"})

    movie_id = (card or {}).get("movie_id")
    if with_sentiment:
        if movie_id is not None:
            _step(
                client, results, "POST /sentiment", "POST", f"/sentiment/{movie_id}",
                skip_statuses={400: "this movie has no reviews on TMDB"}, json={"reviews": []},
            )
        else:
            _skip(results, "POST /sentiment", "opened movie is not in the catalogue")
    if with_chat:
        _step(client, results, "POST /chat", "POST", "/chat", json={"message": rng.choice(CHAT_MESSAGES)})
    return results


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Args:
        values: Samples (any order).
        pct: Percentile between 0 and 100.

    Returns:
        float: The percentile value, or 0.0 for an empty list.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(results: list[StepResult]) -> list[dict[str, Any]]:
    """Aggregate steps per endpoint.

    Args:
        results: Every step of every session.

    Returns:
        list[dict[str, Any]]: One row per endpoint (first-seen order) with ``calls``, ``ok``,
        ``failed``, ``skipped`` and latency percentiles in milliseconds.
    """
    names = list(dict.fromkeys(r.name for r in results))
    rows = []
    for name in names:
        group = [r for r in results if r.name == name]
        ok_latencies = [r.latency_s * 1000 for r in group if r.status == "ok"]
        rows.append(
            {
                "endpoint": name,
                "calls": len(group),
                "ok": sum(r.status == "ok" for r in group),
                "failed": sum(r.status == "failed" for r in group),
                "skipped": sum(r.status == "skipped" for r in group),
                "p50_ms": round(percentile(ok_latencies, 50), 1),
                "p95_ms": round(percentile(ok_latencies, 95), 1),
                "max_ms": round(max(ok_latencies), 1) if ok_latencies else 0.0,
            }
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    """Render the summary as an aligned text table.

    Args:
        rows: Output of `summarize`.

    Returns:
        str: A printable table.
    """
    header = f"{'Endpoint':<22}{'calls':>6}{'ok':>5}{'fail':>6}{'skip':>6}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['endpoint']:<22}{r['calls']:>6}{r['ok']:>5}{r['failed']:>6}{r['skipped']:>6}"
            f"{r['p50_ms']:>10}{r['p95_ms']:>10}{r['max_ms']:>10}"
        )
    return "\n".join(lines)


def failure_report(results: list[StepResult]) -> list[str]:
    """Group failed steps by endpoint and reason.

    Args:
        results: Every step of every session.

    Returns:
        list[str]: Lines such as ``"POST /chat — HTTP 503 (2×)"``; empty when nothing failed.
    """
    counts: dict[tuple[str, str], int] = {}
    for r in results:
        if r.status == "failed":
            counts[(r.name, r.detail)] = counts.get((r.name, r.detail), 0) + 1
    return [f"{name} — {detail} ({n}×)" for (name, detail), n in counts.items()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Simulate user sessions against the CineMatch Pro API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API root (default: %(default)s)")
    parser.add_argument("--sessions", type=int, default=5, help="Number of simulated visitors (default: %(default)s)")
    parser.add_argument("--with-chat", action="store_true", help="Include a CineBot question per session (needs GEMINI_API_KEY)")
    parser.add_argument("--with-sentiment", action="store_true", help="Include review sentiment (needs the RoBERTa engine)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--report", type=Path, default=None, help="Write the full JSON report to this path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the simulation and print the latency table.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        int: 0 when every executed step succeeded, 1 otherwise.
    """
    args = parse_args(argv)
    if args.sessions < 1:
        logger.error("--sessions must be at least 1.")
        return 1
    client = ApiClient(args.base_url, timeout=args.timeout)
    rng = random.Random(args.seed)
    results: list[StepResult] = []
    for number in range(1, args.sessions + 1):
        session = run_session(client, rng, with_chat=args.with_chat, with_sentiment=args.with_sentiment)
        results.extend(session)
        failed = [s for s in session if s.status == "failed"]
        logger.info("Session %d/%d: %d steps, %d failed", number, args.sessions, len(session), len(failed))

    rows = summarize(results)
    print("\n" + format_table(rows))
    total_failed = sum(r["failed"] for r in rows)
    print(f"\n{args.sessions} sessions · {len(results)} steps · {total_failed} failed")
    for line in failure_report(results):
        print(f"  ✗ {line}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"base_url": args.base_url, "summary": rows, "steps": [asdict(r) for r in results]}, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved report to %s", args.report)
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
