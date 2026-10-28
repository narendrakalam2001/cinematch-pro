"""Tests for scripts/simulate_user_journey.py."""

from __future__ import annotations

import json
import random
from unittest.mock import patch

import requests

from scripts import simulate_user_journey as sim


class FakeClient:
    """Scripted stand-in for `ApiClient`: maps (method, path) -> (status, body)."""

    def __init__(self, routes: dict[tuple[str, str], tuple[int, object]], latency: float = 0.05) -> None:
        self.routes, self.latency, self.calls = routes, latency, []

    def call(self, method, path, **kwargs):
        self.calls.append((method, path))
        if (method, path) not in self.routes:
            return 404, {"detail": "nope"}, self.latency
        status, body = self.routes[(method, path)]
        if isinstance(body, Exception):
            raise body
        return status, body, self.latency


CARD = {"movie_id": 7, "cast": [{"id": 99, "name": "A"}],
        "rows": [{"title": "More like this", "movies": [{"movie_id": 8, "title": "B"}]}]}
HAPPY = {
    ("GET", "/health"): (200, {"status": "ok"}),
    ("GET", "/search"): (200, {"results": [{"movie_id": 7, "title": "Iron Man"}]}),
    ("GET", "/movie/7"): (200, CARD),
    ("GET", "/movie/8"): (200, {"movie_id": 8, "rows": []}),
    ("GET", "/person/99"): (200, {"person": {}}),
    ("GET", "/trending"): (200, {"movies": []}),
    ("POST", "/sentiment/7"): (200, {"overall_positive_pct": 80}),
    ("POST", "/chat"): (200, {"reply": "hi", "movies": []}),
}


def test_a_full_session_follows_search_card_related_person_and_trending():
    client = FakeClient(HAPPY)
    results = sim.run_session(client, random.Random(1), with_chat=True, with_sentiment=True)
    assert [r.name for r in results] == [
        "GET /health", "GET /search", "GET /movie (card)", "GET /movie (related)",
        "GET /person", "GET /trending", "POST /sentiment", "POST /chat",
    ]
    assert all(r.status == "ok" for r in results)
    assert ("GET", "/movie/8") in client.calls  # the related movie really is opened (the discovery loop)


def test_steps_that_cannot_run_are_skipped_not_failed():
    routes = dict(HAPPY)
    routes[("GET", "/search")] = (200, {"results": []})
    results = sim.run_session(FakeClient(routes), random.Random(1))
    by_name = {r.name: r for r in results}
    assert by_name["GET /movie (card)"].status == "skipped" and "no search results" in by_name["GET /movie (card)"].detail
    assert by_name["GET /movie (related)"].status == "skipped" and by_name["GET /person"].status == "skipped"
    assert not any(r.status == "failed" for r in results)


def test_tmdb_only_results_open_through_the_tmdb_endpoint():
    routes = dict(HAPPY)
    routes[("GET", "/search")] = (200, {"results": [{"movie_id": None, "tmdb_id": 555}]})
    routes[("GET", "/tmdb-movie/555")] = (200, {"movie_id": None, "cast": [], "rows": []})
    client = FakeClient(routes)
    results = sim.run_session(client, random.Random(1))
    assert ("GET", "/tmdb-movie/555") in client.calls
    assert {r.name: r.status for r in results}["GET /movie (card)"] == "ok"


def test_http_errors_and_network_failures_are_recorded_as_failed():
    routes = dict(HAPPY)
    routes[("GET", "/trending")] = (503, {"detail": "TMDB down"})
    routes[("GET", "/person/99")] = (200, requests.exceptions.ConnectionError("refused"))
    results = {r.name: r for r in sim.run_session(FakeClient(routes), random.Random(1))}
    assert results["GET /trending"].status == "failed" and results["GET /trending"].detail == "HTTP 503"
    assert results["GET /person"].status == "failed" and "ConnectionError" in results["GET /person"].detail


def test_percentile_uses_nearest_rank_and_handles_empty():
    assert sim.percentile([], 95) == 0.0
    assert sim.percentile([5, 1, 3, 2, 4], 50) == 3
    assert sim.percentile(list(range(1, 101)), 95) == 95


def test_summarize_counts_and_latency_columns():
    results = [
        sim.StepResult("GET /search", "ok", 0.10), sim.StepResult("GET /search", "ok", 0.30),
        sim.StepResult("GET /search", "failed", 0.0, "HTTP 500"), sim.StepResult("GET /person", "skipped"),
    ]
    rows = {r["endpoint"]: r for r in sim.summarize(results)}
    assert rows["GET /search"] == {"endpoint": "GET /search", "calls": 3, "ok": 2, "failed": 1, "skipped": 0,
                                   "p50_ms": 100.0, "p95_ms": 300.0, "max_ms": 300.0}
    assert rows["GET /person"]["skipped"] == 1 and rows["GET /person"]["max_ms"] == 0.0
    table = sim.format_table(list(rows.values()))
    assert "GET /search" in table and "p95 ms" in table


def test_main_prints_table_writes_report_and_sets_exit_code(tmp_path, capsys):
    report = tmp_path / "out" / "report.json"
    with patch.object(sim, "ApiClient", return_value=FakeClient(HAPPY)):
        code = sim.main(["--sessions", "3", "--with-chat", "--report", str(report)])
    assert code == 0
    assert "3 sessions" in capsys.readouterr().out
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["summary"] and len(saved["steps"]) == 3 * 7  # health, search, card, related, person, trending, chat

    broken = dict(HAPPY)
    broken[("GET", "/health")] = (500, {})
    with patch.object(sim, "ApiClient", return_value=FakeClient(broken)):
        assert sim.main(["--sessions", "1"]) == 1  # any failed step -> non-zero exit (usable in CI)
    assert sim.main(["--sessions", "0"]) == 1


def test_sentiment_for_a_movie_without_reviews_is_skipped_not_failed():
    """HTTP 400 from /sentiment means "no reviews on TMDB" — a data condition, not an outage."""
    routes = dict(HAPPY)
    routes[("POST", "/sentiment/7")] = (400, {"detail": "No reviews found"})
    results = {r.name: r for r in sim.run_session(FakeClient(routes), random.Random(1), with_sentiment=True)}
    assert results["POST /sentiment"].status == "skipped" and "no reviews" in results["POST /sentiment"].detail

    routes[("POST", "/sentiment/7")] = (503, {"detail": "engine down"})  # a real outage is still a failure
    results = {r.name: r for r in sim.run_session(FakeClient(routes), random.Random(1), with_sentiment=True)}
    assert results["POST /sentiment"].status == "failed"


def test_failure_report_lists_each_reason_with_a_count(capsys):
    results = [sim.StepResult("POST /chat", "failed", 0, "HTTP 503")] * 2 + [sim.StepResult("GET /person", "failed", 0, "HTTP 404")]
    assert sim.failure_report(results) == ["POST /chat — HTTP 503 (2×)", "GET /person — HTTP 404 (1×)"]
    assert sim.failure_report([sim.StepResult("GET /health", "ok", 0.1)]) == []

    broken = dict(HAPPY)
    broken[("GET", "/trending")] = (503, {})
    with patch.object(sim, "ApiClient", return_value=FakeClient(broken)):
        assert sim.main(["--sessions", "1"]) == 1
    assert "GET /trending — HTTP 503" in capsys.readouterr().out  # the reason is printed under the table


def test_default_base_url_avoids_the_windows_localhost_ipv6_delay():
    assert "127.0.0.1" in sim.DEFAULT_BASE_URL
