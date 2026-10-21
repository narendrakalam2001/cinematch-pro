"""
Shared in-memory caching utility for CineMatch Pro's API clients.

A tiny thread-safe, dict-based TTL cache used as a lightweight stand-in
for Redis in this demo deployment. Any client (TMDB, OMDB, etc.) can
import `TTLCache` from here to avoid duplicating cache logic; swap for a
real Redis-backed implementation in production by preserving the same
`get`/`set` interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class _CacheEntry:
    """A single TTL-cache entry.

    Attributes:
        value: The cached response payload.
        expires_at: Unix timestamp after which this entry is stale.
    """

    value: Any
    expires_at: float


class TTLCache:
    """A tiny thread-safe, dict-based time-to-live cache.

    Attributes:
        ttl_seconds: How long an entry remains valid after being set.
        max_entries: Maximum number of entries kept before the oldest
            (by expiry time) is evicted to make room for a new one.
    """

    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, _CacheEntry] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        """Fetch a cached value if present and not expired.

        Args:
            key: Cache key.

        Returns:
            Any | None: The cached value, or `None` on miss/expiry.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del self._store[key]
                return None
            return entry.value

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache, evicting the oldest entry if full.

        Args:
            key: Cache key.
            value: Value to store.
        """
        with self._lock:
            if len(self._store) >= self.max_entries:
                oldest_key = min(self._store, key=lambda k: self._store[k].expires_at)
                del self._store[oldest_key]
            self._store[key] = _CacheEntry(value=value, expires_at=time.time() + self.ttl_seconds)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        """Return the current number of (possibly expired) entries.

        Returns:
            int: Entry count.
        """
        return len(self._store)
