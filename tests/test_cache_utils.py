"""Tests for src/cache_utils.py."""

from __future__ import annotations

import time

from src.cache_utils import TTLCache


def test_set_and_get_returns_value():
    """A value stored with set() is retrievable via get()."""
    cache = TTLCache(ttl_seconds=60, max_entries=10)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_get_missing_key_returns_none():
    """A key that was never set returns None, not an exception."""
    cache = TTLCache(ttl_seconds=60, max_entries=10)
    assert cache.get("does_not_exist") is None


def test_entry_expires_after_ttl():
    """An entry becomes unreachable once its TTL has elapsed."""
    cache = TTLCache(ttl_seconds=1, max_entries=10)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    time.sleep(1.1)
    assert cache.get("key1") is None


def test_max_entries_evicts_oldest():
    """Once at capacity, adding a new entry evicts the closest-to-expiring one."""
    cache = TTLCache(ttl_seconds=100, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert len(cache) == 2
    assert cache.get("c") == 3


def test_clear_removes_all_entries():
    """clear() empties the cache."""
    cache = TTLCache(ttl_seconds=60, max_entries=10)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None
