"""Tests for src/config.py."""

from __future__ import annotations

import pytest

from src.config import require_api_key, settings


def test_fusion_weights_sum_to_one():
    """The default fusion weights must sum to 1.0 for scores to stay in [0,1]."""
    total = (
        settings.fusion.collaborative_weight
        + settings.fusion.content_weight
        + settings.fusion.semantic_weight
    )
    assert total == pytest.approx(1.0)


def test_require_api_key_returns_value_when_present():
    """require_api_key passes through a non-empty key unchanged."""
    assert require_api_key("TEST_KEY", "abc123") == "abc123"


def test_require_api_key_raises_when_missing():
    """require_api_key raises a RuntimeError with a helpful message when unset."""
    with pytest.raises(RuntimeError, match="TEST_KEY"):
        require_api_key("TEST_KEY", None)


def test_require_api_key_raises_on_empty_string():
    """An empty string is treated the same as a missing key."""
    with pytest.raises(RuntimeError):
        require_api_key("TEST_KEY", "")
