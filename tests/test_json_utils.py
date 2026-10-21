"""Tests for src/json_utils.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.json_utils import redact_secrets, to_jsonable


def test_to_jsonable_converts_numpy_and_pandas_values():
    """The original GET /movie 500: numpy arrays/scalars, NaN and timestamps become plain JSON types."""
    payload = {
        "genres": np.array(["Action", "Drama"]),
        "id": np.int64(7),
        "score": np.float32(1.5),
        "missing": float("nan"),
        "na": pd.NA,
        "when": pd.Timestamp("2020-01-02"),
        "nested": [{"cast": np.array([])}],
    }
    result = to_jsonable(payload)
    assert result["genres"] == ["Action", "Drama"]
    assert result["id"] == 7 and result["score"] == 1.5
    assert result["missing"] is None and result["na"] is None
    assert result["when"].startswith("2020-01-02")
    assert result["nested"] == [{"cast": []}]
    json.dumps(result)  # must not raise


def test_redact_secrets_masks_keys_in_urls():
    """API keys embedded in request URLs never reach the logs."""
    message = "401 for url: https://www.omdbapi.com/?i=tt1&apikey=d7ef3f40&plot=full and ?api_key=abc123&x=1"
    cleaned = redact_secrets(message)
    assert "d7ef3f40" not in cleaned and "abc123" not in cleaned
    assert "apikey=***" in cleaned and "api_key=***" in cleaned
    assert "plot=full" in cleaned
