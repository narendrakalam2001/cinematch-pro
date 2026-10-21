"""JSON-safety and secret-redaction helpers shared by the API and the clients."""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

_SECRET_PATTERN = re.compile(r"(?i)((?:api_?key|apikey|token)=)[^&\s)'\"]+")


def redact_secrets(message: object) -> str:
    """Mask API keys / tokens that appear in URLs inside a message.

    Requests' exceptions embed the full request URL (including ``api_key=...``);
    logging them verbatim would leak the key into log files and screenshots.

    Args:
        message: Any object (usually an exception) whose string form may
            contain ``api_key=``, ``apikey=`` or ``token=`` query parameters.

    Returns:
        str: The message with each secret value replaced by ``***``.
    """
    return _SECRET_PATTERN.sub(r"\1***", str(message))


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas values into plain JSON-safe types.

    FastAPI cannot serialise ``numpy.ndarray`` (the cause of the HTTP 500 on
    ``GET /movie/{id}``), numpy scalars, ``NaN`` or pandas timestamps.

    Args:
        value: Any (possibly nested) Python / numpy / pandas value.

    Returns:
        Any: The same structure using only ``dict``, ``list``, ``str``,
        ``int``, ``float``, ``bool`` and ``None``. ``NaN``/``NaT``/``NA``
        become ``None``.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, int):
        return value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    return str(value)
