"""Streamlit Community Cloud entrypoint for the free LITE deployment.

Everything runs in ONE process: the Streamlit UI calls the FastAPI app in-process (no separate API
server), and the heavy ML engines (PyTorch, CLIP, RoBERTa, SBERT, ChromaDB, SVD) are not loaded, so the
app fits in the free tier's memory. On first start the small artifact subset (catalogue + TF-IDF,
~70 MB) is downloaded from the private Hugging Face Hub repo named by the HF_ARTIFACTS_REPO secret.

Streamlit Cloud settings: Main file path ``lite/streamlit_app.py``. Its dependencies come from
``lite/requirements.txt`` (the entrypoint's folder is searched before the repo root).
Secrets: HF_ARTIFACTS_REPO, HF_TOKEN (read), TMDB_API_KEY, GEMINI_API_KEY, OMDB_API_KEY (optional).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CINEMATCH_LITE", "1")
os.environ.setdefault("CINEMATCH_EMBEDDED", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Community Cloud exports top-level secrets as env vars already; this is a safety net.
    import streamlit as st

    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:  # noqa: BLE001 - no secrets file when running locally
    pass

runpy.run_path(str(ROOT / "frontend" / "app.py"), run_name="__main__")
