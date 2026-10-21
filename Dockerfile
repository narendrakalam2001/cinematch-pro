# CineMatch Pro — FastAPI backend image.
#
# Code only: the trained artifacts (~1 GB) are NOT baked in. At startup the API
# downloads them from the Hugging Face Hub (HF_ARTIFACTS_REPO + HF_TOKEN) when they
# are not already on disk — see src/artifact_store.py. Locally you can instead
# mount them:  -v ./models:/app/models -v ./chromadb_store:/app/chromadb_store -v ./data:/app/data
#
# Multi-stage build: dependencies are installed into a virtualenv in the build stage
# and the whole venv is copied into a slim runtime stage.

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build

# CPU-only PyTorch first: the default PyPI wheel drags in ~2 GB of CUDA libraries that
# are useless on CPU hosts. Installing it in the same venv makes requirements.txt see it as satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Runtime-only system deps (libgomp for faiss, curl for the healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# The venv lives in /opt/venv (readable by every user) — NOT under /root, which the
# non-root runtime user below could not read.
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# spaCy's English pipeline is a separate download, needed by aspect_sentiment.py.
RUN python -m spacy download en_core_web_sm

# Application code.
COPY src/ src/
COPY services/ services/
COPY serving/ serving/
COPY scripts/ scripts/

# Non-root user (uid 1000 — also what Hugging Face Spaces runs containers as).
RUN useradd --create-home --uid 1000 cinematch \
    && chown -R cinematch:cinematch /app
USER cinematch
ENV HF_HOME=/home/cinematch/.cache/huggingface

EXPOSE 8000

# start-period is long: on a fresh container the artifacts are downloaded before the API answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Render injects $PORT; Hugging Face Spaces uses app_port 8000 (see scripts/deploy_hf_space.py).
CMD ["sh", "-c", "uvicorn serving.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
