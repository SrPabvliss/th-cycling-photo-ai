# Cycling Photo AI — Detection + OCR + Color Pipeline
# Deployment target: Hetzner CPX21 (3 vCPU AMD, 4GB RAM + 2GB swap)
# CPU-only inference, lazy model loading

# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/

# Install project
RUN uv sync --frozen --no-dev

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy source
COPY --from=builder /app/src /app/src

# Copy production weights only (ablations excluded via .dockerignore)
# YOLO11m detection (~39MB) + PARSeq OCR (~91MB) = ~130MB total
COPY weights/yolo11m_v3cleaned/ /app/weights/yolo11m_v3cleaned/
COPY weights/parseq_4phase/ /app/weights/parseq_4phase/

# Production pipeline defaults (matches code defaults — set explicitly for clarity)
ENV DETECTOR_TYPE=yolo
ENV OCR_TYPE=parseq
ENV COLOR_STRATEGY_TYPE=gemini
ENV YOLO_WEIGHTS=/app/weights/yolo11m_v3cleaned/best.pt
ENV PARSEQ_WEIGHTS=/app/weights/parseq_4phase
ENV OCR_CONFIDENCE_THRESHOLD=0.70

# Color pipeline parallelism (added 2026-05-17)
ENV COLOR_PARALLEL_WORKERS=4
ENV GEMINI_MAX_CONCURRENCY=12

# Server
ENV HOST=0.0.0.0
ENV PORT=8001

# GEMINI_API_KEY (or GOOGLE_AI_API_KEY) MUST be injected at runtime (Dokploy env).
# Not baked here — secret stays out of image.

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

# Run with single worker (CPU inference, lazy loading)
CMD ["uvicorn", "cycling_photo_ai.pipeline.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
