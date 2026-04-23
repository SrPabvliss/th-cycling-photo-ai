# Cycling Photo AI — Detection + OCR Pipeline
# Deployment target: Hetzner CPX31 (4 vCPU AMD, 8GB RAM)
# CPU-only inference, lazy model loading

# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
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
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy source
COPY --from=builder /app/src /app/src

# Copy model weights (mounted or baked in)
# Detection: RF-DETR-M (~128MB)
# OCR: TrOCR-small (~235MB)
# Total: ~363MB in container
COPY weights/ /app/weights/

# Environment variables
ENV RFDETR_WEIGHTS=/app/weights/rfdetr_best.pth
ENV TROCR_WEIGHTS=/app/weights/trocr_bib
ENV OCR_CONFIDENCE_THRESHOLD=0.70
ENV HOST=0.0.0.0
ENV PORT=8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

# Run with single worker (CPU inference, lazy loading)
CMD ["uvicorn", "cycling_photo_ai.pipeline.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
