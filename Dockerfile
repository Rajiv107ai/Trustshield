# Multi-stage production container build for TrustShield AI
# Stage 1: Builder stage for compiling wheels and installing dependencies
FROM python:3.14-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Minimal runtime image
FROM python:3.14-slim as runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/core:/app/backend:/app:/app/trustshield_project" \
    PORT=8000

WORKDIR /app

# Install minimal runtime utility for readiness probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy application source and pre-trained model artifacts
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
COPY core/ ./core/
COPY trustshield_project/ ./trustshield_project/
COPY scripts/ ./scripts/
COPY data/synthetic_v2_1/orders.csv ./data/synthetic_v2_1/orders.csv
COPY models/ ./models/
RUN pip install --no-cache-dir --no-deps -e .

# Security: non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Readiness and Healthcheck (using orchestrator readiness probe)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

# Production server entrypoint
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
