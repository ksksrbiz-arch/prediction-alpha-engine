# Production-oriented Dockerfile for the Prediction Alpha Engine
# Sovereign-friendly: runs everywhere you can run Python + Postgres + optional Ollama

FROM python:3.12-slim AS builder

WORKDIR /app

# System deps for psycopg / asyncpg wheels + build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 alpha

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# App code
COPY . .

# Non-root
USER alpha

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# Default command: one-shot demo (override for continuous)
CMD ["python", "run.py", "--once", "--pages", "3"]

# Healthcheck (used by compose / orchestrators)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

EXPOSE 8000
