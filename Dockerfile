# Clean single-stage image for Render (API by default; worker overrides CMD via render.yaml)
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev libpq5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src
EXPOSE 8000
CMD ["sh", "-c", "uvicorn prediction_alpha.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]