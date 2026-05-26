# Deployment Guide — Prediction Alpha Engine

## Quick Local / VPS

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
# Edit .env with your Kalshi keys (public endpoints work for read-only)
python run.py --once --pages 2
```

For 24/7:

```bash
python run.py --continuous
# or under systemd / supervisord / your process manager
```

## Docker (Recommended for Production)

```bash
cp docker-compose.example.yml docker-compose.yml
# Edit .env (DATABASE_URL must point to the 'db' service inside the network)
docker compose up -d --build
```

To run the continuous background worker:

```bash
docker compose run --rm engine python run.py --continuous
```

## With Local LLM (Ollama)

```bash
docker compose --profile with-llm up -d
# Then in .env set:
# OLLAMA_BASE_URL=http://ollama:11434
# LLM_PROVIDER=ollama
```

Pull a small model on the ollama container:

```bash
docker compose exec ollama ollama pull llama3.2
```

## API Surface

```bash
uvicorn prediction_alpha.api.app:create_app --factory --host 0.0.0.0 --port 8000
curl http://localhost:8000/opportunities?min_score=0.55
curl http://localhost:8000/health
```

## Environment Variables (all secrets)

See `.env.example`. Never commit real keys.

Key ones for production:
- `KALSHI_API_KEY` / `KALSHI_API_SECRET` (optional for public data)
- `DATABASE_URL`
- `OLLAMA_BASE_URL` + `OLLAMA_MODEL`
- `NOTIFY_EMAIL_TO` + SMTP_* for real email

## Scaling Notes (Phase 4+)

- Run multiple engine workers behind a queue (arq / RQ) for high throughput.
- Per-user `ScoringConfig` + agent profiles stored in DB.
- Prometheus metrics endpoint can be added on top of the existing FastAPI app.
- The Brain export payloads are designed to be consumed by a separate ETL or pushed to your True Neutral Brain service.

This system is built to run on a $5–10 VPS indefinitely with very low resource usage when the strict filters are doing their job.
