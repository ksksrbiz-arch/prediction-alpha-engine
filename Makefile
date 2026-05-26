# Prediction Alpha Engine - Developer & Production shortcuts

.PHONY: help run build up down logs shell test

help:
	@echo "Prediction Alpha Engine - Common commands"
	@echo ""
	@echo "  make run          - Start full stack with docker compose (production mode)"
	@echo "  make up           - Same as run"
	@echo "  make down         - Stop everything"
	@echo "  make logs         - Tail logs from all services"
	@echo "  make build        - Rebuild the engine image"
	@echo "  make shell        - Open shell inside the engine container"
	@echo "  make test         - Run the test suite locally"
	@echo ""
	@echo "For local development without Docker:"
	@echo "  python -m venv .venv && source .venv/bin/activate"
	@echo "  pip install -e ."
	@echo "  python run.py --continuous"

run up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

build:
	docker compose build --no-cache engine

shell:
	docker compose exec engine bash

test:
	PYTHONPATH=src python -m pytest tests/ -q
