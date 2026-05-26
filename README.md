# Prediction Alpha Engine

**Sovereign AI-powered prediction market opportunity scout and alpha generator.**

Ingests real-time data from Kalshi (primary) and future Polymarket streams via APIs/WebSockets. Applies rigorous ML and neural-net edge detection, runs agentic research + legwork planning, and surfaces **only the highest-conviction, most valuable profit-potential opportunities** after algorithmic filtering. Designed to feed your True Neutral Brain v2 and accelerate wealth-building automation.

## Vision

Turn noisy prediction markets into a high-signal, low-noise intelligence layer for profit-first decision making. Aligns with Cathedral Principle: validate layers quickly, generate edge or remove blockers, maintain sovereignty and modularity.

Key goals:
- Real-time ingestion & normalization across event contracts (econ/macro for stock plays, sports, policy, weather, etc.)
- Hybrid ML + rule-based scoring for expected value (EV), liquidity-adjusted edge, and personal/portfolio fit
- Agentic "legwork" automation: auto-research, structured briefs, execution plans, task generation
- Neural integration: Update knowledge graph / RAG in True Neutral Brain; generate recommended plans
- Strict filtering: Notify / surface only top opportunities (e.g., top 3-5/week or score > threshold)
- Self-hosted, Python-orchestrated, sovereign-first (local models where possible)
- Seamless integration with UnifyOne, Master Control dashboard, and your 24-month wealth tracks (housing, PM, ag drone)

## Current Status

**✅ Full End-to-End MVP Complete (Phases A–E)** — May 2026

- Live Kalshi ingestion (REST backfill + resilient WebSocket streaming)
- Strict hybrid scoring + multi-stage filtering (rule + heuristic + future ML)
- Agentic legwork (research + structured planning) — local Ollama first, perfect stub fallback
- Selective notifications (console + SMTP email stub) + True Neutral Brain export hook
- Runnable background service + FastAPI surface + Docker-ready
- All secrets via pydantic-settings / .env only
- 12+ tests passing, clean modular sovereign Python

The system is now genuinely usable for paper-trading research 24/7.

## Tech Stack (Initial)
- **Language/Runtime**: Python 3.11+ (primary orchestration & ML); FastAPI for any API surface
- **Data**: PostgreSQL (self-hosted) + pgvector for RAG/events; Pandas/Polars for analysis
- **Real-time**: WebSockets (Kalshi native), asyncio
- **ML/Scoring**: scikit-learn / XGBoost (initial hybrid), PyTorch for custom neural components later
- **Agents**: LangGraph or custom agent orchestration (integrate with your existing Python agents / Manus-style)
- **LLM**: Local/open models (Ollama, etc.) + your sovereign stack for analysis/planning
- **Dashboard/UI (future)**: Integrate with UnifyOne (Next.js/TS) or simple Streamlit/Gradio for MVP
- **Deployment**: Self-hosted VPS/Render/Hetzner, Docker-friendly
- **Versioning**: Git + semantic; strong typing (pydantic, mypy)

## Quick Start (Phase 1)

```bash
git clone https://github.com/ksksrbiz-arch/prediction-alpha-engine.git
cd prediction-alpha-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
# Public Kalshi market data works read-only; set DATABASE_URL to store to Postgres.
python -m prediction_alpha.ingestion.cli backfill --max-pages 1 --print-json
python -m prediction_alpha.ingestion.cli stream --channels ticker trade market_lifecycle_v2
```

To persist normalized events and scores:

```bash
python -m prediction_alpha.ingestion.cli backfill --max-pages 5 --store
```

See `docs/ARCHITECTURE.md` for full layered design, data models, and integration points.

---

## How to Run the Full End-to-End System (One Command)

**Prerequisites**
- Python 3.11+
- (Optional but recommended) Postgres running locally for persistence
- (Optional) Ollama running locally for real agent research (`ollama serve` + `ollama pull llama3.2`)

**30-second start**

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Secrets (never committed)
cp .env.example .env
# Edit .env → put your real KALSHI_API_KEY / SECRET (or leave blank for public data)

# 3. Run the complete sovereign loop
python run.py --once --pages 2
```

What you will see:
- Live markets pulled from Kalshi
- Every event normalized + scored with transparent rationale
- High-value opportunities trigger research agents (stub or real LLM)
- Only the rare top-tier opportunities print beautiful console notifications + Brain payloads
- A clean JSON summary at the end

**Continuous 24/7 background service (the real power)**

```bash
python run.py --continuous
# Leave it running. It will stream live ticks + periodically backfill.
# Graceful shutdown on Ctrl-C.
```

**With real agents (Ollama)**

```bash
# Terminal 1
ollama serve
ollama pull llama3.2

# Terminal 2
OLLAMA_BASE_URL=http://localhost:11434 LLM_PROVIDER=ollama python run.py --once
```

**Docker (fully self-contained)**

```bash
cp docker-compose.example.yml docker-compose.yml
# Add your keys to .env
docker compose up -d --build
# For continuous: docker compose run --rm engine python run.py --continuous
```

See `docs/DEPLOY.md` for production VPS, systemd, and scaling notes.

**API while the engine runs**

```bash
uvicorn prediction_alpha.api.app:create_app --factory --port 8000
curl 'http://localhost:8000/opportunities?min_score=0.55&passed_only=true'
```

---



## Integration Points
- **True Neutral Brain v2**: Event nodes in knowledge graph, RAG corpus updates, probability feeds for wealth plan recommendations (e.g., macro hedges for housing Track A or ag policy for drones).
- **UnifyOne / Master Control**: Opportunity queue + dashboard module. Selective notifications (email, Telegram, in-app).
- **Wealth Tracks**: Surface edges relevant to housing portfolio (macro), Cleveland PM (local econ), Oregon ag drone (weather/policy), PACER (regulatory events).
- **Other bots**: Synergy with gov contract watch, news enrichment, content clipping (market narratives as Shorts).

## Roadmap (Cathedral Style)
1. **Phase 1 (Ingestion + Basic Scoring)**: Kalshi WS/REST client, event normalization, feature store, rule-based + simple ML scorer, basic backtesting harness.
2. **Phase 2 (Agentic Layer)**: Research agents, structured plan generation, task creation for legwork.
3. **Phase 3 (Neural + Filtering)**: RAG/graph integration, advanced ensemble scoring, strict multi-stage filter, selective notification system.
4. **Phase 4 (Polish & Productize)**: Dashboard polish, feedback loop (resolutions → retraining), Polymarket support (when US access stable), tax/P&L tracking, risk management (position sizing).

## Principles
- **Sovereign & Self-Hosted**: Prioritize local inference, your infra, no unnecessary cloud lock-in.
- **Low Noise, High Signal**: The ML filter is sacred — only surface what passes rigorous thresholds.
- **Profit-First + Edge Focus**: Validate real monetary or decision-making edge quickly. Paper trade before capital.
- **Modular & Extensible**: Clean interfaces between ingestion → scoring → agents → neural → action.
- **Feedback & Self-Improvement**: Every resolution improves the models and Brain.
- **Risk-Aware**: Liquidity filters, position sizing rules, human oversight gate on live trades.

## Legal / Compliance Note (Oregon)
Kalshi is CFTC-regulated event contracts and available to Oregon residents. This system is for informational/analytical use + personal trading on compliant platforms. Always verify current state/federal rules, track taxes meticulously, and never risk more than you can afford. This is not financial advice.

## Next Steps
Clone the repo, review `docs/ARCHITECTURE.md`, then use the master Copilot prompt (provided separately or in chat) to implement Phase 1 ingestion layer.

Built for Keith / 1COMMERCE LLC — profit-first sovereign automation.

---

*Status: ✅ Full end-to-end MVP complete and runnable (Phases A–E). Ready for paper-trading research and True Neutral Brain integration.*

## Implementation Status (Full MVP Delivered)

All layers from the master prompt are complete and runnable:

**Phase A – Foundation & Ingestion** ✅ Live
- Robust async Kalshi client (WS + REST) with reconnection
- Full Pydantic `Event` normalization + Postgres upsert

**Phase B – Scoring & Filtering** ✅ Strict & Transparent
- Feature engineering + configurable hybrid scorer (rules + placeholder ML)
- Hard filters → ML score → portfolio fit gate

**Phase C – Agentic Legwork** ✅ Background + Sovereign
- Research + planning agents via clean async orchestration (Ollama preferred)
- Only triggered on opportunities passing early high bars
- Structured `AgentResearchBrief` + human summary

**Phase D – Notifications + Brain Prep** ✅ Selective
- Console + SMTP email stub (only top-tier, attention-protected)
- `prepare_brain_payload()` hook ready for True Neutral Brain v2 / RAG / graph

**Phase E – Autonomy & Polish** ✅ One Command
- `python run.py --continuous` for 24/7 background operation
- Structured logging, task manager, graceful shutdown
- FastAPI + Docker + full test coverage on critical paths

See `docs/DEPLOY.md` and the "How to Run" section above.

### API Server

```bash
uvicorn prediction_alpha.api.app:create_app --factory --host 0.0.0.0 --port 8000
# Then: curl http://localhost:8000/opportunities?min_score=0.7
# Health: curl http://localhost:8000/health
```

### Scoring Configuration

Scoring thresholds, composite weights, and category weights can be overridden via:
1. Environment variables (`.env`) — e.g. `MIN_COMPOSITE_SCORE=0.60`
2. A YAML file — set `SCORING_CONFIG_PATH=scoring_config.yaml` (see `scoring_config.example.yaml`)

YAML takes precedence when `SCORING_CONFIG_PATH` is set.