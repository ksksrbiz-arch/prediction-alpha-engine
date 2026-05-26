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

**Phase 0/1 Bootstrap** — Repo initialized. Core architecture defined. Ready for ingestion layer implementation.

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

*Status: Active development — Phase 1 in progress.*

## Phase 1 Implementation Notes

- `src/prediction_alpha/ingestion/kalshi_client.py` provides async REST backfill plus resilient WebSocket streaming for `ticker`, `trade`, and `market_lifecycle_v2`.
- `src/prediction_alpha/ingestion/normalizer.py` maps Kalshi payloads into the canonical `Event` model and keeps raw payloads for replay/Brain integration.
- `src/prediction_alpha/ingestion/storage.py` upserts events into Postgres and records opportunity scores.
- `src/prediction_alpha/scoring/` computes implied probability, liquidity, horizon, optional volume trend, strict filters, and a transparent heuristic scorer with a future ML hook.
- `python -m prediction_alpha.ingestion.cli backtest --max-pages 1` runs a resolved-market calibration skeleton when Kalshi historical/resolved payloads include outcomes.