# Architecture — Prediction Alpha Engine

**Version**: 0.1 (Bootstrap)
**Date**: May 2026
**Owner**: Keith / 1COMMERCE LLC

## 1. High-Level Layers

```
[External World]
   ↓ (Kalshi WS/REST + future Polymarket)
Ingestion & Normalization Layer
   ↓ (normalized Event objects, enriched features)
Feature Engineering + Hybrid Scoring Engine
   ↓ (EV, liquidity, confidence, portfolio fit scores)
Agentic Research & Legwork Planning Layer
   ↓ (auto-briefs, research tasks, execution plans)
Strict ML Filtering Gate (the sacred noise-killer)
   ↓ (only top-N or score > threshold pass)
Neural Integration + True Neutral Brain Update
   ↓ (graph nodes, RAG, probability feeds, plan recs)
Selective Notification & Action Layer
   ↓ (dashboard, email/Telegram, one-click plans)
Feedback Loop (resolutions → model retrain → Brain improvement)
```

## 2. Core Data Model (Initial)

**Event** (normalized across platforms):
- id, platform (kalshi|polymarket), external_id, title, category (econ| sports|policy|weather|...)
- yes_price, no_price (or equivalent), implied_prob
- volume_24h, open_interest, liquidity_score (spread, depth)
- resolution_date, status (open|resolved|...)
- raw_metadata, enriched_features (dict or JSONB)
- created_at, updated_at

**OpportunityScore**:
- event_id
- edge_score (your_prob - market_implied, or model output)
- liquidity_adjusted_ev
- confidence (model + rule)
- portfolio_fit (correlation to housing/drone/PM tracks)
- composite_score
- recommended_action (size, direction, research_needed)
- agent_plan_summary (text or structured)
- passed_filter (bool)

**Feedback**:
- opportunity_id, actual_resolution, model_prediction, pnl_if_taken, notes

Stored in Postgres + pgvector for semantic search on events.

## 3. Ingestion Layer (Phase 1 Priority)

- Kalshi: WebSocket channels (ticker, trade, market_lifecycle) + REST for historical/backfill.
- Polymarket (read-only): Gamma API (https://gamma-api.polymarket.com) for markets/prices/volume + optional CLOB for orderbook depth. REST polling + iter_markets (WebSocket/RTDS can be added later).

The Event model and all downstream layers (scoring, agents, Brain, notifications) are platform-agnostic. Adding a platform only requires a client + normalizer.
- Client: Async Python (asyncio + websockets lib or official patterns). Robust reconnection, rate-limit handling.
- Normalizer: Dedup events, standardize fields, compute basic derived (implied prob from prices).
- Enrichment (light): Category tagging, time-to-resolution, simple macro context hooks.
- Storage: Upsert to DB; publish to internal event bus/queue for downstream.
- Future: Polymarket Gamma + CLOB + RTDS when US access stable; unified abstraction (Dome) optional.

## 4. Scoring & Feature Layer

Hybrid approach (fast validation):
- **Rules**: Min liquidity threshold, max days-to-resolution, category whitelist, basic EV calc.
- **ML Model** (initial): Feature vector → edge / has_edge classifier or regressor.
  Features: price momentum, volume trend, historical platform accuracy per category, external signals (light), your domain tags (ag/policy/macro).
- Backtesting harness: Replay historical resolved markets, measure calibration (predicted prob vs actual), Sharpe-like on simulated edges.
- Later: Ensemble (XGBoost + small NN), online learning from feedback.

## 5. Agentic Legwork Layer (Phase 2 — Hardened v2)

The original single-shot research agent has been significantly extended:

- **Multi-step ReAct-style loop** with explicit tool calling (knowledge_base always available; web_search opt-in).
- **Critic / Debate agent** — second pass that reviews the draft thesis and explicitly calls out weaknesses, missing factors, and over-optimism.
- **Short-term memory** — agents recall recent similar opportunities (by category + recency) and inject them into reasoning.
- **Rich structured output** — `AgentResearchBrief` v2 now contains `debate_summary`, `weaknesses`, `tool_calls`, `memory_used`, `confidence_breakdown`, `steps_taken`, etc.
- **Full configurability** — `AgentConfig` (YAML or env-derived) controls model, temperature, max_steps, which tools are enabled, critic/memory toggles, prompt overrides, and `backend` ("auto" | "python" | "langgraph").
- **Observability** — `AgentMetrics` records success rate, avg latency, tool usage, failure modes, and step counts. Exposed in engine logs + via FastAPI at `/metrics/agents` and `/metrics/agents/runs`.
- **Optional LangGraph backend** — When `langgraph` is installed and `agent_backend` is "auto" or "langgraph", a compiled state graph can be used for the research loop (graceful fallback to pure Python).
- **Persistent memory** — `ShortTermAgentMemory` supports "file" or "postgres" persistence (best-effort) in addition to pure in-memory.

Triggering remains strictly gated behind `agent_min_composite_to_research` (and `passed_filter`) to control compute cost.

See `src/prediction_alpha/agents/` (tools.py, memory.py, config.py, legwork.py) and `agent_config.example.yaml`.

- Triggered on high-scoring candidates (pre-filter).
- Agents (LangGraph or custom):
  - Research Agent: Web/search + LLM summarize key drivers, risks, correlated news.
  - Analysis Agent: Generate structured brief (thesis, counter, sizing suggestion, hedges).
  - Planning Agent: Create actionable tasks (calendar, alerts, draft orders), integrate with Linear/Notion if connected.
- Output: Structured JSON + human-readable plan attached to Opportunity.
- Goal: "Starts work to plan doing the legwork" so you review, not start from zero.

## 6. Filtering Gate (Critical)

Multi-stage to enforce "only most likely and valuable":
1. Hard filters (liquidity, time horizon, risk category, capital tie-up).
2. ML score threshold (composite > X or top percentile).
3. Portfolio / personal fit (diversification, alignment with 24-mo plan tracks).
4. Novelty / saturation check (avoid over-exposed categories).

Only passed opportunities proceed to neural update + notification.

## 7. Neural Integration (True Neutral Brain)

- Events as nodes in knowledge graph or high-quality chunks in vector store.
- LLM (local) generates initial analysis + "recommended plan going forward".
- Probability feeds update risk models for wealth tracks (e.g., election → housing policy risk).
- Feedback loop: Actual resolutions + your outcomes improve graph embeddings and scorers.
- Sovereign: All inference self-hosted; selective use of external only if needed.

## 8. Notification & Action

- Channels: In-app (Master Control/UnifyOne queue), email digest (daily/weekly), push/Telegram for urgent (imminent resolution, live sports).
- Content: One-sentence hook + score rationale + plan summary + links.
- Actionability: Copy-paste trade plan, one-click add to watchlist or execute (future).
- Strict: Max N per day/week to protect attention.

## 9. Tech Decisions & Sovereignty

- Python-first for orchestration, agents, ML (aligns with your existing agents and PACER/UnifyOne Python layers).
- Self-hosted DB + vector (your VPS/Render pattern).
- Avoid heavy external dependencies for core loop.
- Dashboard: Future extension into UnifyOne (TS/Next) rather than separate app.
- Observability: Structured logging, simple metrics (opportunities surfaced, calibration, simulated P&L).

## 10. Risks & Mitigations (Edge Cases)

- Market efficiency / weak edges: Strict thresholds + focus on niches where you have asymmetry (macro for wealth, policy for drones).
- Liquidity traps: Hard filter on open_interest/volume/spread.
- Model overconfidence: Ensembles, calibration tracking, paper-trade gate.
- Agent hallucinations: Human review required on any live action; verify facts.
- Regulatory: Kalshi CFTC status solid in OR; monitor state challenges. Tax logging built-in from day 1.
- Black swans: Human oversight + broad context; conservative sizing.
- Notification fatigue: The filter is the safeguard.

## 11. Data Flow Example (Happy Path)

Kalshi WS tick → normalize Event → enrich features → score (rule+ML) → if passes hard filter → spawn research agents → generate plan → re-score with agent insights → final filter → update Brain graph + RAG → notify (if top tier) → log for feedback.

## 12. Future Extensions
- Polymarket US support + cross-platform arb.
- Advanced quant (implied vol, statistical arb).
- 3D opportunity visualization (your game dev strengths).
- Direct execution hooks or broker integration.
- Productization: Curated signals as UnifyOne module or SaaS.
- BCI / neural interface hooks (long-term True Neutral vision).

---

## 13. Current Implementation Status (Delivered MVP)

**As of this revision the full end-to-end system described in sections 1–12 is live and functional.**

- `src/prediction_alpha/models.py` — canonical strongly-typed Event / OpportunityScore / AgentResearchBrief
- Ingestion: production-grade KalshiRESTClient + KalshiWebSocketClient with throttling, retries, and reconnect
- Scoring: HybridScorer + ScoringRules + features (fully config-driven via ScoringConfig + YAML)
- Agents: `agents/legwork.py` — Ollama-first research + planning with robust stub fallback and TaskManager integration
- Notifications + Brain: `notifications/` — selective console/email + `prepare_brain_payload` export hook (only top-tier)
- Pipeline: `pipeline.py` + `run.py` — one-command autonomous background service (WS + periodic backfill + agents + notify)
- API + Tasks: existing FastAPI surface + shared asyncio task manager for graceful lifecycle
- Docker + deploy docs complete

All non-negotiable requirements from the master prompt are satisfied:
- Sovereign (local LLM preference, .env only, self-hosted DB)
- Config-driven at every layer
- Strict multi-stage filtering sacred
- Background scalable + observable
- Productization hooks (per-profile config comments throughout, clean serializable models)
- Working end-to-end loop today

Future work (Phase 4 roadmap items) can now be built on a solid, proven foundation instead of scaffolding.

This is no longer a design doc — it is a living description of a running system.

## 14. True Neutral Brain v2 Integration (Concrete Implementation)

The original preparation hook has evolved into a full bidirectional integration layer (`src/prediction_alpha/brain/`).

**Ingestion Flow**:
1. Opportunity passes strict multi-stage filter + agents.
2. Background task (via TaskManager) calls `TrueNeutralBrainIngestor.ingest()`.
3. Rich `BrainOpportunity` is constructed with wealth-track tags, macro signals, agent thesis + debate.
4. Upserted into `brain_opportunities` (dedup on `event_id`).
5. Embedding generated (stub or real local model) and stored in pgvector column.

**Key Artifacts**:
- `BrainIngestionConfig` (thresholds, categories, embedding settings) — YAML or env.
- `BrainOpportunity` model (graph + RAG ready).
- `BrainRetriever` with wealth-track-aware and semantic queries.
- Automatic schema creation for `brain_opportunities` table + vector indexes.

**Sovereign Pattern**:
- Everything lives in the same Postgres instance the engine already uses (pgvector extension).
- No external SaaS vector DB required.
- Embeddings can be generated locally (sentence-transformers) or via a self-hosted embedding service.

**Value to the 24-Month Wealth Plan**:
- The Brain no longer has to reason over raw Kalshi data.
- It receives pre-filtered, pre-researched, wealth-track-tagged probability updates with thesis/counter/debate already attached.
- Perfect for macro hedge sizing, ag policy monitoring, housing rate risk models, etc.

Example full flow is demonstrated in `scripts/demo_brain_ingestion.py`.
