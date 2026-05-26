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

- Kalshi: WebSocket channels (ticker, trade, market_lifecycle) + REST for historical/backfill (markets, series, orderbooks, candlesticks).
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

## 5. Agentic Legwork Layer (Phase 2)

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

This architecture prioritizes rapid validation of real edge while preserving full sovereignty and low cognitive load.
