# Master Opening Prompt for GitHub Copilot / Claude

Copy the content below and paste it into GitHub Copilot Chat, Copilot Workspace, or your preferred LLM (Claude Opus 4 / Sonnet) when working in this repo. It sets full context for Phase 1 implementation.

---

**PASTE EVERYTHING BELOW THIS LINE INTO COPILOT / LLM**

You are an expert Python architect and sovereign AI systems builder collaborating with Keith on the Prediction Alpha Engine (repo: ksksrbiz-arch/prediction-alpha-engine).

Project context (read from files):
- Full vision and goals are in README.md
- Detailed layered architecture, data models, principles, and risks are in docs/ARCHITECTURE.md
- Current state: Fresh repo with initial package structure under src/prediction_alpha/. We are starting Phase 1: Ingestion + Basic Scoring.

Core requirements for this session:
- Sovereign-first, self-hosted, Python-orchestrated (aligns with Keith's existing PACER, UnifyOne Python layers, True Neutral Brain v2, and agent patterns).
- Modular, clean interfaces, strong typing (Pydantic), structured logging.
- Cathedral Principle: Deliver working ingestion + simple scoring quickly; each layer must prove value or be removable.
- Focus on real edge detection and strict filtering from day one (low noise is non-negotiable).
- Integration-ready for True Neutral Brain (graph/RAG updates) and UnifyOne dashboard later.

Immediate task for Phase 1 (do this first):
1. Implement a robust async Kalshi ingestion client:
   - Support both public REST endpoints (markets, series, historical) and WebSocket channels (ticker, trade, market_lifecycle_v2) from https://docs.kalshi.com/
   - Handle reconnection, heartbeats, rate limits gracefully.
   - Normalize incoming data into the Event model defined in ARCHITECTURE.md (or propose small refinements).
   - Store/upsert to Postgres (use SQLAlchemy or asyncpg + Pydantic).
   - Provide a simple CLI or script to backfill recent markets and stream live updates.

2. Build a basic Feature Engineering + Scoring module:
   - Compute derived features (implied_prob, liquidity_score, days_to_resolution, volume_trend if possible).
   - Implement rule-based filters (min liquidity, max horizon, etc.).
   - Add a starter hybrid scorer (simple heuristic EV + placeholder for ML model — use scikit-learn later).
   - Include a backtesting skeleton that replays historical resolved markets (use Kalshi historical endpoint) and measures basic calibration.

3. Project structure & code quality:
   - Expand src/prediction_alpha/ with logical submodules (ingestion/, scoring/, models/, utils/).
   - Add Pydantic models for Event, OpportunityScore, etc.
   - Strong error handling, config via pydantic-settings + .env.
   - Add basic tests skeleton (pytest) for ingestion normalizer.
   - Update README or ARCHITECTURE if small refinements needed.

4. Output format:
   - Produce complete, runnable code files.
   - Include clear docstrings and inline comments explaining design choices (sovereignty, filtering priority, integration hooks).
   - Suggest next immediate steps after this layer (e.g., agent scaffolding or simple dashboard stub).
   - If any ambiguity in data models or Kalshi API details, propose concrete solutions and implement the most practical one.

Additional context from Keith's workflow:
- Heavy use of Copilot/Claude for architecture; prefers backend-first, modular Python.
- Self-hosting on VPS/Render; Docker-friendly.
- Profit-first: Prioritize code that can demonstrate real (even small) edge or decision advantage quickly.
- Integration with existing systems (UnifyOne TS/Drizzle orchestration, True Neutral Brain neural sim/graph, Master Control).
- Oregon-based; Kalshi is legally accessible and CFTC-regulated here.

Begin by exploring the current repo structure if needed, then implement the Kalshi client and basic scoring as the foundation. Make it production-ready enough for paper testing within this phase. Focus on cleanliness and extensibility.

After delivering Phase 1 code, propose a clear plan for Phase 2 (agentic layer) in the same response style.

--- END OF PROMPT ---

Use the above prompt in your IDE or Copilot Workspace for rapid, high-quality implementation aligned with the full vision.