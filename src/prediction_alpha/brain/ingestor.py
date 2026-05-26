"""True Neutral Brain v2 Ingestor.

This is the concrete implementation that takes high-value, agent-enriched
opportunities from the Prediction Alpha Engine and writes them into the
Brain's Postgres + pgvector store in a graph + vector friendly way.

Design goals:
- Sovereign (uses the engine's existing PostgresStore)
- Deduplicating (upsert on event_id)
- Richly tagged for the 24-month wealth plan
- Background-friendly (never blocks the main pipeline)
- Pluggable embedding generation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from prediction_alpha.brain.config import BrainIngestionConfig
from prediction_alpha.brain.models import BrainOpportunity
from prediction_alpha.ingestion.storage import PostgresStore
from prediction_alpha.models import Event, OpportunityScore
from prediction_alpha.utils.logging import get_logger

_log = get_logger("brain.ingestor")


# Default embedding stub (sovereign-safe, never fails)
def _stub_embedding(text: str, dim: int = 768) -> list[float]:
    """Deterministic stub embedding for testing / when no real model is configured."""
    # Simple hash-based pseudo-vector (good enough for demo & tests)
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    vec = [((b % 100) / 50.0 - 1.0) for b in h[:dim]]
    # Pad or truncate
    if len(vec) < dim:
        vec += [0.0] * (dim - len(vec))
    return vec[:dim]


class TrueNeutralBrainIngestor:
    """Main class for writing Prediction Alpha opportunities into the Brain."""

    def __init__(
        self,
        store: PostgresStore,
        config: BrainIngestionConfig,
        *,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ):
        self.store = store
        self.config = config
        self.embedding_fn = embedding_fn or (lambda t: _stub_embedding(t, config.embedding_dimension))
        self._log = get_logger("true_neutral_brain_ingestor")

    async def ensure_schema(self) -> None:
        """Create the brain_opportunities table + pgvector extension + indexes."""
        async with self.store.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS brain_opportunities (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    platform TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,

                    implied_prob DOUBLE PRECISION,
                    liquidity_score DOUBLE PRECISION,
                    days_to_resolution DOUBLE PRECISION,
                    volume_24h DOUBLE PRECISION,
                    open_interest DOUBLE PRECISION,

                    edge_score DOUBLE PRECISION NOT NULL,
                    composite_score DOUBLE PRECISION NOT NULL,
                    confidence DOUBLE PRECISION,
                    recommended_action TEXT,

                    agent_thesis TEXT,
                    agent_counter_thesis TEXT,
                    agent_sizing TEXT,
                    debate_summary TEXT,

                    rationale JSONB,
                    agent_drivers JSONB,
                    agent_risks JSONB,
                    weaknesses JSONB,
                    macro_signals JSONB,

                    wealth_tracks TEXT[] NOT NULL DEFAULT '{{}}',
                    policy_tags TEXT[] NOT NULL DEFAULT '{{}}',

                    text_for_embedding TEXT NOT NULL,
                    embedding VECTOR({self.config.embedding_dimension}),

                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    source_engine_version TEXT,

                    CONSTRAINT brain_opp_event_fk FOREIGN KEY (event_id)
                        REFERENCES events(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_brain_opp_composite
                    ON brain_opportunities (composite_score DESC);

                CREATE INDEX IF NOT EXISTS idx_brain_opp_wealth_tracks
                    ON brain_opportunities USING GIN (wealth_tracks);

                -- Vector similarity index (cosine)
                CREATE INDEX IF NOT EXISTS idx_brain_opp_embedding
                    ON brain_opportunities USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """
            )

    def _compute_wealth_tracks(self, category: str, brief: dict[str, Any] | None) -> list[str]:
        """Apply wealth track tagging logic."""
        tracks: set[str] = set(self.config.always_tag_tracks)

        mapping = self.config.wealth_track_mapping
        cat = category.lower()

        if cat in mapping.housing:
            tracks.add("housing")
        if cat in mapping.ag_drone:
            tracks.add("ag_drone")
        if cat in mapping.property_management:
            tracks.add("property_management")
        if cat in mapping.general_macro:
            tracks.add("general_macro")

        # Simple macro signal boosting from agent brief
        if brief:
            text = " ".join([
                brief.get("thesis", ""),
                brief.get("debate_summary", ""),
                " ".join(brief.get("key_drivers", [])),
            ]).lower()
            if any(k in text for k in ["fed", "rate", "inflation", "housing", "mortgage"]):
                tracks.add("housing")
            if any(k in text for k in ["tariff", "trade", "ag", "crop", "water"]):
                tracks.add("ag_drone")

        return sorted(tracks)

    async def ingest(
        self,
        event: Event,
        score: OpportunityScore,
        agent_brief: dict[str, Any] | None = None,
    ) -> bool:
        """Ingest a single high-value opportunity. Returns True if written."""
        if not self.config.enabled:
            return False

        # Apply filters
        if score.composite_score < self.config.min_composite_score:
            return False
        if score.edge_score < self.config.min_edge:
            return False
        if self.config.allowed_categories and event.category not in self.config.allowed_categories:
            return False

        wealth_tracks = self._compute_wealth_tracks(event.category, agent_brief)

        # Build rich payload
        brain_opp = BrainOpportunity.from_engine_data(
            event, score, agent_brief,
            wealth_tracks=wealth_tracks,
            macro_signals={"fed_policy": 0.8} if "econ" in event.category else {},
        )

        # Generate embedding if enabled
        embedding = None
        if self.config.embedding_enabled:
            try:
                embedding = self.embedding_fn(brain_opp.text_for_embedding)
                brain_opp.embedding = embedding
            except Exception as exc:  # noqa: BLE001
                self._log.warning("embedding_generation_failed", event_id=event.id, error=str(exc)[:120])

        # Persist
        await self._upsert_brain_opportunity(brain_opp)

        if self.config.log_ingests:
            self._log.info(
                "brain_ingested",
                event_id=event.id,
                composite=round(score.composite_score, 3),
                tracks=wealth_tracks,
                embedding=bool(embedding),
            )

        return True

    async def _upsert_brain_opportunity(self, opp: BrainOpportunity) -> None:
        """Atomic upsert with deduplication on event_id."""
        async with self.store.connection() as conn:
            await conn.execute(
                """
                INSERT INTO brain_opportunities (
                    id, event_id, platform, external_id, title, category,
                    implied_prob, liquidity_score, days_to_resolution, volume_24h, open_interest,
                    edge_score, composite_score, confidence, recommended_action,
                    agent_thesis, agent_counter_thesis, agent_sizing, debate_summary,
                    rationale, agent_drivers, agent_risks, weaknesses, macro_signals,
                    wealth_tracks, policy_tags,
                    text_for_embedding, embedding,
                    ingested_at, last_updated, source_engine_version
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10, $11,
                    $12, $13, $14, $15,
                    $16, $17, $18, $19,
                    $20::jsonb, $21::jsonb, $22::jsonb, $23::jsonb, $24::jsonb,
                    $25, $26,
                    $27, $28::vector,
                    $29, $30, $31
                )
                ON CONFLICT (event_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    composite_score = EXCLUDED.composite_score,
                    edge_score = EXCLUDED.edge_score,
                    agent_thesis = EXCLUDED.agent_thesis,
                    debate_summary = EXCLUDED.debate_summary,
                    wealth_tracks = EXCLUDED.wealth_tracks,
                    macro_signals = EXCLUDED.macro_signals,
                    text_for_embedding = EXCLUDED.text_for_embedding,
                    embedding = COALESCE(EXCLUDED.embedding, brain_opportunities.embedding),
                    last_updated = NOW();
                """,
                opp.id,
                opp.event_id,
                opp.platform,
                opp.external_id,
                opp.title,
                opp.category,
                opp.implied_prob,
                opp.liquidity_score,
                opp.days_to_resolution,
                opp.volume_24h,
                opp.open_interest,
                opp.edge_score,
                opp.composite_score,
                opp.confidence,
                opp.recommended_action,
                opp.agent_thesis,
                opp.agent_counter_thesis,
                opp.agent_sizing,
                opp.debate_summary,
                opp.rationale,
                opp.agent_drivers,
                opp.agent_risks,
                opp.weaknesses,
                opp.macro_signals,
                opp.wealth_tracks,
                opp.policy_tags,
                opp.text_for_embedding,
                opp.embedding,
                opp.ingested_at,
                opp.last_updated,
                opp.source_engine_version,
            )

    async def ingest_batch(
        self,
        items: list[dict[str, Any]],  # [{"event": Event, "score": OpportunityScore, "brief": ...}]
    ) -> int:
        """Batch ingest (useful for backfills)."""
        count = 0
        for item in items:
            ev = item.get("event")
            sc = item.get("score")
            brief = item.get("brief")
            if ev and sc:
                if await self.ingest(ev, sc, brief):
                    count += 1
        return count
