"""Async Postgres persistence for normalized events and scores."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from pydantic import TypeAdapter

from prediction_alpha.models import Event, OpportunityScore


class PostgresStore:
    """Small asyncpg repository.

    SQL is intentionally direct: fewer moving pieces in Phase 1, clean upgrade path
    to SQLAlchemy repositories if the schema grows.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self.database_url, min_size=1, max_size=5)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        if self._pool is None:
            await self.connect()
        if self._pool is None:
            raise RuntimeError(
                "Postgres pool is not initialized. Ensure connect() was called before using "
                "the connection."
            )
        async with self._pool.acquire() as conn:
            yield conn

    async def create_schema(self) -> None:
        """Create Phase 1 tables if they do not exist."""

        async with self.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    yes_price DOUBLE PRECISION,
                    no_price DOUBLE PRECISION,
                    implied_prob DOUBLE PRECISION,
                    volume_24h DOUBLE PRECISION NOT NULL DEFAULT 0,
                    open_interest DOUBLE PRECISION NOT NULL DEFAULT 0,
                    liquidity_score DOUBLE PRECISION NOT NULL DEFAULT 0,
                    resolution_date TIMESTAMPTZ,
                    status TEXT NOT NULL,
                    raw_metadata JSONB NOT NULL,
                    enriched_features JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    UNIQUE (platform, external_id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_status_resolution
                    ON events (status, resolution_date);
                CREATE TABLE IF NOT EXISTS opportunity_scores (
                    id BIGSERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    edge_score DOUBLE PRECISION NOT NULL,
                    liquidity_adjusted_ev DOUBLE PRECISION NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    portfolio_fit DOUBLE PRECISION NOT NULL,
                    composite_score DOUBLE PRECISION NOT NULL,
                    recommended_action TEXT NOT NULL,
                    agent_plan_summary TEXT,
                    passed_filter BOOLEAN NOT NULL,
                    rationale JSONB NOT NULL,
                    features JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_opportunity_scores_passed
                    ON opportunity_scores (passed_filter, composite_score DESC);
                """
            )

    async def upsert_event(self, event: Event) -> None:
        payload = event.model_dump(mode="json")
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO events (
                    id, platform, external_id, title, category, yes_price, no_price, implied_prob,
                    volume_24h, open_interest, liquidity_score, resolution_date, status,
                    raw_metadata, enriched_features, created_at, updated_at
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8,
                    $9, $10, $11, $12, $13, $14::jsonb, $15::jsonb, $16, $17
                )
                ON CONFLICT (platform, external_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    category = EXCLUDED.category,
                    yes_price = EXCLUDED.yes_price,
                    no_price = EXCLUDED.no_price,
                    implied_prob = EXCLUDED.implied_prob,
                    volume_24h = EXCLUDED.volume_24h,
                    open_interest = EXCLUDED.open_interest,
                    liquidity_score = EXCLUDED.liquidity_score,
                    resolution_date = EXCLUDED.resolution_date,
                    status = EXCLUDED.status,
                    raw_metadata = EXCLUDED.raw_metadata,
                    enriched_features = EXCLUDED.enriched_features,
                    updated_at = EXCLUDED.updated_at;
                """,
                payload["id"],
                payload["platform"],
                payload["external_id"],
                payload["title"],
                payload["category"],
                payload["yes_price"],
                payload["no_price"],
                payload["implied_prob"],
                payload["volume_24h"],
                payload["open_interest"],
                payload["liquidity_score"],
                payload["resolution_date"],
                payload["status"],
                json.dumps(payload["raw_metadata"]),
                json.dumps(payload["enriched_features"]),
                payload["created_at"],
                payload["updated_at"],
            )

    async def insert_score(self, score: OpportunityScore) -> None:
        payload = score.model_dump(mode="json")
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO opportunity_scores (
                    event_id, edge_score, liquidity_adjusted_ev, confidence, portfolio_fit,
                    composite_score, recommended_action, agent_plan_summary, passed_filter,
                    rationale, features, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::jsonb, $12);
                """,
                payload["event_id"],
                payload["edge_score"],
                payload["liquidity_adjusted_ev"],
                payload["confidence"],
                payload["portfolio_fit"],
                payload["composite_score"],
                payload["recommended_action"],
                payload["agent_plan_summary"],
                payload["passed_filter"],
                json.dumps(payload["rationale"]),
                json.dumps(payload["features"]),
                payload["created_at"],
            )


event_adapter = TypeAdapter(Event)
score_adapter = TypeAdapter(OpportunityScore)
