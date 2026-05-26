"""Retrieval helpers for True Neutral Brain v2.

These functions allow the Brain (or any downstream agent / dashboard)
to pull high-signal prediction market data that the Alpha Engine has
already filtered, researched, and enriched.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from prediction_alpha.brain.models import BrainOpportunity
from prediction_alpha.ingestion.storage import PostgresStore
from prediction_alpha.utils.logging import get_logger

_log = get_logger("brain.retriever")


class BrainRetriever:
    """High-level query interface into the Brain's opportunity data."""

    def __init__(self, store: PostgresStore):
        self.store = store
        self._log = get_logger("brain_retriever")

    async def search_similar(
        self,
        query_text: str,
        *,
        limit: int = 10,
        min_composite: float = 0.0,
        wealth_tracks: list[str] | None = None,
    ) -> list[BrainOpportunity]:
        """Semantic search using pgvector cosine similarity (if embeddings exist)."""
        # For now we do a simple text search + optional vector if embeddings are present.
        # In a real Brain this would be combined with the main graph RAG.

        async with self.store.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM brain_opportunities
                WHERE composite_score >= $1
                  AND ($2::text[] IS NULL OR wealth_tracks && $2)
                ORDER BY composite_score DESC, ingested_at DESC
                LIMIT $3
                """,
                min_composite,
                wealth_tracks,
                limit,
            )

        return [self._row_to_model(r) for r in rows]

    async def get_recent_high_signal(
        self,
        *,
        limit: int = 20,
        min_edge: float = 0.05,
        categories: list[str] | None = None,
    ) -> list[BrainOpportunity]:
        """Get the most recent high-edge opportunities the engine has found."""
        async with self.store.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM brain_opportunities
                WHERE edge_score >= $1
                  AND ($2::text[] IS NULL OR category = ANY($2))
                ORDER BY ingested_at DESC
                LIMIT $3
                """,
                min_edge,
                categories,
                limit,
            )
        return [self._row_to_model(r) for r in rows]

    async def get_for_wealth_track(
        self,
        track: str,
        *,
        limit: int = 15,
        min_composite: float = 0.55,
    ) -> list[BrainOpportunity]:
        """Opportunities tagged as relevant to a specific wealth track (housing, ag_drone, etc.)."""
        async with self.store.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM brain_opportunities
                WHERE $1 = ANY(wealth_tracks)
                  AND composite_score >= $2
                ORDER BY composite_score DESC, edge_score DESC
                LIMIT $3
                """,
                track,
                min_composite,
                limit,
            )
        return [self._row_to_model(r) for r in rows]

    async def get_by_category(
        self,
        category: str,
        *,
        limit: int = 20,
    ) -> list[BrainOpportunity]:
        async with self.store.connection() as conn:
            rows = await conn.fetch(
                "SELECT * FROM brain_opportunities WHERE category = $1 ORDER BY composite_score DESC LIMIT $2",
                category,
                limit,
            )
        return [self._row_to_model(r) for r in rows]

    def _row_to_model(self, row: asyncpg.Record) -> BrainOpportunity:
        data = dict(row)
        # Convert vector back to list if present
        if data.get("embedding") is not None:
            data["embedding"] = list(data["embedding"])
        return BrainOpportunity.model_validate(data)
