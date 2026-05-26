"""Pydantic models for data going into / coming out of True Neutral Brain v2.

These are the canonical shapes used for both storage and retrieval.
They are deliberately rich so the Brain (graph + RAG) gets high-signal, well-tagged data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prediction_alpha.models import Event, OpportunityScore


class WealthTrackRelevance(BaseModel):
    """How relevant this opportunity is to specific wealth tracks."""

    housing: float = 0.0       # 0..1
    ag_drone: float = 0.0
    property_management: float = 0.0
    general_macro: float = 0.0


class BrainOpportunity(BaseModel):
    """The primary object written to True Neutral Brain v2.

    Contains everything the Brain needs for both graph nodes and vector search.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    # Identity & provenance
    id: str  # same as Event.id for easy correlation
    event_id: str
    platform: str
    external_id: str
    title: str
    category: str

    # Core prediction market data
    implied_prob: float | None
    yes_price: float | None
    no_price: float | None
    resolution_date: datetime | None
    days_to_resolution: float | None
    liquidity_score: float
    volume_24h: float
    open_interest: float

    # Alpha Engine scoring (the "why this matters" signal)
    edge_score: float
    composite_score: float
    confidence: float
    portfolio_fit: float
    recommended_action: str
    rationale: list[str] = Field(default_factory=list)

    # Agent intelligence (the highest value part)
    agent_thesis: str | None = None
    agent_counter_thesis: str | None = None
    agent_drivers: list[str] = Field(default_factory=list)
    agent_risks: list[str] = Field(default_factory=list)
    agent_sizing: str | None = None
    agent_confidence: float | None = None
    debate_summary: str | None = None
    weaknesses: list[str] = Field(default_factory=list)

    # Brain-specific enrichment
    wealth_tracks: list[str] = Field(default_factory=list)  # e.g. ["housing", "ag_drone"]
    wealth_track_relevance: WealthTrackRelevance = Field(default_factory=WealthTrackRelevance)
    macro_signals: dict[str, Any] = Field(default_factory=dict)  # e.g. {"fed": 0.9, "tariff_risk": 0.6}
    policy_tags: list[str] = Field(default_factory=list)

    # Vector + retrieval
    text_for_embedding: str
    embedding: list[float] | None = None  # populated at ingest time if enabled

    # Lifecycle
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_engine_version: str = "prediction-alpha-v2"

    @classmethod
    def from_engine_data(
        cls,
        event: Event,
        score: OpportunityScore,
        agent_brief: dict[str, Any] | None,
        wealth_tracks: list[str] | None = None,
        macro_signals: dict[str, Any] | None = None,
        text_for_embedding: str | None = None,
    ) -> BrainOpportunity:
        """Convenience constructor used by the ingestor."""
        brief = agent_brief or {}

        return cls(
            id=event.id,
            event_id=event.id,
            platform=event.platform.value,
            external_id=event.external_id,
            title=event.title,
            category=event.category,
            implied_prob=event.implied_prob,
            yes_price=event.yes_price,
            no_price=event.no_price,
            resolution_date=event.resolution_date,
            days_to_resolution=event.days_to_resolution,
            liquidity_score=event.liquidity_score,
            volume_24h=event.volume_24h,
            open_interest=event.open_interest,
            edge_score=score.edge_score,
            composite_score=score.composite_score,
            confidence=score.confidence,
            portfolio_fit=score.portfolio_fit,
            recommended_action=score.recommended_action.value,
            rationale=score.rationale,
            agent_thesis=brief.get("thesis"),
            agent_counter_thesis=brief.get("counter_thesis"),
            agent_drivers=brief.get("key_drivers", []),
            agent_risks=brief.get("risks", []),
            agent_sizing=brief.get("recommended_sizing"),
            agent_confidence=brief.get("confidence_in_edge"),
            debate_summary=brief.get("debate_summary"),
            weaknesses=brief.get("weaknesses", []),
            wealth_tracks=wealth_tracks or [],
            macro_signals=macro_signals or {},
            text_for_embedding=text_for_embedding or "",
        )
