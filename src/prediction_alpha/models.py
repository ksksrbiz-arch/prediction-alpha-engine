"""Core Pydantic data models for the Prediction Alpha Engine.

Sovereignty note: These models are the single source of truth for events,
scores, and opportunities. They are fully serializable (model_dump), DB-friendly
(JSONB storage), and designed for both personal use and future multi-tenant
productization (UnifyOne integration, per-profile overrides).

All timestamps are timezone-aware (UTC). No secrets or PII ever stored here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Platform(StrEnum):
    """Supported prediction market platforms.

    Productization note: Adding a new platform (e.g. Polymarket) only requires
    a normalizer + client; the rest of the pipeline is platform-agnostic.
    """

    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class EventStatus(StrEnum):
    """Lifecycle states for a prediction market event/contract."""

    OPEN = "open"
    RESOLVED = "resolved"
    CLOSED = "closed"
    PAUSED = "paused"
    UNKNOWN = "unknown"


class RecommendedAction(StrEnum):
    """Actionable recommendation produced by the hybrid scorer + agents.

    These drive downstream behavior (paper trading, research queue, alerts).
    """

    REJECT = "reject"
    RESEARCH = "research"
    PAPER_YES = "paper_yes"
    # Future: PAPER_NO, SCALE_IN, HEDGE, etc. when position sizing lands.


class Event(BaseModel):
    """Canonical normalized event from any prediction market platform.

    This is the core data unit flowing through ingestion → scoring → agents →
    filtering → Brain update → notifications.

    Design decisions:
    - raw_metadata preserved verbatim for replay, forensics, and future feature
      engineering (no data loss on Kalshi schema changes).
    - enriched_features is the mutable scratchpad for scoring, candles, agent
      context, etc.
    - Computed fields (days_to_resolution, etc.) keep downstream code clean.
    - Strict typing + Pydantic v2 guarantees safe serialization to JSON/JSONB.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

    # Identity
    id: str = Field(..., description="Deterministic internal ID (platform + hash)")
    platform: Platform
    external_id: str = Field(..., description="Native ticker/id from the platform")
    title: str

    # Classification
    category: str = "unknown"  # econ, policy, weather, sports, ...

    # Prices (normalized to [0, 1] probability scale)
    yes_price: float | None = None
    no_price: float | None = None
    implied_prob: float | None = None

    # Liquidity & activity signals
    volume_24h: float = 0.0
    open_interest: float = 0.0
    liquidity_score: float = 0.0  # 0..1 normalized

    # Timing
    resolution_date: datetime | None = None
    status: EventStatus = EventStatus.UNKNOWN

    # Full fidelity + enrichment
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    enriched_features: dict[str, Any] = Field(default_factory=dict)

    # Audit
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def days_to_resolution(self) -> float | None:
        """Days until resolution (fractional). None if unknown or already resolved."""
        if self.resolution_date is None:
            return None
        delta = self.resolution_date - datetime.now(UTC)
        return max(delta.total_seconds() / 86_400.0, 0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_live(self) -> bool:
        """True when the market is still tradable / unresolved."""
        return self.status in {EventStatus.OPEN, EventStatus.UNKNOWN}

    def to_brief(self) -> dict[str, Any]:
        """Compact human/LLM-friendly summary for agents and notifications."""
        return {
            "id": self.id,
            "platform": self.platform.value,
            "title": self.title,
            "category": self.category,
            "implied_prob": self.implied_prob,
            "liquidity_score": round(self.liquidity_score, 3),
            "days_to_resolution": round(self.days_to_resolution, 1) if self.days_to_resolution else None,
            "status": self.status.value,
        }


class OpportunityScore(BaseModel):
    """Hybrid scorer output + agent enrichment for a single Event.

    This is the primary object that feeds the strict multi-stage filter.
    Only entries with passed_filter=True proceed to Brain + notifications.

    Productization note: composite_score + passed_filter are the sacred gates.
    Everything downstream (agents, email, Brain) must respect this.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    event_id: str
    edge_score: float  # your_prob - market_implied (can be negative)
    liquidity_adjusted_ev: float
    confidence: float  # 0..1
    portfolio_fit: float  # 0..1, driven by category weights
    composite_score: float  # 0..1 final ranking signal

    recommended_action: RecommendedAction
    agent_plan_summary: str | None = None  # populated by Phase C agents

    passed_filter: bool
    rationale: list[str] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Future Phase 3/4 extension points (kept optional for backward compat)
    research_brief: dict[str, Any] | None = None  # structured agent output
    execution_plan: dict[str, Any] | None = None  # tasks, sizing, hedges


class AgentResearchBrief(BaseModel):
    """Rich structured output from the hardened agentic legwork layer (Phase C+).

    Supports multi-step research, tool use, critic/debate review, and memory.
    Backward compatible with previous single-shot briefs.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str

    # Core thesis (improved)
    thesis: str
    counter_thesis: str

    # Enhanced analysis
    key_drivers: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_sizing: str | None = None

    # New hardened fields
    debate_summary: str | None = None  # output from Critic/Debate agent
    weaknesses: list[str] = Field(default_factory=list)  # flagged by critic
    additional_factors: list[str] = Field(default_factory=list)

    # Execution trace
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # [{tool, args, result_summary}]
    steps_taken: int = 1
    memory_used: list[str] = Field(default_factory=list)  # summaries of recalled similar opportunities

    confidence_in_edge: float  # overall 0..1
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)  # e.g. {"macro": 0.8, "liquidity": 0.6}

    sources: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_agent_output: str | None = None  # last or concatenated LLM responses for audit

    # For observability / Brain
    agent_version: str = "2.0-hardened"
    processing_time_seconds: float | None = None


# Convenience type aliases for downstream code
EventList = list[Event]
ScoredOpportunity = dict[str, Any]  # {"event": Event, "score": OpportunityScore} shape used in caches/API
