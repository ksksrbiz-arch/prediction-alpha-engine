"""True Neutral Brain v2 integration prep layer.

Produces clean, high-signal payloads that can be:
- Written to a vector store (pgvector, Chroma, etc.)
- Posted to the Brain's /ingest endpoint
- Used to create knowledge-graph nodes + relationships

Nothing here talks directly to the Brain — it is a pure preparation hook so the
Prediction Alpha Engine stays loosely coupled and sovereign.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from prediction_alpha.models import Event, OpportunityScore


def prepare_brain_payload(
    event: Event,
    score: OpportunityScore,
    agent_brief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a RAG / graph-ready document for the True Neutral Brain.

    This is the contract surface. The Brain (or an ETL job) consumes these dicts.
    """

    payload: dict[str, Any] = {
        "type": "prediction_alpha_opportunity",
        "id": event.id,
        "platform": event.platform.value,
        "external_id": event.external_id,
        "title": event.title,
        "category": event.category,
        "implied_prob": event.implied_prob,
        "edge_score": score.edge_score,
        "composite_score": score.composite_score,
        "recommended_action": score.recommended_action.value,
        "liquidity_score": event.liquidity_score,
        "days_to_resolution": event.days_to_resolution,
        "volume_24h": event.volume_24h,
        "rationale": score.rationale,
        "features": score.features,
        "created_at": datetime.now(UTC).isoformat(),
        "raw_event": {
            "title": event.title,
            "resolution_date": event.resolution_date.isoformat() if event.resolution_date else None,
            "status": event.status.value,
        },
    }

    if agent_brief:
        payload["agent_thesis"] = agent_brief.get("thesis")
        payload["agent_counter"] = agent_brief.get("counter_thesis")
        payload["agent_drivers"] = agent_brief.get("key_drivers")
        payload["agent_risks"] = agent_brief.get("risks")
        payload["agent_sizing"] = agent_brief.get("recommended_sizing")
        payload["agent_confidence"] = agent_brief.get("confidence_in_edge")

    # Vector-friendly text blob for embedding
    text_parts = [
        event.title,
        f"Category: {event.category}",
        f"Edge: {score.edge_score:+.2%}",
        " ".join(score.rationale),
    ]
    if agent_brief:
        text_parts.extend([agent_brief.get("thesis", ""), agent_brief.get("counter_thesis", "")])
    payload["text_for_embedding"] = " | ".join(p for p in text_parts if p)

    return payload


def export_brain_batch(
    opportunities: list[dict[str, Any]],  # [{"event": Event, "score": OpportunityScore, "brief": ...}]
) -> list[dict[str, Any]]:
    """Convenience batch exporter for the full pipeline or nightly jobs."""

    out: list[dict[str, Any]] = []
    for item in opportunities:
        ev = item.get("event")
        sc = item.get("score")
        brief = item.get("brief")
        if ev and sc:
            out.append(prepare_brain_payload(ev, sc, brief))
    return out
