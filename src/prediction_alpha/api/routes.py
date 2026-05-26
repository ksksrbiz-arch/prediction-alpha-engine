"""API routes for the Prediction Alpha Engine.

Phase 1 ships a single endpoint:

    GET /opportunities?min_score=0.7&category=econ&limit=20

This becomes the product API surface that UnifyOne / Master Control dashboard
and future mobile clients consume.

Productization note: when per-user profiles land, the endpoint will accept an
``X-Profile-Id`` header (or JWT claim) and apply that profile's ``ScoringConfig``
instead of the global defaults.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from prediction_alpha.models import Event, OpportunityScore
from prediction_alpha.scoring.scorer import HybridScorer
from prediction_alpha.utils.logging import get_logger

router = APIRouter(tags=["opportunities"])

_log = get_logger("api.opportunities")

# ---------------------------------------------------------------------------
# In-memory opportunity cache (Phase 1 stub — replaced by Postgres query in
# Phase 2 once the ingestion worker persists scored events continuously).
# ---------------------------------------------------------------------------
_scored_cache: list[dict[str, Any]] = []


def update_scored_cache(events: list[Event], scorer: HybridScorer) -> int:
    """Score a batch of events and refresh the in-memory cache.

    Returns the number of opportunities that passed the filter.

    Productization note: this function is called by the background task manager
    after each ingestion cycle.  In Phase 2 it will be replaced by a Postgres
    query so results survive restarts.
    """

    global _scored_cache  # noqa: PLW0603
    new_cache: list[dict[str, Any]] = []
    passed = 0
    for event in events:
        score = scorer.score(event)
        entry = {
            "event": event.model_dump(mode="json"),
            "score": score.model_dump(mode="json"),
        }
        new_cache.append(entry)
        if score.passed_filter:
            passed += 1
    _scored_cache = new_cache
    _log.info("cache_refreshed", total=len(new_cache), passed=passed)
    return passed


# ---------------------------------------------------------------------------
# Response models (thin wrappers for OpenAPI docs)
# ---------------------------------------------------------------------------


class OpportunityResponse(OpportunityScore):
    """Score with the parent event inlined for API convenience."""

    event: Event


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/opportunities")
async def list_opportunities(
    request: Request,
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum composite score"),
    category: str | None = Query(default=None, description="Filter by event category"),
    passed_only: bool = Query(default=True, description="Only return filter-passing opportunities"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
) -> list[dict[str, Any]]:
    """Return scored opportunities, filtered by composite score and category.

    Productization note: this config/filter will support per-user profiles later.
    """

    results: list[dict[str, Any]] = []
    for entry in _scored_cache:
        score_data = entry["score"]
        event_data = entry["event"]
        if score_data["composite_score"] < min_score:
            continue
        if passed_only and not score_data["passed_filter"]:
            continue
        if category and event_data.get("category") != category:
            continue
        results.append(entry)
        if len(results) >= limit:
            break

    # Feedback-loop log: record every API query for usage analytics and
    # future per-user preference learning.
    _log.info(
        "opportunities_queried",
        min_score=min_score,
        category=category,
        passed_only=passed_only,
        results_returned=len(results),
    )
    return results


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "cached_opportunities": str(len(_scored_cache))}
