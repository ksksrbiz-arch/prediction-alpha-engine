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

from prediction_alpha.agents.legwork import get_agent_metrics
from prediction_alpha.api.tasks import task_manager
from prediction_alpha.feedback.loop import FeedbackLoop
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
async def health() -> dict[str, Any]:
    """Liveness probe - is the process alive?"""
    return {
        "status": "ok",
        "cached_opportunities": len(_scored_cache),
        "pending_tasks": task_manager.pending_count if 'task_manager' in globals() else 0,
    }

@router.get("/ready")
async def ready() -> dict[str, Any]:
    """Readiness probe - is the service ready to accept traffic?"""
    return {
        "status": "ready",
        "cached_opportunities": len(_scored_cache),
    }


# ---------------------------------------------------------------------------
# Cortex bridge — callable voice for the day-trade platform
# ---------------------------------------------------------------------------


@router.get("/cortex/signal", tags=["cortex"])
async def cortex_signal(
    symbol: str = Query(..., min_length=1, max_length=12, description="Ticker, e.g. SPY, QQQ, BTC"),
) -> dict[str, Any]:
    """Per-symbol directional signal in [-1, 1] for the day-trade cortex.

    Aggregates every cached live prediction market that maps to the symbol
    (see ``integrations.cortex``). ``score`` is null when nothing maps — the
    trader treats that as an abstaining voice, not a zero vote.
    """

    from prediction_alpha.integrations.cortex import signal_for

    sig = signal_for(symbol, _scored_cache)
    _log.info("cortex_signal_queried", symbol=sig["symbol"], score=sig["score"], n=sig["n"])
    return sig


@router.get("/cortex/signals", tags=["cortex"])
async def cortex_signals(
    symbols: str = Query(default="SPY,QQQ,BTC", description="Comma-separated tickers"),
) -> list[dict[str, Any]]:
    """Batch form of /cortex/signal for the trader's watchlist sweep."""

    from prediction_alpha.integrations.cortex import signals_for

    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:25]
    return signals_for(syms, _scored_cache)


# ---------------------------------------------------------------------------
# Simple Production Status Dashboard
# ---------------------------------------------------------------------------

@router.get("/status", include_in_schema=False)
async def status_dashboard() -> str:
    """Minimal HTML dashboard for quick operational visibility."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Prediction Alpha Engine - Status</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
        .card {{ background: #1e2937; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; }}
        h1 {{ color: #60a5fa; }}
        .metric {{ font-size: 2rem; font-weight: bold; color: #34d399; }}
        .label {{ color: #94a3b8; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <h1>Prediction Alpha Engine</h1>
    <div class="card">
        <div class="label">Cached Opportunities</div>
        <div class="metric">{len(_scored_cache)}</div>
    </div>
    <div class="card">
        <div class="label">Pending Background Tasks</div>
        <div class="metric">{task_manager.pending_count}</div>
    </div>
    <div class="card">
        <div class="label">Agent Metrics (last 5 min summary)</div>
        <pre>{get_agent_metrics().summary()}</pre>
    </div>
    <p style="color:#64748b">Auto-refreshes every 30s. For production use Prometheus + Grafana or your Master Control dashboard.</p>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Agent Metrics (v2 hardened legwork layer observability)
# ---------------------------------------------------------------------------


@router.get("/metrics/agents", tags=["metrics"])
async def get_agent_metrics_summary() -> dict[str, Any]:
    """Return aggregated observability for the agentic legwork layer.

    Includes success rate, average latency, tool usage, critic runs, etc.
    This is the primary endpoint for dashboards and monitoring.
    """
    metrics = get_agent_metrics()
    return metrics.summary()


@router.get("/metrics/agents/runs", tags=["metrics"])
async def get_agent_recent_runs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    """Return the most recent individual agent run records (lightweight).

    Useful for debugging specific research jobs. Raw LLM output is truncated.
    """
    metrics = get_agent_metrics()
    recent = metrics.runs[-limit:][::-1]  # newest first

    sanitized = []
    for run in recent:
        sanitized.append({
            "event_id": run.event_id,
            "started_at": run.started_at.isoformat(),
            "ended_at": run.ended_at.isoformat() if run.ended_at else None,
            "success": run.success,
            "steps_taken": run.steps_taken,
            "tools_used": run.tools_used,
            "critic_used": run.critic_used,
            "memory_items_recalled": run.memory_items_recalled,
            "llm_calls": run.llm_calls,
            "approx_tokens": run.approx_tokens,
            "processing_time_seconds": round(run.processing_time_seconds, 2),
            "failure_reason": run.failure_reason,
            "model": run.model,
        })
    return sanitized


# ---------------------------------------------------------------------------
# Feedback / Self-Improvement Endpoints
# ---------------------------------------------------------------------------


_feedback_loop: FeedbackLoop | None = None  # populated at app startup or lazily


@router.get("/metrics/feedback", tags=["metrics"])
async def get_feedback_calibration() -> dict[str, Any]:
    """Return current calibration metrics from the feedback loop."""
    global _feedback_loop
    if _feedback_loop is None:
        # In real usage this would be injected; for now return empty
        return {"count": 0, "message": "FeedbackLoop not initialized in this process"}
    return _feedback_loop.get_calibration_summary()


@router.post("/feedback/log_resolution", tags=["feedback"])
async def log_market_resolution(payload: dict[str, Any]) -> dict[str, str]:
    """Manual or webhook endpoint to log a market's actual resolution.

    Expected payload: {"event_id": "...", "actual": 1.0 or 0.0, "predicted": 0.XX (optional)}
    In production this would be called by a settlement watcher.
    """
    # Minimal implementation — in a real system we'd look up the event/score from DB
    global _feedback_loop
    if _feedback_loop is None:
        return {"status": "error", "message": "Feedback loop not available"}

    # For demo we just accept and log
    _feedback_loop.log_resolution(
        # We don't have the full Event here in this minimal endpoint; create a stub
        type("StubEvent", (), {
            "id": payload.get("event_id"),
            "platform": type("p", (), {"value": "unknown"})(),
            "category": "unknown",
            "implied_prob": payload.get("predicted"),
        })(),
        actual_outcome=float(payload.get("actual", 0.5)),
        predicted_prob=payload.get("predicted"),
    )
    return {"status": "logged"}
