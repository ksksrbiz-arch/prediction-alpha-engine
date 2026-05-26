#!/usr/bin/env python3
"""
Demo script for the hardened Agentic Legwork Layer (v2.1+).

Demonstrates:
- Multi-step research + tools + Critic/Debate agent
- Short-term memory (with optional persistence)
- Optional LangGraph backend (graceful fallback)
- Rich AgentResearchBrief v2 + AgentMetrics

Run:
    PYTHONPATH=src python scripts/demo_enhanced_agents.py

With real Ollama:
    OLLAMA_BASE_URL=http://localhost:11434 python scripts/demo_enhanced_agents.py

New in this version: try hitting the API after starting the server:
    curl http://localhost:8000/metrics/agents
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

from prediction_alpha.agents import (
    AgentOrchestrator,
    get_agent_metrics,
    get_default_memory,
)
from prediction_alpha.config import Settings
from prediction_alpha.models import Event, EventStatus, OpportunityScore, Platform, RecommendedAction


def make_high_value_macro_event() -> tuple[Event, OpportunityScore]:
    """Create a realistic high-conviction macro opportunity (Fed / housing relevant)."""
    now = datetime.now(UTC)
    event = Event(
        id="kalshi-demo-fed-cut-2026-07",
        platform=Platform.KALSHI,
        external_id="FED-26JUL-CUT",
        title="Will the Federal Reserve cut the federal funds rate by July 2026?",
        category="econ",
        yes_price=0.41,
        no_price=0.59,
        implied_prob=0.41,
        volume_24h=18500.0,
        open_interest=27500.0,
        liquidity_score=0.78,
        resolution_date=now + timedelta(days=38),
        status=EventStatus.OPEN,
        raw_metadata={"source": "demo"},
        enriched_features={},
    )

    score = OpportunityScore(
        event_id=event.id,
        edge_score=0.095,
        liquidity_adjusted_ev=0.074,
        confidence=0.81,
        portfolio_fit=0.78,
        composite_score=0.71,
        recommended_action=RecommendedAction.PAPER_YES,
        passed_filter=True,
        rationale=[
            "Housing macro alignment (mortgage rates)",
            "Strong liquidity + open interest",
            "Model sees 9.5pp edge vs market",
            "Category weight boost applied",
        ],
        features={
            "implied_prob": 0.41,
            "volume_trend": 0.22,
            "days_to_resolution": 38.0,
        },
    )
    return event, score


async def main() -> None:
    print("=" * 72)
    print("PREDICTION ALPHA ENGINE — HARDENED AGENTIC LEGWORK DEMO (v2)")
    print("=" * 72)

    settings = Settings(
        environment="test",
        agent_enabled=True,
        llm_provider="ollama",  # will gracefully fall back if unreachable
        agent_min_composite_to_research=0.55,
        agent_critic_enabled=True,
        agent_memory_enabled=True,
        agent_enable_web_search=False,  # sovereign default
    )

    event, score = make_high_value_macro_event()

    print("\n[INPUT] High-value scored opportunity")
    print(f"  Title: {event.title}")
    print(f"  Category: {event.category} | Implied: {event.implied_prob:.0%}")
    print(f"  Edge: {score.edge_score:+.2%} | Composite: {score.composite_score:.2%}")
    print(f"  Liquidity: {event.liquidity_score:.2f} | Horizon: {event.days_to_resolution:.0f}d")
    print(f"  Rationale: {score.rationale[0]}")

    orchestrator = AgentOrchestrator(settings)

    print("\n[RUN] Executing hardened agent pipeline (tools + critic + memory)...")
    start = asyncio.get_event_loop().time()
    enriched = await orchestrator.enrich_score_with_plan(event, score)
    elapsed = asyncio.get_event_loop().time() - start

    brief = enriched.research_brief or {}
    print(f"\n[RESULT] Agent run completed in {elapsed:.1f}s")

    print("\n" + "-" * 72)
    print("STRUCTURED AGENT RESEARCH BRIEF (v2)")
    print("-" * 72)
    print(json.dumps(brief, indent=2, default=str))

    print("\n" + "-" * 72)
    print("HUMAN-READABLE PLAN SUMMARY (for notifications / Brain)")
    print("-" * 72)
    print(enriched.agent_plan_summary or "(none)")

    # Memory state
    mem = get_default_memory()
    print(f"\n[MEMORY] Short-term memory now holds {len(mem)} episode(s)")
    if len(mem) > 0:
        recent = mem.get_recent(1)[0]
        print(f"  Last remembered: {recent.title[:60]}... (edge {recent.edge:+.1%})")

    # Observability
    metrics = get_agent_metrics().summary()
    print("\n[OBSERVABILITY] AgentMetrics summary")
    print(json.dumps(metrics, indent=2))

    print("\n" + "=" * 72)
    print("Demo complete. Try with real Ollama running for multi-step + tool behavior.")
    print("  ollama serve & ollama pull llama3.2")
    print("  Then re-run this script.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
