"""Research + Planning agents for high-value prediction market opportunities.

Sovereign design:
- Prefers local Ollama (no data leaves your machine).
- Falls back to deterministic high-quality stub when LLM is unavailable or disabled.
- Never blocks the ingestion pipeline; always runs via background TaskManager or
  explicit await in batch jobs.
- Output is structured (AgentResearchBrief) + human summary for notifications/Brain.

Trigger rule (enforced by caller): only on scores with composite >=
settings.agent_min_composite_to_research AND passed_filter (or explicit force).

The prompt is deliberately wealth-building oriented (macro hedges for housing,
policy for ag/drone, econ for PM portfolio) per the Cathedral vision.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from prediction_alpha.config import Settings, get_settings
from prediction_alpha.models import AgentResearchBrief, Event, OpportunityScore
from prediction_alpha.utils.logging import get_logger

_log = get_logger("agents.legwork")


# ---------------------------------------------------------------------------
# Prompt engineering (wealth-first, edge-realistic, low hallucination)
# ---------------------------------------------------------------------------


RESEARCH_PROMPT_TEMPLATE = """You are a sovereign, profit-first research analyst for prediction markets.
Your only goal is to help the user detect REAL edge on Kalshi event contracts that can compound wealth.

Focus on the user's actual portfolio context:
- Housing / real-estate macro exposure (Fed, inflation, rates, regional econ)
- Oregon agriculture + drone operations (weather, water policy, tariffs, crop yields)
- Cleveland-area property management (local labor, permitting, energy)
- General macro/policy events that affect capital allocation

Event under review:
Title: {title}
Category: {category}
Market-implied probability (YES): {implied_prob}
Your current edge estimate (model - market): {edge}
Liquidity score (0-1): {liquidity}
Days to resolution: {days}
Volume 24h: {volume}
Open interest: {oi}

Raw signals / rationale so far:
{rationale}

1. Write a tight, evidence-based THESIS (2-4 sentences) explaining why the market may be mispricing.
2. Write a disciplined COUNTER-THESIS (2-4 sentences) — the strongest honest case that the market is correct or your edge is an illusion.
3. List the 3-5 most important DRIVERS (catalysts or data releases) that will move this contract.
4. List the 3-5 biggest RISKS or ways this could go wrong (including liquidity traps, correlated events, black swans).
5. Give a one-sentence RECOMMENDED ACTION for a conservative wealth-builder (paper trade size, watchlist only, skip, etc.).

Respond ONLY with valid JSON in this exact schema (no markdown fences, no extra text):
{{
  "thesis": "...",
  "counter_thesis": "...",
  "key_drivers": ["...", "..."],
  "risks": ["...", "..."],
  "recommended_sizing": "...",
  "confidence_in_edge": 0.0
}}
Be brutally realistic. If there is no credible edge, say so in the counter-thesis.
"""


def _build_research_prompt(event: Event, score: OpportunityScore) -> str:
    return RESEARCH_PROMPT_TEMPLATE.format(
        title=event.title,
        category=event.category,
        implied_prob=event.implied_prob or 0.5,
        edge=round(score.edge_score, 4),
        liquidity=round(event.liquidity_score, 3),
        days=round(event.days_to_resolution or 999, 1),
        volume=int(event.volume_24h),
        oi=int(event.open_interest),
        rationale="; ".join(score.rationale[:6]),
    )


# ---------------------------------------------------------------------------
# Ollama client (minimal, resilient, no langchain dependency required)
# ---------------------------------------------------------------------------


async def _call_ollama(
    prompt: str,
    settings: Settings,
    *,
    system: str | None = None,
) -> str:
    """Call local Ollama /api/generate. Returns raw text or raises on hard failure."""

    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # favor precision over creativity for research
            "num_predict": 800,
            "top_p": 0.9,
        },
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=settings.agent_request_timeout_seconds) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", "")).strip()


def _extract_json(text: str) -> dict[str, Any] | None:
    """Robustly pull a JSON object from LLM output (handles ```json fences and noise)."""

    # Strip common code fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last-ditch: try to repair trailing commas etc. (very defensive)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Public agent API
# ---------------------------------------------------------------------------


def stub_research_brief(event: Event, score: OpportunityScore) -> AgentResearchBrief:
    """High-signal deterministic stub used when LLM is disabled or unreachable.

    This is intentionally opinionated and conservative — never hallucinates facts.
    """

    edge = score.edge_score
    category = event.category
    thesis = (
        f"Model sees modest edge of {edge:+.2%} vs market. "
        f"Category '{category}' receives elevated portfolio weight in current config. "
        "Liquidity and horizon filters already passed."
    )
    counter = (
        "Edge may be illusory due to unmodeled information asymmetry or rapid "
        "news flow. Many Kalshi markets are efficient on high-volume econ events. "
        "Paper trade only until resolution feedback improves the model."
    )
    drivers = ["Upcoming data releases", "Policy announcements", "Liquidity shifts"]
    risks = ["Low liquidity exit", "Correlated macro shock", "Resolution ambiguity"]
    sizing = "Paper position only (1-2 contracts). Add to watchlist for resolution tracking."

    if category in {"econ", "policy"}:
        thesis = f"Macro/policy event in '{category}' — aligns with wealth-track monitoring. " + thesis
        risks.append("Fed or legislative surprise")

    return AgentResearchBrief(
        event_id=event.id,
        thesis=thesis,
        counter_thesis=counter,
        key_drivers=drivers,
        risks=risks,
        recommended_sizing=sizing,
        confidence_in_edge=max(0.25, min(0.65, 0.4 + abs(edge) * 2)),
        sources=["heuristic-stub-v1"],
        generated_at=datetime.now(UTC),
        raw_agent_output=None,
    )


async def run_legwork_for_score(
    event: Event,
    score: OpportunityScore,
    settings: Settings | None = None,
) -> AgentResearchBrief:
    """Run the full research + planning legwork for one high-potential opportunity.

    This is the main entry point. It is safe to call from any context (sync or async).
    Returns a structured brief (real LLM or stub).
    """

    if settings is None:
        settings = get_settings()

    if not settings.agent_enabled or settings.llm_provider == "stub":
        brief = stub_research_brief(event, score)
        _log.info("agent_stub_used", event_id=event.id, reason="disabled_or_stub_provider")
        return brief

    prompt = _build_research_prompt(event, score)

    try:
        raw = await _call_ollama(prompt, settings)
        parsed = _extract_json(raw) or {}
        brief = AgentResearchBrief(
            event_id=event.id,
            thesis=str(parsed.get("thesis", "LLM returned no thesis."))[:1200],
            counter_thesis=str(parsed.get("counter_thesis", "No counter provided."))[:1200],
            key_drivers=[str(d) for d in parsed.get("key_drivers", [])][:6],
            risks=[str(r) for r in parsed.get("risks", [])][:6],
            recommended_sizing=str(parsed.get("recommended_sizing", "Research further / paper only."))[:200],
            confidence_in_edge=float(parsed.get("confidence_in_edge", 0.5)),
            sources=["ollama:" + settings.ollama_model],
            generated_at=datetime.now(UTC),
            raw_agent_output=raw[:4000],  # audit trail, truncated
        )
        _log.info(
            "agent_research_complete",
            event_id=event.id,
            model=settings.ollama_model,
            confidence=round(brief.confidence_in_edge, 2),
        )
        return brief

    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "agent_llm_failed_fallback_to_stub",
            event_id=event.id,
            error=str(exc)[:200],
            provider=settings.llm_provider,
        )
        return stub_research_brief(event, score)


class AgentOrchestrator:
    """Lightweight coordinator for future expansion (multi-step graphs, critique loops).

    Current implementation is sequential research → brief. When you are ready for
    LangGraph, replace the body of run() with a compiled graph while keeping this
    class signature stable for callers.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._log = get_logger("agents.orchestrator")

    async def run_for_score(
        self, event: Event, score: OpportunityScore
    ) -> AgentResearchBrief:
        """Entry point used by the pipeline / background workers."""
        return await run_legwork_for_score(event, score, self.settings)

    async def enrich_score_with_plan(
        self, event: Event, score: OpportunityScore
    ) -> OpportunityScore:
        """Run agents (if threshold met) and return an updated OpportunityScore
        with agent_plan_summary + research_brief populated.
        """
        if not score.passed_filter:
            return score

        min_score = self.settings.agent_min_composite_to_research
        if score.composite_score < min_score:
            score.agent_plan_summary = (
                f"Below agent research threshold ({min_score:.2f}). "
                "Passed hard filters but not high-conviction enough for auto-legwork."
            )
            return score

        brief = await self.run_for_score(event, score)

        # Mutate a copy (Pydantic models are immutable by default in v2 unless we use model_copy)
        updated = score.model_copy(update={
            "research_brief": brief.model_dump(mode="json"),
            "agent_plan_summary": (
                f"Thesis: {brief.thesis[:180]}... | "
                f"Sizing: {brief.recommended_sizing} | "
                f"Confidence: {brief.confidence_in_edge:.0%}"
            ),
        })
        return updated
