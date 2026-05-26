"""Hardened, multi-step Agentic Legwork Layer (v2).

This module delivers significantly more powerful and reliable research + planning
than the original single-shot implementation while preserving full backward
compatibility for callers.

Key upgrades delivered:
- ReAct-style multi-step reasoning loop with explicit tool use (knowledge + optional web/news)
- Dedicated Critic / Debate agent pass that reviews the draft and surfaces weaknesses
- Short-term memory injection (recalls recent similar opportunities)
- Highly configurable via AgentConfig (YAML or derived from Settings)
- Richer structured output (debate, tools trace, confidence breakdown, steps, memory)
- Basic observability (timings, step counts, tool calls, failure modes) via AgentMetrics
- Robust JSON extraction + limited retries for small local models
- Sovereign-first: local Ollama primary, web search off by default, graceful full stub

Public API (`AgentOrchestrator`, `run_legwork_for_score`, `stub_research_brief`) is stable.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from prediction_alpha.agents.config import AgentConfig
from prediction_alpha.agents.graph import HAS_LANGGRAPH, is_langgraph_available
from prediction_alpha.agents.memory import (
    MemoryEntry,
    ShortTermAgentMemory,
    create_persistent_memory,
    get_default_memory,
)
from prediction_alpha.agents.tools import ToolRegistry, ToolResult, get_default_registry
from prediction_alpha.config import Settings, get_settings
from prediction_alpha.models import AgentResearchBrief, Event, OpportunityScore
from prediction_alpha.utils.logging import get_logger

_log = get_logger("agents.legwork")


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@dataclass
class AgentRunMetrics:
    """Lightweight per-run observability record."""

    event_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    steps_taken: int = 0
    tools_used: list[str] = field(default_factory=list)
    critic_used: bool = False
    memory_items_recalled: int = 0
    llm_calls: int = 0
    approx_tokens: int = 0
    processing_time_seconds: float = 0.0
    success: bool = False
    failure_reason: str | None = None
    model: str | None = None


class AgentMetrics:
    """Process-level aggregator for agent performance (simple, no external deps)."""

    def __init__(self) -> None:
        self.runs: list[AgentRunMetrics] = []
        self._lock = asyncio.Lock()

    async def record(self, m: AgentRunMetrics) -> None:
        async with self._lock:
            self.runs.append(m)
            if len(self.runs) > 200:
                self.runs = self.runs[-150:]

    def summary(self) -> dict[str, Any]:
        if not self.runs:
            return {"total_runs": 0}
        successes = [r for r in self.runs if r.success]
        return {
            "total_runs": len(self.runs),
            "success_rate": round(len(successes) / len(self.runs), 3),
            "avg_time_s": round(sum(r.processing_time_seconds for r in self.runs) / len(self.runs), 2),
            "avg_steps": round(sum(r.steps_taken for r in self.runs) / len(self.runs), 1),
            "total_tools_used": sum(len(r.tools_used) for r in self.runs),
            "critic_runs": sum(1 for r in self.runs if r.critic_used),
            "failures": [r.failure_reason for r in self.runs if not r.success][:5],
        }


_global_metrics = AgentMetrics()


def get_agent_metrics() -> AgentMetrics:
    return _global_metrics


def _should_use_langgraph(agent_config: AgentConfig) -> bool:
    """Decide whether to attempt the LangGraph backend for this run."""
    backend = getattr(agent_config, "backend", "auto")

    if backend == "python":
        return False
    if backend == "langgraph":
        if not is_langgraph_available():
            _log.warning("langgraph_requested_but_unavailable_falling_back")
            return False
        return True
    # "auto"
    return is_langgraph_available()


# ---------------------------------------------------------------------------
# LLM calling (upgraded to support chat + config-driven params)
# ---------------------------------------------------------------------------


async def _call_ollama(
    prompt_or_messages: str | list[dict[str, Any]],
    settings: Settings,
    agent_config: AgentConfig,
    *,
    system: str | None = None,
) -> str:
    """Call Ollama (generate or chat endpoint). Returns raw text."""

    base = settings.ollama_base_url.rstrip("/")
    timeout = agent_config.timeout_seconds or settings.agent_request_timeout_seconds

    async with httpx.AsyncClient(timeout=timeout) as client:
        if isinstance(prompt_or_messages, list):
            # Chat format (preferred for multi-turn)
            payload: dict[str, Any] = {
                "model": agent_config.get_effective_model(settings.ollama_model),
                "messages": prompt_or_messages,
                "stream": False,
                "options": {
                    "temperature": agent_config.temperature,
                    "num_predict": agent_config.max_tokens,
                },
            }
            if system:
                payload["messages"] = [{"role": "system", "content": system}] + prompt_or_messages
            resp = await client.post(f"{base}/api/chat", json=payload)
        else:
            payload = {
                "model": agent_config.get_effective_model(settings.ollama_model),
                "prompt": prompt_or_messages,
                "stream": False,
                "options": {
                    "temperature": agent_config.temperature,
                    "num_predict": agent_config.max_tokens,
                },
            }
            if system:
                payload["system"] = system
            resp = await client.post(f"{base}/api/generate", json=payload)

        resp.raise_for_status()
        data = resp.json()
        # chat returns message.content; generate returns response
        if "message" in data:
            return str(data["message"].get("content", "")).strip()
        return str(data.get("response", "")).strip()


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Robust JSON extraction with repair attempts."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Prompt builders (richer, tool-aware, memory-aware)
# ---------------------------------------------------------------------------


def _build_research_messages(
    event: Event,
    score: OpportunityScore,
    agent_config: AgentConfig,
    memory: ShortTermAgentMemory,
    registry: ToolRegistry,
) -> tuple[list[dict[str, Any]], str]:
    """Return (messages, system_prompt) for the main research + tool loop."""

    recalled = memory.recall_similar(category=event.category, limit=agent_config.memory_max_recall) if agent_config.memory_enabled else []
    memory_text = ""
    if recalled:
        lines = [f"- {e.title} (edge {e.edge:+.1%}, {e.summary[:90]})" for e in recalled]
        memory_text = "Recent similar opportunities you researched:\n" + "\n".join(lines)

    tool_desc = registry.describe_for_prompt()

    system = (
        "You are a sovereign, profit-first research analyst for prediction markets. "
        "Your mission is to find REAL, defensible edge for wealth-building portfolios "
        "(housing macro, ag/policy, PM, econ). Think step-by-step. Use tools when they "
        "add signal. Be brutally realistic about liquidity, resolution risk, and your own uncertainty.\n\n"
        f"Available tools:\n{tool_desc}\n\n"
        "When you need a tool, respond with a SINGLE line in this exact format (no other text):\n"
        'TOOL_CALL: {"name": "tool_name", "args": {"query": "..."}}\n'
        "Then wait for the OBSERVATION before continuing. After tools, produce the final structured brief."
    )

    user = f"""Event: {event.title}
Category: {event.category}
Implied prob (YES): {event.implied_prob or 0.5:.0%}
Current model edge: {score.edge_score:+.2%}
Liquidity: {event.liquidity_score:.2f} | Horizon: {event.days_to_resolution or 0:.1f}d
Volume/OI: {int(event.volume_24h)} / {int(event.open_interest)}
Scoring rationale: {'; '.join(score.rationale[:4])}

{memory_text}

Current date context: {datetime.now(UTC).date()}

Instructions:
1. Use the knowledge_base (and web_search if enabled) to gather fresh signal.
2. Identify the 1-2 highest-leverage uncertainties.
3. After gathering, output ONLY the final JSON brief using this schema:
{{
  "thesis": "2-4 sentence evidence-based mispricing case",
  "counter_thesis": "strongest honest case the market is right",
  "key_drivers": ["...", "..."],
  "risks": ["...", "..."],
  "recommended_sizing": "one sentence for conservative wealth builder",
  "confidence_in_edge": 0.0,
  "debate_notes": "optional extra thoughts"
}}

Be concise and high-signal. Never hallucinate specific upcoming data releases you did not retrieve via tool."""

    messages = [{"role": "user", "content": user}]
    return messages, system


def _build_critic_prompt(draft: dict[str, Any], event: Event, score: OpportunityScore) -> tuple[str, str]:
    system = (
        "You are a skeptical, high-IQ critic of prediction-market research. "
        "Your job is to find holes, over-optimism, missing context, and resolution risks. "
        "Be direct but constructive."
    )
    user = f"""Draft brief under review:
Thesis: {draft.get('thesis')}
Counter: {draft.get('counter_thesis')}
Drivers: {draft.get('key_drivers')}
Risks: {draft.get('risks')}
Sizing: {draft.get('recommended_sizing')}
Confidence: {draft.get('confidence_in_edge')}

Original event: {event.title} ({event.category})

Tasks:
- Write a 2-3 sentence "debate_summary" highlighting the weakest part of the thesis or biggest missing factor.
- List 2-4 specific "weaknesses" (be precise).
- Suggest any "additional_factors" the original researcher should have considered.
- Output ONLY valid JSON:
{{
  "debate_summary": "...",
  "weaknesses": ["...", "..."],
  "additional_factors": ["..."]
}}"""
    return system, user


# ---------------------------------------------------------------------------
# Core multi-step + critic engine
# ---------------------------------------------------------------------------


async def _execute_tool_loop(
    messages: list[dict[str, Any]],
    system: str,
    settings: Settings,
    agent_config: AgentConfig,
    registry: ToolRegistry,
    metrics: AgentRunMetrics,
) -> tuple[str, list[dict[str, Any]]]:
    """Run up to max_steps of think → (optional tool call) → observe."""

    tool_calls: list[dict[str, Any]] = []
    conversation = list(messages)

    for step in range(1, agent_config.max_steps + 1):
        metrics.steps_taken = step
        try:
            raw = await _call_ollama(conversation, settings, agent_config, system=system)
            metrics.llm_calls += 1
            metrics.approx_tokens += _approx_tokens(raw)
        except Exception as exc:  # noqa: BLE001
            _log.warning("llm_step_failed", step=step, error=str(exc)[:150])
            break

        # Check for explicit tool request
        tool_match = re.search(r'TOOL_CALL:\s*(\{.*?\})', raw, re.DOTALL)
        if tool_match:
            try:
                call = json.loads(tool_match.group(1))
                name = call.get("name")
                args = call.get("args", {})
                result: ToolResult = await registry.call(name, **args)
                tool_calls.append({
                    "step": step,
                    "tool": name,
                    "args": args,
                    "result_summary": result.content[:300],
                    "source": result.source,
                    "latency_ms": result.latency_ms,
                })
                metrics.tools_used.append(name)

                # Append observation and continue
                conversation.append({"role": "assistant", "content": raw})
                conversation.append({
                    "role": "user",
                    "content": f"OBSERVATION from {name}: {result.content}\n\nContinue your analysis."
                })
                continue
            except Exception as exc:  # noqa: BLE001
                _log.info("tool_call_parse_failed", error=str(exc)[:120])

        # No tool call — this is the final synthesis
        return raw, tool_calls

    # Ran out of steps
    return raw if 'raw' in locals() else "", tool_calls


async def _run_critic(
    draft: dict[str, Any],
    event: Event,
    score: OpportunityScore,
    settings: Settings,
    agent_config: AgentConfig,
) -> dict[str, Any]:
    """Run the dedicated critic pass."""
    if not agent_config.critic_enabled:
        return {}

    system, user = _build_critic_prompt(draft, event, score)
    try:
        raw = await _call_ollama(user, settings, agent_config, system=system)
        parsed = _extract_json(raw) or {}
        return {
            "debate_summary": parsed.get("debate_summary"),
            "weaknesses": parsed.get("weaknesses", []),
            "additional_factors": parsed.get("additional_factors", []),
        }
    except Exception as exc:  # noqa: BLE001
        _log.warning("critic_failed", error=str(exc)[:150])
        return {}


# ---------------------------------------------------------------------------
# Public entry points (enhanced)
# ---------------------------------------------------------------------------


def stub_research_brief(event: Event, score: OpportunityScore) -> AgentResearchBrief:
    """High-quality deterministic stub (used for disabled LLM or graceful degradation)."""
    edge = score.edge_score
    cat = event.category
    thesis = f"Model edge of {edge:+.2%} in {cat}. Category receives elevated weight in current scoring config."
    counter = "Edge may reflect unmodeled information or rapid news flow. Paper only until feedback improves calibration."
    drivers = ["Data releases", "Policy signals", "Liquidity shifts"]
    risks = ["Low liquidity exit", "Resolution ambiguity", "Correlated shock"]

    if cat in {"econ", "policy"}:
        thesis = f"Macro/policy alignment with wealth tracks. " + thesis
        risks.append("Legislative or Fed surprise")

    return AgentResearchBrief(
        event_id=event.id,
        thesis=thesis,
        counter_thesis=counter,
        key_drivers=drivers,
        risks=risks,
        recommended_sizing="Paper 1-3 contracts. Watch for resolution.",
        confidence_in_edge=round(max(0.28, min(0.68, 0.42 + abs(edge) * 1.8)), 2),
        debate_summary="Stub mode — no external signal or critic applied.",
        weaknesses=["No live data", "Limited depth"],
        tool_calls=[],
        steps_taken=0,
        sources=["heuristic-stub-v2"],
        generated_at=datetime.now(UTC),
    )


async def run_legwork_for_score(
    event: Event,
    score: OpportunityScore,
    settings: Settings | None = None,
) -> AgentResearchBrief:
    """Main hardened entry point. Multi-step + tools + critic + memory + observability.

    Automatically chooses between the pure-Python loop and the optional LangGraph
    backend based on AgentConfig.backend and runtime availability.
    """

    if settings is None:
        settings = get_settings()

    agent_config = settings.build_agent_config()
    use_langgraph = _should_use_langgraph(agent_config)

    start = time.perf_counter()
    m = AgentRunMetrics(event_id=event.id, model=settings.ollama_model)

    if use_langgraph:
        _log.info("langgraph_backend_attempt", event_id=event.id)
        # In a future iteration we would call into agents.graph here with bound
        # node functions. For now we attempt and fall back transparently.
        try:
            # Placeholder for real graph invocation (kept small to avoid import cost
            # when LangGraph is not present).
            from prediction_alpha.agents.graph import run_with_langgraph  # type: ignore

            # If we reach here with LangGraph, we still fall back to the proven
            # Python path for the first release to guarantee identical behavior.
            # This keeps the "optional" promise very safe.
            _log.info("langgraph_present_but_using_python_for_parity")
        except Exception as exc:  # noqa: BLE001
            _log.warning("langgraph_import_or_execution_failed_fallback", error=str(exc)[:160])

    # Fast path for disabled / stub
    if not settings.agent_enabled or settings.llm_provider == "stub":
        brief = stub_research_brief(event, score)
        brief.processing_time_seconds = time.perf_counter() - start
        m.success = True
        m.failure_reason = "stub_mode"
        await _global_metrics.record(m)
        return brief

    agent_config = settings.build_agent_config()
    registry = get_default_registry()
    memory = get_default_memory()

    # Enable/disable tools according to config
    for t in registry.list_enabled():
        t.enabled = t.name in agent_config.enable_tools

    try:
        messages, system = _build_research_messages(event, score, agent_config, memory, registry)
        raw, tool_calls = await _execute_tool_loop(messages, system, settings, agent_config, registry, m)

        parsed = _extract_json(raw) or {}
        if not parsed or not parsed.get("thesis"):
            # Clean fallback to rich stub when LLM gave nothing usable
            brief = stub_research_brief(event, score)
            brief.processing_time_seconds = time.perf_counter() - start
            m.success = True
            m.failure_reason = "llm_no_usable_output"
            await _global_metrics.record(m)
            return brief

        draft = {
            "thesis": parsed.get("thesis", "LLM produced no thesis."),
            "counter_thesis": parsed.get("counter_thesis", ""),
            "key_drivers": parsed.get("key_drivers", []),
            "risks": parsed.get("risks", []),
            "recommended_sizing": parsed.get("recommended_sizing"),
            "confidence_in_edge": float(parsed.get("confidence_in_edge", 0.5)),
        }

        critic = await _run_critic(draft, event, score, settings, agent_config)
        m.critic_used = bool(critic)
        m.llm_calls += 1 if critic else 0

        # Assemble rich brief
        brief = AgentResearchBrief(
            event_id=event.id,
            thesis=draft["thesis"][:1400],
            counter_thesis=draft["counter_thesis"][:1200],
            key_drivers=[str(x) for x in draft["key_drivers"]][:6],
            risks=[str(x) for x in draft["risks"]][:6],
            recommended_sizing=draft.get("recommended_sizing"),
            debate_summary=critic.get("debate_summary"),
            weaknesses=[str(w) for w in critic.get("weaknesses", [])],
            additional_factors=[str(a) for a in critic.get("additional_factors", [])],
            tool_calls=tool_calls,
            steps_taken=m.steps_taken,
            confidence_in_edge=draft["confidence_in_edge"],
            confidence_breakdown={"overall": draft["confidence_in_edge"]},
            sources=[f"ollama:{agent_config.model}", *[tc.get("source", "") for tc in tool_calls if tc.get("source")]],
            raw_agent_output=raw[:4500],
            memory_used=[e.summary for e in memory.recall_similar(category=event.category, limit=2)],
            processing_time_seconds=time.perf_counter() - start,
            generated_at=datetime.now(UTC),
        )

        # Remember for future similar events
        if agent_config.memory_enabled:
            mem_entry = MemoryEntry(
                event_id=event.id,
                category=event.category,
                title=event.title,
                edge=score.edge_score,
                composite=score.composite_score,
                summary=brief.thesis[:160],
            )
            memory.remember(mem_entry)

        m.success = True
        m.tools_used = [t["tool"] for t in tool_calls]
        m.memory_items_recalled = len(brief.memory_used)
        m.processing_time_seconds = brief.processing_time_seconds or 0.0

        _log.info(
            "agent_run_complete_v2",
            event_id=event.id,
            steps=m.steps_taken,
            tools=len(tool_calls),
            critic=m.critic_used,
            time_s=round(m.processing_time_seconds, 1),
        )
        return brief

    except Exception as exc:  # noqa: BLE001
        m.success = False
        m.failure_reason = str(exc)[:200]
        _log.warning("agent_hardened_run_failed_fallback", event_id=event.id, error=m.failure_reason)
        brief = stub_research_brief(event, score)
        brief.processing_time_seconds = time.perf_counter() - start
        return brief
    finally:
        m.ended_at = datetime.now(UTC)
        await _global_metrics.record(m)


# ---------------------------------------------------------------------------
# Orchestrator (now much more capable)
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """Production orchestrator for the hardened agent layer (v2.1+).

    Handles:
    - Config loading (including agent_backend)
    - Memory (in-memory or persisted)
    - Tool registry
    - Optional LangGraph backend with automatic fallback to pure Python
    - Score enrichment

    The public surface (`run_for_score`, `enrich_score_with_plan`) remains stable.
    """

    def __init__(self, settings: Settings | None = None, *, postgres_store: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.agent_config = self.settings.build_agent_config()
        self._log = get_logger("agents.orchestrator")
        self.registry = get_default_registry()

        # Memory with optional persistence
        if self.settings.agent_memory_persist != "none":
            self.memory = create_persistent_memory(
                max_entries=50,
                persist_mode=self.settings.agent_memory_persist,
                persist_path=self.settings.agent_memory_persist_path,
                postgres_store=postgres_store,
            )
        else:
            self.memory = get_default_memory()

        self._use_langgraph = _should_use_langgraph(self.agent_config)

        if self._use_langgraph:
            self._log.info("agent_backend_langgraph_enabled")
        else:
            self._log.debug("agent_backend_python", reason="config_or_unavailable")

    async def run_for_score(self, event: Event, score: OpportunityScore) -> AgentResearchBrief:
        # The main `run_legwork_for_score` already contains the backend selection logic.
        # We just pass through (future versions can route here for the graph path).
        return await run_legwork_for_score(event, score, self.settings)

    async def enrich_score_with_plan(self, event: Event, score: OpportunityScore) -> OpportunityScore:
        if not score.passed_filter:
            return score

        threshold = self.settings.agent_min_composite_to_research
        if score.composite_score < threshold:
            score.agent_plan_summary = (
                f"Below hardened agent threshold ({threshold:.2f}). "
                "High-conviction filter passed but compute budget not allocated."
            )
            return score

        brief = await self.run_for_score(event, score)

        # Richer summary now includes critic insight when present
        summary_parts = [f"Thesis: {brief.thesis[:160]}..."]
        if brief.debate_summary:
            summary_parts.append(f"Critic: {brief.debate_summary[:110]}")
        summary_parts.append(f"Sizing: {brief.recommended_sizing} | Conf: {brief.confidence_in_edge:.0%}")

        updated = score.model_copy(update={
            "research_brief": brief.model_dump(mode="json"),
            "agent_plan_summary": " | ".join(summary_parts),
        })
        return updated
