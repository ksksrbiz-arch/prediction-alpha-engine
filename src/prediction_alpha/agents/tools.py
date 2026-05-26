"""Pluggable tool system for the hardened agentic legwork layer.

Sovereign-first design:
- Tools run locally or via user-controlled providers.
- Web/news search is **disabled by default** (no surprise egress or costs).
- A rich `KnowledgeTool` provides high-signal, curated context for the wealth
  tracks (housing macro, ag/policy, PM, econ) without any external calls.
- Easy to extend: subclass `BaseTool` and register it. Production users can
  plug in Tavily, SearxNG (self-hosted), NewsAPI, etc. via environment or
  a small adapter.

Tool calling in the agent loop uses a simple, robust JSON protocol that works
well with small local models (Ollama) as well as larger ones.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from prediction_alpha.utils.logging import get_logger

_log = get_logger("agents.tools")


@dataclass
class ToolResult:
    """Standardized result from any tool execution."""

    content: str
    source: str
    confidence: float = 0.7  # 0..1, how trustworthy this snippet is
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


class BaseTool(ABC):
    """Base class for all agent tools.

    Implement `name`, `description` (used in system prompt), and `run`.
    Tools must be async-safe and never raise — return a ToolResult with
    low confidence on failure.
    """

    name: str = "base_tool"
    description: str = "A generic tool. Override in subclass."

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        ...

    def to_prompt_description(self) -> str:
        return f"- {self.name}: {self.description}"


class KnowledgeTool(BaseTool):
    """High-signal, zero-egress curated knowledge for wealth-building categories.

    This is the primary "search" tool when external web access is disabled
    (the recommended sovereign default for most runs).
    """

    name = "knowledge_base"
    description = (
        "Returns curated, high-signal context about macro, policy, ag, housing, "
        "Fed, inflation, weather, and regional econ factors relevant to US "
        "wealth tracks. Always prefer this for baseline understanding."
    )

    # Curated knowledge snippets (expand over time; versioned by category)
    _KNOWLEDGE: dict[str, list[str]] = {
        "econ": [
            "Fed policy shifts have strong lagged effects on housing affordability and mortgage rates (typically 3-9 months).",
            "CPI prints and PCE are the highest-impact scheduled catalysts for rate-sensitive contracts.",
            "Labor market data (NFP, JOLTS) often moves probability mass more than the Fed's own dot plot.",
        ],
        "policy": [
            "Tariffs and trade policy have direct P&L impact on ag exporters and domestic processors.",
            "Water rights, permitting reform, and infrastructure bills are slow but high-conviction multi-year drivers for Western ag and housing.",
            "Oregon-specific legislation (energy, land use) can create localized edges vs national Kalshi contracts.",
        ],
        "weather": [
            "Drought and heat anomalies in the Pacific Northwest and California have measurable effects on almond, berry, and dairy production.",
            "Hurricane landfall probability in the Gulf affects both energy and certain crop insurance proxies.",
            "Long-range forecasts (CFS, ECMWF) are noisy but useful for 14-45 day horizons when combined with soil moisture data.",
        ],
        "default": [
            "High-liquidity macro contracts tend to be more efficient than niche or low-volume names.",
            "Resolution risk (ambiguous wording) is a frequently under-priced source of edge or loss.",
        ],
    }

    async def run(self, *, category: str = "default", query: str | None = None, **kwargs: Any) -> ToolResult:
        start = time.perf_counter()
        cats = [category.lower(), "default"]
        snippets: list[str] = []
        for c in cats:
            snippets.extend(self._KNOWLEDGE.get(c, []))
        if not snippets:
            snippets = self._KNOWLEDGE["default"]

        # Simple relevance filter if query provided
        if query:
            q = query.lower()
            snippets = [s for s in snippets if any(w in s.lower() for w in q.split()[:4])][:4] or snippets[:3]

        content = " | ".join(snippets[:5])
        latency = (time.perf_counter() - start) * 1000

        return ToolResult(
            content=content,
            source="curated_knowledge_v2",
            confidence=0.85,
            metadata={"category": category, "matched": len(snippets)},
            latency_ms=latency,
        )


class SimulatedSearchTool(BaseTool):
    """Simulated but realistic web/news search results.

    Used when external search is disabled or as a safe fallback. The results
    are crafted to be useful for demoing the full agent pipeline and reflect
    real-world macro/policy dynamics without leaking or depending on live data.

    In production, users replace this with a real provider (Tavily, self-hosted
    Searx, etc.) by subclassing and registering under the same name.
    """

    name = "web_search"
    description = (
        "Performs a web/news search for recent developments, scheduled releases, "
        "expert commentary, or correlated events. Returns 2-4 high-relevance snippets. "
        "When AGENT_ENABLE_WEB_SEARCH=false (default), this returns high-quality simulated results."
    )

    async def run(self, *, query: str, category: str = "default", **kwargs: Any) -> ToolResult:
        start = time.perf_counter()
        q = (query or "").lower()

        # Realistic simulated results tailored to common wealth-track queries
        if "fed" in q or "rate" in q or "inflation" in q:
            snippets = [
                "Recent FOMC minutes showed increased focus on shelter inflation persistence; markets pricing ~65% chance of cut by Sept.",
                "Core PCE came in 0.1% hotter than expected — analysts at major banks revised H2 cut expectations downward.",
                "Housing starts data released this morning showed continued weakness in multifamily, supportive of rate-sensitive hedges.",
            ]
        elif "tariff" in q or "trade" in q or "ag" in q:
            snippets = [
                "New proposed China tariffs on ag machinery could raise input costs for large Western row-crop operations.",
                "USDA WASDE report next week expected to adjust corn and soybean ending stocks; weather in Brazil remains the dominant variable.",
                "Oregon legislative session considering new port infrastructure funding that may benefit export-oriented producers.",
            ]
        elif "weather" in q or "drought" in q:
            snippets = [
                "NOAA seasonal outlook continues to favor above-normal temperatures across the PNW through September.",
                "Current USDM maps show expanding moderate drought in parts of the Columbia Basin — watch for hay and forage price spikes.",
                "European medium-range models are trending drier for the Central Valley in the 15-30 day window.",
            ]
        else:
            snippets = [
                f"Recent coverage on '{query}' highlights elevated uncertainty around resolution language and liquidity in the final week.",
                "Analyst notes suggest retail flow has been one-sided; smart money appears to be fading the move.",
                "Correlated contracts on other platforms show similar mispricing on the same macro driver.",
            ]

        content = " || ".join(snippets)
        latency = (time.perf_counter() - start) * 1000

        return ToolResult(
            content=content,
            source="simulated_search_v2 (replace with real provider for live data)",
            confidence=0.65,  # lower than curated knowledge — clearly labeled
            metadata={"query": query[:80], "simulated": True},
            latency_ms=latency,
        )


class ToolRegistry:
    """Central registry for agent tools. Supports enable/disable and easy extension."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(KnowledgeTool())
        self.register(SimulatedSearchTool(), enabled=False)  # disabled until user opts in

    def register(self, tool: BaseTool, *, enabled: bool | None = None) -> None:
        if enabled is not None:
            tool.enabled = enabled
        self._tools[tool.name] = tool
        _log.debug("tool_registered", name=tool.name, enabled=tool.enabled)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_enabled(self) -> list[BaseTool]:
        return [t for t in self._tools.values() if t.enabled]

    def describe_for_prompt(self) -> str:
        enabled = self.list_enabled()
        if not enabled:
            return "No tools currently enabled."
        return "\n".join(t.to_prompt_description() for t in enabled)

    async def call(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(name)
        if not tool or not tool.enabled:
            return ToolResult(
                content=f"Tool '{name}' is not available or disabled.",
                source="registry",
                confidence=0.1,
            )
        try:
            t0 = time.perf_counter()
            result = await tool.run(**kwargs)
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result
        except Exception as exc:  # noqa: BLE001
            _log.warning("tool_execution_failed", tool=name, error=str(exc)[:200])
            return ToolResult(
                content=f"Tool {name} failed: {exc}",
                source=f"{name}_error",
                confidence=0.0,
            )


# Global default registry (can be overridden per orchestrator instance)
default_registry = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    return default_registry
