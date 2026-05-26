"""Agentic legwork layer (v2 hardened) — multi-step research, tool use, critic, memory, observability.

All agents remain sovereign-first: local Ollama primary, web search off by default,
graceful deterministic fallback.
"""

from prediction_alpha.agents.config import AgentConfig
from prediction_alpha.agents.legwork import (
    AgentMetrics,
    AgentOrchestrator,
    AgentRunMetrics,
    get_agent_metrics,
    run_legwork_for_score,
    stub_research_brief,
)
from prediction_alpha.agents.memory import ShortTermAgentMemory, get_default_memory
from prediction_alpha.agents.tools import (
    BaseTool,
    KnowledgeTool,
    SimulatedSearchTool,
    ToolRegistry,
    ToolResult,
    get_default_registry,
)

__all__ = [
    "AgentOrchestrator",
    "run_legwork_for_score",
    "stub_research_brief",
    "AgentConfig",
    "AgentMetrics",
    "AgentRunMetrics",
    "get_agent_metrics",
    "ShortTermAgentMemory",
    "get_default_memory",
    "BaseTool",
    "KnowledgeTool",
    "SimulatedSearchTool",
    "ToolRegistry",
    "ToolResult",
    "get_default_registry",
]
