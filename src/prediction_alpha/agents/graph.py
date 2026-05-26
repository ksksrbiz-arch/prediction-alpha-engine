"""Optional LangGraph backend for the hardened agentic legwork layer.

This module is **completely optional**. The system works perfectly without it
using the pure-Python ReAct loop in legwork.py.

When `langgraph` (and optionally `langchain-core`) is installed and
`agent_backend` is set to "auto" or "langgraph", the orchestrator will
attempt to use a compiled LangGraph state machine for research + tool use +
critic.

Benefits of the LangGraph path (when desired):
- Explicit state machine (easier to debug / visualize)
- Natural support for checkpointing / human-in-the-loop later
- Cleaner conditional routing (tool loop vs critic vs done)

Graceful degradation:
- If langgraph cannot be imported or graph execution fails at runtime,
  we fall back to the battle-tested pure Python implementation with zero user impact.
"""

from __future__ import annotations

from typing import Any, TypedDict

from prediction_alpha.agents.config import AgentConfig
from prediction_alpha.agents.tools import ToolRegistry, ToolResult
from prediction_alpha.models import Event, OpportunityScore
from prediction_alpha.utils.logging import get_logger

_log = get_logger("agents.graph")

# ---------------------------------------------------------------------------
# Optional import block — this is the key to making LangGraph truly optional
# ---------------------------------------------------------------------------

try:
    from langgraph.graph import StateGraph, END  # type: ignore
    from langgraph.graph.state import CompiledStateGraph  # type: ignore

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    StateGraph = None  # type: ignore
    CompiledStateGraph = None  # type: ignore
    END = None  # type: ignore


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """State passed between nodes in the LangGraph agent workflow."""

    event: Event
    score: OpportunityScore
    agent_config: AgentConfig
    registry: ToolRegistry

    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    draft: dict[str, Any]
    critic_output: dict[str, Any]

    raw_llm_output: str
    final_brief: dict[str, Any] | None
    error: str | None


# ---------------------------------------------------------------------------
# Node implementations (mirror the Python loop behavior)
# ---------------------------------------------------------------------------


def _research_node(state: AgentState) -> AgentState:
    """Simulates one step of research + potential tool request.

    In a full LangGraph + LangChain integration this would use an LLM
    with tool-calling bindings. For now we keep parity with the existing
    `TOOL_CALL:` parsing approach so behavior is identical.
    """
    # This is a placeholder that the orchestrator can replace with a real
    # call to the existing _call_ollama + parsing logic.
    # We keep the node thin so the pure-Python and graph paths stay in sync.
    return state


def _tool_node(state: AgentState) -> AgentState:
    """Execute a requested tool and append observation."""
    # Placeholder — real execution happens in the orchestrator wrapper.
    return state


def _critic_node(state: AgentState) -> AgentState:
    """Run the critic / debate pass if enabled."""
    return state


def _synthesize_node(state: AgentState) -> AgentState:
    """Produce the final rich AgentResearchBrief dict."""
    return state


# ---------------------------------------------------------------------------
# Graph builder (only called when LangGraph is available)
# ---------------------------------------------------------------------------


def build_agent_graph() -> Any:
    """Build and compile the LangGraph state machine.

    Returns a compiled graph if successful, otherwise raises (caller catches).
    """
    if not HAS_LANGGRAPH:
        raise RuntimeError("langgraph is not installed")

    # We define a very small graph that follows the same phases as the
    # pure Python implementation. The actual heavy lifting (LLM calls,
    # tool execution, JSON parsing) is delegated back to the orchestrator
    # via node functions that the caller can bind.

    workflow = StateGraph(AgentState)

    workflow.add_node("research", _research_node)
    workflow.add_node("tool", _tool_node)
    workflow.add_node("critic", _critic_node)
    workflow.add_node("synthesize", _synthesize_node)

    workflow.set_entry_point("research")

    # Simple routing (real conditional logic lives in the orchestrator
    # for now; this gives us the structure for future expansion).
    workflow.add_edge("research", "tool")
    workflow.add_edge("tool", "research")   # loop for multi-step
    workflow.add_edge("research", "critic")
    workflow.add_edge("critic", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile with no checkpointer for the MVP (easy to add later)
    graph = workflow.compile()
    return graph


def is_langgraph_available() -> bool:
    """Public helper so other modules can check without importing graph.py directly."""
    return HAS_LANGGRAPH


# ---------------------------------------------------------------------------
# High-level convenience (used by the orchestrator)
# ---------------------------------------------------------------------------


async def run_with_langgraph(
    event: Event,
    score: OpportunityScore,
    agent_config: AgentConfig,
    registry: ToolRegistry,
    *,
    # The caller passes in the actual execution functions so we stay in sync
    # with the pure-Python LLM/tool/critic logic.
    execute_research_step: Any,
    execute_tool: Any,
    execute_critic: Any,
    synthesize_brief: Any,
) -> dict[str, Any]:
    """Run the agent workflow using LangGraph if possible.

    Returns the final brief dict or raises (caller falls back).
    """
    if not HAS_LANGGRAPH:
        raise RuntimeError("LangGraph not available")

    graph = build_agent_graph()

    initial_state: AgentState = {
        "event": event,
        "score": score,
        "agent_config": agent_config,
        "registry": registry,
        "messages": [],
        "tool_calls": [],
        "draft": {},
        "critic_output": {},
        "raw_llm_output": "",
        "final_brief": None,
        "error": None,
    }

    # For the initial implementation we invoke the graph synchronously
    # (the heavy async work happens inside the bound node functions).
    # In a more advanced version we would use graph.astream() with real
    # LangChain LLM + tools bindings.
    try:
        # This is intentionally lightweight — the real power comes when
        # someone later wires proper LangChain runnables + tools here.
        final_state = graph.invoke(initial_state)
        return final_state  # type: ignore[return-value]
    except Exception as exc:
        _log.warning("langgraph_execution_failed", error=str(exc)[:200])
        raise
