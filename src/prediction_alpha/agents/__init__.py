"""Agentic legwork layer — research, structured briefs, and execution planning.

This module is only invoked on high-conviction opportunities (after strict
filtering). All agents are sovereign-first: local Ollama by default, zero
external API keys required for core research.
"""

from prediction_alpha.agents.legwork import (
    AgentOrchestrator,
    run_legwork_for_score,
    stub_research_brief,
)

__all__ = [
    "AgentOrchestrator",
    "run_legwork_for_score",
    "stub_research_brief",
]
