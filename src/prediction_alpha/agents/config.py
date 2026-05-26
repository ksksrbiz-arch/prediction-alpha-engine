"""Agent-specific configuration for the hardened legwork layer.

Follows the same philosophy as ScoringConfig:
- Serializable, YAML-overridable, per-profile ready.
- All prompts, model behavior, tool selection, and reasoning depth are
  controllable without code changes.
- Supports both global Settings defaults and a dedicated agent_config.yaml.
"""

from __future__ import annotations

import pathlib
from typing import Any

from typing import Literal

from pydantic import BaseModel, Field

from prediction_alpha.utils.logging import get_logger

_log = get_logger("agents.config")


class AgentPrompts(BaseModel):
    """Customizable prompt templates (advanced users can override sections)."""

    research_system: str | None = None
    research_user_template: str | None = None
    critic_system: str | None = None
    synthesis_instructions: str | None = None


class AgentConfig(BaseModel):
    """Full runtime configuration for the agentic research layer.

    Loaded via `AgentConfig.from_yaml(path)` or constructed from Settings.
    This is the single place to tune cost vs quality for your hardware and risk tolerance.
    """

    # Model & inference behavior
    model: str = "llama3.2"
    temperature: float = 0.28
    max_tokens: int = 900
    timeout_seconds: float = 55.0
    max_steps: int = 5  # ReAct / tool-use loop depth

    # Reasoning features
    critic_enabled: bool = True
    memory_enabled: bool = True
    memory_max_recall: int = 3

    # Tooling (sovereign & cost control)
    enable_tools: list[str] = Field(default_factory=lambda: ["knowledge_base"])
    # "knowledge_base" is always safe. Add "web_search" only when you have a real provider wired.
    tool_timeout_seconds: float = 20.0

    # Output quality
    require_structured_output: bool = True
    json_retry_attempts: int = 2

    # Execution backend (new in v2.1)
    # "auto"   = try LangGraph if available, else fall back to pure Python
    # "python" = always use the reliable pure-Python ReAct loop
    # "langgraph" = require LangGraph (fail hard if unavailable)
    backend: Literal["auto", "python", "langgraph"] = "auto"

    # Prompt overrides (advanced)
    prompts: AgentPrompts = Field(default_factory=AgentPrompts)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path | None) -> AgentConfig:
        """Load from YAML, falling back gracefully to defaults on any issue."""

        import yaml  # lazy

        if not path:
            return cls()

        file_path = pathlib.Path(path)
        if not file_path.is_file():
            _log.warning("agent_config_not_found", path=str(path))
            return cls()

        try:
            with open(file_path) as fh:
                data: Any = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                return cls()
            # Allow nested prompts
            return cls.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            _log.error("agent_config_parse_failed", path=str(path), error=str(exc))
            return cls()

    def get_effective_model(self, override: str | None = None) -> str:
        return override or self.model
