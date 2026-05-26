"""Configuration for True Neutral Brain v2 ingestion.

Follows the exact same patterns as ScoringConfig and AgentConfig for consistency:
- YAML overridable via brain_config_path
- Environment variable fallbacks
- Per-user / per-profile ready in the future
- Sovereign and explicit
"""

from __future__ import annotations

import pathlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from prediction_alpha.utils.logging import get_logger

_log = get_logger("brain.config")


class WealthTrackMapping(BaseModel):
    """Maps prediction market categories to the user's 24-month wealth tracks."""

    housing: list[str] = Field(default_factory=lambda: ["econ", "policy", "weather"])
    ag_drone: list[str] = Field(default_factory=lambda: ["weather", "policy", "econ"])
    property_management: list[str] = Field(default_factory=lambda: ["econ", "policy"])
    general_macro: list[str] = Field(default_factory=lambda: ["econ", "policy"])


class BrainIngestionConfig(BaseModel):
    """Controls what gets written into True Neutral Brain v2 and how."""

    # Ingestion gates (very important — protect the Brain from noise)
    enabled: bool = True
    min_composite_score: float = 0.62
    min_edge: float = 0.03
    allowed_categories: list[str] = Field(default_factory=list)  # empty = all

    # Wealth track relevance (used for tagging and retrieval)
    wealth_track_mapping: WealthTrackMapping = Field(default_factory=WealthTrackMapping)
    always_tag_tracks: list[str] = Field(default_factory=list)  # e.g. ["housing"]

    # Embedding settings (pgvector)
    embedding_enabled: bool = True
    embedding_dimension: int = 768  # common for many local models (e.g. all-MiniLM-L6-v2)
    # Pluggable: user can inject a real embedding function at runtime
    # "stub", "local_sentence_transformers", or "remote"
    embedding_provider: Literal["stub", "local", "remote"] = "stub"

    # Deduplication & freshness
    update_existing: bool = True  # upsert on event_id

    # Observability
    log_ingests: bool = True

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path | None) -> BrainIngestionConfig:
        """Load from YAML with graceful fallback (same pattern as other configs)."""
        import yaml  # lazy import

        if not path:
            return cls()

        file_path = pathlib.Path(path)
        if not file_path.is_file():
            _log.warning("brain_config_not_found", path=str(path))
            return cls()

        try:
            with open(file_path) as fh:
                data: Any = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                return cls()
            return cls.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            _log.error("brain_config_parse_failed", path=str(path), error=str(exc)[:200])
            return cls()


# Module-level helper (used by Settings)
def build_brain_config(settings: Any) -> BrainIngestionConfig:
    """Build BrainIngestionConfig honoring YAML override then env defaults."""
    if getattr(settings, "brain_config_path", None):
        return BrainIngestionConfig.from_yaml(settings.brain_config_path)

    return BrainIngestionConfig(
        enabled=getattr(settings, "brain_ingest_enabled", True),
        min_composite_score=getattr(settings, "brain_ingest_min_composite", 0.62),
        min_edge=getattr(settings, "brain_ingest_min_edge", 0.03),
        allowed_categories=getattr(settings, "brain_ingest_allowed_categories", []),
        embedding_enabled=getattr(settings, "brain_embedding_enabled", True),
        embedding_dimension=getattr(settings, "brain_embedding_dimension", 768),
        embedding_provider=getattr(settings, "brain_embedding_provider", "stub"),
    )
