"""Runtime configuration for the sovereign Prediction Alpha Engine.

Productization note: all scoring/filter thresholds live in ``ScoringConfig`` and can
be loaded from environment variables *or* a YAML file (``scoring_config_path``).  This
design supports per-user profile overrides later — each profile would carry its own
``ScoringConfig`` instance while sharing the global ``Settings`` for infra.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Scoring / filter configuration (YAML-friendly, per-user extensible)
# ---------------------------------------------------------------------------


class CompositeWeights(BaseModel):
    """Weights used to combine sub-scores into the composite opportunity score.

    Productization note: this config/filter will support per-user profiles later.
    """

    edge: float = 3.0
    liquidity_adjusted_ev: float = 4.0
    confidence: float = 0.25
    portfolio_fit: float = 0.20


class CategoryWeights(BaseModel):
    """Portfolio-fit weights by event category.

    Productization note: when multi-user support lands, each user profile will
    carry its own ``CategoryWeights`` reflecting personal portfolio tracks.
    """

    econ: float = 0.75
    policy: float = 0.75
    weather: float = 0.75
    sports: float = 0.35
    default: float = 0.50


class ScoringConfig(BaseModel):
    """All tunable scoring and filter knobs in one serializable block.

    Can be loaded from a YAML file via ``ScoringConfig.from_yaml(path)`` or
    constructed from environment defaults via the parent ``Settings`` object.

    Productization note: this config/filter will support per-user profiles later.
    Every threshold, weight, and category list here becomes a per-profile override
    in Phase 4.
    """

    min_liquidity_score: float = 0.20
    min_volume_24h: float = 100.0
    max_days_to_resolution: int = 60
    min_composite_score: float = 0.55
    min_edge: float = 0.02
    allowed_categories: list[str] = Field(default_factory=list)
    composite_weights: CompositeWeights = Field(default_factory=CompositeWeights)
    category_weights: CategoryWeights = Field(default_factory=CategoryWeights)

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> ScoringConfig:
        """Load scoring config from a YAML file.

        Falls back to defaults for any missing keys, so partial YAML overrides
        are supported.
        """

        import yaml  # lazy import — yaml is only needed when a config file is used

        file_path = pathlib.Path(path)
        if not file_path.is_file():
            return cls()
        with open(file_path) as fh:
            data: Any = yaml.safe_load(fh)
        if not isinstance(data, dict):
            return cls()
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Global settings (environment + .env)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Environment-backed settings.

    Defaults favor read-only paper testing: public Kalshi endpoints, local Postgres,
    conservative filters, and no mandatory external services.

    Productization note: ``scoring_config_path`` lets operators drop a YAML file
    to override scoring defaults without touching environment variables.  In a
    multi-user deployment each user profile would reference its own YAML.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # --- API surface ---
    # Productization note: 0.0.0.0 is intentional for container / Docker / Render
    # deployments.  Production should front this with a reverse proxy (nginx, Caddy)
    # or restrict access via firewall rules.
    api_host: str = "0.0.0.0"  # noqa: S104
    api_port: int = 8000

    # --- Kalshi ---
    kalshi_api_key: str | None = None
    kalshi_api_secret: str | None = None
    kalshi_rest_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    kalshi_request_timeout_seconds: float = 20.0
    kalshi_requests_per_second: float = 8.0
    kalshi_ws_ping_interval_seconds: float = 20.0
    kalshi_ws_reconnect_initial_seconds: float = 1.0
    kalshi_ws_reconnect_max_seconds: float = 60.0

    # --- Database ---
    database_url: str = (
        "postgresql://prediction_alpha:prediction_alpha@localhost:5432/prediction_alpha"
    )

    # --- Agents / LLM (sovereign-first: local Ollama preferred) ---
    # Productization note: per-profile agent config (model choice, temperature,
    # research depth) will be added in Phase 4. For now global with .env override.
    llm_provider: Literal["ollama", "stub"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"  # or phi3, mistral, qwen2.5:3b — small & fast for MVP
    agent_request_timeout_seconds: float = 45.0
    agent_min_composite_to_research: float = 0.60  # only spawn expensive research on promising candidates
    agent_enabled: bool = True

    # --- Notifications (selective, top-tier only) ---
    # Productization note: in a real deployment each user profile gets its own
    # notification preferences, digest frequency, and channels (email, Telegram,
    # UnifyOne push). Start simple: console + SMTP stub.
    notifications_enabled: bool = True
    notify_min_composite: float = 0.68  # very selective — protects attention
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "prediction-alpha@yourdomain.com"
    notify_email_to: str | None = None  # comma-separated list

    # --- Scoring (env-level overrides; YAML takes precedence if file exists) ---
    scoring_config_path: str | None = None
    min_liquidity_score: float = 0.20
    min_volume_24h: float = 100.0
    max_days_to_resolution: int = 60
    min_composite_score: float = 0.55
    allowed_categories: list[str] = Field(default_factory=list)

    @field_validator("kalshi_rest_base_url")
    @classmethod
    def trim_rest_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        # asyncpg expects postgresql:// rather than SQLAlchemy's postgresql+asyncpg:// form.
        return value.replace("postgresql+asyncpg://", "postgresql://", 1)

    def build_scoring_config(self) -> ScoringConfig:
        """Build a ``ScoringConfig`` — YAML file wins if ``scoring_config_path`` is set."""

        if self.scoring_config_path:
            return ScoringConfig.from_yaml(self.scoring_config_path)
        return ScoringConfig(
            min_liquidity_score=self.min_liquidity_score,
            min_volume_24h=self.min_volume_24h,
            max_days_to_resolution=self.max_days_to_resolution,
            min_composite_score=self.min_composite_score,
            allowed_categories=self.allowed_categories,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process settings."""

    return Settings()
