"""Runtime configuration for the sovereign Prediction Alpha Engine."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings.

    Defaults favor read-only paper testing: public Kalshi endpoints, local Postgres,
    conservative filters, and no mandatory external services.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    kalshi_api_key: str | None = None
    kalshi_api_secret: str | None = None
    kalshi_rest_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    kalshi_request_timeout_seconds: float = 20.0
    kalshi_requests_per_second: float = 8.0
    kalshi_ws_ping_interval_seconds: float = 20.0
    kalshi_ws_reconnect_initial_seconds: float = 1.0
    kalshi_ws_reconnect_max_seconds: float = 60.0

    database_url: str = "postgresql://prediction_alpha:prediction_alpha@localhost:5432/prediction_alpha"

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process settings."""

    return Settings()
