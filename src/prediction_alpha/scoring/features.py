"""Feature engineering for low-noise opportunity detection."""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean
from typing import Any

from prediction_alpha.models import Event


def compute_features(event: Event) -> dict[str, Any]:
    """Compute Phase 1 derived features from a normalized event."""

    implied_prob = event.implied_prob
    if implied_prob is None and event.yes_price is not None:
        implied_prob = event.yes_price
    days_to_resolution = event.days_to_resolution
    if days_to_resolution is None and event.resolution_date:
        days_to_resolution = max(
            (event.resolution_date - datetime.now(UTC)).total_seconds() / 86_400,
            0.0,
        )
    volume_trend = _volume_trend(event.enriched_features.get("candles") or event.raw_metadata.get("candles"))
    return {
        "implied_prob": implied_prob,
        "liquidity_score": event.liquidity_score,
        "days_to_resolution": days_to_resolution,
        "volume_24h": event.volume_24h,
        "open_interest": event.open_interest,
        "volume_trend": volume_trend,
        "category": event.category,
        "status": event.status.value,
    }


def _volume_trend(candles: Any) -> float | None:
    """Return recent/older volume ratio minus one when candle history exists."""

    if not isinstance(candles, list) or len(candles) < 4:
        return None
    volumes: list[float] = []
    for candle in candles:
        if isinstance(candle, dict):
            value = candle.get("volume") or candle.get("volume_contracts")
            try:
                volumes.append(float(value or 0))
            except (TypeError, ValueError):
                continue
    if len(volumes) < 4:
        return None
    half = len(volumes) // 2
    older = mean(volumes[:half]) or 1.0
    recent = mean(volumes[half:])
    return (recent / older) - 1.0
