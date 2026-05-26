"""Kalshi normalization into the canonical Event model."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from typing import Any

from prediction_alpha.models import Event, EventStatus, Platform


def _first(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value is not None:
            return value
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _price_to_probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    if number > 1.0:
        number = number / 100.0
    return min(max(number, 0.0), 1.0)


def _normalize_status(value: Any) -> EventStatus:
    text = str(value or "").lower()
    if text in {"active", "open", "initialized"}:
        return EventStatus.OPEN
    if text in {"settled", "resolved", "finalized"}:
        return EventStatus.RESOLVED
    if text in {"closed", "expired"}:
        return EventStatus.CLOSED
    if text in {"paused", "halted"}:
        return EventStatus.PAUSED
    return EventStatus.UNKNOWN


def _infer_category(raw: dict[str, Any]) -> str:
    source = " ".join(
        str(_first(raw, "category", "category_name", "series_ticker", "event_ticker", "title") or "")
        .lower()
        .split()
    )
    keyword_map = {
        "econ": ("inflation", "fed", "rate", "gdp", "cpi", "jobs", "unemployment"),
        "policy": ("election", "senate", "house", "president", "law", "tariff"),
        "weather": ("weather", "temperature", "hurricane", "rain", "snow"),
        "sports": ("nba", "nfl", "mlb", "nhl", "soccer", "game"),
    }
    for category, keywords in keyword_map.items():
        if any(keyword in source for keyword in keywords):
            return category
    category = _first(raw, "category", "category_name")
    return str(category).lower() if category else "unknown"


def _event_id(platform: Platform, external_id: str) -> str:
    digest = hashlib.sha256(f"{platform}:{external_id}".encode("utf-8")).hexdigest()
    return f"{platform}-{digest[:24]}"


def _liquidity_score(raw: dict[str, Any], yes_price: float | None) -> float:
    volume = float(_first(raw, "volume_24h", "volume_24h_contracts", "volume") or 0)
    open_interest = float(_first(raw, "open_interest", "open_interest_count") or 0)
    bid = _price_to_probability(_first(raw, "yes_bid", "bid"))
    ask = _price_to_probability(_first(raw, "yes_ask", "ask"))
    spread_penalty = 0.0
    if bid is not None and ask is not None and ask >= bid:
        spread_penalty = min(ask - bid, 1.0)
    depth_component = min((volume + open_interest) / 10_000, 1.0)
    price_component = 0.5 if yes_price is None else 1.0 - abs(yes_price - 0.5)
    return max(min((0.70 * depth_component) + (0.30 * price_component) - spread_penalty, 1.0), 0.0)


def normalize_market(raw: dict[str, Any]) -> Event:
    """Normalize a Kalshi REST market payload.

    Kalshi fields vary across endpoints; this keeps aliases explicit and stores
    the untouched payload so later scoring/backtesting can evolve without data loss.
    """

    external_id = str(_first(raw, "ticker", "market_ticker", "id"))
    yes_price = _price_to_probability(
        _first(raw, "yes_price", "last_price", "yes_ask", "yes_bid", "price")
    )
    no_price = _price_to_probability(_first(raw, "no_price", "no_ask", "no_bid"))
    if no_price is None and yes_price is not None:
        no_price = 1.0 - yes_price
    implied_prob = yes_price if yes_price is not None else (1.0 - no_price if no_price else None)
    resolution_date = _parse_datetime(
        _first(
            raw,
            "resolution_date",
            "close_time",
            "expiration_time",
            "expected_expiration_time",
            "settlement_timer_datetime",
        )
    )
    now = datetime.now(UTC)
    return Event(
        id=_event_id(Platform.KALSHI, external_id),
        platform=Platform.KALSHI,
        external_id=external_id,
        title=str(_first(raw, "title", "subtitle", "name") or external_id),
        category=_infer_category(raw),
        yes_price=yes_price,
        no_price=no_price,
        implied_prob=implied_prob,
        volume_24h=float(_first(raw, "volume_24h", "volume_24h_contracts", "volume") or 0),
        open_interest=float(_first(raw, "open_interest", "open_interest_count") or 0),
        liquidity_score=_liquidity_score(raw, yes_price),
        resolution_date=resolution_date,
        status=_normalize_status(_first(raw, "status", "market_status")),
        raw_metadata=raw,
        enriched_features={},
        created_at=_parse_datetime(_first(raw, "created_time", "created_at")) or now,
        updated_at=_parse_datetime(_first(raw, "updated_time", "updated_at")) or now,
    )


def normalize_ws_message(message: dict[str, Any]) -> Event | None:
    """Normalize Kalshi WebSocket ticker/trade/lifecycle messages when market-like."""

    payload = message.get("msg") if isinstance(message.get("msg"), dict) else message
    if not isinstance(payload, dict):
        return None
    market_payload = payload.get("market") if isinstance(payload.get("market"), dict) else payload
    external_id = _first(market_payload, "market_ticker", "ticker", "id")
    if external_id is None:
        return None
    normalized = dict(market_payload)
    normalized.setdefault("ticker", external_id)
    normalized.setdefault("title", str(external_id))
    normalized.setdefault("status", "open")
    normalized["ws_type"] = message.get("type")
    return normalize_market(normalized)
