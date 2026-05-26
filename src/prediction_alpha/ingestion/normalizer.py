"""Kalshi normalization into the canonical Event model."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from typing import Any

from prediction_alpha.ingestion.constants import CATEGORY_KEYWORDS
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


def _calculate_implied_prob(yes_price: float | None, no_price: float | None) -> float | None:
    if yes_price is not None:
        return yes_price
    if no_price is not None:
        return 1.0 - no_price
    return None


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
    source_value = _first(
        raw,
        "category",
        "category_name",
        "series_ticker",
        "event_ticker",
        "title",
    )
    source = " ".join(str(source_value or "").lower().split())
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in source for keyword in keywords):
            return category
    category = _first(raw, "category", "category_name")
    return str(category).lower() if category else "unknown"


def _event_id(platform: Platform, external_id: str) -> str:
    digest = hashlib.sha256(f"{platform}:{external_id}".encode()).hexdigest()
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
    implied_prob = _calculate_implied_prob(yes_price, no_price)
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
    candidate = payload.get("market")
    market_payload: dict[str, Any] = candidate if isinstance(candidate, dict) else payload
    external_id = _first(market_payload, "market_ticker", "ticker", "id")
    if external_id is None:
        return None
    normalized = dict(market_payload)
    normalized.setdefault("ticker", external_id)
    normalized.setdefault("title", str(external_id))
    normalized.setdefault("status", "open")
    normalized["ws_type"] = message.get("type")
    return normalize_market(normalized)


# ---------------------------------------------------------------------------
# Polymarket normalization (Gamma API payloads)
# ---------------------------------------------------------------------------

def normalize_polymarket_market(raw: dict[str, Any]) -> Event:
    """Normalize a Polymarket market from the Gamma API into the canonical Event.

    Polymarket specifics:
    - Markets can be binary (2 outcomes) or multi-outcome.
    - Prices come as strings in "outcomePrices" (e.g. ["0.42", "0.58"]).
    - Volume/liquidity in "volumeNum", "liquidityNum", "volume24hr".
    - Resolution in "endDate" / "resolutionSource".
    - Full outcome data preserved in raw_metadata.

    For multi-outcome markets, we map the first (or highest-prob) outcome
    to the yes_price for compatibility with existing scoring. The full set
    of outcomes/prices lives in enriched_features["polymarket_outcomes"].
    """
    external_id = str(_first(raw, "id", "conditionId", "slug") or "")

    # Outcomes & prices
    outcomes = raw.get("outcomes") or raw.get("outcomeNames") or []
    prices_raw = raw.get("outcomePrices") or raw.get("prices") or []
    if isinstance(prices_raw, str):
        try:
            import json
            prices_raw = json.loads(prices_raw)
        except Exception:
            prices_raw = []

    # Normalize first outcome as "yes" for binary compatibility
    yes_price = None
    if prices_raw:
        yes_price = _price_to_probability(prices_raw[0] if isinstance(prices_raw, (list, tuple)) else prices_raw)

    no_price = None
    if len(prices_raw) > 1:
        no_price = _price_to_probability(prices_raw[1])
    elif yes_price is not None:
        no_price = 1.0 - yes_price

    implied_prob = _calculate_implied_prob(yes_price, no_price)

    # Volume & liquidity (Polymarket uses different field names)
    volume_24h = float(_first(raw, "volume24hr", "volume24Hr", "volumeNum") or 0)
    liquidity = float(_first(raw, "liquidityNum", "liquidity") or 0)
    open_interest = float(_first(raw, "openInterest", "open_interest") or liquidity * 0.6)  # rough proxy

    # Liquidity score (reuse Kalshi-style with Polymarket fields)
    liq_score = _liquidity_score(raw, yes_price)
    # Boost slightly for on-chain transparency signal (Polymarket is on Polygon)
    if volume_24h > 1000 or liquidity > 5000:
        liq_score = min(1.0, liq_score + 0.05)

    resolution_date = _parse_datetime(
        _first(raw, "endDate", "end_date", "resolutionDate", "createdAt")
    )

    now = datetime.now(UTC)

    # Store full Polymarket structure for downstream (agents, Brain, features)
    enriched = {
        "platform": "polymarket",
        "outcomes": outcomes,
        "outcome_prices": prices_raw,
        "clob_token_ids": raw.get("clobTokenIds") or raw.get("clob_token_ids"),
        "resolution_source": raw.get("resolutionSource"),
        "image": raw.get("image"),
        "slug": raw.get("slug"),
        "on_chain": True,  # Polymarket is on-chain (Polygon)
    }

    return Event(
        id=_event_id(Platform.POLYMARKET, external_id),
        platform=Platform.POLYMARKET,
        external_id=external_id,
        title=str(_first(raw, "question", "title", "slug") or external_id),
        category=_infer_category(raw),
        yes_price=yes_price,
        no_price=no_price,
        implied_prob=implied_prob,
        volume_24h=volume_24h,
        open_interest=open_interest,
        liquidity_score=liq_score,
        resolution_date=resolution_date,
        status=_normalize_status(_first(raw, "active", "closed", "status")),
        raw_metadata=raw,
        enriched_features=enriched,
        created_at=_parse_datetime(_first(raw, "createdAt", "created_at")) or now,
        updated_at=_parse_datetime(_first(raw, "updatedAt", "updated_at")) or now,
    )

