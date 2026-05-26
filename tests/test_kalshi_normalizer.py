from datetime import UTC, datetime, timedelta

from prediction_alpha.ingestion.normalizer import normalize_market, normalize_ws_message
from prediction_alpha.models import EventStatus


def test_normalize_market_computes_prices_and_liquidity() -> None:
    raw = {
        "ticker": "KXCPICORE-26MAY-T3.0",
        "title": "Will CPI be above 3.0?",
        "category": "Economics",
        "yes_bid": 42,
        "yes_ask": 46,
        "volume_24h": 2_500,
        "open_interest": 8_000,
        "close_time": (datetime.now(UTC) + timedelta(days=15)).isoformat(),
        "status": "active",
    }

    event = normalize_market(raw)

    assert event.external_id == "KXCPICORE-26MAY-T3.0"
    assert event.yes_price == 0.46
    assert event.no_price == 0.54
    assert event.implied_prob == 0.46
    assert event.category == "econ"
    assert event.status == EventStatus.OPEN
    assert event.liquidity_score > 0


def test_normalize_ws_message_accepts_nested_market_payload() -> None:
    event = normalize_ws_message(
        {
            "type": "ticker",
            "msg": {
                "market_ticker": "KXFED-26MAY-HIKE",
                "yes_price": 61,
                "volume": 900,
            },
        }
    )

    assert event is not None
    assert event.external_id == "KXFED-26MAY-HIKE"
    assert event.title == "KXFED-26MAY-HIKE"
    assert event.implied_prob == 0.61


# --- Polymarket normalization tests ---

from prediction_alpha.ingestion.normalizer import normalize_polymarket_market


def test_normalize_polymarket_market_binary() -> None:
    raw = {
        "id": "0xpolymarket123",
        "question": "Will ETH be above $3000 by end of 2026?",
        "slug": "eth-3000-eoy-2026",
        "outcomes": ["Yes", "No"],
        "outcomePrices": ["0.65", "0.35"],
        "volume24hr": 125000,
        "liquidityNum": 45000,
        "endDate": "2026-12-31T23:59:59Z",
        "active": True,
        "category": "Crypto",
    }

    event = normalize_polymarket_market(raw)

    assert event.platform.value == "polymarket"
    assert event.external_id == "0xpolymarket123"
    assert event.yes_price == 0.65
    assert event.no_price == 0.35
    assert event.implied_prob == 0.65
    assert event.volume_24h == 125000
    assert "on_chain" in event.enriched_features
    assert event.enriched_features["on_chain"] is True
    assert event.category in ("crypto", "unknown")  # depending on keyword match
