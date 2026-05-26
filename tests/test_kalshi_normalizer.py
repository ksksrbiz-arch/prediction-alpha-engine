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
