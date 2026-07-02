"""Tests for the day-trade cortex bridge (integrations.cortex + /cortex API)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from prediction_alpha.api.app import create_app
from prediction_alpha.api.routes import update_scored_cache
from prediction_alpha.config import Settings
from prediction_alpha.integrations.cortex import map_event, signal_for
from prediction_alpha.models import Event, EventStatus, Platform
from prediction_alpha.scoring.scorer import HybridScorer


def _event(title: str, yes_price: float = 0.70, status: EventStatus = EventStatus.OPEN,
           enriched: dict | None = None, external_id: str = "MKT") -> Event:
    return Event(
        id=f"kalshi-{external_id}",
        platform=Platform.KALSHI,
        external_id=external_id,
        title=title,
        category="econ",
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        implied_prob=yes_price,
        volume_24h=5_000.0,
        open_interest=10_000.0,
        liquidity_score=0.65,
        resolution_date=datetime.now(UTC) + timedelta(days=10),
        status=status,
        enriched_features=enriched or {},
    )


def _entry(event: Event, composite: float = 0.6) -> dict:
    return {
        "event": event.model_dump(mode="json"),
        "score": {"composite_score": composite, "passed_filter": True},
    }


# ---------------------------------------------------------------------------
# map_event — rule table
# ---------------------------------------------------------------------------


def test_asset_rule_needs_direction_cue() -> None:
    up = map_event(_event("Will Bitcoin close above $150,000 this year?").model_dump(mode="json"))
    down = map_event(_event("Will Bitcoin drop below $50,000?").model_dump(mode="json"))
    none = map_event(_event("Will Bitcoin be discussed at the G7?").model_dump(mode="json"))
    assert up == {"BTC": 1.0}
    assert down == {"BTC": -1.0}
    assert none == {}


def test_macro_rule_carries_its_own_sense() -> None:
    m = map_event(_event("Will the US enter a recession in 2026?").model_dump(mode="json"))
    assert m["SPY"] == -1.0 and m["QQQ"] == -1.0
    cut = map_event(_event("Will the Fed announce a rate cut in September?").model_dump(mode="json"))
    assert cut["SPY"] == 1.0 and cut["TLT"] == 1.0


def test_enriched_cortex_map_overrides_rules() -> None:
    ev = _event("Will the US enter a recession in 2026?",
                enriched={"cortex_map": {"iwm": -2.0}}).model_dump(mode="json")
    assert map_event(ev) == {"IWM": -1.0}  # upper-cased and clamped


# ---------------------------------------------------------------------------
# signal_for — aggregation
# ---------------------------------------------------------------------------


def test_signal_direction_follows_market_probability() -> None:
    # 80% chance of recession -> bearish SPY (sense -1 * (2*0.8-1) < 0)
    entries = [_entry(_event("Will the US enter a recession in 2026?", yes_price=0.80))]
    sig = signal_for("SPY", entries)
    assert sig["n"] == 1 and sig["score"] < 0
    # 20% chance -> the market prices "no recession" -> mildly bullish
    entries = [_entry(_event("Will the US enter a recession in 2026?", yes_price=0.20))]
    assert signal_for("spy", entries)["score"] > 0


def test_signal_abstains_when_nothing_maps() -> None:
    entries = [_entry(_event("Will it rain in Miami on Friday?"))]
    sig = signal_for("SPY", entries)
    assert sig["score"] is None and sig["n"] == 0


def test_signal_skips_resolved_markets_and_stays_bounded() -> None:
    entries = [
        _entry(_event("Will the US enter a recession in 2026?", yes_price=0.99,
                      status=EventStatus.RESOLVED)),
        _entry(_event("Will the Fed announce a rate cut in September?", yes_price=0.95,
                      external_id="CUT"), composite=0.9),
    ]
    sig = signal_for("SPY", entries)
    assert sig["n"] == 1               # resolved market ignored
    assert -1.0 <= sig["score"] <= 1.0
    assert sig["components"][0]["event_id"] == "kalshi-CUT"


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_cortex_endpoints_serve_signals() -> None:
    settings = Settings(environment="test")
    app = create_app(settings)
    scorer = HybridScorer.from_settings(settings)
    update_scored_cache(
        [_event("Will Bitcoin close above $150,000 this year?", yes_price=0.65, external_id="BTCHI")],
        scorer,
    )
    client = TestClient(app)

    resp = client.get("/cortex/signal", params={"symbol": "btc"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTC" and body["score"] is not None and body["score"] > 0

    resp = client.get("/cortex/signals", params={"symbols": "BTC,SPY"})
    assert resp.status_code == 200
    by_sym = {s["symbol"]: s for s in resp.json()}
    assert by_sym["BTC"]["n"] == 1
    assert by_sym["SPY"]["score"] is None  # abstains: nothing maps
