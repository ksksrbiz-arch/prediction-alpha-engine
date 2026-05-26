"""Tests for the FastAPI /opportunities endpoint and scoring config."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from prediction_alpha.api.app import create_app
from prediction_alpha.api.routes import update_scored_cache
from prediction_alpha.config import ScoringConfig, Settings
from prediction_alpha.models import Event, EventStatus, Platform
from prediction_alpha.scoring.scorer import HybridScorer


def _make_event(
    external_id: str = "TEST-MARKET",
    yes_price: float = 0.60,
    volume: float = 5_000.0,
    category: str = "econ",
) -> Event:
    return Event(
        id=f"kalshi-{external_id}",
        platform=Platform.KALSHI,
        external_id=external_id,
        title=f"Test market {external_id}",
        category=category,
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        implied_prob=yes_price,
        volume_24h=volume,
        open_interest=10_000.0,
        liquidity_score=0.65,
        resolution_date=datetime.now(UTC) + timedelta(days=10),
        status=EventStatus.OPEN,
    )


def test_opportunities_returns_scored_events() -> None:
    settings = Settings(environment="test")
    app = create_app(settings)
    scorer = HybridScorer.from_settings(settings)
    events = [_make_event("MKT-1"), _make_event("MKT-2", yes_price=0.90)]
    update_scored_cache(events, scorer)

    client = TestClient(app)
    resp = client.get("/opportunities?min_score=0&passed_only=false")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert "event" in data[0]
    assert "score" in data[0]


def test_opportunities_filters_by_min_score() -> None:
    settings = Settings(environment="test")
    app = create_app(settings)
    scorer = HybridScorer.from_settings(settings)
    events = [_make_event("MKT-HI"), _make_event("MKT-LO", yes_price=0.50)]
    update_scored_cache(events, scorer)

    client = TestClient(app)
    resp = client.get("/opportunities?min_score=0.99&passed_only=false")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


def test_opportunities_filters_by_category() -> None:
    settings = Settings(environment="test")
    app = create_app(settings)
    scorer = HybridScorer.from_settings(settings)
    events = [
        _make_event("MKT-ECON", category="econ"),
        _make_event("MKT-SPORT", category="sports"),
    ]
    update_scored_cache(events, scorer)

    client = TestClient(app)
    resp = client.get("/opportunities?min_score=0&passed_only=false&category=econ")
    assert resp.status_code == 200
    data = resp.json()
    assert all(e["event"]["category"] == "econ" for e in data)


def test_health_endpoint() -> None:
    app = create_app(Settings(environment="test"))
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_scoring_config_defaults() -> None:
    config = ScoringConfig()
    assert config.min_liquidity_score == 0.20
    assert config.composite_weights.edge == 3.0
    assert config.category_weights.econ == 0.75


def test_scoring_config_from_yaml(tmp_path: object) -> None:
    import pathlib

    yaml_path = pathlib.Path(str(tmp_path)) / "scoring.yaml"
    yaml_path.write_text(
        "min_liquidity_score: 0.50\ncomposite_weights:\n  edge: 5.0\n"
    )
    config = ScoringConfig.from_yaml(yaml_path)
    assert config.min_liquidity_score == 0.50
    assert config.composite_weights.edge == 5.0
    # Non-overridden values keep defaults
    assert config.composite_weights.confidence == 0.25
