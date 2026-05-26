"""Backtesting skeleton for resolved Kalshi markets."""

from __future__ import annotations

from statistics import mean
from typing import Any

from prediction_alpha.ingestion.kalshi_client import KalshiRESTClient
from prediction_alpha.models import Event
from prediction_alpha.scoring.scorer import HybridScorer


def _actual_resolution(event: Event) -> float | None:
    raw = event.raw_metadata
    result = raw.get("result") or raw.get("settlement_value") or raw.get("yes_result")
    if result is None:
        return None
    text = str(result).lower()
    if text in {"yes", "true", "1", "1.0"}:
        return 1.0
    if text in {"no", "false", "0", "0.0"}:
        return 0.0
    return None


async def run_backtest(client: KalshiRESTClient, *, max_pages: int = 1) -> dict[str, Any]:
    """Replay resolved markets and report basic calibration metrics."""

    scorer = HybridScorer.from_settings(client.settings)
    predictions: list[float] = []
    outcomes: list[float] = []
    scored = 0
    passed = 0
    async for event in client.iter_markets(status="settled", max_pages=max_pages):
        score = scorer.score(event)
        scored += 1
        passed += int(score.passed_filter)
        actual = _actual_resolution(event)
        predicted = score.features.get("implied_prob")
        if actual is not None and isinstance(predicted, int | float):
            outcomes.append(actual)
            predictions.append(float(predicted))
    brier = mean([(prediction - outcome) ** 2 for prediction, outcome in zip(predictions, outcomes)])
    return {
        "markets_scored": scored,
        "passed_filter": passed,
        "resolved_with_outcomes": len(outcomes),
        "brier_score": brier if predictions else None,
        "note": "Skeleton metric uses market implied probability until ML predictions are trained.",
    }
