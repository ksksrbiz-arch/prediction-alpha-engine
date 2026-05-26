"""Ingestion layer: Multi-platform (Kalshi + Polymarket) clients and normalization.

Platforms are designed to be independently enabled via Settings.
The Event model and downstream scoring/agents/Brain are platform-agnostic.
"""

from prediction_alpha.ingestion.kalshi_client import KalshiRESTClient, KalshiWebSocketClient
from prediction_alpha.ingestion.normalizer import (
    normalize_market,
    normalize_polymarket_market,
    normalize_ws_message,
)
from prediction_alpha.ingestion.polymarket_client import PolymarketRESTClient

__all__ = [
    "KalshiRESTClient",
    "KalshiWebSocketClient",
    "normalize_market",
    "normalize_ws_message",
    "PolymarketRESTClient",
    "normalize_polymarket_market",
]
