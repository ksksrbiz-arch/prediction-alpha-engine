"""Ingestion layer: Kalshi WebSocket + REST clients, normalization."""

from prediction_alpha.ingestion.kalshi_client import KalshiRESTClient, KalshiWebSocketClient
from prediction_alpha.ingestion.normalizer import normalize_market, normalize_ws_message

__all__ = [
    "KalshiRESTClient",
    "KalshiWebSocketClient",
    "normalize_market",
    "normalize_ws_message",
]
