"""Async Kalshi REST and WebSocket clients."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any, cast

import httpx
import websockets

from prediction_alpha.config import Settings
from prediction_alpha.ingestion.normalizer import normalize_market, normalize_ws_message
from prediction_alpha.models import Event
from prediction_alpha.utils.logging import get_logger


class KalshiRateLimitError(RuntimeError):
    """Raised when Kalshi rate limits persist after retries."""


class KalshiRESTClient:
    """Read-oriented public REST client for Kalshi market data."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.kalshi_rest_base_url,
            timeout=settings.kalshi_request_timeout_seconds,
            headers=self._headers(),
        )
        self._lock = asyncio.Lock()
        self._min_interval = 1.0 / max(settings.kalshi_requests_per_second, 0.1)
        self._last_request_at = 0.0
        self._log = get_logger("kalshi.rest")

    async def __aenter__(self) -> KalshiRESTClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "prediction-alpha-engine/0.1"}
        if self.settings.kalshi_api_key:
            # Public market endpoints work without auth. The key is included only for deployments
            # where Kalshi enables read scopes; private trading signatures are intentionally out
            # of scope for this paper-testing layer.
            headers["KALSHI-ACCESS-KEY"] = self.settings.kalshi_api_key
        return headers

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait_for = self._min_interval - (now - self._last_request_at)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_request_at = asyncio.get_running_loop().time()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(4):
            await self._throttle()
            response = await self._client.get(path, params=params)
            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                retry_after = float(response.headers.get("Retry-After", 2**attempt))
                self._log.warning("kalshi_rate_limited", path=path, retry_after=retry_after)
                await asyncio.sleep(retry_after)
                continue
            try:
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
            except (httpx.HTTPStatusError, ValueError) as exc:
                last_exc = exc
                if response.status_code >= 500:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
        raise KalshiRateLimitError(f"Kalshi request failed after retries: {path}") from last_exc

    async def list_markets(
        self,
        *,
        status: str | None = "open",
        limit: int = 100,
        cursor: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        return await self._get("/markets", params=params)

    async def iter_markets(
        self, *, status: str | None = "open", limit: int = 100, max_pages: int | None = None
    ) -> AsyncIterator[Event]:
        cursor: str | None = None
        pages = 0
        while True:
            payload = await self.list_markets(status=status, limit=limit, cursor=cursor)
            for market in payload.get("markets", []):
                yield normalize_market(market)
            pages += 1
            cursor = payload.get("cursor")
            if not cursor or (max_pages is not None and pages >= max_pages):
                break

    async def get_series(self, series_ticker: str) -> dict[str, Any]:
        return await self._get(f"/series/{series_ticker}")

    async def get_market(self, market_ticker: str) -> Event:
        payload = await self._get(f"/markets/{market_ticker}")
        return normalize_market(payload.get("market", payload))

    async def get_orderbook(self, market_ticker: str, depth: int = 20) -> dict[str, Any]:
        return await self._get(f"/markets/{market_ticker}/orderbook", params={"depth": depth})

    async def get_historical_candlesticks(
        self, market_ticker: str, *, start_ts: int, end_ts: int, period_interval: int = 60
    ) -> dict[str, Any]:
        """Fetch historical market candles.

        Kalshi exposes history through market candlesticks; this wrapper keeps the
        backtester isolated if endpoint naming changes.
        """

        return await self._get(
            f"/markets/{market_ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )


class KalshiWebSocketClient:
    """Resilient Kalshi WebSocket stream for ticker/trade/lifecycle channels."""

    def __init__(self, settings: Settings, rest_client: KalshiRESTClient | None = None) -> None:
        self.settings = settings
        self.rest_client = rest_client
        self._log = get_logger("kalshi.websocket")
        self._message_id = 0

    def _next_id(self) -> int:
        self._message_id += 1
        return self._message_id

    async def _subscribe(
        self,
        ws: Any,
        *,
        channels: Iterable[str],
        market_tickers: Iterable[str] | None,
    ) -> None:
        params: dict[str, Any] = {"channels": list(channels)}
        if market_tickers:
            params["market_tickers"] = list(market_tickers)
        await ws.send(json.dumps({"id": self._next_id(), "cmd": "subscribe", "params": params}))

    async def stream(
        self,
        *,
        channels: Iterable[str] = ("ticker", "trade", "market_lifecycle_v2"),
        market_tickers: Iterable[str] | None = None,
    ) -> AsyncIterator[Event]:
        """Yield normalized live events forever, reconnecting with exponential backoff."""

        reconnect_delay = self.settings.kalshi_ws_reconnect_initial_seconds
        while True:
            try:
                async with websockets.connect(
                    self.settings.kalshi_ws_url,
                    ping_interval=self.settings.kalshi_ws_ping_interval_seconds,
                    ping_timeout=self.settings.kalshi_ws_ping_interval_seconds,
                    additional_headers=self._headers(),
                ) as ws:
                    await self._subscribe(ws, channels=channels, market_tickers=market_tickers)
                    reconnect_delay = self.settings.kalshi_ws_reconnect_initial_seconds
                    self._log.info("kalshi_ws_connected", channels=list(channels))
                    async for raw_message in ws:
                        message = self._decode(raw_message)
                        if message is None:
                            continue
                        event = normalize_ws_message(message)
                        if event is not None:
                            yield event
            except (OSError, websockets.WebSocketException, TimeoutError) as exc:
                self._log.warning(
                    "kalshi_ws_reconnect",
                    error=str(exc),
                    reconnect_delay=reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(
                    reconnect_delay * 2,
                    self.settings.kalshi_ws_reconnect_max_seconds,
                )

    @staticmethod
    def _headers() -> dict[str, str]:
        return {"User-Agent": "prediction-alpha-engine/0.1"}

    @staticmethod
    def _decode(raw_message: str | bytes) -> dict[str, Any] | None:
        try:
            return cast(dict[str, Any], json.loads(raw_message))
        except (TypeError, ValueError):
            return None
