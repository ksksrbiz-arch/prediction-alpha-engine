"""Async Polymarket REST client using the public Gamma API (and CLOB where useful).

Read-only focus for compliance and US access limitations (as of May 2026,
Polymarket is invite-only / geo-restricted for US persons; this client performs
no authentication or trading actions).

Primary data source: https://gamma-api.polymarket.com/
- Markets list & details (including prices, volume, liquidity)
- Supports pagination, filtering by active/closed, ordering by volume/liquidity

Optional CLOB integration (https://clob.polymarket.com/):
- Order books for more accurate liquidity/depth signals (read-only)

Design mirrors Kalshi client:
- Throttling + retries
- Async iteration for backfill
- No secrets required for public data
- raw_metadata preserved for future on-chain / crypto correlation features

Future extensibility:
- WebSocket/RTDS (Polymarket has limited public real-time feeds; can be added)
- Trading (CLOB auth + orders) can be added in a PolymarketTradingClient subclass
  without changing the read path or Event normalization.

Polymarket markets can be binary or multi-outcome. The normalizer maps them
to the canonical Event (for multi-outcome, the primary/highest-prob outcome
is used for yes_price; full outcomes live in raw_metadata + enriched_features).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from prediction_alpha.config import Settings
from prediction_alpha.ingestion.normalizer import normalize_polymarket_market
from prediction_alpha.models import Event
from prediction_alpha.utils.logging import get_logger


class PolymarketRESTClient:
    """Read-oriented public REST client for Polymarket via Gamma (and CLOB)."""

    _MIN_REQUESTS_PER_SECOND = 0.5  # Polymarket Gamma is more generous than Kalshi

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._gamma_base = getattr(
            settings, "polymarket_gamma_api_url", "https://gamma-api.polymarket.com"
        ).rstrip("/")
        self._clob_base = getattr(
            settings, "polymarket_clob_url", "https://clob.polymarket.com"
        ).rstrip("/")

        self._client = httpx.AsyncClient(
            timeout=getattr(settings, "polymarket_request_timeout_seconds", 20.0),
            headers={"User-Agent": "prediction-alpha-engine/0.2 (Polymarket read-only)"},
        )
        self._lock = asyncio.Lock()
        rps = getattr(settings, "polymarket_requests_per_second", 5.0)
        self._min_interval = 1.0 / max(rps, self._MIN_REQUESTS_PER_SECOND)
        self._last_request_at = 0.0
        self._log = get_logger("polymarket.rest")

    async def __aenter__(self) -> PolymarketRESTClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait_for = self._min_interval - (now - self._last_request_at)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_request_at = asyncio.get_running_loop().time()

    async def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generic GET with throttling + basic retry."""
        last_exc: Exception | None = None
        for attempt in range(4):
            await self._throttle()
            try:
                resp = await self._client.get(url, params=params)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2**attempt))
                    self._log.warning("polymarket_rate_limited", url=url, retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return cast(dict[str, Any], resp.json())
            except (httpx.HTTPStatusError, ValueError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError(f"Polymarket request failed after retries: {url}") from last_exc

    async def list_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        order: str = "volume24hr",
        ascending: bool = False,
        slug: str | None = None,
    ) -> list[dict[str, Any]]:
        """List markets via Gamma API."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if slug:
            params["slug"] = slug

        url = f"{self._gamma_base}/markets"
        data = await self._get(url, params=params)
        # Gamma returns a list directly or {"data": [...] } in some versions
        if isinstance(data, list):
            return data
        return data.get("data", data) if isinstance(data, dict) else []

    async def iter_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        max_pages: int | None = None,
        order: str = "volume24hr",
    ) -> AsyncIterator[Event]:
        """Yield normalized Polymarket events with pagination."""
        offset = 0
        pages = 0
        while True:
            markets = await self.list_markets(
                active=active,
                closed=closed,
                limit=limit,
                offset=offset,
                order=order,
            )
            if not markets:
                break

            for m in markets:
                try:
                    yield normalize_polymarket_market(m)
                except Exception as exc:  # noqa: BLE001
                    self._log.warning("polymarket_normalize_failed", market=m.get("slug"), error=str(exc)[:120])

            pages += 1
            offset += len(markets)
            if max_pages is not None and pages >= max_pages:
                break
            if len(markets) < limit:
                break

    async def get_market(self, slug_or_id: str) -> dict[str, Any]:
        """Fetch a single market by slug or condition id."""
        # Gamma supports /markets/{slug} or query by id
        url = f"{self._gamma_base}/markets"
        data = await self._get(url, params={"slug": slug_or_id})
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}

    async def get_orderbook(self, token_id: str, depth: int = 20) -> dict[str, Any]:
        """Optional: Fetch CLOB orderbook for deeper liquidity signals (read-only)."""
        url = f"{self._clob_base}/book"
        try:
            return await self._get(url, params={"token_id": token_id, "depth": depth})
        except Exception as exc:  # noqa: BLE001
            self._log.debug("clob_orderbook_unavailable", token_id=token_id, error=str(exc)[:80])
            return {"bids": [], "asks": []}


# Convenience alias for future WebSocket client (not implemented in initial read-only MVP)
PolymarketWebSocketClient = None  # type: ignore  # Placeholder; real WS can be added later
