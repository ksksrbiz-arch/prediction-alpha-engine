"""Cortex signal bridge — make the engine a callable voice for day-trade.

The day-trade platform's confluence brain / neural core ("cortex") fuses
per-method scores, each a scalar in [-1, 1] for a tradable symbol. This module
translates the engine's scored prediction-market opportunities into exactly
that shape so the trader can call us like any other voice:

    GET /cortex/signal?symbol=SPY   ->  {"symbol": "SPY", "score": 0.31, ...}

How a market becomes a ticker signal
------------------------------------
1. ``map_event`` matches an event's title/category against a transparent rule
   table.  A rule yields ``{symbol: sense}`` where sense=+1 means "this market
   resolving YES is bullish for the symbol" and -1 the opposite.
   Asset rules (bitcoin/S&P/nasdaq/gold/oil/rates...) combine with a
   direction cue in the title (above/reach vs below/drop); macro rules
   (recession, shutdown, rate cut/hike...) carry their sense directly.
   Agents can override per event via ``enriched_features["cortex_map"]``.
2. Each mapped live market contributes ``sense * (2 * implied_prob - 1)`` —
   the direction the market itself is pricing — weighted by the engine's
   composite score and liquidity.
3. ``signal_for`` aggregates contributions into a weighted mean in [-1, 1]
   with a components trace, so the trader's dashboard can show *why*.

Pure functions over the scored-cache entries; no I/O, trivially testable.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Rule tables (transparent + extensible; keep patterns lowercase)
# ---------------------------------------------------------------------------

# Direction cues inside a market title: does YES mean "price goes up"?
_UP_CUE = re.compile(r"\b(above|over|exceed|reach|hit|higher|close above|at least|rise)\b")
_DOWN_CUE = re.compile(r"\b(below|under|drop|fall|lower|close below|less than|decline)\b")

# Asset rules: title pattern -> symbol. Sense comes from the direction cue.
_ASSET_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(bitcoin|btc)\b"), "BTC"),
    (re.compile(r"\b(ethereum|eth)\b"), "ETH"),
    (re.compile(r"\b(s&p ?500|s&p|spx|sp500)\b"), "SPY"),
    (re.compile(r"\bnasdaq\b"), "QQQ"),
    (re.compile(r"\b(dow jones|djia)\b"), "DIA"),
    (re.compile(r"\bgold\b"), "GLD"),
    (re.compile(r"\b(crude oil|wti|brent)\b"), "USO"),
]

# Macro rules: title pattern -> {symbol: sense} directly (no direction cue).
_MACRO_RULES: list[tuple[re.Pattern[str], dict[str, float]]] = [
    (re.compile(r"\brecession\b"), {"SPY": -1.0, "QQQ": -1.0}),
    (re.compile(r"\bgovernment shutdown\b"), {"SPY": -0.5}),
    (re.compile(r"\b(debt ceiling|default on .*debt)\b"), {"SPY": -1.0, "TLT": -1.0}),
    (re.compile(r"\b(rate cut|cut (interest )?rates|lower (interest )?rates)\b"),
     {"SPY": 1.0, "QQQ": 1.0, "TLT": 1.0}),
    (re.compile(r"\b(rate hike|raise (interest )?rates|hike (interest )?rates)\b"),
     {"SPY": -1.0, "QQQ": -1.0, "TLT": -1.0}),
    (re.compile(r"\b(inflation|cpi).{0,40}\b(above|exceed|higher|hot)\b", re.DOTALL),
     {"TLT": -1.0, "SPY": -0.5}),
    (re.compile(r"\bunemployment.{0,40}\b(above|exceed|rise)\b", re.DOTALL), {"SPY": -0.5}),
]

_LIVE_STATUSES = {"open", "unknown"}


def map_event(event: dict[str, Any]) -> dict[str, float]:
    """Map one normalized event (as a plain dict) to ``{symbol: sense}``.

    Precedence: explicit agent-enriched ``cortex_map`` > macro rules > asset
    rules (which need a direction cue in the title). Unmapped events return {}.
    """

    enriched = event.get("enriched_features") or {}
    override = enriched.get("cortex_map")
    if isinstance(override, dict) and override:
        out: dict[str, float] = {}
        for sym, sense in override.items():
            try:
                out[str(sym).upper()] = max(-1.0, min(1.0, float(sense)))
            except (TypeError, ValueError):
                continue
        if out:
            return out

    title = str(event.get("title", "")).lower()
    senses: dict[str, float] = {}
    for pat, mapping in _MACRO_RULES:
        if pat.search(title):
            for sym, sense in mapping.items():
                senses[sym] = max(-1.0, min(1.0, senses.get(sym, 0.0) + sense))

    cue = 1.0 if _UP_CUE.search(title) else -1.0 if _DOWN_CUE.search(title) else 0.0
    if cue:
        for pat, sym in _ASSET_RULES:
            if pat.search(title) and sym not in senses:
                senses[sym] = cue
    return senses


def _contribution(entry: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    """One scored-cache entry's weighted directional contribution for symbol."""

    event = entry.get("event") or {}
    score = entry.get("score") or {}
    if str(event.get("status", "unknown")).lower() not in _LIVE_STATUSES:
        return None
    sense = map_event(event).get(symbol)
    implied = event.get("implied_prob")
    if sense is None or implied is None:
        return None
    direction = sense * (2.0 * float(implied) - 1.0)          # what the market prices
    composite = float(score.get("composite_score") or 0.0)
    liquidity = float(event.get("liquidity_score") or 0.0)
    weight = max(composite, 0.05) * (0.5 + 0.5 * liquidity)   # engine conviction x liquidity
    return {
        "event_id": event.get("id"),
        "title": event.get("title"),
        "sense": sense,
        "implied_prob": implied,
        "direction": round(direction, 4),
        "weight": round(weight, 4),
    }


def signal_for(symbol: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all mapped live markets into one cortex-shaped signal.

    ``score`` is None when no market maps to the symbol — the trader's voice
    abstains rather than voting 0 (an explicit "no opinion", matching how the
    day-trade confluence treats absent methods).
    """

    symbol = symbol.upper()
    components = []
    for entry in entries:
        c = _contribution(entry, symbol)
        if c:
            components.append(c)
    if not components:
        return {"symbol": symbol, "score": None, "n": 0, "confidence": 0.0, "components": []}

    num = sum(c["direction"] * c["weight"] for c in components)
    den = sum(c["weight"] for c in components) or 1.0
    score = max(-1.0, min(1.0, num / den))
    # confidence grows with total evidence weight, saturating around ~2.0
    confidence = min(1.0, den / 2.0)
    components.sort(key=lambda c: c["weight"], reverse=True)
    return {
        "symbol": symbol,
        "score": round(score, 4),
        "n": len(components),
        "confidence": round(confidence, 3),
        "components": components[:10],
    }


def signals_for(symbols: list[str], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch form of ``signal_for`` (one pass per symbol; cache is small)."""

    return [signal_for(s, entries) for s in symbols if s.strip()]
