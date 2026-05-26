"""Command-line entrypoints for Phase 1 backfill, stream, and paper scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from prediction_alpha.config import get_settings
from prediction_alpha.ingestion.kalshi_client import KalshiRESTClient, KalshiWebSocketClient
from prediction_alpha.ingestion.storage import PostgresStore
from prediction_alpha.scoring.scorer import HybridScorer
from prediction_alpha.utils.logging import configure_logging, get_logger


async def _maybe_store(store: PostgresStore | None, event: Any, score: Any | None = None) -> None:
    if store is None:
        return
    await store.upsert_event(event)
    if score is not None:
        await store.insert_score(score)


async def backfill(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("cli.backfill")
    store = PostgresStore(settings.database_url) if args.store else None
    if store:
        await store.create_schema()
    scorer = HybridScorer.from_settings(settings)
    count = 0
    async with KalshiRESTClient(settings) as client:
        async for event in client.iter_markets(
            status=args.status, limit=args.limit, max_pages=args.max_pages
        ):
            score = scorer.score(event)
            await _maybe_store(store, event, score)
            if args.print_json:
                print(
                    json.dumps(
                        {
                            "event": event.model_dump(mode="json"),
                            "score": score.model_dump(mode="json"),
                        }
                    )
                )
            count += 1
    if store:
        await store.close()
    log.info("backfill_complete", markets=count, stored=bool(store))


async def stream(args: argparse.Namespace) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    store = PostgresStore(settings.database_url) if args.store else None
    if store:
        await store.create_schema()
    scorer = HybridScorer.from_settings(settings)
    ws = KalshiWebSocketClient(settings)
    async for event in ws.stream(
        channels=args.channels,
        market_tickers=args.market_tickers or None,
    ):
        score = scorer.score(event)
        await _maybe_store(store, event, score)
        print(
            json.dumps(
                {
                    "event": event.model_dump(mode="json"),
                    "score": score.model_dump(mode="json"),
                }
            )
        )


async def backtest(args: argparse.Namespace) -> None:
    from prediction_alpha.scoring.backtesting import run_backtest

    settings = get_settings()
    configure_logging(settings.log_level)
    async with KalshiRESTClient(settings) as client:
        result = await run_backtest(client, max_pages=args.max_pages)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prediction Alpha Phase 1 Kalshi tools")
    sub = parser.add_subparsers(dest="command", required=True)

    backfill_parser = sub.add_parser("backfill", help="Backfill recent Kalshi markets")
    backfill_parser.add_argument("--status", default="open")
    backfill_parser.add_argument("--limit", type=int, default=100)
    backfill_parser.add_argument("--max-pages", type=int, default=1)
    backfill_parser.add_argument("--store", action="store_true", help="Upsert to Postgres")
    backfill_parser.add_argument("--print-json", action="store_true")
    backfill_parser.set_defaults(func=backfill)

    stream_parser = sub.add_parser("stream", help="Stream live Kalshi updates")
    stream_parser.add_argument(
        "--channels",
        nargs="+",
        default=["ticker", "trade", "market_lifecycle_v2"],
    )
    stream_parser.add_argument("--market-tickers", nargs="*", default=[])
    stream_parser.add_argument("--store", action="store_true", help="Upsert to Postgres")
    stream_parser.set_defaults(func=stream)

    backtest_parser = sub.add_parser("backtest", help="Replay resolved markets skeleton")
    backtest_parser.add_argument("--max-pages", type=int, default=1)
    backtest_parser.set_defaults(func=backtest)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
