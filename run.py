#!/usr/bin/env python3
"""Prediction Alpha Engine — primary executable entrypoint.

This is the file you run to start the full sovereign system.

Usage examples:

    # One-shot demo (ingest recent markets, score, run agents on gems, notify on elites)
    python run.py --once --pages 2

    # Continuous 24/7 background service (recommended)
    python run.py --continuous

    # With custom config
    SCORING_CONFIG_PATH=scoring_config.yaml python run.py --once

The engine will:
1. Load secrets only from environment (never hardcoded)
2. Ingest live + historical Kalshi data
3. Score every event with the strict hybrid filter
4. Spawn research agents ONLY on high-potential names
5. Dispatch console (and optional email) notifications ONLY for top-tier
6. Prepare clean payloads for your True Neutral Brain

Press Ctrl-C for graceful shutdown (background tasks are cancelled cleanly).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from prediction_alpha.config import get_settings
from prediction_alpha.pipeline import main_once, main_forever
from prediction_alpha.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prediction Alpha Engine — full end-to-end sovereign alpha scout"
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run a single ingestion+scoring+agent+notify cycle then exit (great for testing)",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        help="Run forever as a background service with live WS + periodic backfills",
    )
    p.add_argument(
        "--pages",
        type=int,
        default=2,
        help="How many pages of markets to backfill per cycle (default: 2)",
    )
    p.add_argument(
        "--log-level",
        default=None,
        help="Override log level (DEBUG, INFO, WARNING...)",
    )
    return p


async def async_main(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.log_level:
        settings.log_level = args.log_level  # type: ignore[attr-defined]
    configure_logging(settings.log_level)

    print("🚀 Prediction Alpha Engine starting")
    print(f"   Environment: {settings.environment}")
    print(f"   LLM provider: {settings.llm_provider} / {settings.ollama_model}")
    print(f"   Notifications min score: {settings.notify_min_composite}")
    print(f"   Agent research threshold: {settings.agent_min_composite_to_research}")
    print("   (All secrets from .env — never in code)")

    try:
        if args.continuous:
            await main_forever(settings, backfill_interval_seconds=300, max_pages_per_cycle=args.pages)
        else:
            # default to --once behavior for safety
            result = await main_once(settings, max_pages=args.pages)
            print("\n✅ One-shot run complete:")
            print(result)
            print("\nNext steps:")
            print("  - Try: python run.py --continuous   (for live background)")
            print("  - API:  uvicorn prediction_alpha.api.app:create_app --factory")
            print("  - Full docs: see README.md")
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested — waiting for background tasks (max 8s)...")
        # The pipeline + task_manager already handle cancellation
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.once and not args.continuous:
        # Friendly default: do a small once run so first `python run.py` always does something useful
        args.once = True

    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
