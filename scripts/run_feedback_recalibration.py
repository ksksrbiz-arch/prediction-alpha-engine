#!/usr/bin/env python3
"""
Simple feedback / self-improvement job.

Run periodically (cron / systemd timer) after markets resolve:
    PYTHONPATH=src python scripts/run_feedback_recalibration.py

It logs any newly resolved markets you feed it (or pulls from DB in a real impl)
and runs the lightweight recalibration hook.
"""

import asyncio
from prediction_alpha.feedback.loop import FeedbackLoop
from prediction_alpha.ingestion.storage import PostgresStore
from prediction_alpha.config import get_settings


async def main():
    settings = get_settings()
    store = PostgresStore(settings.database_url)
    await store.create_schema()  # safe

    feedback = FeedbackLoop(store)
    await feedback.ensure_schema()

    # In a real system you would query recently settled markets here
    # and call feedback.log_resolution(...) for each.

    print("Running simple recalibration analysis...")
    result = await feedback.run_simple_recalibration()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
