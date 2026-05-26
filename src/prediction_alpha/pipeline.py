"""End-to-end orchestration layer: the beating heart of the Prediction Alpha Engine.

This module wires:
  Ingestion (REST backfill + WS stream)
    → Feature + Hybrid Scoring (strict filters)
    → Agentic Legwork (only on high-conviction candidates)
    → Final Gate + Notifications (ultra-selective)
    → Brain Export Hook

It is deliberately background-friendly and can be run as:
- One-shot batch (for testing / cron)
- Long-lived continuous service (recommended for 24/7 operation)

All heavy work (agents, notifications) happens via the shared TaskManager so
FastAPI lifespan can shut everything down cleanly.

Sovereignty & productization notes are repeated in comments because this is the
file a new operator or future SaaS team will read first.
"""

from __future__ import annotations

import asyncio
from typing import Any

from prediction_alpha.agents.legwork import AgentOrchestrator
from prediction_alpha.config import Settings, get_settings
from prediction_alpha.ingestion.kalshi_client import KalshiRESTClient
from prediction_alpha.ingestion.storage import PostgresStore
from prediction_alpha.models import Event, OpportunityScore
from prediction_alpha.notifications.brain import prepare_brain_payload
from prediction_alpha.notifications.notifier import get_notifier
from prediction_alpha.scoring.scorer import HybridScorer
from prediction_alpha.utils.logging import configure_logging, get_logger

from prediction_alpha.api.tasks import task_manager  # shared background manager

_log = get_logger("pipeline")


class PredictionAlphaEngine:
    """The runnable sovereign alpha engine.

    Instantiate once, then call run_once() or run_forever().
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.scorer = HybridScorer.from_settings(self.settings)
        self.notifier = get_notifier(self.settings)
        self.agent_orchestrator = AgentOrchestrator(self.settings)
        self.store: PostgresStore | None = None
        self._log = get_logger("engine")

    async def _ensure_store(self) -> PostgresStore | None:
        if self.store is None and self.settings.database_url:
            self.store = PostgresStore(self.settings.database_url)
            try:
                await self.store.create_schema()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("db_schema_failed_continue_without_persist", error=str(exc))
                self.store = None
        return self.store

    async def process_event(
        self, event: Event, *, force_agent: bool = False
    ) -> OpportunityScore:
        """Score one event, optionally run agents, notify if elite, return the score."""

        score = self.scorer.score(event)

        # Persist every scored event (cheap and invaluable for feedback)
        store = await self._ensure_store()
        if store:
            try:
                await store.upsert_event(event)
                await store.insert_score(score)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("persist_failed", event_id=event.id, error=str(exc))

        if not score.passed_filter:
            return score

        # Spawn agent legwork in background ONLY for promising names
        if force_agent or (
            self.settings.agent_enabled
            and score.composite_score >= self.settings.agent_min_composite_to_research
        ):
            async def _agent_job() -> None:
                enriched = await self.agent_orchestrator.enrich_score_with_plan(event, score)
                # Re-evaluate notification gate after agent insights (future: re-score)
                if self.notifier.should_notify(enriched):
                    notif = self.notifier.build_notification(
                        event, enriched, enriched.research_brief
                    )
                    await self.notifier.dispatch(notif)

                    # Brain prep hook (fire-and-forget friendly)
                    brain_payload = prepare_brain_payload(event, enriched, enriched.research_brief)
                    self._log.info(
                        "brain_payload_ready",
                        event_id=event.id,
                        score=round(enriched.composite_score, 3),
                        # In real life: await brain_client.ingest(brain_payload)
                    )
                # TODO: later re-persist the enriched score with agent data

            task_manager.submit(_agent_job(), name=f"agent-{event.id[:12]}")
        else:
            # Even without full agents, if it is *extremely* high value, still notify
            if self.notifier.should_notify(score):
                notif = self.notifier.build_notification(event, score)
                await self.notifier.dispatch(notif)
                brain_payload = prepare_brain_payload(event, score)
                self._log.info("brain_payload_ready", event_id=event.id, score=score.composite_score)

        return score

    async def run_once(
        self,
        *,
        max_pages: int = 2,
        status: str = "open",
        only_high_value: bool = True,
    ) -> dict[str, Any]:
        """One complete ingestion + scoring + selective agent + notify pass.

        Ideal for testing, CI, or scheduled cron jobs.
        """

        configure_logging(self.settings.log_level)
        self._log.info("engine_run_once_start", max_pages=max_pages)

        processed = 0
        high_value = 0
        notified = 0

        async with KalshiRESTClient(self.settings) as client:
            async for event in client.iter_markets(status=status, max_pages=max_pages):
                score = await self.process_event(event)
                processed += 1
                if score.passed_filter:
                    high_value += 1
                    if score.composite_score >= self.settings.notify_min_composite:
                        notified += 1

        self._log.info(
            "engine_run_once_complete",
            processed=processed,
            high_value=high_value,
            notified=notified,
            pending_tasks=task_manager.pending_count,
        )
        return {
            "processed": processed,
            "high_value": high_value,
            "notified": notified,
            "pending_background": task_manager.pending_count,
        }

    async def run_forever(
        self,
        *,
        backfill_interval_seconds: float = 300.0,  # 5 min
        max_pages_per_cycle: int = 1,
    ) -> None:
        """Long-lived background service.

        - Does an initial backfill
        - Then streams live WS ticks (scoring everything in real time)
        - Periodically does larger backfills to catch anything missed
        - Agents + notifications fire only on the tiny fraction that survives filters

        This is the command you put in systemd / Docker / your VPS.
        """

        configure_logging(self.settings.log_level)
        self._log.info("engine_run_forever_start")

        # Initial backfill so we have a warm cache
        await self.run_once(max_pages=max_pages_per_cycle)

        # Background WS stream (never stops unless cancelled)
        async def _ws_worker() -> None:
            from prediction_alpha.ingestion.kalshi_client import KalshiWebSocketClient

            ws = KalshiWebSocketClient(self.settings)
            self._log.info("ws_stream_worker_started")
            try:
                async for event in ws.stream():
                    await self.process_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log.error("ws_worker_crashed", error=str(exc))

        # Periodic larger backfill (catches markets that never tick)
        async def _periodic_backfill() -> None:
            while True:
                await asyncio.sleep(backfill_interval_seconds)
                try:
                    await self.run_once(max_pages=max_pages_per_cycle)
                except Exception as exc:  # noqa: BLE001
                    self._log.warning("periodic_backfill_error", error=str(exc))

        # Launch workers
        ws_task = task_manager.submit(_ws_worker(), name="ws-ingest")
        backfill_task = task_manager.submit(_periodic_backfill(), name="periodic-backfill")

        self._log.info("engine_background_workers_launched", tasks=2)

        # Wait forever (or until shutdown)
        try:
            await asyncio.gather(ws_task, backfill_task, return_exceptions=True)
        except asyncio.CancelledError:
            self._log.info("engine_shutdown_requested")
            await task_manager.shutdown()
            raise


# ---------------------------------------------------------------------------
# CLI-friendly entry points (used by run.py and `python -m prediction_alpha.pipeline`)
# ---------------------------------------------------------------------------


async def main_once(settings: Settings | None = None, **kwargs: Any) -> dict[str, Any]:
    engine = PredictionAlphaEngine(settings)
    return await engine.run_once(**kwargs)


async def main_forever(settings: Settings | None = None, **kwargs: Any) -> None:
    engine = PredictionAlphaEngine(settings)
    await engine.run_forever(**kwargs)
