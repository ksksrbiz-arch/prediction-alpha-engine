"""Lightweight background task manager for Phase 1.

Provides an ``asyncio.create_task`` wrapper that tracks running tasks and
supports graceful shutdown.  This is the hook point for Phase 2 agents —
research tasks, plan generation, and notification dispatch will all be
submitted through ``task_manager.submit()``.

Productization note: in Phase 2 this can be swapped for a heavier job queue
(RQ, Celery, or arq) without changing the call sites.  The current design
deliberately stays asyncio-native to avoid new infrastructure dependencies
during paper testing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from prediction_alpha.utils.logging import get_logger

_log = get_logger("tasks")


class TaskManager:
    """Thin wrapper around ``asyncio.create_task`` with lifecycle tracking.

    Productization note: when per-user agent jobs land, each task will carry a
    ``profile_id`` for quota enforcement and result routing.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        """Schedule a coroutine as a tracked background task."""

        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        _log.info("task_submitted", task_name=name or task.get_name())
        return task

    def _on_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            _log.info("task_cancelled", task_name=task.get_name())
        elif exc := task.exception():
            _log.error("task_failed", task_name=task.get_name(), error=str(exc))
        else:
            _log.info("task_completed", task_name=task.get_name())

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel all running tasks and wait for them to finish."""

        if not self._tasks:
            return
        _log.info("task_manager_shutdown", pending=len(self._tasks))
        for task in self._tasks:
            task.cancel()
        await asyncio.wait(self._tasks, timeout=timeout)
        self._tasks.clear()

    @property
    def pending_count(self) -> int:
        return len(self._tasks)


# Module-level singleton so the FastAPI lifespan and route handlers share state.
task_manager = TaskManager()
