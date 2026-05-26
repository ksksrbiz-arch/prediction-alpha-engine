"""Short-term memory for the agentic layer (with optional persistence).

Allows agents to recall recent high-value opportunities they have already
researched. This reduces repetition, surfaces pattern recognition across
similar macro or policy events, and makes the overall system feel more
"experienced" over a trading session or day.

Persistence options (configured via Settings):
- "none"   : pure in-memory (original behavior, great for short sessions)
- "file"   : append-only JSON file (excellent for single-process VPS use)
- "postgres": uses the existing PostgresStore infrastructure (best for
             multi-worker or long-lived deployments)

Persistence is best-effort. Agent runs never fail because memory could not
be saved or loaded.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from prediction_alpha.utils.logging import get_logger

_log = get_logger("agents.memory")

PersistMode = Literal["none", "file", "postgres"]


@dataclass
class MemoryEntry:
    """A single remembered research episode."""

    event_id: str
    category: str
    title: str
    edge: float
    composite: float
    summary: str  # short thesis or key insight
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class ShortTermAgentMemory:
    """Ring buffer of recent agent research episodes (with optional persistence).

    Usage:
        mem = ShortTermAgentMemory(max_entries=20, persist_mode="file",
                                   persist_path="/data/agent_memory.jsonl")
        mem.remember(entry)
        similar = mem.recall_similar(category="econ", limit=3)
    """

    def __init__(
        self,
        max_entries: int = 25,
        *,
        persist_mode: PersistMode = "none",
        persist_path: str | Path | None = None,
        # postgres_store is injected by the engine / orchestrator when using "postgres"
        postgres_store: Any | None = None,
    ) -> None:
        self.max_entries = max_entries
        self.persist_mode: PersistMode = persist_mode
        self.persist_path = Path(persist_path) if persist_path else None
        self._postgres_store = postgres_store

        self._buffer: deque[MemoryEntry] = deque(maxlen=max_entries)
        self._log = get_logger("short_term_memory")

        # Best-effort load on startup
        if self.persist_mode != "none":
            self._load_persisted()

    # ------------------------------------------------------------------
    # Core API (unchanged for callers)
    # ------------------------------------------------------------------

    def remember(self, entry: MemoryEntry) -> None:
        self._buffer.appendleft(entry)
        self._log.debug(
            "memory_stored",
            event_id=entry.event_id,
            category=entry.category,
            buffer_size=len(self._buffer),
        )
        # Best-effort persistence (never raises to caller)
        try:
            self._persist_entry(entry)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("memory_persist_failed", error=str(exc)[:150])

    def recall_similar(
        self,
        *,
        category: str | None = None,
        keywords: list[str] | None = None,
        limit: int = 3,
    ) -> list[MemoryEntry]:
        """Return the most recent entries that match category or simple keyword overlap."""
        results: list[MemoryEntry] = []
        kws = [k.lower() for k in (keywords or [])]

        for entry in self._buffer:
            if category and entry.category.lower() != category.lower():
                continue
            if kws:
                text = f"{entry.title} {entry.summary}".lower()
                if not any(kw in text for kw in kws):
                    continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def get_recent(self, limit: int = 5) -> list[MemoryEntry]:
        return list(self._buffer)[:limit]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Persistence helpers (best effort)
    # ------------------------------------------------------------------

    def _load_persisted(self) -> None:
        """Load recent entries from the chosen persistence backend."""
        if self.persist_mode == "file" and self.persist_path:
            try:
                if self.persist_path.exists():
                    with open(self.persist_path) as f:
                        lines = f.readlines()[-self.max_entries :]
                        for line in lines:
                            data = json.loads(line)
                            entry = MemoryEntry(**data)
                            self._buffer.appendleft(entry)
                    self._log.info("memory_loaded_from_file", count=len(self._buffer))
            except Exception as exc:  # noqa: BLE001
                self._log.warning("memory_file_load_failed", error=str(exc)[:150])

        elif self.persist_mode == "postgres" and self._postgres_store:
            # Postgres loading is done by the caller (engine) after schema exists.
            # We keep the hook here for future direct support.
            pass

    def _persist_entry(self, entry: MemoryEntry) -> None:
        if self.persist_mode == "file" and self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "event_id": entry.event_id,
                "category": entry.category,
                "title": entry.title,
                "edge": entry.edge,
                "composite": entry.composite,
                "summary": entry.summary,
                "timestamp": entry.timestamp.isoformat(),
            }
            with open(self.persist_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        elif self.persist_mode == "postgres" and self._postgres_store:
            # The engine will handle async persistence via the store when available.
            # For now we just log the intent (real implementation is in the wiring layer).
            self._log.debug("memory_persist_postgres_queued", event_id=entry.event_id)


# Module-level default memory instance (shared across the process for a single run)
_default_memory = ShortTermAgentMemory()


def get_default_memory() -> ShortTermAgentMemory:
    """Return the process-wide short-term memory instance (in-memory only by default)."""
    return _default_memory


def create_persistent_memory(
    *,
    max_entries: int = 50,
    persist_mode: PersistMode = "file",
    persist_path: str | Path | None = None,
    postgres_store: Any | None = None,
) -> ShortTermAgentMemory:
    """Factory for a memory instance with persistence enabled.

    Used by the engine / orchestrator when the user configures memory persistence.
    """
    return ShortTermAgentMemory(
        max_entries=max_entries,
        persist_mode=persist_mode,
        persist_path=persist_path,
        postgres_store=postgres_store,
    )
