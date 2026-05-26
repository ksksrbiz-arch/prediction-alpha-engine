"""Lightweight resilience utilities for production hardening.

Includes a simple circuit breaker suitable for external API calls (Kalshi, Ollama, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    name: str = "default"

    def __post_init__(self):
        self._failures = 0
        self._last_failure_time = 0.0
        self._state = CircuitState.CLOSED

    def call(self, func: Callable[[], T]) -> T:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
            else:
                raise RuntimeError(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failures = 0
            return result
        except Exception:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
            raise
