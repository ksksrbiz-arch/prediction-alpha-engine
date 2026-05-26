"""Structured logging helpers."""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, cast

import structlog

# Correlation ID context variable (useful for tracing requests across logs)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def configure_logging(level: str = "INFO") -> None:
    """Configure JSON structured logs for VPS/Render-friendly ingestion workers."""

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    def add_correlation_id(logger, method_name, event_dict):
        cid = correlation_id_var.get("")
        if cid:
            event_dict["correlation_id"] = cid
        return event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **context: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger."""

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name).bind(**context))


def set_correlation_id(cid: str | None = None) -> str:
    """Set a correlation ID for the current context (useful in request handlers)."""
    if cid is None:
        cid = str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid
