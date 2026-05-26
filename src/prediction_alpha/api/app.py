"""FastAPI application factory for the Prediction Alpha Engine.

Productization note: ``create_app`` is a factory so tests and different deployment
targets (uvicorn, Render, Docker) can construct the app with custom settings.
When multi-user support lands the app will accept auth middleware and per-user
scoring config injection here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from prediction_alpha.api.routes import router
from prediction_alpha.api.tasks import task_manager
from prediction_alpha.config import Settings, get_settings
from prediction_alpha.feedback.loop import FeedbackLoop
from prediction_alpha.ingestion.storage import PostgresStore
from prediction_alpha.utils.logging import configure_logging


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks — wire up background task manager + feedback loop."""

    settings: Settings = app.state.settings
    configure_logging(settings.log_level)

    # Initialize feedback loop for calibration tracking
    store = PostgresStore(settings.database_url) if settings.database_url else None
    if store:
        try:
            feedback = FeedbackLoop(store)
            await feedback.ensure_schema()
            # Inject into routes (simple global for MVP)
            from prediction_alpha.api import routes as api_routes
            api_routes._feedback_loop = feedback
        except Exception:
            pass  # non-fatal

    yield
    await task_manager.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI application."""

    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Prediction Alpha Engine",
        version="0.1.0",
        description=(
            "Sovereign prediction-market opportunity scout.  "
            "Surfaces filtered, scored events from Kalshi (and future platforms)."
        ),
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.include_router(router)
    return app
