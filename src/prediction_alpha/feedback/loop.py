"""Core Feedback & Self-Improvement Loop.

Logs resolutions, computes running calibration metrics, and provides hooks
for simple retraining / prompt adaptation.

Usage in pipeline or a periodic job:
    feedback = FeedbackLoop(store)
    feedback.log_resolution(event, actual_outcome=1.0 or 0.0)
    metrics = feedback.get_calibration_summary()
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import mean
from typing import Any

from prediction_alpha.ingestion.storage import PostgresStore
from prediction_alpha.models import Event
from prediction_alpha.utils.logging import get_logger

_log = get_logger("feedback.loop")


@dataclass
class ResolutionRecord:
    event_id: str
    platform: str
    category: str
    predicted: float  # model's or market's implied prob at scoring time
    actual: float     # 0.0 or 1.0
    composite_at_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class FeedbackLoop:
    """Lightweight feedback manager.

    In production this would persist to DB and run as a background job.
    For MVP we keep an in-memory buffer + optional Postgres persistence.
    """

    def __init__(self, store: PostgresStore | None = None):
        self.store = store
        self._records: list[ResolutionRecord] = []
        self._log = get_logger("feedback")

    async def ensure_schema(self) -> None:
        if not self.store:
            return
        async with self.store.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_resolutions (
                    id BIGSERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    category TEXT,
                    predicted DOUBLE PRECISION NOT NULL,
                    actual DOUBLE PRECISION NOT NULL,
                    composite_at_time DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_event ON feedback_resolutions (event_id);
                """
            )

    def log_resolution(
        self,
        event: Event,
        actual_outcome: float,
        predicted_prob: float | None = None,
        composite_at_time: float | None = None,
    ) -> None:
        """Record the actual resolution of a market.

        Call this when you learn the outcome (from Kalshi/Polymarket settlement feed
        or manual input).
        """
        if predicted_prob is None:
            # Fall back to the last known implied_prob if not provided
            predicted_prob = event.implied_prob or 0.5

        rec = ResolutionRecord(
            event_id=event.id,
            platform=event.platform.value,
            category=event.category,
            predicted=float(predicted_prob),
            actual=float(actual_outcome),
            composite_at_time=composite_at_time or 0.0,
        )
        self._records.append(rec)

        _log.info(
            "resolution_logged",
            event_id=event.id,
            platform=rec.platform,
            predicted=round(rec.predicted, 3),
            actual=rec.actual,
            brier_contrib=round((rec.predicted - rec.actual) ** 2, 4),
        )

        # Best-effort async persist (fire-and-forget in real use)
        # For now just log; a real job would call an async version.

    def get_calibration_summary(self, last_n: int | None = None) -> dict[str, Any]:
        """Return running calibration metrics (Brier score primary)."""
        records = self._records[-last_n:] if last_n else self._records
        if not records:
            return {"count": 0, "brier": None}

        briers = [(r.predicted - r.actual) ** 2 for r in records]
        brier = mean(briers)

        by_category: dict[str, list[float]] = defaultdict(list)
        for r in records:
            by_category[r.category].append((r.predicted - r.actual) ** 2)

        cat_brier = {cat: round(mean(vals), 4) for cat, vals in by_category.items()}

        return {
            "count": len(records),
            "brier_score": round(brier, 4),
            "brier_by_category": cat_brier,
            "note": "Lower Brier is better. 0 = perfect calibration.",
            "last_updated": datetime.now(UTC).isoformat(),
        }

    async def run_simple_recalibration(self) -> dict[str, Any]:
        """Very lightweight 'retraining' hook.

        In a real system this could:
        - Adjust category weights in ScoringConfig
        - Suggest prompt changes for agents
        - Trigger a scikit-learn / XGBoost retrain on logged (features, actual) pairs

        For MVP we just compute fresh metrics and log recommendations.
        """
        summary = self.get_calibration_summary()
        if summary["count"] < 20:
            return {"action": "collect_more_data", "current": summary}

        # Toy example: if certain categories have bad Brier, suggest boosting their weight
        recommendations = []
        for cat, b in summary.get("brier_by_category", {}).items():
            if b > 0.25:  # arbitrary threshold
                recommendations.append(f"Increase portfolio_fit weight for category '{cat}'")

        _log.info("recalibration_run", brier=summary["brier_score"], recs=recommendations)

        return {
            "action": "recommendations_generated",
            "metrics": summary,
            "recommendations": recommendations or ["No major adjustments suggested"],
        }
