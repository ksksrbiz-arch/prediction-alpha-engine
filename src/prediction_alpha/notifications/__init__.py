"""Selective notification layer + True Neutral Brain integration prep.

Only the very best opportunities (after agents + final filter) ever reach a human.
Everything else is logged for feedback and silently discarded.
"""

from prediction_alpha.notifications.notifier import (
    Notification,
    NotificationChannel,
    get_notifier,
)
from prediction_alpha.notifications.brain import prepare_brain_payload

__all__ = [
    "Notification",
    "NotificationChannel",
    "get_notifier",
    "prepare_brain_payload",
]
