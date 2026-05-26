"""Selective, low-noise notification system.

Design principles (Cathedral + attention protection):
- Nothing reaches a human unless it passed the sacred multi-stage filter + agent
  research + final high composite threshold.
- Multiple channels supported; console is always on for observability.
- Email is a stub that logs the full message and optionally sends via SMTP
  (no external deps beyond stdlib).
- Future: Telegram, UnifyOne webhook, push — all behind the same interface.
"""

from __future__ import annotations

import asyncio
import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from enum import StrEnum
from typing import Any

from prediction_alpha.config import Settings, get_settings
from prediction_alpha.models import Event, OpportunityScore
from prediction_alpha.utils.logging import get_logger

_log = get_logger("notifications")


class NotificationChannel(StrEnum):
    CONSOLE = "console"
    EMAIL = "email"
    # TELEGRAM = "telegram"  # future


@dataclass
class Notification:
    """A single high-signal alert ready for dispatch."""

    title: str
    body: str
    event_id: str
    composite_score: float
    channels: list[NotificationChannel]
    metadata: dict[str, Any]


class Notifier:
    """Central dispatcher. Respects global kill switch and min-score threshold."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._log = get_logger("notifier")

    def should_notify(self, score: OpportunityScore) -> bool:
        if not self.settings.notifications_enabled:
            return False
        return (
            score.passed_filter
            and score.composite_score >= self.settings.notify_min_composite
        )

    def build_notification(
        self, event: Event, score: OpportunityScore, agent_brief: dict[str, Any] | None = None
    ) -> Notification:
        """Create a polished, actionable notification payload."""

        thesis = ""
        if agent_brief:
            thesis = agent_brief.get("thesis", "")[:220]

        title = f"ALPHA {score.composite_score:.0%} | {event.title[:70]}"
        body = self._format_body(event, score, thesis)

        channels: list[NotificationChannel] = [NotificationChannel.CONSOLE]
        if self.settings.notify_email_to and self.settings.smtp_host:
            channels.append(NotificationChannel.EMAIL)

        return Notification(
            title=title,
            body=body,
            event_id=event.id,
            composite_score=score.composite_score,
            channels=channels,
            metadata={
                "edge": round(score.edge_score, 3),
                "category": event.category,
                "days": event.days_to_resolution,
                "action": score.recommended_action.value,
            },
        )

    def _format_body(self, event: Event, score: OpportunityScore, thesis: str) -> str:
        lines = [
            f"Event: {event.title}",
            f"Platform: {event.platform.value} | Ticker: {event.external_id}",
            f"Category: {event.category} | Implied: {event.implied_prob:.1%}" if event.implied_prob else f"Category: {event.category}",
            f"Edge: {score.edge_score:+.2%} | Composite: {score.composite_score:.2%}",
            f"Liquidity: {event.liquidity_score:.2f} | Horizon: {event.days_to_resolution:.1f}d" if event.days_to_resolution else "",
            f"Recommended: {score.recommended_action.value.upper()}",
            "",
            "Rationale:",
        ]
        for r in score.rationale[:5]:
            lines.append(f"  - {r}")
        if thesis:
            lines.extend(["", "Agent Thesis (excerpt):", thesis])
        lines.extend([
            "",
            "Links: https://kalshi.com/markets (search ticker)",
            "Next: Review agent brief in logs or /opportunities API. Paper trade first.",
        ])
        return "\n".join(lines)

    async def dispatch(self, notif: Notification) -> None:
        """Fire all requested channels. Never raises — notifications are best-effort."""

        for ch in notif.channels:
            try:
                if ch == NotificationChannel.CONSOLE:
                    self._send_console(notif)
                elif ch == NotificationChannel.EMAIL:
                    await self._send_email(notif)
            except Exception as exc:  # noqa: BLE001
                self._log.error("notification_channel_failed", channel=ch.value, error=str(exc))

        self._log.info(
            "notification_dispatched",
            event_id=notif.event_id,
            score=round(notif.composite_score, 3),
            channels=[c.value for c in notif.channels],
        )

    def _send_console(self, notif: Notification) -> None:
        # Structured log + pretty print for humans watching the terminal
        print("\n" + "=" * 70)
        print(f"🔔  {notif.title}")
        print("-" * 70)
        print(notif.body)
        print("=" * 70 + "\n")

    async def _send_email(self, notif: Notification) -> None:
        if not self.settings.notify_email_to:
            return

        recipients = [r.strip() for r in self.settings.notify_email_to.split(",") if r.strip()]
        if not recipients:
            return

        msg = MIMEText(notif.body, "plain", "utf-8")
        msg["Subject"] = f"[Prediction Alpha] {notif.title}"
        msg["From"] = self.settings.smtp_from
        msg["To"] = ", ".join(recipients)

        # Run blocking smtplib in thread to keep async clean
        def _send() -> None:
            if not self.settings.smtp_host:
                self._log.info("email_stub_only_no_smtp_host", subject=msg["Subject"])
                return
            with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=10) as smtp:
                smtp.starttls()
                if self.settings.smtp_user and self.settings.smtp_password:
                    smtp.login(self.settings.smtp_user, self.settings.smtp_password)
                smtp.sendmail(self.settings.smtp_from, recipients, msg.as_string())

        try:
            await asyncio.to_thread(_send)
            self._log.info("email_sent", to=recipients, subject=msg["Subject"])
        except Exception as exc:  # noqa: BLE001
            self._log.warning("email_send_failed_stub_logged", error=str(exc)[:120])
            # Still log the full body so nothing is lost
            self._log.info("email_content_fallback", body=notif.body[:800])


def get_notifier(settings: Settings | None = None) -> Notifier:
    """Singleton-friendly accessor."""
    return Notifier(settings)
