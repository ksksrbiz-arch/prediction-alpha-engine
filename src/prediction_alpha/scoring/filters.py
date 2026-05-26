"""Strict hard filters; low noise is non-negotiable.

Productization note: this config/filter will support per-user profiles later.
Each ``ScoringRules`` instance maps to a user profile, allowing different
liquidity thresholds, horizons, and category whitelists per subscriber.
"""

from dataclasses import dataclass, field

from prediction_alpha.models import Event, EventStatus


@dataclass(frozen=True)
class ScoringRules:
    """Hard-filter thresholds.

    Productization note: these values come from ``ScoringConfig`` (YAML or env)
    and will be overridable per-user profile in Phase 4.
    """

    min_liquidity_score: float = 0.20
    min_volume_24h: float = 100.0
    max_days_to_resolution: int = 60
    allowed_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def apply_hard_filters(event: Event, rules: ScoringRules) -> FilterResult:
    reasons: list[str] = []
    if event.status not in {EventStatus.OPEN, EventStatus.UNKNOWN}:
        reasons.append(f"status={event.status.value}")
    if event.liquidity_score < rules.min_liquidity_score:
        reasons.append(f"liquidity<{rules.min_liquidity_score}")
    if event.volume_24h < rules.min_volume_24h:
        reasons.append(f"volume_24h<{rules.min_volume_24h}")
    days = event.days_to_resolution
    if days is not None and days > rules.max_days_to_resolution:
        reasons.append(f"horizon>{rules.max_days_to_resolution}d")
    if rules.allowed_categories and event.category not in rules.allowed_categories:
        reasons.append(f"category_not_allowed={event.category}")
    return FilterResult(passed=not reasons, reasons=reasons)
