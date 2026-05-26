"""Starter hybrid scorer: rule gate + transparent heuristic edge estimate.

Productization note: all thresholds, weights, and category boosts are driven by
``ScoringConfig`` (env vars or YAML).  This config/filter will support per-user
profiles later — each profile will carry its own ``ScoringConfig`` and the scorer
will be instantiated per-request with the caller's config.
"""

from __future__ import annotations

from typing import TypeGuard

from prediction_alpha.config import ScoringConfig, Settings
from prediction_alpha.models import Event, OpportunityScore, RecommendedAction
from prediction_alpha.scoring.features import compute_features
from prediction_alpha.scoring.filters import ScoringRules, apply_hard_filters
from prediction_alpha.utils.logging import get_logger

# Feedback-loop logger: every scoring decision is recorded so Phase 3 retraining
# jobs can replay decisions and measure calibration drift.
_score_log = get_logger("scoring.feedback")


def _is_numeric(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float)


class HybridScorer:
    """Simple, inspectable scoring layer with a future ML hook.

    The heuristic intentionally errs conservative: it is useful for paper testing
    and can be replaced by calibrated scikit-learn/XGBoost models once feedback
    data exists.

    Productization note: this config/filter will support per-user profiles later.
    """

    def __init__(self, scoring_config: ScoringConfig) -> None:
        self.config = scoring_config
        self.rules = ScoringRules(
            min_liquidity_score=scoring_config.min_liquidity_score,
            min_volume_24h=scoring_config.min_volume_24h,
            max_days_to_resolution=scoring_config.max_days_to_resolution,
            allowed_categories=tuple(scoring_config.allowed_categories),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> HybridScorer:
        return cls(scoring_config=settings.build_scoring_config())

    def score(self, event: Event) -> OpportunityScore:
        features = compute_features(event)
        filter_result = apply_hard_filters(event, self.rules)
        implied_raw = features.get("implied_prob")
        if not _is_numeric(implied_raw):
            result = OpportunityScore(
                event_id=event.id,
                edge_score=0.0,
                liquidity_adjusted_ev=0.0,
                confidence=0.0,
                portfolio_fit=self._portfolio_fit(event),
                composite_score=0.0,
                recommended_action=RecommendedAction.REJECT,
                agent_plan_summary=None,
                passed_filter=False,
                rationale=["missing_implied_probability", *filter_result.reasons],
                features=features,
            )
            # Feedback-loop log: record every rejection for calibration analysis.
            _score_log.info(
                "score_computed",
                event_id=event.id,
                passed=False,
                composite=0.0,
                reason="missing_implied_probability",
            )
            return result
        implied_prob = float(implied_raw)
        model_prob = self._placeholder_model_probability(event, features)
        edge = model_prob - implied_prob
        liquidity_adjusted_ev = edge * event.liquidity_score
        confidence = min(0.35 + (event.liquidity_score * 0.35) + (abs(edge) * 0.30), 0.95)
        portfolio_fit = self._portfolio_fit(event)
        composite = self._composite_score(edge, liquidity_adjusted_ev, confidence, portfolio_fit)
        passed = (
            filter_result.passed
            and composite >= self.config.min_composite_score
            and edge > self.config.min_edge
        )
        action = self._recommended_action(passed, edge)
        rationale = [
            f"model_prob={model_prob:.3f}",
            f"implied_prob={implied_prob:.3f}",
            f"edge={edge:.3f}",
            *filter_result.reasons,
        ]
        result = OpportunityScore(
            event_id=event.id,
            edge_score=edge,
            liquidity_adjusted_ev=liquidity_adjusted_ev,
            confidence=confidence,
            portfolio_fit=portfolio_fit,
            composite_score=composite,
            recommended_action=action,
            agent_plan_summary=(
                "Phase 2 hook: spawn research agent for thesis/counter-thesis before capital."
                if passed
                else None
            ),
            passed_filter=passed,
            rationale=rationale,
            features=features,
        )
        # Feedback-loop log: captures all dimensions needed for Phase 3 retraining.
        _score_log.info(
            "score_computed",
            event_id=event.id,
            passed=passed,
            composite=round(composite, 4),
            edge=round(edge, 4),
            confidence=round(confidence, 4),
            action=action.value,
            category=event.category,
        )
        return result

    def _placeholder_model_probability(self, event: Event, features: dict[str, object]) -> float:
        """Transparent EV proxy until enough feedback exists for real ML calibration."""

        implied_raw = features.get("implied_prob")
        implied = float(implied_raw) if isinstance(implied_raw, int | float) else 0.5
        trend = features.get("volume_trend")
        trend_boost = 0.0
        if _is_numeric(trend):
            trend_boost = max(min(float(trend) * 0.02, 0.04), -0.04)
        liquidity_boost = (event.liquidity_score - 0.5) * 0.03
        # Productization note: category boosts should eventually be driven by
        # per-user domain expertise signals from the profile config.
        cw = self.config.category_weights
        category_boost = 0.015 if event.category in {"econ", "policy", "weather"} else 0.0
        _ = cw  # reserved for upcoming weighted-category model
        return max(min(implied + trend_boost + liquidity_boost + category_boost, 0.98), 0.02)

    def _portfolio_fit(self, event: Event) -> float:
        """Return portfolio-fit score driven by configurable category weights."""

        cw = self.config.category_weights
        weights: dict[str, float] = {
            "econ": cw.econ,
            "policy": cw.policy,
            "weather": cw.weather,
            "sports": cw.sports,
        }
        return weights.get(event.category, cw.default)

    def _composite_score(
        self, edge: float, liquidity_adjusted_ev: float, confidence: float, portfolio_fit: float
    ) -> float:
        w = self.config.composite_weights
        raw = (
            max(edge, 0.0) * w.edge
            + max(liquidity_adjusted_ev, 0.0) * w.liquidity_adjusted_ev
            + confidence * w.confidence
            + portfolio_fit * w.portfolio_fit
        )
        return max(min(raw, 1.0), 0.0)

    @staticmethod
    def _recommended_action(passed: bool, edge: float) -> RecommendedAction:
        if not passed:
            return RecommendedAction.REJECT
        if edge >= 0.05:
            return RecommendedAction.PAPER_YES
        return RecommendedAction.RESEARCH
