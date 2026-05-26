"""Starter hybrid scorer: rule gate + transparent heuristic edge estimate."""

from __future__ import annotations

from prediction_alpha.config import Settings
from prediction_alpha.models import Event, OpportunityScore, RecommendedAction
from prediction_alpha.scoring.features import compute_features
from prediction_alpha.scoring.filters import ScoringRules, apply_hard_filters


class HybridScorer:
    """Simple, inspectable scoring layer with a future ML hook.

    The heuristic intentionally errs conservative: it is useful for paper testing
    and can be replaced by calibrated scikit-learn/XGBoost models once feedback
    data exists.
    """

    def __init__(self, rules: ScoringRules, min_composite_score: float = 0.55) -> None:
        self.rules = rules
        self.min_composite_score = min_composite_score

    @classmethod
    def from_settings(cls, settings: Settings) -> HybridScorer:
        return cls(
            rules=ScoringRules(
                min_liquidity_score=settings.min_liquidity_score,
                min_volume_24h=settings.min_volume_24h,
                max_days_to_resolution=settings.max_days_to_resolution,
                allowed_categories=tuple(settings.allowed_categories),
            ),
            min_composite_score=settings.min_composite_score,
        )

    def score(self, event: Event) -> OpportunityScore:
        features = compute_features(event)
        filter_result = apply_hard_filters(event, self.rules)
        implied_raw = features.get("implied_prob")
        if not isinstance(implied_raw, int | float):
            return OpportunityScore(
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
        implied_prob = float(implied_raw)
        model_prob = self._placeholder_model_probability(event, features)
        edge = model_prob - implied_prob
        liquidity_adjusted_ev = edge * event.liquidity_score
        confidence = min(0.35 + (event.liquidity_score * 0.35) + (abs(edge) * 0.30), 0.95)
        portfolio_fit = self._portfolio_fit(event)
        composite = self._composite_score(edge, liquidity_adjusted_ev, confidence, portfolio_fit)
        passed = filter_result.passed and composite >= self.min_composite_score and edge > 0.02
        action = self._recommended_action(passed, edge)
        rationale = [
            f"model_prob={model_prob:.3f}",
            f"implied_prob={implied_prob:.3f}",
            f"edge={edge:.3f}",
            *filter_result.reasons,
        ]
        return OpportunityScore(
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

    def _placeholder_model_probability(self, event: Event, features: dict[str, object]) -> float:
        """Transparent EV proxy until enough feedback exists for real ML calibration."""

        implied_raw = features.get("implied_prob")
        implied = float(implied_raw) if isinstance(implied_raw, int | float) else 0.5
        trend = features.get("volume_trend")
        trend_boost = 0.0
        if isinstance(trend, int | float):
            trend_boost = max(min(float(trend) * 0.02, 0.04), -0.04)
        liquidity_boost = (event.liquidity_score - 0.5) * 0.03
        category_boost = 0.015 if event.category in {"econ", "policy", "weather"} else 0.0
        return max(min(implied + trend_boost + liquidity_boost + category_boost, 0.98), 0.02)

    @staticmethod
    def _portfolio_fit(event: Event) -> float:
        if event.category in {"econ", "policy", "weather"}:
            return 0.75
        if event.category == "sports":
            return 0.35
        return 0.50

    @staticmethod
    def _composite_score(
        edge: float, liquidity_adjusted_ev: float, confidence: float, portfolio_fit: float
    ) -> float:
        raw = (
            max(edge, 0.0) * 3.0
            + max(liquidity_adjusted_ev, 0.0) * 4.0
            + confidence * 0.25
            + portfolio_fit * 0.20
        )
        return max(min(raw, 1.0), 0.0)

    @staticmethod
    def _recommended_action(passed: bool, edge: float) -> RecommendedAction:
        if not passed:
            return RecommendedAction.REJECT
        if edge >= 0.05:
            return RecommendedAction.PAPER_YES
        return RecommendedAction.RESEARCH
