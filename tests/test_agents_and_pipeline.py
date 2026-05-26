"""Tests for the agentic legwork layer and high-level pipeline orchestration."""

from datetime import UTC, datetime, timedelta

from prediction_alpha.agents.legwork import stub_research_brief, AgentOrchestrator
from prediction_alpha.agents import AgentOrchestrator as AgentOrchestratorV2  # for v2 tests
from prediction_alpha.agents.memory import MemoryEntry
from prediction_alpha.config import Settings
from prediction_alpha.models import Event, EventStatus, OpportunityScore, Platform, RecommendedAction
from prediction_alpha.notifications.notifier import get_notifier
from prediction_alpha.pipeline import PredictionAlphaEngine


def _make_high_value_event() -> Event:
    return Event(
        id="kalshi-test-alpha-1",
        platform=Platform.KALSHI,
        external_id="TEST-ALPHA-1",
        title="Will the Fed cut 25bp at the next meeting?",
        category="econ",
        yes_price=0.38,
        no_price=0.62,
        implied_prob=0.38,
        volume_24h=9500.0,
        open_interest=14500.0,
        liquidity_score=0.71,
        resolution_date=datetime.now(UTC) + timedelta(days=12),
        status=EventStatus.OPEN,
    )


def _make_high_value_score(event: Event) -> OpportunityScore:
    return OpportunityScore(
        event_id=event.id,
        edge_score=0.08,
        liquidity_adjusted_ev=0.056,
        confidence=0.78,
        portfolio_fit=0.75,
        composite_score=0.69,
        recommended_action=RecommendedAction.PAPER_YES,
        passed_filter=True,
        rationale=["macro tailwind", "liquidity ok", "portfolio aligned"],
        features={"implied_prob": 0.38},
    )


def test_stub_agent_produces_structured_brief() -> None:
    ev = _make_high_value_event()
    sc = _make_high_value_score(ev)
    brief = stub_research_brief(ev, sc)
    assert brief.event_id == ev.id
    assert len(brief.thesis) > 20
    assert len(brief.counter_thesis) > 10
    assert brief.confidence_in_edge > 0.2


def test_agent_orchestrator_enriches_score() -> None:
    settings = Settings(agent_enabled=False)  # force stub path
    orch = AgentOrchestrator(settings)
    ev = _make_high_value_event()
    sc = _make_high_value_score(ev)

    # Run the enrich path (should not raise even with agents disabled)
    import asyncio
    enriched = asyncio.run(orch.enrich_score_with_plan(ev, sc))
    assert enriched.passed_filter is True
    assert enriched.agent_plan_summary is not None


def test_notifier_selective_gate() -> None:
    settings = Settings(notifications_enabled=True, notify_min_composite=0.68)
    notifier = get_notifier(settings)
    ev = _make_high_value_event()
    sc = _make_high_value_score(ev)  # 0.69 > 0.68

    assert notifier.should_notify(sc) is True

    sc_low = sc.model_copy(update={"composite_score": 0.55})
    assert notifier.should_notify(sc_low) is False


def test_engine_run_once_smoke_no_crash() -> None:
    """The engine must be importable and runnable even without DB or LLM."""
    settings = Settings(
        environment="test",
        database_url="postgresql://fake",  # will gracefully degrade
        agent_enabled=False,
        notifications_enabled=True,
        notify_min_composite=0.99,  # nothing will notify in test data
    )
    engine = PredictionAlphaEngine(settings)

    import asyncio
    result = asyncio.run(engine.run_once(max_pages=1, status="open"))
    assert "processed" in result
    assert result["processed"] >= 0


# ---------------------------------------------------------------------------
# Hardened v2 agent layer tests
# ---------------------------------------------------------------------------


def test_stub_produces_rich_v2_brief() -> None:
    from prediction_alpha.agents import stub_research_brief

    ev = _make_high_value_event()
    sc = _make_high_value_score(ev)
    brief = stub_research_brief(ev, sc)

    assert brief.event_id == ev.id
    assert brief.debate_summary is not None
    assert len(brief.weaknesses) >= 1 or "Stub" in (brief.debate_summary or "")
    assert brief.agent_version.startswith("2.")


def test_tools_knowledge_and_registry() -> None:
    from prediction_alpha.agents.tools import KnowledgeTool, ToolRegistry, get_default_registry

    reg = ToolRegistry()
    assert "knowledge_base" in [t.name for t in reg.list_enabled()]

    tool = KnowledgeTool()
    import asyncio

    res = asyncio.run(tool.run(category="econ"))
    assert "Fed" in res.content or "inflation" in res.content.lower()
    assert res.confidence > 0.7


def test_memory_recall() -> None:
    from prediction_alpha.agents.memory import MemoryEntry, ShortTermAgentMemory

    mem = ShortTermAgentMemory(max_entries=5)
    mem.remember(MemoryEntry("e1", "econ", "Fed cut", 0.08, 0.65, "Strong housing tailwind"))
    mem.remember(MemoryEntry("e2", "policy", "Tariff", -0.03, 0.40, "Ag input costs"))

    econ = mem.recall_similar(category="econ")
    assert len(econ) == 1
    assert "housing" in econ[0].summary.lower()


def test_hardened_orchestrator_stub_path_and_rich_output() -> None:
    """Even in stub mode (or no LLM), the new rich fields are populated."""
    settings = Settings(agent_enabled=False)  # forces stub
    orch = AgentOrchestrator(settings)

    ev = _make_high_value_event()
    sc = _make_high_value_score(ev)

    import asyncio

    brief = asyncio.run(orch.run_for_score(ev, sc))

    assert brief.event_id == ev.id
    assert brief.steps_taken == 0  # stub path
    assert "Stub" in (brief.debate_summary or "") or brief.debate_summary is not None
    assert isinstance(brief.tool_calls, list)


def test_agent_config_backend_and_persist_fields() -> None:
    from prediction_alpha.agents.config import AgentConfig
    from prediction_alpha.config import Settings

    cfg = AgentConfig(backend="auto")
    assert cfg.backend in {"auto", "python", "langgraph"}

    s = Settings(agent_backend="python", agent_memory_persist="file")
    assert s.agent_backend == "python"
    assert s.agent_memory_persist == "file"


def test_persistent_memory_factory() -> None:
    from prediction_alpha.agents.memory import create_persistent_memory

    mem = create_persistent_memory(persist_mode="none")
    assert len(mem) == 0
    mem.remember(MemoryEntry("x1", "econ", "Test", 0.1, 0.6, "demo"))
    assert len(mem) == 1
