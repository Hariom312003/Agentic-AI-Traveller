"""
Graph + checkpointing integration tests (review issue #1).

Runs the real compiled graphs end-to-end with ZERO LLM provider keys
configured, on purpose: this is both the hardest case (every agent must
gracefully degrade) and the one that doesn't need network access or a paid
API key to run in CI. It exercises the rule-based fallback planner, the
keyword-based query fallback, and the rule-based refinement fallback all
at once — see src/planning_engine/fallback_planner.py and
src/agents/refinement_agent.py.
"""
from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.models.request import TripRequest
from src.models.state import new_state


@pytest.fixture()
def graph_manager(tmp_path, monkeypatch):
    """Fresh GraphManager per test, pointed at an isolated tmp_path so tests
    never share checkpoint/vector-store state with each other or with a
    real local dev environment."""
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma_db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    get_settings.cache_clear()

    import src.agents.rag_agent as rag_agent_module
    import src.graph.workflow as workflow_module
    import src.memory.store as memory_store_module

    rag_agent_module._retriever_singleton = None
    workflow_module._manager_singleton = None
    memory_store_module._store_singleton = None

    from src.graph.workflow import get_graph_manager

    manager = get_graph_manager()
    yield manager
    manager.close()


def _make_trip_state(session_id: str, destination: str = "Goa", duration_days: int = 3):
    trip_request = TripRequest(
        raw_query=f"Plan a {duration_days} day budget trip to {destination} with nightlife and culture",
        user_id="test-user",
        session_id=session_id,
        destination=destination,
        duration_days=duration_days,
        travelers_count=2,
        interests=["nightlife", "culture"],
    )
    return new_state(mode="plan", trip_request=trip_request, user_id="test-user", session_id=session_id)


def test_plan_graph_runs_end_to_end_with_zero_api_keys(graph_manager):
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    state = _make_trip_state(session_id)
    result = graph_manager.plan_graph.invoke(state, config=graph_manager.config_for(session_id))

    assert result["itinerary"] is not None
    assert len(result["itinerary"].days) == 3
    assert result["budget_breakdown"].total > 0
    assert result["validation_report"] is not None
    assert result["final_response"] is not None
    assert len(result["execution_trace"]) >= 7  # at least one record per core agent


def test_checkpoint_history_is_recorded(graph_manager):
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    state = _make_trip_state(session_id)
    graph_manager.plan_graph.invoke(state, config=graph_manager.config_for(session_id))

    history = graph_manager.history(session_id)
    assert len(history) > 1, "expected multiple checkpoints (one per superstep), not a single stateless run"


def test_rollback_recovers_an_earlier_itinerary_version(graph_manager):
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    state = _make_trip_state(session_id)
    config = graph_manager.config_for(session_id)

    result = graph_manager.plan_graph.invoke(state, config=config)
    history_after_plan = graph_manager.history(session_id)
    early_checkpoint_id = history_after_plan[-1].config["configurable"]["checkpoint_id"]  # oldest = last in history

    refine_state = dict(state)
    refine_state.update({
        "mode": "refine", "itinerary": result["itinerary"], "refinement_instruction": "Replace day 2",
        "target_days": [2], "planner_attempts": 0, "validation_report": None, "execution_trace": [],
    })
    refined_result = graph_manager.refine_graph.invoke(refine_state, config=config)
    assert refined_result["itinerary"].version > result["itinerary"].version

    rollback_cfg = graph_manager.rollback_config(session_id, early_checkpoint_id)
    assert rollback_cfg is not None
    snapshot = graph_manager.plan_graph.get_state(rollback_cfg)
    # the earliest checkpoint predates the planner even running once
    assert snapshot.values.get("itinerary") is None or snapshot.values["itinerary"].version == 1


def test_rollback_with_unknown_checkpoint_id_returns_none(graph_manager):
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    state = _make_trip_state(session_id)
    graph_manager.plan_graph.invoke(state, config=graph_manager.config_for(session_id))
    assert graph_manager.rollback_config(session_id, "not-a-real-checkpoint-id") is None


def test_refine_graph_preserves_locks_on_untouched_days(graph_manager):
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    state = _make_trip_state(session_id, duration_days=3)
    config = graph_manager.config_for(session_id)

    result = graph_manager.plan_graph.invoke(state, config=config)
    original = result["itinerary"]
    original_hashes = original.day_hashes()

    refine_state = dict(state)
    refine_state.update({
        "mode": "refine", "itinerary": original, "refinement_instruction": "Replace day 2 with adventure activities",
        "target_days": [2], "planner_attempts": 0, "validation_report": None, "execution_trace": [],
    })
    refined_result = graph_manager.refine_graph.invoke(refine_state, config=config)
    updated = refined_result["itinerary"]

    assert updated.day(1).content_hash() == original_hashes[1]
    assert updated.day(3).content_hash() == original_hashes[3]
    assert updated.day(1).locked is True
    assert updated.day(3).locked is True


def test_two_different_sessions_do_not_share_checkpoint_history(graph_manager):
    session_a = f"test-a-{uuid.uuid4().hex[:8]}"
    session_b = f"test-b-{uuid.uuid4().hex[:8]}"
    graph_manager.plan_graph.invoke(_make_trip_state(session_a), config=graph_manager.config_for(session_a))
    graph_manager.plan_graph.invoke(_make_trip_state(session_b), config=graph_manager.config_for(session_b))

    state_a = graph_manager.get_state(session_a)
    state_b = graph_manager.get_state(session_b)
    assert state_a.values["itinerary"].itinerary_id != state_b.values["itinerary"].itinerary_id
