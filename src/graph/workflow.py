"""
LangGraph workflow construction.

REVIEW FIX (#1 — "No checkpointing or rollback in the LangGraph pipeline"):
the previous graph was `.compile()`d with no checkpointer at all, so every
run was stateless — there was nothing to roll back TO, and a crash mid-run
lost everything. Here, both graphs are compiled with a shared, persistent
`SqliteSaver`, keyed by `session_id` as LangGraph's `thread_id`. That gets
us, for free, from LangGraph itself:

- Automatic checkpointing after every node ("superstep").
- `GraphManager.history(session_id)` — list every checkpoint for a session
  (used by the /rollback API to show the user what they can revert to).
- `GraphManager.rollback(session_id, checkpoint_id)` — fork execution back
  to an earlier checkpoint's state, so a bad refinement can be undone
  instead of being permanent.

Two separate compiled graphs (`plan_graph`, `refine_graph`) share the same
node functions and the same checkpointer/thread_id space, so a session's
history is continuous across an initial plan and any number of later
refinements — not two disconnected systems that both happen to use the
word "checkpoint".
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator, Literal

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.budget_agent import run_budget_agent
from src.agents.memory_agent import run_memory_agent, run_memory_update_agent
from src.agents.planner_agent import run_planner_agent
from src.agents.query_agent import run_query_agent
from src.agents.rag_agent import run_rag_agent
from src.agents.refinement_agent import run_refinement_agent
from src.agents.rewards_agent import run_rewards_agent
from src.agents.summary_agent import run_summary_agent
from src.agents.validator_agent import run_validator_agent
from src.agents.evaluator_agent import run_evaluator_agent
from src.config import get_settings
from src.models.budget import BudgetBreakdown, RewardRecommendation, RewardsSummary
from src.models.itinerary import Activity, ActivityCategory, ActivitySource, Day, Itinerary, TimeSlotName
from src.models.request import RefinementRequest, RollbackRequest, TripRequest
from src.models.state import TripState
from src.models.user import BehavioralPreferences, PastTrip, UserProfile
from src.models.validation import IssueCategory, IssueSeverity, ValidationIssue, ValidationReport
from src.monitoring.telemetry import AgentExecutionRecord

# Every custom type that can end up inside `TripState` needs to be
# explicitly allow-listed for the checkpoint serializer. Without this,
# LangGraph's default msgpack codec still works today but emits a
# "will be blocked in a future version" warning per type (permissive mode) —
# explicitly listing our own types now is the forward-compatible fix, and
# also the more secure one: `allowed_msgpack_modules=True` (the permissive
# default) would happily deserialize *any* importable class from the
# checkpoint file, not just ours.
from src.models.trip_summary import TripSummary, TripOverview, TripHighlights, BudgetSummary, QuickStatistics, WeatherOverview, FoodRecommendations, TransportationSummary, ImportantTravelTips

_CHECKPOINT_ALLOWED_TYPES = [
    TripRequest, RefinementRequest, RollbackRequest,
    UserProfile, BehavioralPreferences, PastTrip,
    Activity, ActivityCategory, ActivitySource, Day, Itinerary, TimeSlotName,
    BudgetBreakdown, RewardRecommendation, RewardsSummary,
    ValidationIssue, ValidationReport, IssueCategory, IssueSeverity,
    AgentExecutionRecord,
    TripSummary, TripOverview, TripHighlights, BudgetSummary, QuickStatistics, WeatherOverview, FoodRecommendations, TransportationSummary, ImportantTravelTips,
]


def _build_checkpoint_serde() -> JsonPlusSerializer:
    allowed = [(cls.__module__, cls.__qualname__) for cls in _CHECKPOINT_ALLOWED_TYPES]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def _should_replan(state: TripState) -> Literal["planner_agent", "evaluator_agent"]:
    settings = get_settings()
    report = state.get("validation_report")
    attempts = state.get("planner_attempts", 0)
    if report is not None and report.needs_replan() and attempts < settings.max_planner_repair_attempts:
        return "planner_agent"
    return "evaluator_agent"


def _should_replan_eval(state: TripState) -> Literal["planner_agent", "memory_update_agent"]:
    replan = state.get("replan_needed", False)
    if replan:
        return "planner_agent"
    return "memory_update_agent"


def _build_plan_graph(checkpointer) -> CompiledStateGraph:
    graph = StateGraph(TripState)
    graph.add_node("query_agent", run_query_agent)
    graph.add_node("memory_agent", run_memory_agent)
    graph.add_node("rag_agent", run_rag_agent)
    graph.add_node("planner_agent", run_planner_agent)
    graph.add_node("budget_agent", run_budget_agent)
    graph.add_node("rewards_agent", run_rewards_agent)
    graph.add_node("validator_agent", run_validator_agent)
    graph.add_node("evaluator_agent", run_evaluator_agent)
    graph.add_node("memory_update_agent", run_memory_update_agent)
    graph.add_node("summary_agent", run_summary_agent)

    graph.add_edge(START, "query_agent")
    graph.add_edge("query_agent", "memory_agent")
    graph.add_edge("memory_agent", "rag_agent")
    graph.add_edge("rag_agent", "planner_agent")
    graph.add_edge("planner_agent", "budget_agent")
    graph.add_edge("budget_agent", "rewards_agent")
    graph.add_edge("rewards_agent", "validator_agent")
    graph.add_conditional_edges(
        "validator_agent", _should_replan, {"planner_agent": "planner_agent", "evaluator_agent": "evaluator_agent"}
    )
    graph.add_conditional_edges(
        "evaluator_agent", _should_replan_eval, {"planner_agent": "planner_agent", "memory_update_agent": "memory_update_agent"}
    )
    graph.add_edge("memory_update_agent", "summary_agent")
    graph.add_edge("summary_agent", END)

    return graph.compile(checkpointer=checkpointer)


def _should_replan_refine(state: TripState) -> Literal["refinement_agent", "summary_agent"]:
    settings = get_settings()
    report = state.get("validation_report")
    attempts = state.get("planner_attempts", 0)
    if report is not None and report.needs_replan() and attempts < settings.max_planner_repair_attempts:
        return "refinement_agent"
    return "summary_agent"


def _build_refine_graph(checkpointer) -> CompiledStateGraph:
    graph = StateGraph(TripState)
    graph.add_node("refinement_agent", run_refinement_agent)
    graph.add_node("validator_agent", run_validator_agent)
    graph.add_node("summary_agent", run_summary_agent)

    graph.add_edge(START, "refinement_agent")
    graph.add_edge("refinement_agent", "validator_agent")
    graph.add_conditional_edges(
        "validator_agent", _should_replan_refine, {"refinement_agent": "refinement_agent", "summary_agent": "summary_agent"}
    )
    graph.add_edge("summary_agent", END)

    return graph.compile(checkpointer=checkpointer)


class GraphManager:
    """Owns the single long-lived SQLite connection + checkpointer for the
    process, and both compiled graphs built on top of it. Constructed once
    at API startup (see src/api/main.py's lifespan) and closed at shutdown.
    """

    def __init__(self, checkpoint_db_path: str):
        Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI/Streamlit both call into this from
        # a thread pool, not always the thread that opened the connection.
        # SQLite itself serializes writes, so this is safe for the
        # single-process deployment this project targets — see
        # docs/deployment.md for the multi-process/Postgres-checkpointer
        # scaling note.
        self._conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._conn, serde=_build_checkpoint_serde())
        self.checkpointer.setup()
        self.plan_graph = _build_plan_graph(self.checkpointer)
        self.refine_graph = _build_refine_graph(self.checkpointer)

    def config_for(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def get_state(self, session_id: str):
        return self.plan_graph.get_state(self.config_for(session_id))

    def history(self, session_id: str, limit: int = 25) -> list:
        return list(self.plan_graph.get_state_history(self.config_for(session_id), limit=limit))

    def rollback_config(self, session_id: str, checkpoint_id: str) -> dict | None:
        """Finds the historical checkpoint matching `checkpoint_id` and
        returns the config that, if used for the next `invoke`, forks
        execution from that point — LangGraph's built-in time-travel
        mechanism. Returns None if no matching checkpoint exists."""
        for snapshot in self.history(session_id, limit=200):
            if snapshot.config["configurable"].get("checkpoint_id") == checkpoint_id:
                return snapshot.config
        return None

    def close(self) -> None:
        self._conn.close()


_manager_singleton: GraphManager | None = None


def get_graph_manager() -> GraphManager:
    global _manager_singleton
    if _manager_singleton is None:
        _manager_singleton = GraphManager(get_settings().checkpoint_db_path)
    return _manager_singleton
