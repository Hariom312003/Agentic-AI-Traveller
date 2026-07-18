"""
Shared LangGraph state.

`TripState` is a TypedDict (LangGraph's native state shape) rather than a
pydantic model directly, because LangGraph merges partial node returns into
state key-by-key, and `Annotated[..., operator.add]` reducers only work
cleanly on TypedDict fields. Pydantic models are nested *inside* as values
(`itinerary: Itinerary | None`) so we still get full validation on the
things that matter.

Two fields use reducers instead of overwrite-on-return:
- `execution_trace`: every agent appends its own record; nobody should have
  to know the full trace to add to it.
- `errors`: same idea, non-fatal warnings accumulate instead of clobbering.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from src.models.budget import BudgetBreakdown, RewardsSummary
from src.models.itinerary import Itinerary
from src.models.request import TripRequest
from src.models.trip_summary import TripSummary
from src.models.user import UserProfile
from src.models.validation import ValidationReport
from src.monitoring.telemetry import AgentExecutionRecord


class TripState(TypedDict, total=False):
    # ---- input ----
    mode: str  # "plan" | "refine"
    trip_request: TripRequest
    user_id: str
    session_id: str

    # ---- query understanding ----
    parsed_request: dict[str, Any]

    # ---- memory ----
    user_profile: UserProfile | None

    # ---- RAG ----
    retrieved_context: list[dict[str, Any]]
    grounded_ratio: float

    # ---- planning ----
    itinerary: Itinerary | None
    planner_attempts: int

    # ---- budget / rewards ----
    budget_breakdown: BudgetBreakdown | None
    rewards_summary: RewardsSummary | None

    # ---- validation ----
    validation_report: ValidationReport | None

    # ---- refinement ----
    refinement_instruction: str | None
    target_days: list[int] | None
    pre_refinement_hashes: dict[int, str]

    # ---- summary / output ----
    trip_summary: TripSummary | None
    final_response: dict[str, Any] | None

    # ---- evaluation ----
    evaluator_score: float
    evaluator_feedback: str | None
    replan_needed: bool
    replan_reason: str | None

    # ---- explainability (append-only across the whole run) ----
    execution_trace: Annotated[list[AgentExecutionRecord], operator.add]
    errors: Annotated[list[str], operator.add]


def new_state(
    *,
    mode: str,
    trip_request: TripRequest,
    user_id: str,
    session_id: str,
) -> TripState:
    """Factory for a fresh state dict — keeps node code from having to know
    every key's default."""
    return TripState(
        mode=mode,
        trip_request=trip_request,
        user_id=user_id,
        session_id=session_id,
        parsed_request={},
        user_profile=None,
        retrieved_context=[],
        grounded_ratio=0.0,
        itinerary=None,
        planner_attempts=0,
        budget_breakdown=None,
        rewards_summary=None,
        validation_report=None,
        refinement_instruction=None,
        target_days=None,
        pre_refinement_hashes={},
        trip_summary=None,
        final_response=None,
        execution_trace=[],
        errors=[],
    )
