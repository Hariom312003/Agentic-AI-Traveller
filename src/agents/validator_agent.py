"""
Validator Agent.

Runs every check in src/validation/rules.py against the current itinerary
and produces a `ValidationReport`. `ValidationReport.needs_replan()` (any
CRITICAL issue) is what the graph's conditional edge reads to decide
"loop back to the Planner" vs. "proceed to Summary" — see
src/graph/workflow.py. Capped by `max_planner_repair_attempts` so a
persistently broken destination/config can't loop forever.
"""
from __future__ import annotations

from src.config import get_settings
from src.models.state import TripState
from src.models.validation import ValidationReport
from src.monitoring.telemetry import track_agent
from src.validation.rules import run_all_checks


def run_validator_agent(state: TripState) -> dict:
    with track_agent("validator_agent") as handle:
        settings = get_settings()
        itinerary = state["itinerary"]
        budget = state.get("budget_breakdown")
        trip_request = state["trip_request"]

        issues, grounded_ratio, route_score = run_all_checks(
            itinerary=itinerary,
            budget_total=trip_request.budget_amount,
            actual_total=budget.total if budget else 0.0,
            duplicate_threshold=settings.duplicate_fuzzy_threshold,
        )

        duplicate_count = sum(1 for i in issues if i.category.value == "duplicate_activity")
        budget_status = "unknown"
        if trip_request.budget_amount and budget:
            budget_status = "over_budget" if budget.total > trip_request.budget_amount else "within_budget"

        report = ValidationReport(
            is_valid=not any(i.severity.value == "critical" for i in issues),
            issues=issues,
            duplicate_count=duplicate_count,
            budget_status=budget_status,
            route_efficiency_score=route_score,
            grounded_ratio=grounded_ratio,
        )

        attempts = state.get("planner_attempts", 0)
        will_replan = report.needs_replan() and attempts < settings.max_planner_repair_attempts

        handle.record.reasoning_summary = (
            f"{len(issues)} issue(s) found ({sum(1 for i in issues if i.severity.value=='critical')} critical); "
            f"{'requesting replan' if will_replan else 'proceeding'}"
        )
        handle.record.extra["will_replan"] = will_replan

        return {"validation_report": report, "execution_trace": [handle.record]}
