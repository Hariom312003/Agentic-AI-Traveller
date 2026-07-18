"""
Self-Evaluation Critic Agent.
"""
from __future__ import annotations

from src.models.state import TripState
from src.monitoring.telemetry import track_agent
from src.validation.rules import route_efficiency_score, check_duplicates


def evaluate_itinerary(state: TripState) -> tuple[float, dict[str, float], str | None]:
    itinerary = state.get("itinerary")
    if not itinerary:
        return 0.0, {}, "No itinerary generated to evaluate"

    scores = {}
    feedback_notes = []

    # 1. Diversity Score (max 2.5)
    # Check for duplicates using the same threshold (80.0) as the validator
    dup_issues = check_duplicates(itinerary, threshold=80.0)
    if not dup_issues:
        scores["diversity"] = 2.5
    else:
        # Penalize duplicate occurrences
        scores["diversity"] = max(0.0, 2.5 - len(dup_issues) * 0.5)
        feedback_notes.append(f"Contains {len(dup_issues)} potential duplicate activities.")

    # 2. Travel Efficiency Score (max 2.5)
    eff_val = route_efficiency_score(itinerary)
    scores["efficiency"] = round(eff_val * 2.5, 2)
    if eff_val < 0.7:
        feedback_notes.append(f"Route efficiency is low ({eff_val:.2f}) — attractions are geographically spread.")

    # 3. Preference Alignment Score (max 2.5)
    trip_request = state.get("trip_request")
    interests = [i.lower().strip() for i in (trip_request.interests if trip_request else []) if i]
    if not interests:
        scores["alignment"] = 2.5
    else:
        activities = itinerary.all_activities()
        if not activities:
            scores["alignment"] = 0.0
        else:
            matches = 0
            for act in activities:
                title_desc_cat = (act.title + " " + act.description + " " + (act.category.value if hasattr(act.category, "value") else str(act.category))).lower()
                if any(interest in title_desc_cat for interest in interests):
                    matches += 1
            ratio = matches / len(activities)
            scores["alignment"] = round(ratio * 2.5, 2)
            if ratio < 0.4:
                feedback_notes.append(f"Low alignment to user interests ({ratio * 100:.0f}% matched).")

    # 4. Budget Compliance Score (max 2.5)
    budget = state.get("budget_breakdown")
    if not budget or not trip_request or not trip_request.budget_amount or trip_request.budget_amount <= 0:
        scores["budget"] = 2.5
    else:
        limit = trip_request.budget_amount
        actual = budget.total
        if actual <= limit:
            scores["budget"] = 2.5
        elif actual <= limit * 1.15:
            scores["budget"] = 1.5
            feedback_notes.append("Itinerary slightly exceeds the requested budget limit.")
        else:
            scores["budget"] = 0.0
            feedback_notes.append(f"Itinerary significantly exceeds the budget limit ({actual:,.0f} vs {limit:,.0f}).")

    total_score = round(sum(scores.values()), 2)
    feedback_str = " | ".join(feedback_notes) if feedback_notes else None

    return total_score, scores, feedback_str


def run_evaluator_agent(state: TripState) -> dict:
    with track_agent("evaluator_agent") as handle:
        score, scores, feedback = evaluate_itinerary(state)
        attempts = state.get("planner_attempts", 1)
        
        # Decide if we need to replan
        replan = False
        replan_reason = None
        if score < 8.5 and attempts < 3:
            replan = True
            replan_reason = f"Itinerary evaluation score was {score:.2f}/10.0 (below threshold 8.5). Critic Feedback: {feedback or 'None'}"
            
        handle.record.reasoning_summary = (
            f"Critic score: {score:.2f}/10.0 (Diversity: {scores.get('diversity', 0.0)}, "
            f"Efficiency: {scores.get('efficiency', 0.0)}, Alignment: {scores.get('alignment', 0.0)}, "
            f"Budget: {scores.get('budget', 0.0)}). Replan: {replan}"
        )
        if replan_reason:
            handle.record.extra["replan_reason"] = replan_reason
            
        return {
            "execution_trace": [handle.record],
            "evaluator_score": score,
            "evaluator_feedback": feedback,
            "replan_needed": replan,
            "replan_reason": replan_reason
        }
