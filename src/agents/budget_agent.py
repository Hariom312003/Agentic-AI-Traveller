"""
Budget Agent.

Estimates a full cost breakdown from the generated itinerary. Flights and
hotels are the two line items no destination-knowledge-base chunk can ever
supply (they depend on live fares/rates, which need a real booking API
integration — a documented extension point, not something to fake with a
static number dressed up as real-time). Everything else is derived from
the itinerary's own `estimated_cost` fields (set by the Planner, or by the
knowledge base's `budget_category` as a fallback estimate) plus simple,
clearly-labeled heuristic multipliers for hotels/flights/taxes so the total
is at least directionally useful.
"""
from __future__ import annotations

from src.models.budget import BudgetBreakdown
from src.models.itinerary import ActivityCategory, Itinerary
from src.models.state import TripState
from src.monitoring.telemetry import track_agent

# Heuristic nightly hotel cost bands by style, in INR — clearly a
# simplification (see module docstring); exposed via config-like constants
# here rather than scattered magic numbers so they're easy to override.
_HOTEL_NIGHTLY_INR = {"budget": 1800, "mid_range": 4500, "luxury": 12000}
_FLIGHT_ESTIMATE_INR = {"budget": 6000, "mid_range": 12000, "luxury": 30000}
_BUDGET_CATEGORY_COST_INR = {"free": 0, "low-cost entry": 200, "budget": 300, "mid-range": 900, "high": 2500}


def _style_tier(travel_style: str | None) -> str:
    style = (travel_style or "").lower()
    if "luxury" in style or "5 star" in style or "premium" in style:
        return "luxury"
    if "budget" in style or "backpack" in style:
        return "budget"
    return "mid_range"


def _estimate_activity_cost(activity, tier: str) -> float:
    if activity.estimated_cost is not None:
        return activity.estimated_cost
    budget_hint = (activity.notes or "").lower()
    for key, value in _BUDGET_CATEGORY_COST_INR.items():
        if key in budget_hint:
            return float(value)
    defaults = {"budget": 250.0, "mid_range": 600.0, "luxury": 1500.0}
    return defaults[tier] if activity.category != ActivityCategory.TRANSPORT else defaults[tier] * 0.4


def estimate_budget(itinerary: Itinerary, travelers_count: int, travel_style: str | None, currency: str) -> BudgetBreakdown:
    tier = _style_tier(travel_style)
    nights = max(itinerary.duration_days - 1, 0)

    activities_total = 0.0
    shopping_total = 0.0
    food_total = 0.0
    transport_total = 0.0

    for activity in itinerary.all_activities():
        cost = _estimate_activity_cost(activity, tier) * travelers_count
        if activity.category == ActivityCategory.SHOPPING:
            shopping_total += cost
        elif activity.category == ActivityCategory.FOOD:
            food_total += cost
        elif activity.category == ActivityCategory.TRANSPORT:
            transport_total += cost
        else:
            activities_total += cost

    hotels = _HOTEL_NIGHTLY_INR[tier] * nights * max(1, (travelers_count + 1) // 2)  # ~2 travelers/room
    flights = _FLIGHT_ESTIMATE_INR[tier] * travelers_count
    subtotal = hotels + flights + activities_total + shopping_total + food_total + transport_total
    emergency_buffer = round(subtotal * 0.08, 2)
    taxes = round(subtotal * 0.05, 2)

    return BudgetBreakdown(
        currency=currency,
        flights=round(flights, 2),
        hotels=round(hotels, 2),
        food=round(food_total, 2),
        activities=round(activities_total, 2),
        shopping=round(shopping_total, 2),
        local_transport=round(transport_total, 2),
        emergency_buffer=emergency_buffer,
        taxes_and_fees=taxes,
    )


def run_budget_agent(state: TripState) -> dict:
    with track_agent("budget_agent") as handle:
        itinerary = state["itinerary"]
        trip_request = state["trip_request"]
        breakdown = estimate_budget(
            itinerary, trip_request.travelers_count or 1, trip_request.travel_style, trip_request.budget_currency
        )
        handle.record.reasoning_summary = (
            f"Estimated total {breakdown.total:,.0f} {breakdown.currency} "
            f"({breakdown.per_day(itinerary.duration_days):,.0f}/day)"
        )
        return {"budget_breakdown": breakdown, "execution_trace": [handle.record]}
