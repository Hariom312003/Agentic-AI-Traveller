"""
Rewards Agent.

Matches budget line items (flights, hotels, food, ...) to illustrative
reward-earning categories from `data/rewards_catalogue.json`. This is
explicitly reference data, not live financial product data — see that
file's `_disclaimer` field and `RewardRecommendation.is_illustrative` in
src/models/budget.py. Fabricating specific real card names/current offers
here would be worse than not having this feature at all, since a stale or
made-up offer is actively misleading; a clearly-labeled example structure
the user can swap in their own (or a live comparison API's) data is the
honest version of this feature.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.models.budget import BudgetBreakdown, RewardRecommendation, RewardsSummary
from src.models.state import TripState
from src.monitoring.telemetry import track_agent

_CATALOGUE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "rewards_catalogue.json"

_BUDGET_FIELD_TO_CATEGORY = {
    "flights": "Flight Bookings",
    "hotels": "Hotel Bookings",
    "food": "Dining & Food",
    "shopping": "General Shopping",
}

# Rough illustrative savings rates used only to show an *order of magnitude*
# potential saving next to each recommendation — not a promise.
_ILLUSTRATIVE_RATE = {
    "Flight Bookings": 0.04,
    "Hotel Bookings": 0.06,
    "Dining & Food": 0.04,
    "General Shopping": 0.015,
}


def _load_catalogue() -> list[dict]:
    if not _CATALOGUE_PATH.exists():
        return []
    return json.loads(_CATALOGUE_PATH.read_text(encoding="utf-8")).get("categories", [])


def build_rewards_summary(budget: BudgetBreakdown) -> RewardsSummary:
    catalogue = {c["category"]: c for c in _load_catalogue()}
    recommendations: list[RewardRecommendation] = []
    spend_by_field = {
        "flights": budget.flights,
        "hotels": budget.hotels,
        "food": budget.food,
        "shopping": budget.shopping,
    }

    for field, category_name in _BUDGET_FIELD_TO_CATEGORY.items():
        spend = spend_by_field.get(field, 0.0)
        entry = catalogue.get(category_name)
        if not entry or spend <= 0:
            continue
        rate = _ILLUSTRATIVE_RATE.get(category_name, 0.02)
        recommendations.append(RewardRecommendation(
            category=category_name,
            instrument=entry["instrument"],
            reason=f"{entry['typical_benefit']}. {entry['notes']}.",
            estimated_savings=round(spend * rate, 2),
            currency=budget.currency,
            is_illustrative=True,
        ))

    # forex is relevant to the whole trip whenever there's any international-shaped spend
    forex_entry = catalogue.get("Forex / International Spend")
    if forex_entry and budget.total > 0:
        recommendations.append(RewardRecommendation(
            category="Forex / International Spend",
            instrument=forex_entry["instrument"],
            reason=f"{forex_entry['typical_benefit']}. {forex_entry['notes']}.",
            estimated_savings=round(budget.total * 0.02, 2),
            currency=budget.currency,
            is_illustrative=True,
        ))

    total_savings = round(sum(r.estimated_savings for r in recommendations), 2)
    return RewardsSummary(recommendations=recommendations, total_estimated_savings=total_savings, currency=budget.currency)


def run_rewards_agent(state: TripState) -> dict:
    with track_agent("rewards_agent") as handle:
        budget = state["budget_breakdown"]
        summary = build_rewards_summary(budget)
        handle.record.reasoning_summary = (
            f"{len(summary.recommendations)} illustrative reward categor{'y' if len(summary.recommendations)==1 else 'ies'} matched; "
            f"potential order-of-magnitude saving ~{summary.total_estimated_savings:,.0f} {summary.currency}"
        )
        return {"rewards_summary": summary, "execution_trace": [handle.record]}
