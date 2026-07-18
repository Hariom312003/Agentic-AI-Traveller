from __future__ import annotations

from src.agents.evaluator_agent import evaluate_itinerary
from src.models.itinerary import Itinerary, Day, Activity, ActivityCategory, ActivitySource
from src.models.budget import BudgetBreakdown
from src.models.request import TripRequest


def test_evaluate_itinerary_basic():
    # Construct a minimal valid itinerary
    activity = Activity(
        id="act-1",
        title="Senso-ji Temple",
        description="Famous temple in Tokyo",
        category=ActivityCategory.CULTURE,
        source=ActivitySource.KNOWLEDGE_BASE,
        latitude=35.7147,
        longitude=139.7966,
    )
    day = Day(day_number=1)
    day.morning.append(activity)
    
    itinerary = Itinerary(
        itinerary_id="trip-123",
        destination="Tokyo",
        duration_days=1,
        travelers_count=1,
        currency="INR",
        days=[day],
    )
    
    request = TripRequest(
        raw_query="1 day trip to Tokyo",
        destination="Tokyo",
        duration_days=1,
        budget_amount=50000.0,
        budget_currency="INR"
    )
    
    budget = BudgetBreakdown(
        currency="INR",
        flights=10000.0,
        hotels=5000.0,
        food=2000.0,
        activities=1000.0,
        shopping=0.0,
        local_transport=500.0,
        emergency_buffer=1000.0,
        taxes_and_fees=500.0
    )
    
    state = {
        "itinerary": itinerary,
        "trip_request": request,
        "budget_breakdown": budget
    }
    
    total_score, scores, feedback = evaluate_itinerary(state)
    
    assert total_score > 5.0
    assert scores["diversity"] == 2.5  # No duplicates
    assert scores["budget"] == 2.5     # Within budget
