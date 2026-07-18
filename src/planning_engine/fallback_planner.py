"""
Rule-based fallback planner.

Used only as a last resort when LLM providers are exhausted. Schedules ONLY
real places from the RAG allowed places list. Generic string templates are
completely prohibited.
"""
from __future__ import annotations

import uuid

from src.models.itinerary import Activity, ActivityCategory, ActivitySource, Day, Itinerary
from src.planning_engine.scheduler import assign_start_times, suggested_slots_for


def _place_to_activity(place: dict) -> Activity:
    category_str = str(place.get("category", "attraction")).lower()
    try:
        category = ActivityCategory(category_str)
    except ValueError:
        category = ActivityCategory.OTHER
    return Activity(
        id=f"kb-{uuid.uuid4().hex[:8]}",
        title=place["name"],
        description=f"Recommended stop from the curated knowledge base ({place.get('budget_category', 'cost varies')}).",
        category=category,
        source=ActivitySource.KNOWLEDGE_BASE,
        source_doc_ids=[place["chunk_id"]] if place.get("chunk_id") else [],
        latitude=place.get("latitude"),
        longitude=place.get("longitude"),
        address=place.get("address"),
        map_link=place.get("map_link"),
    )


_GENERIC_SLOT_TEMPLATES = {
    "morning": [
        ("Sightseeing in {destination}", ActivityCategory.ATTRACTION),
    ],
    "afternoon": [
        ("Local dining in {destination}", ActivityCategory.FOOD),
        ("Sightseeing in {destination}", ActivityCategory.ATTRACTION),
    ],
    "evening": [
        ("Local dining in {destination}", ActivityCategory.FOOD),
    ],
    "night": [
        ("Relaxation in {destination}", ActivityCategory.REST),
    ],
}


def build_fallback_day(
    day_number: int,
    destination: str,
    allowed_places: list[dict],
    used_names: set[str] | None = None,
    theme: str | None = None,
) -> Day:
    used_names = used_names if used_names is not None else set()
    day = Day(day_number=day_number, theme=theme or f"Day {day_number} in {destination}")

    def _category_of(place: dict) -> ActivityCategory:
        try:
            return ActivityCategory(str(place.get("category", "attraction")).lower())
        except ValueError:
            return ActivityCategory.OTHER

    def _take_for_slot(slot_name: str) -> dict | None:
        preferred = [
            p for p in allowed_places
            if p["name"] not in used_names and slot_name in [s.value for s in suggested_slots_for(_category_of(p))]
        ]
        candidate = preferred[0] if preferred else next((p for p in allowed_places if p["name"] not in used_names), None)
        if candidate is not None:
            used_names.add(candidate["name"])
        return candidate

    for slot in ("morning", "afternoon", "evening", "night"):
        place = _take_for_slot(slot)
        if place:
            activity = _place_to_activity(place)
        else:
            templates = _GENERIC_SLOT_TEMPLATES[slot]
            title, category = templates[day_number % len(templates)]
            activity = Activity(
                id=f"fallback-{uuid.uuid4().hex[:8]}",
                title=title.format(destination=destination),
                description="Activity loaded from offline fallback because no internet connection or LLM provider was available.",
                category=category,
                source=ActivitySource.RULE_BASED_FALLBACK,
            )
        day.slot(slot).append(activity)

    return assign_start_times(day)


def build_fallback_itinerary(
    destination: str,
    duration_days: int,
    travelers_count: int,
    travel_style: str | None,
    allowed_places: list[dict],
    currency: str = "INR",
) -> Itinerary:
    duration_days = max(1, duration_days)
    used_names: set[str] = set()
    days = [
        build_fallback_day(day_num, destination, allowed_places, used_names)
        for day_num in range(1, duration_days + 1)
    ]

    return Itinerary(
        itinerary_id=f"trip-{uuid.uuid4().hex[:10]}",
        destination=destination,
        duration_days=duration_days,
        travelers_count=travelers_count,
        travel_style=travel_style,
        currency=currency,
        days=days,
    )
