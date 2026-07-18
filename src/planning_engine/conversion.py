"""
LLM planner-output schema <-> domain model conversion.

Shared by `src/agents/planner_agent.py` (full itinerary generation) and
`src/agents/refinement_agent.py` (single-day regeneration) so there is
exactly one place that defines "what a day/activity looks like coming out
of the model" and exactly one place that does the groundedness check
(`verify_source`) — a second copy of that logic drifting out of sync would
be its own kind of bug.
"""
from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from src.models.itinerary import Activity, ActivityCategory, ActivitySource, Day, TimeSlotName
from src.planning_engine.scheduler import assign_start_times
from src.validation.duplicate_checker import is_near_duplicate


class PlannerActivityOut(BaseModel):
    title: str
    description: str = ""
    category: str = "attraction"
    time_slot: str
    location: str | None = None
    estimated_cost: float | None = None
    duration_minutes: int | None = None
    notes: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None
    map_link: str | None = None


class PlannerDayOut(BaseModel):
    day_number: int
    theme: str | None = None
    activities: list[PlannerActivityOut] = Field(default_factory=list)


class PlannerItineraryOut(BaseModel):
    days: list[PlannerDayOut]


def verify_source(title: str, allowed_places: list[dict], grounded: bool) -> tuple[ActivitySource, list[str]]:
    """Never trust the model's self-reported groundedness — independently
    fuzzy-match the returned title against what was actually retrieved."""
    if not grounded:
        return ActivitySource.MODEL_KNOWLEDGE, []
    for place in allowed_places:
        if is_near_duplicate(title, place["name"]):
            return ActivitySource.KNOWLEDGE_BASE, [place["chunk_id"]] if place.get("chunk_id") else []
    return ActivitySource.MODEL_KNOWLEDGE, []


def to_domain_day(day_out: PlannerDayOut, allowed_places: list[dict], grounded: bool, currency: str) -> Day:
    day = Day(day_number=day_out.day_number, theme=day_out.theme)
    for act_out in day_out.activities:
        try:
            category = ActivityCategory(act_out.category.lower().strip())
        except ValueError:
            category = ActivityCategory.OTHER
        try:
            slot = TimeSlotName(act_out.time_slot.lower().strip())
        except ValueError:
            slot = TimeSlotName.MORNING

        source, doc_ids = verify_source(act_out.title, allowed_places, grounded)
        
        # Look up matched place in allowed_places to inherit coordinates/map links if not output by model
        lat = act_out.latitude
        lng = act_out.longitude
        addr = act_out.address
        mlink = act_out.map_link
        
        if source == ActivitySource.KNOWLEDGE_BASE:
            for place in allowed_places:
                if is_near_duplicate(act_out.title, place["name"]):
                    lat = lat or place.get("latitude")
                    lng = lng or place.get("longitude")
                    addr = addr or place.get("address")
                    mlink = mlink or place.get("map_link")
                    break

        activity = Activity(
            id=f"act-{uuid.uuid4().hex[:8]}",
            title=act_out.title,
            description=act_out.description,
            category=category,
            location=act_out.location,
            duration_minutes=act_out.duration_minutes,
            estimated_cost=act_out.estimated_cost,
            currency=currency,
            source=source,
            source_doc_ids=doc_ids,
            notes=act_out.notes,
            latitude=lat,
            longitude=lng,
            address=addr,
            map_link=mlink,
        )
        day.slot(slot).append(activity)
    return assign_start_times(day)
