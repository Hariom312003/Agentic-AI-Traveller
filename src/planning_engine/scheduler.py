"""
Scheduling heuristics.

Deliberately NOT a live opening-hours / traffic API integration — there is
no such API wired into this project (that would need a Google Places/Maps
key and a live quota), so this module is explicit about being a heuristic
approximation, not a promise of real-time accuracy. It's used to (a) assign
plausible start times within a slot and (b) suggest which slot a category
of activity usually belongs in. `docs/architecture.md` documents wiring a
real hours/traffic API as a concrete extension point.
"""
from __future__ import annotations

from src.models.itinerary import Activity, ActivityCategory, Day, TimeSlotName

DEFAULT_SLOT_START = {
    TimeSlotName.MORNING: "09:00",
    TimeSlotName.AFTERNOON: "14:00",
    TimeSlotName.EVENING: "18:30",
    TimeSlotName.NIGHT: "21:00",
}

_CATEGORY_SLOT_PREFERENCE: dict[ActivityCategory, list[TimeSlotName]] = {
    ActivityCategory.NIGHTLIFE: [TimeSlotName.NIGHT, TimeSlotName.EVENING],
    ActivityCategory.CULTURE: [TimeSlotName.MORNING, TimeSlotName.AFTERNOON],
    ActivityCategory.ATTRACTION: [TimeSlotName.MORNING, TimeSlotName.AFTERNOON],
    ActivityCategory.ADVENTURE: [TimeSlotName.MORNING],
    ActivityCategory.NATURE: [TimeSlotName.MORNING, TimeSlotName.AFTERNOON],
    ActivityCategory.SHOPPING: [TimeSlotName.AFTERNOON, TimeSlotName.EVENING],
    ActivityCategory.FOOD: [TimeSlotName.AFTERNOON, TimeSlotName.EVENING],
    ActivityCategory.REST: [TimeSlotName.AFTERNOON],
    ActivityCategory.TRANSPORT: [TimeSlotName.MORNING],
}


def suggested_slots_for(category: ActivityCategory) -> list[TimeSlotName]:
    return _CATEGORY_SLOT_PREFERENCE.get(category, [TimeSlotName.MORNING, TimeSlotName.AFTERNOON])


def _minutes_to_hhmm(total_minutes: int) -> str:
    total_minutes %= 24 * 60
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def _hhmm_to_minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def assign_start_times(day: Day, default_duration_minutes: int = 120) -> Day:
    """Fills in `start_time` for any activity missing one, sequencing
    activities within each slot back-to-back from the slot's default start
    time using each activity's `duration_minutes` (or a default)."""
    for slot_name in TimeSlotName:
        activities = day.slot(slot_name)
        if not activities:
            continue
        cursor = _hhmm_to_minutes(DEFAULT_SLOT_START[slot_name])
        for activity in activities:
            if not activity.start_time:
                activity.start_time = _minutes_to_hhmm(cursor)
            duration = activity.duration_minutes or default_duration_minutes
            cursor += duration + 20  # +20 min buffer for local transit between stops
    return day


def rebalance_pacing(day: Day, pacing: str) -> Day:
    """'relaxed' trims each slot down to at most one activity; 'packed'
    is a no-op (packed is already the default density from the planner).
    Used by the Refinement Agent for "make Day X more relaxed" requests."""
    if pacing != "relaxed":
        return day
    for slot_name in TimeSlotName:
        activities = day.slot(slot_name)
        if len(activities) > 1:
            day.set_slot(slot_name, activities[:1])
    day.pacing = pacing
    return day
