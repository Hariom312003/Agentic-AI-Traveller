"""
Memory Agent.

Two jobs, split across the graph so the write only happens once we have a
finished itinerary to learn from:

- `run_memory_agent` (early in the graph): loads the user's profile so the
  Planner Agent can bias toward known preferences.
- `run_memory_update_agent` (after summary, end of the "plan" path):
  folds this trip's signals back into the profile — see
  src/memory/behavioral.py for the destination-agnostic scrubbing that
  makes this useful across trips instead of overfitting to one city.
"""
from __future__ import annotations

from src.memory.behavioral import preferences_as_prompt_context, update_preferences_from_trip
from src.memory.store import get_memory_store
from src.models.state import TripState
from src.monitoring.telemetry import track_agent


def run_memory_agent(state: TripState) -> dict:
    with track_agent("memory_agent") as handle:
        store = get_memory_store()
        profile = store.get_profile(state["user_id"])
        handle.record.reasoning_summary = (
            f"Loaded profile with {len(profile.past_trips)} past trip(s) on file"
            if profile.past_trips
            else "No prior profile found — starting fresh"
        )
        handle.record.extra["preference_context"] = preferences_as_prompt_context(profile.behavioral_preferences)
        return {"user_profile": profile, "execution_trace": [handle.record]}


def run_memory_update_agent(state: TripState) -> dict:
    with track_agent("memory_update_agent") as handle:
        store = get_memory_store()
        profile = state.get("user_profile") or store.get_profile(state["user_id"])
        updated = update_preferences_from_trip(profile, state["trip_request"], state.get("itinerary"))
        store.save_profile(updated)
        handle.record.reasoning_summary = "Behavioral preferences updated from this trip (destination-agnostic)"
        return {"user_profile": updated, "execution_trace": [handle.record]}
