"""
Memory tests. `test_preferences_generalize_without_place_names` is the one
that matters most: it proves a preference learned on one destination is
represented in a way that could apply to *any* destination, which is the
whole point of `src/memory/behavioral.py`'s scrubbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.behavioral import (
    infer_style_signals,
    scrub_place_names,
    update_preferences_from_trip,
)
from src.memory.store import MemoryStore
from src.models.request import TripRequest
from src.models.user import UserProfile


def test_scrub_place_names_removes_exact_and_case_insensitive_matches():
    text = "I loved Baga Beach and baga beach sunsets"
    scrubbed = scrub_place_names(text, ["Baga Beach"])
    assert "baga" not in scrubbed.lower()
    assert "[place]" in scrubbed


def test_infer_style_signals_detects_relaxed_pace():
    request = TripRequest(raw_query="I want a relaxed, chill trip with nothing packed", destination="Bali")
    assert infer_style_signals(request)["pace"] == "relaxed"


def test_infer_style_signals_detects_budget_tier():
    request = TripRequest(raw_query="Looking for a backpacking trip on a tight budget", destination="Vietnam")
    assert infer_style_signals(request)["budget_tier"] == "budget"


def test_infer_style_signals_returns_empty_when_nothing_matches():
    request = TripRequest(raw_query="Plan something for me", destination="Rome")
    assert infer_style_signals(request) == {}


def test_preferences_generalize_without_place_names():
    """The core behavioral-memory guarantee: after a Goa trip mentioning
    Goa-specific places, none of those place names should appear anywhere
    in the persisted profile — only destination-agnostic style signals."""
    profile = UserProfile(user_id="u1")
    request = TripRequest(
        raw_query="relaxed budget trip",
        destination="Goa",
        interests=["nightlife", "beaches"],
        special_requests="I really loved Baga Beach and Tito's Lane last time, want something similar",
    )
    updated = update_preferences_from_trip(profile, request, itinerary=None)

    serialized = updated.model_dump_json().lower()
    assert "baga beach" not in serialized
    assert "tito's lane" not in serialized
    assert updated.behavioral_preferences.interest_weights.get("nightlife", 0) > 0


def test_interest_weights_increase_with_repeated_trips():
    profile = UserProfile(user_id="u1")
    request = TripRequest(raw_query="trip", destination="Goa", interests=["nightlife"])
    profile = update_preferences_from_trip(profile, request, itinerary=None)
    first_weight = profile.behavioral_preferences.interest_weights["nightlife"]
    profile = update_preferences_from_trip(profile, request, itinerary=None)
    second_weight = profile.behavioral_preferences.interest_weights["nightlife"]
    assert second_weight > first_weight


def test_past_trips_are_bounded_to_recent_25():
    profile = UserProfile(user_id="u1")
    for i in range(30):
        request = TripRequest(raw_query="trip", destination=f"City{i}")
        profile = update_preferences_from_trip(profile, request, itinerary=None)
    assert len(profile.past_trips) == 25
    assert profile.past_trips[-1].destination == "City29"


def test_memory_store_roundtrip(tmp_path):
    store = MemoryStore(str(tmp_path / "test_memory.db"))
    profile = store.get_profile("new-user")
    assert profile.user_id == "new-user"
    assert profile.past_trips == []

    profile.behavioral_preferences.pace = "relaxed"
    profile.behavioral_preferences.interest_weights["nightlife"] = 0.8
    store.save_profile(profile)

    reloaded = store.get_profile("new-user")
    assert reloaded.behavioral_preferences.pace == "relaxed"
    assert reloaded.behavioral_preferences.interest_weights["nightlife"] == 0.8


def test_memory_store_upsert_does_not_duplicate_rows(tmp_path):
    store = MemoryStore(str(tmp_path / "test_memory.db"))
    profile = store.get_profile("u1")
    store.save_profile(profile)
    profile.behavioral_preferences.pace = "packed"
    store.save_profile(profile)
    assert store.all_user_ids().count("u1") == 1


def test_memory_store_isolated_between_users(tmp_path):
    store = MemoryStore(str(tmp_path / "test_memory.db"))
    p1 = store.get_profile("user-1")
    p1.behavioral_preferences.pace = "relaxed"
    store.save_profile(p1)

    p2 = store.get_profile("user-2")
    assert p2.behavioral_preferences.pace is None
