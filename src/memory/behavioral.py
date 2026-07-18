"""
Behavioral memory — style-transfer preference learning.

The point of this module: a preference learned on a Goa trip ("prefers
relaxed pacing", "leans budget", "likes nightlife") should usefully inform
a Bali trip planned six months later. It should NOT store "liked Baga
Beach" and then be useless anywhere that isn't Goa. So every update here
strips place names, hotel names, and destination-specific nouns before
anything is persisted — the profile only ever holds *style* signals.

`scrub_place_names` is intentionally conservative: given a free-text note
and a list of known place names from the current trip's context, it removes
those exact names (case-insensitively) and collapses whitespace. It does
not try to be a general NER system — for this system's purposes, "the
current itinerary's place names" is exactly the set of things that must
never leak into behavioral memory.
"""
from __future__ import annotations

import re

from src.models.itinerary import Itinerary
from src.models.request import TripRequest
from src.models.user import BehavioralPreferences, PastTrip, UserProfile
from src.utils.timeutil import utc_now

_PACE_KEYWORDS = {
    "relaxed": ["relaxed", "slow", "chill", "laid back", "leisurely", "less hectic"],
    "packed": ["packed", "jam-packed", "fast paced", "action packed", "see everything"],
}

_BUDGET_KEYWORDS = {
    "budget": ["budget", "cheap", "backpacking", "affordable", "economical"],
    "luxury": ["luxury", "5 star", "five star", "premium", "high end", "lavish"],
}

# Heuristic, not an NER model: 2+ consecutive Capitalized Words (optionally
# joined by "of"/"the"/"and"/"de"/"da") almost always means a proper noun —
# in travel free-text, overwhelmingly a place name ("Baga Beach", "Gateway
# of India", "Hidimba Devi Temple"). This exists because `scrub_place_names`
# can only strip names it's explicitly told about (the current trip's
# destination + any itinerary activity locations); a user's free-text note
# can casually mention OTHER specific places (a beach, a restaurant) that
# never appear in that explicit list. Since under-scrubbing here means a
# destination-specific name leaking into cross-destination memory (the
# exact thing this module exists to prevent) and over-scrubbing only costs
# a little color in a stored note, this deliberately errs aggressive.
_PROPER_NOUN_RUN_RE = re.compile(
    r"\b[A-Z][a-zA-Z']*(?:\s+(?:of|the|and|de|da)\s+[A-Z][a-zA-Z']*|\s+[A-Z][a-zA-Z']*)+\b"
)


def scrub_place_names(text: str, place_names: list[str]) -> str:
    """Two passes: first strip every explicitly-known place name (the
    current destination, past destinations, itinerary activity locations),
    then run the proper-noun heuristic for anything specific that slipped
    through free text but wasn't in that list."""
    scrubbed = text
    for name in sorted(place_names, key=len, reverse=True):
        if not name:
            continue
        scrubbed = re.sub(re.escape(name), "[place]", scrubbed, flags=re.IGNORECASE)
    scrubbed = _PROPER_NOUN_RUN_RE.sub("[place]", scrubbed)
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    return scrubbed


def infer_style_signals(trip_request: TripRequest) -> dict[str, str]:
    """Cheap keyword-based style inference from the free-text request +
    explicit fields — no LLM call needed for this, and it's easy to unit
    test deterministically."""
    haystack = " ".join(
        filter(None, [trip_request.raw_query, trip_request.travel_style, trip_request.special_requests])
    ).lower()

    signals: dict[str, str] = {}
    for pace, keywords in _PACE_KEYWORDS.items():
        if any(k in haystack for k in keywords):
            signals["pace"] = pace
            break
    for tier, keywords in _BUDGET_KEYWORDS.items():
        if any(k in haystack for k in keywords):
            signals["budget_tier"] = tier
            break
    return signals


def update_preferences_from_trip(
    profile: UserProfile,
    trip_request: TripRequest,
    itinerary: Itinerary | None,
    learning_rate: float = 0.35,
) -> UserProfile:
    """Incrementally folds one trip's signals into the persisted,
    destination-agnostic profile. Interest weights use exponential
    smoothing (`learning_rate`) so a single trip nudges the profile without
    overwriting years of history in one request."""
    prefs = profile.behavioral_preferences

    signals = infer_style_signals(trip_request)
    if "pace" in signals:
        prefs.pace = signals["pace"]
    if "budget_tier" in signals:
        prefs.budget_tier = signals["budget_tier"]

    for interest in trip_request.interests:
        key = interest.strip().lower()
        if not key:
            continue
        current = prefs.interest_weights.get(key, 0.5)
        prefs.interest_weights[key] = round(current + learning_rate * (1.0 - current), 3)

    for food in trip_request.food_preferences:
        if food and food not in prefs.food_preferences:
            prefs.food_preferences.append(food)

    if trip_request.transport_preference:
        prefs.preferred_transport = trip_request.transport_preference

    if trip_request.special_requests:
        place_names = [d.destination for d in profile.past_trips] + (
            [itinerary.destination] if itinerary else []
        )
        if trip_request.destination:
            place_names.append(trip_request.destination)
        if itinerary:
            place_names.extend(a.location or "" for a in itinerary.all_activities())
        note = scrub_place_names(trip_request.special_requests, [p for p in place_names if p])
        if note and note not in prefs.notes:
            prefs.notes.append(note)
            prefs.notes = prefs.notes[-10:]  # bounded history

    destination = trip_request.destination or (itinerary.destination if itinerary else None)
    if destination:
        profile.past_trips.append(
            PastTrip(
                destination=destination,
                duration_days=trip_request.duration_days or (itinerary.duration_days if itinerary else None),
                travel_style=trip_request.travel_style,
            )
        )
        profile.past_trips = profile.past_trips[-25:]

    profile.behavioral_preferences = prefs
    profile.updated_at = utc_now()
    return profile


def preferences_as_prompt_context(prefs: BehavioralPreferences) -> str:
    """Renders the profile as a short natural-language block the Planner
    Agent can drop straight into its prompt — deliberately place-name-free."""
    lines = []
    if prefs.pace:
        lines.append(f"- Preferred pacing: {prefs.pace}")
    if prefs.budget_tier:
        lines.append(f"- Budget tier leaning: {prefs.budget_tier}")
    if prefs.interest_weights:
        top = sorted(prefs.interest_weights.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("- Strongest interests: " + ", ".join(f"{k} ({v:.1f})" for k, v in top))
    if prefs.food_preferences:
        lines.append("- Food preferences: " + ", ".join(prefs.food_preferences))
    if prefs.preferred_transport:
        lines.append(f"- Preferred transport: {prefs.preferred_transport}")
    if prefs.notes:
        lines.append("- Past style notes: " + "; ".join(prefs.notes[-3:]))
    return "\n".join(lines) if lines else "No prior preferences on file yet."
