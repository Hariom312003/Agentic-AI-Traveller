"""
User memory models.

`BehavioralPreferences` is deliberately destination-agnostic: no city or
attraction names live here, only *style* signals ("prefers relaxed pacing",
"leans budget", "likes nightlife"). That's what lets a preference learned in
Goa usefully generalize to a Bali trip six months later instead of
overfitting to specific place names. The scrubbing happens in
src/memory/behavioral.py; this model is just the shape the scrubbed data
takes.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field
from src.utils.timeutil import utc_now


class PastTrip(BaseModel):
    destination: str
    start_date: str | None = None
    duration_days: int | None = None
    travel_style: str | None = None
    satisfaction_signal: str | None = None  # "liked" | "neutral" | "disliked", if ever captured


class BehavioralPreferences(BaseModel):
    """Destination-agnostic style profile, updated incrementally."""

    pace: str | None = None  # "relaxed" | "moderate" | "packed"
    budget_tier: str | None = None  # "budget" | "mid_range" | "luxury"
    interest_weights: dict[str, float] = Field(default_factory=dict)  # e.g. {"nightlife": 0.8}
    food_preferences: list[str] = Field(default_factory=list)
    preferred_transport: str | None = None
    favorite_airlines: list[str] = Field(default_factory=list)
    favorite_hotel_chains: list[str] = Field(default_factory=list)
    walking_tolerance: str | None = None  # "low" | "medium" | "high"
    notes: list[str] = Field(default_factory=list)  # short scrubbed style notes


class UserProfile(BaseModel):
    user_id: str
    behavioral_preferences: BehavioralPreferences = Field(default_factory=BehavioralPreferences)
    past_trips: list[PastTrip] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
