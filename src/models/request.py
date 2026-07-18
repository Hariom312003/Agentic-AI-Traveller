"""Inbound request models — what the API and Streamlit frontend send in."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class TripRequest(BaseModel):
    """Structured form of a trip planning ask. `raw_query` is always kept
    so the Query Agent can run NL extraction even when the frontend also
    supplies structured fields (structured fields win on conflict — they
    represent an explicit user choice in the UI, e.g. a form field)."""

    raw_query: str = Field(..., description="Free-text description of the trip")
    user_id: str = "anonymous"
    session_id: str | None = None

    destination: str | None = None
    origin: str | None = None
    duration_days: int | None = Field(default=None, ge=1, le=60)
    start_date: date | None = None
    travelers_count: int | None = Field(default=None, ge=1, le=50)
    travelers_type: str | None = None  # solo, couple, family, friends, business
    travel_style: str | None = None  # luxury, budget, backpacking, honeymoon, adventure...
    budget_amount: float | None = Field(default=None, ge=0)
    budget_currency: str = "INR"
    interests: list[str] = Field(default_factory=list)
    food_preferences: list[str] = Field(default_factory=list)
    transport_preference: str | None = None
    season: str | None = None
    special_requests: str | None = None
    constraints: list[str] = Field(default_factory=list)

    @field_validator("raw_query")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("raw_query cannot be empty")
        return v.strip()


class RefinementRequest(BaseModel):
    """A surgical edit request against an existing itinerary/session.

    `target_days` is the primary, unambiguous mechanism for scoping a change
    (driven by an explicit day-picker in the UI). `instruction` is always
    used for *what* to do; it is also used to *infer* target_days via
    regex ("Day 2", "days 2 and 3") when the caller does not pass
    target_days explicitly — see src/refinement/locking.py::resolve_target_days.
    """

    session_id: str
    instruction: str
    target_days: list[int] | None = None

    @field_validator("instruction")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("instruction cannot be empty")
        return v.strip()


class RollbackRequest(BaseModel):
    session_id: str
    checkpoint_id: str | None = None  # None => list available checkpoints instead of rolling back
