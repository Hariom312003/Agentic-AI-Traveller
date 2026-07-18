"""
Itinerary domain models.

These are the nouns of the whole system: an `Itinerary` is a list of `Day`s,
each `Day` has up to four `TimeSlot` buckets, each holding `Activity` items.

Two fields matter more than they look like they should:

- `Activity.source` / `source_doc_ids` — explainability + hallucination
  control. Every activity is tagged with where it came from so the UI can
  show "grounded in knowledge base" vs "model general knowledge, unverified".
- `Day.content_hash()` — the enforcement mechanism for surgical refinement.
  A day's hash is computed over its actual content, not over a "locked"
  flag someone could forget to set. See src/refinement/locking.py.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from src.utils.timeutil import utc_now


class TimeSlotName(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


class ActivityCategory(str, Enum):
    ATTRACTION = "attraction"
    FOOD = "food"
    TRANSPORT = "transport"
    SHOPPING = "shopping"
    NIGHTLIFE = "nightlife"
    REST = "rest"
    ADVENTURE = "adventure"
    CULTURE = "culture"
    NATURE = "nature"
    OTHER = "other"


class ActivitySource(str, Enum):
    """Where an activity's factual content came from — the backbone of
    hallucination transparency. Nothing is silently presented as verified
    when it wasn't retrieved from the knowledge base."""

    KNOWLEDGE_BASE = "knowledge_base"     # grounded in curated/ingested RAG docs
    MODEL_KNOWLEDGE = "model_knowledge"   # LLM's own world knowledge, unverified
    RULE_BASED_FALLBACK = "rule_based_fallback"  # emergency planner, no LLM involved
    USER_SPECIFIED = "user_specified"     # user explicitly asked for this place


class Activity(BaseModel):
    id: str
    title: str
    description: str = ""
    category: ActivityCategory = ActivityCategory.OTHER
    location: str | None = None
    start_time: str | None = None  # "HH:MM", kept as free string across timezones
    duration_minutes: int | None = None
    estimated_cost: float | None = None
    currency: str = "INR"
    source: ActivitySource = ActivitySource.MODEL_KNOWLEDGE
    source_doc_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    booking_required: bool = False
    notes: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None
    map_link: str | None = None

    def canonical(self) -> dict[str, Any]:
        """Deterministic dict representation used for hashing — excludes
        nothing that affects meaning, includes nothing volatile (no
        timestamps)."""
        return self.model_dump(mode="json")


class TimeSlot(BaseModel):
    name: TimeSlotName
    activities: list[Activity] = Field(default_factory=list)


class Day(BaseModel):
    day_number: int
    # NOTE: named `travel_date`, not `date` — a field named identically to
    # its own type (`date: date | None`) breaks pydantic's forward-ref
    # resolution under `from __future__ import annotations`, because the
    # class-body assignment shadows the imported `date` type before pydantic
    # evaluates the annotation string.
    travel_date: date | None = None
    theme: str | None = None
    morning: list[Activity] = Field(default_factory=list)
    afternoon: list[Activity] = Field(default_factory=list)
    evening: list[Activity] = Field(default_factory=list)
    night: list[Activity] = Field(default_factory=list)
    pacing: str | None = None  # "relaxed" | "moderate" | "packed"
    locked: bool = False

    def all_activities(self) -> list[Activity]:
        return [*self.morning, *self.afternoon, *self.evening, *self.night]

    def slot(self, name: TimeSlotName | str) -> list[Activity]:
        name = TimeSlotName(name) if not isinstance(name, TimeSlotName) else name
        return getattr(self, name.value)

    def set_slot(self, name: TimeSlotName | str, activities: list[Activity]) -> None:
        name = TimeSlotName(name) if not isinstance(name, TimeSlotName) else name
        setattr(self, name.value, activities)

    def content_hash(self) -> str:
        """SHA-256 over the day's actual content (all slots, sorted keys,
        no volatile fields). This is what "locked" means in this system —
        not a prompt instruction, a value that can be recomputed and
        compared. See src/refinement/locking.py for the enforcement path."""
        payload = {
            "day_number": self.day_number,
            "theme": self.theme,
            "pacing": self.pacing,
            "morning": [a.canonical() for a in self.morning],
            "afternoon": [a.canonical() for a in self.afternoon],
            "evening": [a.canonical() for a in self.evening],
            "night": [a.canonical() for a in self.night],
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Itinerary(BaseModel):
    itinerary_id: str
    destination: str
    origin: str | None = None
    duration_days: int
    travelers_count: int = 1
    travel_style: str | None = None
    currency: str = "INR"
    days: list[Day] = Field(default_factory=list)
    version: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("days")
    @classmethod
    def _days_sorted(cls, v: list[Day]) -> list[Day]:
        return sorted(v, key=lambda d: d.day_number)

    def day(self, day_number: int) -> Day | None:
        return next((d for d in self.days if d.day_number == day_number), None)

    def all_activities(self) -> list[Activity]:
        return [a for d in self.days for a in d.all_activities()]

    def day_hashes(self) -> dict[int, str]:
        return {d.day_number: d.content_hash() for d in self.days}

    def bump_version(self) -> "Itinerary":
        self.version += 1
        self.updated_at = utc_now()
        return self
