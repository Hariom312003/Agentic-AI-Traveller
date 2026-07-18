"""
Query Agent.

Turns free-text ("Plan a 5 day trip to Tokyo under ₹120000, adventurous,
love street food") into a structured, validated `TripRequest`. Structured
fields the caller already supplied (e.g. from a UI form) are treated as
authoritative and never overwritten by the LLM's extraction — the model
only fills in what's missing. If the model call fails entirely (all
providers exhausted) we fall back to a cheap regex/keyword extractor so a
destination-only query like "plan a trip to Goa" still produces something
usable instead of hard failing the whole pipeline before it even starts.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.agents.common import SchemaGenerationError, generate_structured
from src.llm.router import AllProvidersExhaustedError
from src.models.request import TripRequest
from src.models.state import TripState
from src.monitoring.telemetry import track_agent

SYSTEM_PROMPT = """You are the Query Understanding agent of a travel planning system.
Extract structured trip details from a traveller's free-text request.
Rules:
- Only extract what is actually stated or strongly implied; use null for anything you're not confident about.
- `destination` should be the primary city/region name only (e.g. "Tokyo", not "a trip to Tokyo").
- `interests` and `food_preferences` are short lowercase tags (e.g. "nightlife", "adventure", "vegetarian").
- Never invent a budget, duration, or destination that wasn't mentioned.
"""

_DURATION_RE = re.compile(r"(\d+)\s*[- ]?\s*(day|days|night|nights)", re.IGNORECASE)
_BUDGET_RE = re.compile(r"(?:under|within|budget of|around)?\s*[₹$€£]\s?([\d,]+)", re.IGNORECASE)
# Catches "trip to Goa", "vacation in Tokyo", "for Paris" etc — a best-effort
# proper-noun grab, not NER. Good enough as a fallback because it only needs
# to beat "no destination at all" (see _keyword_fallback_extract's docstring).
_DESTINATION_PHRASE_RE = re.compile(
    r"\b(?:to|in|for|around|near|visit|visiting|explore|exploring)\s+"
    r"([A-Z][a-zA-Z']*(?:\s+[A-Z][a-zA-Z']*){0,2})\b"
)


class ExtractedFields(BaseModel):
    destination: str | None = None
    origin: str | None = None
    duration_days: int | None = None
    travelers_count: int | None = None
    travelers_type: str | None = None
    travel_style: str | None = None
    budget_amount: float | None = None
    budget_currency: str | None = None
    interests: list[str] = Field(default_factory=list)
    food_preferences: list[str] = Field(default_factory=list)
    transport_preference: str | None = None
    season: str | None = None
    special_requests: str | None = None
    constraints: list[str] = Field(default_factory=list)


def _guess_destination(raw_query: str) -> str | None:
    """Two tiers: first check whether any of the destinations we actually
    have curated knowledge-base data for is mentioned (exact, high
    confidence — and it means the RAG Agent will find real grounded
    content, not just a guessed name). Falls back to a "to/in <Capitalized
    Words>" phrase grab, which is weaker but still far better than leaving
    the destination blank — a blank destination cascades into every
    downstream agent describing the trip as "your destination" instead of
    anything useful. This exists because the LLM-based extraction (the
    primary path) has no such gap; this is specifically the safety net for
    when *that* is also unavailable."""
    from src.rag.chunking import known_destinations
    from src.config import get_settings

    try:
        for name in known_destinations(get_settings().destinations_data_path):
            if re.search(rf"\b{re.escape(name)}\b", raw_query, re.IGNORECASE):
                return name
    except Exception:
        pass  # never let a KB-listing hiccup break the whole fallback path

    match = _DESTINATION_PHRASE_RE.search(raw_query)
    if match:
        candidate = match.group(1).strip()
        # filter out common false positives this pattern tends to grab
        if candidate.lower() not in {"me", "us", "we", "them", "budget", "days", "day"}:
            return candidate
    return None


def _keyword_fallback_extract(raw_query: str) -> ExtractedFields:
    """Zero-LLM safety net: a destination is the one field the rest of the
    pipeline cannot proceed without, so we try hard to recover at least
    that plus duration/budget via regex before giving up."""
    fields = ExtractedFields()
    fields.destination = _guess_destination(raw_query)
    duration_match = _DURATION_RE.search(raw_query)
    if duration_match:
        fields.duration_days = int(duration_match.group(1))
    budget_match = _BUDGET_RE.search(raw_query)
    if budget_match:
        fields.budget_amount = float(budget_match.group(1).replace(",", ""))
    for style in ("luxury", "budget", "backpacking", "honeymoon", "family", "business", "adventure", "solo"):
        if style in raw_query.lower():
            fields.travel_style = style
            break
    return fields


def run_query_agent(state: TripState) -> dict:
    from src.validation.input_safety import validate_input_safety

    with track_agent("query_agent") as handle:
        trip_request: TripRequest = state["trip_request"]
        provider = model = None
        errors: list[str] = []
        
        is_safe, safety_err = validate_input_safety(trip_request.raw_query)
        if not is_safe:
            extracted = ExtractedFields(
                destination="your destination",
                travel_style="standard",
                duration_days=3,
            )
            warn_msg = safety_err or "Flagged by input safety filter"
            errors.append(warn_msg)
            handle.record.reasoning_summary = f"Safety filter active: {warn_msg}. Using safe placeholders."
        else:
            try:
                extracted, meta = generate_structured(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=f"Traveller's request:\n\"{trip_request.raw_query}\"",
                    schema=ExtractedFields,
                    temperature=0.1,
                )
                provider, model = meta.provider, meta.model
                handle.record.retry_count = meta.retries
                handle.record.reasoning_summary = f"Extracted structured fields via {provider}/{model}"
            except (AllProvidersExhaustedError, SchemaGenerationError) as exc:
                extracted = _keyword_fallback_extract(trip_request.raw_query)
                handle.record.reasoning_summary = f"LLM extraction unavailable ({exc}); used keyword fallback"

        handle.record.llm_provider = provider
        handle.record.llm_model = model

        # Structured fields the caller already provided win over extraction.
        merged = trip_request.model_copy(
            update={
                k: getattr(trip_request, k) or v
                for k, v in extracted.model_dump().items()
                if k in TripRequest.model_fields and v not in (None, [], "")
            }
        )
        if not merged.destination:
            merged.destination = "your destination"
        if not merged.travel_style:
            merged.travel_style = "standard"
            
        # interests/food_preferences/constraints should union, not just prefer one side
        merged.interests = list(dict.fromkeys([*trip_request.interests, *extracted.interests]))
        merged.food_preferences = list(dict.fromkeys([*trip_request.food_preferences, *extracted.food_preferences]))
        merged.constraints = list(dict.fromkeys([*trip_request.constraints, *extracted.constraints]))

        return {
            "trip_request": merged,
            "parsed_request": merged.model_dump(mode="json"),
            "execution_trace": [handle.record],
            "errors": errors,
        }
