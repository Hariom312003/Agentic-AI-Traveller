"""
Planner Agent.

Generates the day-by-day itinerary. Two things here are worth calling out
because they're easy to get wrong:

1. We never trust the model's self-reported "this came from the knowledge
   base" claim (models will say that regardless of whether it's true). We
   independently fuzzy-match every returned activity title against the
   `allowed_places` list built by the RAG Agent — the SAME duplicate-
   matching function the Validator uses for a different purpose — and set
   `source`/`source_doc_ids` ourselves. That is the actual hallucination
   check for this system: verify against the retrieved set, don't ask the
   model to grade its own homework.
2. On a repair pass (`state["planner_attempts"] > 0`, i.e. the Validator
   sent us back), we feed the *specific* validation issues into the prompt
   and ask for a full itinerary again — repair happens at the "regenerate
   the plan" granularity, not surgical editing. Surgical, lock-respecting
   editing is a different code path entirely (src/refinement/), used for
   user-requested day-specific changes after a plan already exists.
"""
from __future__ import annotations

import uuid

from src.agents.common import SchemaGenerationError, generate_structured
from src.llm.router import AllProvidersExhaustedError
from src.memory.behavioral import preferences_as_prompt_context
from src.models.itinerary import Itinerary
from src.models.state import TripState
from src.monitoring.telemetry import track_agent
from src.planning_engine.conversion import PlannerDayOut, PlannerItineraryOut, to_domain_day
from src.planning_engine.fallback_planner import build_fallback_itinerary

SYSTEM_PROMPT = """You are the Planner agent of a travel planning system. You produce a
complete, realistic day-by-day itinerary as JSON.

Hard rules:
- You MUST schedule the pre-routed verified places exactly as outlined in the "Verified Curated Knowledge Base Skeleton" for each day.
- Do NOT move places between days, and do NOT repeat places.
- For each activity, write a specific, descriptive summary, provide estimated cost and duration, and explain why it was selected.
- If told there is NO curated knowledge base for this destination, you may use your own world knowledge freely, but schedule real, famous attractions and restaurants (never use generic templates like 'visit a museum' or 'walk downtown').
- Every day should have at least one activity in morning, afternoon, and evening; night is optional.
- Respect the traveler's stated budget tier, interests, and pace.
- `category` must be one of: attraction, food, transport, shopping, nightlife, rest,
  adventure, culture, nature, other.
- `time_slot` must be one of: morning, afternoon, evening, night.
"""


def _build_prompt(state: TripState, allowed_places: list[dict], prompt_block: str, preference_context: str) -> str:
    trip_request = state["trip_request"]
    lines = [
        f"Destination: {trip_request.destination}",
        f"Duration: {trip_request.duration_days or 'unspecified — default to 4'} days",
        f"Travelers: {trip_request.travelers_count or 1} ({trip_request.travelers_type or 'unspecified'})",
        f"Travel style: {trip_request.travel_style or 'unspecified'}",
        f"Budget: {trip_request.budget_amount or 'unspecified'} {trip_request.budget_currency}",
        f"Interests: {', '.join(trip_request.interests) or 'unspecified'}",
        f"Food preferences: {', '.join(trip_request.food_preferences) or 'unspecified'}",
        f"Special requests: {trip_request.special_requests or 'none'}",
        f"Season/timing: {trip_request.season or 'unspecified'}",
        "",
        "Traveler's known preferences from past trips (destination-agnostic):",
        preference_context,
        "",
        prompt_block,
    ]

    if state.get("planner_attempts", 0) > 0 and state.get("validation_report"):
        report = state["validation_report"]
        issues_text = "\n".join(f"- [{i.severity.value}] {i.message}" for i in report.issues)
        lines += [
            "",
            "IMPORTANT — this is a repair pass. The previous itinerary had these validation issues:",
            issues_text,
            "Generate a full corrected itinerary that resolves them (e.g. remove duplicates, reduce cost, "
            "fill empty days) while keeping everything else that was working well.",
        ]

    return "\n".join(lines)


def run_planner_agent(state: TripState) -> dict:
    with track_agent("planner_agent") as handle:
        trip_request = state["trip_request"]
        rag_ctx = (state.get("retrieved_context") or [{}])[0]
        allowed_places = rag_ctx.get("allowed_places", [])
        prompt_block = rag_ctx.get("prompt_block", "")
        grounded = rag_ctx.get("grounded", False)
        preference_context = preferences_as_prompt_context(
            state["user_profile"].behavioral_preferences if state.get("user_profile") else None  # type: ignore[arg-type]
        ) if state.get("user_profile") else "No prior preferences on file yet."

        duration_days = trip_request.duration_days or 4
        provider = model = None
        fallback_used = False

        try:
            prompt = _build_prompt(state, allowed_places, prompt_block, preference_context)
            output, meta = generate_structured(
                system_prompt=SYSTEM_PROMPT, user_prompt=prompt, schema=PlannerItineraryOut, temperature=0.5,
            )
            provider, model = meta.provider, meta.model
            handle.record.retry_count = meta.retries
            days = [to_domain_day(d, allowed_places, grounded, trip_request.budget_currency) for d in output.days]
            if not days:
                raise SchemaGenerationError("Model returned zero days")
            itinerary = Itinerary(
                itinerary_id=f"trip-{uuid.uuid4().hex[:10]}",
                destination=trip_request.destination or "Unknown",
                origin=trip_request.origin,
                duration_days=len(days),
                travelers_count=trip_request.travelers_count or 1,
                travel_style=trip_request.travel_style,
                currency=trip_request.budget_currency,
                days=days,
            )
            handle.record.reasoning_summary = f"Generated {len(days)}-day itinerary via {provider}/{model}"

        except Exception as exc:
            fallback_used = True
            itinerary = build_fallback_itinerary(
                destination=trip_request.destination or "your destination",
                duration_days=duration_days,
                travelers_count=trip_request.travelers_count or 1,
                travel_style=trip_request.travel_style,
                allowed_places=allowed_places,
                currency=trip_request.budget_currency,
            )
            handle.record.reasoning_summary = f"Planner exception triggered ({exc}); used rule-based fallback planner"
            handle.record.error = str(exc)

        handle.record.llm_provider = provider
        handle.record.llm_model = model
        handle.record.extra["fallback_used"] = fallback_used
        handle.record.retrieved_doc_ids = [c for p in allowed_places for c in ([p["chunk_id"]] if p.get("chunk_id") else [])]

        return {
            "itinerary": itinerary,
            "planner_attempts": state.get("planner_attempts", 0) + 1,
            "execution_trace": [handle.record],
        }
