"""
Refinement Agent.

Regenerates ONLY the day(s) resolved by `resolve_target_days` and merges
via `apply_refinement`, which is the actual lock enforcement (see
src/refinement/locking.py's module docstring for the full "why" — this is
review fix #2). The LLM is never shown, and never allowed to influence,
any day outside the target set.
"""
from __future__ import annotations

from src.agents.common import SchemaGenerationError, generate_structured
from src.llm.router import AllProvidersExhaustedError
from src.models.itinerary import Day
from src.models.state import TripState
from src.monitoring.logging_config import get_logger
from src.monitoring.telemetry import track_agent
from src.planning_engine.conversion import PlannerDayOut, to_domain_day
from src.planning_engine.fallback_planner import build_fallback_day
from src.refinement.locking import apply_refinement, compute_pre_hashes, resolve_target_days, verify_locks_held

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Refinement agent of a travel planning system. You are given
ONE existing day of an itinerary and an editing instruction. Regenerate ONLY that single
day as JSON matching the requested change. Keep the same `day_number`. Do not reference or
assume anything about other days — you cannot see them and your output only replaces this
one day.
"""


def _regenerate_single_day(
    day: Day, instruction: str, destination: str, allowed_places: list[dict], grounded: bool, currency: str,
) -> tuple[Day, str | None, str | None, int]:
    current_day_text = "\n".join(
        f"- [{slot}] {a.title}: {a.description}" for slot in ("morning", "afternoon", "evening", "night")
        for a in day.slot(slot)
    ) or "(this day is currently empty)"

    places_text = (
        "\n".join(f"- {p['name']} ({p['category']})" for p in allowed_places)
        if allowed_places else "(no curated knowledge base for this destination — use general knowledge, "
                                 "and it's fine, this will be labeled as unverified automatically)"
    )

    prompt = (
        f"Destination: {destination}\n"
        f"Day number: {day.day_number}\n"
        f"Current content of this day:\n{current_day_text}\n\n"
        f"Available verified places for this destination:\n{places_text}\n\n"
        f"Editing instruction: {instruction}\n\n"
        f"Regenerate this single day to satisfy the instruction."
    )

    output, meta = generate_structured(
        system_prompt=SYSTEM_PROMPT, user_prompt=prompt, schema=PlannerDayOut, temperature=0.5,
    )
    if output.day_number != day.day_number:
        output.day_number = day.day_number  # defensive — never trust the model to echo this correctly
    new_day = to_domain_day(output, allowed_places, grounded, currency)
    return new_day, meta.provider, meta.model, meta.retries


def run_refinement_agent(state: TripState) -> dict:
    with track_agent("refinement_agent") as handle:
        itinerary = state["itinerary"]
        instruction = state["refinement_instruction"]
        explicit_targets = state.get("target_days")
        rag_ctx = (state.get("retrieved_context") or [{}])[0]
        allowed_places = rag_ctx.get("allowed_places", [])
        grounded = rag_ctx.get("grounded", False)

        target_days = resolve_target_days(instruction, itinerary, explicit_targets)
        pre_hashes = compute_pre_hashes(itinerary)

        regenerated: dict[int, Day] = {}
        providers_used = []
        total_retries = 0
        errors: list[str] = []
        fallback_days: list[int] = []
        already_used_names = {
            a.title for d in itinerary.days if d.day_number not in target_days for a in d.all_activities()
        }

        for day_number in target_days:
            day = itinerary.day(day_number)
            if day is None:
                continue
            try:
                new_day, provider, model, retries = _regenerate_single_day(
                    day, instruction, itinerary.destination, allowed_places, grounded, itinerary.currency
                )
                regenerated[day_number] = new_day
                if provider:
                    providers_used.append(provider)
                total_retries += retries
            except (AllProvidersExhaustedError, SchemaGenerationError) as exc:
                logger.warning("refinement_day_llm_unavailable", extra={"day_number": day_number, "error": str(exc)})
                # Degrade to the same rule-based day builder the Planner uses
                # when every provider is down, rather than either silently
                # leaving the (explicitly requested) day untouched or
                # crashing the whole refinement. Still fully transparent:
                # activities are tagged `rule_based_fallback`/`knowledge_base`
                # as appropriate and a warning is surfaced in the response.
                regenerated[day_number] = build_fallback_day(
                    day_number, itinerary.destination, allowed_places, already_used_names, theme=day.theme
                )
                fallback_days.append(day_number)
                errors.append(f"Day {day_number}: AI provider unavailable ({exc}); used rule-based regeneration instead")

        updated_itinerary = apply_refinement(itinerary, regenerated, target_days)
        violated = verify_locks_held(pre_hashes, updated_itinerary, target_days)
        if violated:
            # Should be structurally impossible given apply_refinement's construction
            # (see locking.py docstring) — surface loudly rather than silently proceed.
            msg = f"Lock verification FAILED for days {violated} — this indicates a merge bug, not a prompting issue"
            logger.error("lock_violation_detected", extra={"violated_days": violated})
            errors.append(msg)

        handle.record.llm_provider = providers_used[0] if providers_used else None
        handle.record.retry_count = total_retries
        handle.record.reasoning_summary = (
            f"Regenerated day(s) {target_days} of {len(itinerary.days)} "
            f"({len(fallback_days)} via rule-based fallback); "
            f"{len(itinerary.days) - len(target_days)} day(s) verified byte-identical (locked)"
        )
        handle.record.extra["fallback_days"] = fallback_days
        if errors:
            handle.record.error = "; ".join(errors)

        return {
            "itinerary": updated_itinerary,
            "target_days": target_days,
            "pre_refinement_hashes": pre_hashes,
            "errors": errors,
            "planner_attempts": state.get("planner_attempts", 0) + 1,
            "execution_trace": [handle.record],
        }
