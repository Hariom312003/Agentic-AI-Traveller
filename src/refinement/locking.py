"""
Refinement locking — the actual enforcement mechanism.

REVIEW FIX (#2 — "'Byte-for-byte' locked day editing isn't actually
enforced but was mentioned that it was"): the previous implementation
computed `locked_days` *after* asking the LLM to regenerate the entire
itinerary JSON, purely by diffing which days happened not to change, and
merely told the model in a system prompt to "preserve unaffected days as
much as possible" — a request, not a guarantee. Nothing stopped the model
from subtly rewording, reordering, or dropping an activity in a day that
was supposed to be untouched.

The fix here doesn't ask the model to behave — it makes deviation
structurally impossible and independently checks the guarantee:

1. `resolve_target_days` decides which day(s) are in scope for this edit
   (explicit `target_days` from the API wins; free-text "Day 2" parsing is
   a convenience fallback).
2. The refinement agent (src/agents/refinement_agent.py) NEVER sends
   untouched days to the LLM and never lets the LLM's output replace them —
   `apply_refinement` below copies every non-target `Day` object from the
   ORIGINAL itinerary verbatim into the result. The model physically cannot
   alter a day it was never shown and whose slot in the output is never
   populated from its response.
3. `verify_locks_held` is a defense-in-depth assertion: it recomputes
   `Day.content_hash()` for every non-target day post-merge and confirms it
   is byte-identical to the pre-refinement hash. This should be
   mathematically impossible to fail given step 2's construction — if it
   ever does fail, that's a bug in the merge code, not a prompting issue,
   and `tests/test_refinement_locking.py` asserts it never does.
"""
from __future__ import annotations

import re

from src.models.itinerary import Day, Itinerary

_DAY_ANCHOR_RE = re.compile(r"days?\b", re.IGNORECASE)
_RANGE_RE = re.compile(r"(\d+)\s*(?:-|to|through)\s*(\d+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")
_WINDOW_CHARS = 40


def _extract_day_numbers(instruction: str) -> set[int]:
    """For every 'day'/'days' anchor, look at a short window of text right
    after it and pull out ranges ("2-4", "2 to 4") and individual numbers
    ("2, 3 and 5"). Anchored to the word "day(s)" so we don't misread
    unrelated numbers elsewhere in the instruction (a budget figure, a
    traveler count, etc) as day references."""
    found: set[int] = set()
    for anchor in _DAY_ANCHOR_RE.finditer(instruction):
        window = instruction[anchor.end() : anchor.end() + _WINDOW_CHARS]
        consumed_spans = []
        for m in _RANGE_RE.finditer(window):
            start, end = int(m.group(1)), int(m.group(2))
            if 1 <= start <= 60 and 1 <= end <= 60:  # sanity bound, not a real limit
                found.update(range(min(start, end), max(start, end) + 1))
            consumed_spans.append(m.span())
        for m in _NUMBER_RE.finditer(window):
            if any(s <= m.start() < e for s, e in consumed_spans):
                continue
            found.add(int(m.group()))
    return found


class LockViolationError(Exception):
    """Raised if a post-merge hash check ever disagrees with the pre-check
    — see module docstring. This indicates a bug in `apply_refinement`,
    not a prompting failure, since the LLM is never given the chance to
    touch a locked day's data at all."""


def resolve_target_days(
    instruction: str, itinerary: Itinerary, explicit_target_days: list[int] | None
) -> list[int]:
    """Explicit days from the API/UI always win — that's the reliable path
    (a day-picker control in the Streamlit refinement panel). Free-text
    parsing is a best-effort convenience for callers that only pass an
    instruction string; if neither yields anything, the whole itinerary is
    considered in scope (an unscoped instruction like "make it more
    luxurious" reasonably applies everywhere)."""
    valid_days = {d.day_number for d in itinerary.days}

    if explicit_target_days:
        resolved = [d for d in explicit_target_days if d in valid_days]
        if resolved:
            return sorted(set(resolved))

    found = _extract_day_numbers(instruction)
    resolved = sorted(d for d in found if d in valid_days)
    if resolved:
        return resolved

    return sorted(valid_days)  # unscoped instruction => applies to every day


def compute_pre_hashes(itinerary: Itinerary) -> dict[int, str]:
    return itinerary.day_hashes()


def apply_refinement(
    original: Itinerary,
    regenerated_days_by_number: dict[int, Day],
    target_days: list[int],
) -> Itinerary:
    """Builds the new itinerary. Every day NOT in `target_days` is the
    exact same `Day` object (deep-copied) from `original` — never anything
    that passed through the LLM. Only days in `target_days` are taken from
    `regenerated_days_by_number`."""
    new_days: list[Day] = []
    for day in original.days:
        if day.day_number in target_days and day.day_number in regenerated_days_by_number:
            new_day = regenerated_days_by_number[day.day_number]
            new_day.locked = False
        else:
            new_day = day.model_copy(deep=True)
            new_day.locked = True
        new_days.append(new_day)

    updated = original.model_copy(update={"days": new_days})
    updated.bump_version()
    return updated


def verify_locks_held(pre_hashes: dict[int, str], result: Itinerary, target_days: list[int]) -> list[int]:
    """Returns the list of day numbers whose lock was violated (should
    always be empty by construction — see module docstring). Never raises
    itself; the caller (refinement agent) decides whether a non-empty
    result is a hard error or just a logged anomaly, but it always logs
    loudly either way since this would indicate a real bug."""
    violated = []
    for day in result.days:
        if day.day_number in target_days:
            continue
        if day.day_number not in pre_hashes:
            continue
        if day.content_hash() != pre_hashes[day.day_number]:
            violated.append(day.day_number)
    return violated
