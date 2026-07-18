"""
Refinement locking tests (review issue #2).

The key thing this suite proves that a superficial test wouldn't: it's not
enough that `apply_refinement` *usually* leaves other days alone — we
assert byte-identical content hashes, and in
`test_verify_locks_held_actually_detects_a_violation` we deliberately
construct the exact failure mode the old system had (a day's content
silently changed) and confirm the checker catches it. A checker that can
never fail isn't a checker.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.itinerary import Activity, ActivityCategory, ActivitySource, Day, Itinerary
from src.refinement.locking import apply_refinement, compute_pre_hashes, resolve_target_days, verify_locks_held


def _activity(id_: str, title: str) -> Activity:
    return Activity(id=id_, title=title, category=ActivityCategory.ATTRACTION, source=ActivitySource.KNOWLEDGE_BASE)


def _sample_itinerary(n_days: int = 3) -> Itinerary:
    days = []
    for i in range(1, n_days + 1):
        days.append(Day(
            day_number=i,
            theme=f"Day {i} theme",
            morning=[_activity(f"d{i}-m", f"Day {i} morning activity")],
            afternoon=[_activity(f"d{i}-a", f"Day {i} afternoon activity")],
        ))
    return Itinerary(itinerary_id="test-trip", destination="Goa", duration_days=n_days, days=days)


# ---- resolve_target_days ---------------------------------------------------

def test_explicit_target_days_wins_over_text():
    it = _sample_itinerary(3)
    result = resolve_target_days("please change everything", it, explicit_target_days=[2])
    assert result == [2]


def test_parses_single_day_mention():
    it = _sample_itinerary(3)
    assert resolve_target_days("Replace Day 2 with adventure activities", it, None) == [2]


def test_parses_day_range():
    it = _sample_itinerary(4)
    assert resolve_target_days("Make days 2-4 more relaxed", it, None) == [2, 3, 4]


def test_parses_comma_separated_days():
    it = _sample_itinerary(5)
    assert resolve_target_days("Update days 2, 3 and 5 for more nightlife", it, None) == [2, 3, 5]


def test_unscoped_instruction_applies_to_all_days():
    it = _sample_itinerary(3)
    assert resolve_target_days("Make the whole trip more luxurious", it, None) == [1, 2, 3]


def test_out_of_range_day_is_ignored():
    it = _sample_itinerary(2)
    # Day 9 doesn't exist on a 2-day trip -> falls through to "no valid explicit match"
    result = resolve_target_days("Change day 9", it, None)
    assert result == [1, 2]  # unscoped fallback, since the only mentioned day is invalid


# ---- apply_refinement / lock enforcement -----------------------------------

def test_untouched_days_are_byte_identical_after_refinement():
    original = _sample_itinerary(3)
    pre_hashes = compute_pre_hashes(original)

    new_day_2 = Day(
        day_number=2, theme="New adventure theme",
        morning=[_activity("new-1", "Adventure activity")],
    )
    result = apply_refinement(original, {2: new_day_2}, target_days=[2])

    assert result.day(1).content_hash() == pre_hashes[1]
    assert result.day(3).content_hash() == pre_hashes[3]
    assert result.day(2).content_hash() != pre_hashes[2]
    assert result.day(2).theme == "New adventure theme"


def test_untouched_days_are_marked_locked_target_day_is_not():
    original = _sample_itinerary(3)
    new_day_2 = Day(day_number=2, theme="New theme", morning=[_activity("new-1", "New activity")])
    result = apply_refinement(original, {2: new_day_2}, target_days=[2])

    assert result.day(1).locked is True
    assert result.day(3).locked is True
    assert result.day(2).locked is False


def test_version_bumps_on_refinement():
    original = _sample_itinerary(2)
    assert original.version == 1
    new_day_1 = Day(day_number=1, theme="x", morning=[_activity("a", "a")])
    result = apply_refinement(original, {1: new_day_1}, target_days=[1])
    assert result.version == 2


def test_verify_locks_held_passes_for_correct_refinement():
    original = _sample_itinerary(3)
    pre_hashes = compute_pre_hashes(original)
    new_day_2 = Day(day_number=2, theme="New theme", morning=[_activity("new-1", "New activity")])
    result = apply_refinement(original, {2: new_day_2}, target_days=[2])
    assert verify_locks_held(pre_hashes, result, target_days=[2]) == []


def test_verify_locks_held_actually_detects_a_violation():
    """Simulates the exact bug the code review caught: a day outside the
    target set gets silently altered. This does not go through
    `apply_refinement` (which cannot produce this state) — it directly
    constructs the broken output an unconstrained "ask the LLM nicely"
    approach could produce, to prove the checker is not a tautology."""
    original = _sample_itinerary(3)
    pre_hashes = compute_pre_hashes(original)

    tampered = original.model_copy(deep=True)
    tampered.days[0].morning[0].title = "Silently altered by the model"  # Day 1 was NOT a target day

    violations = verify_locks_held(pre_hashes, tampered, target_days=[2])
    assert violations == [1]


def test_original_object_is_not_mutated_by_refinement():
    """apply_refinement must deep-copy, not alias, untouched days — if it
    aliased them, an in-place mutation on the *result* would corrupt the
    caller's original reference too."""
    original = _sample_itinerary(2)
    new_day_1 = Day(day_number=1, theme="new", morning=[_activity("x", "x")])
    result = apply_refinement(original, {1: new_day_1}, target_days=[1])

    result.day(2).morning[0].title = "mutated after the fact"
    assert original.day(2).morning[0].title == "Day 2 morning activity"
