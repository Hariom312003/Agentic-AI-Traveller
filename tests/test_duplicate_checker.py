"""
Tests for the fuzzy duplicate checker (review issue #3).

The labeled pairs below are the exact cases used to choose the scoring
function and threshold in `src/validation/duplicate_checker.py` — this test
isn't just "does it run", it's "does it still separate these specific
true/false cases", so a future threshold tweak can't silently regress it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.duplicate_checker import (
    duplicate_score,
    find_duplicate_pairs,
    is_near_duplicate,
    normalize_name,
)

LABELED_PAIRS: list[tuple[str, str, bool]] = [
    ("Gateway of India", "The Gateway of India Monument", True),
    ("Baga Beach", "Calangute Beach", False),
    ("Baga Beach", "Baga beach!!", True),
    ("Anjuna Flea Market", "Anjuna Market", True),
    ("Fort Aguada", "Chapora Fort", False),
    ("Dudhsagar Falls", "Dudhsagar Waterfalls", True),
    ("Old Goa Churches", "Basilica of Bom Jesus", False),
    ("Calangute Beach", "Baga Beach Shacks", False),
    ("Anjuna Beach", "Anjuna beach", True),  # pure case difference — old exact-match-after-lower WOULD catch this
    ("Anjuna Beach", "Vagator Beach", False),
]


def test_labeled_pairs_classified_correctly():
    for name_a, name_b, expected in LABELED_PAIRS:
        result = is_near_duplicate(name_a, name_b)
        assert result == expected, (
            f"{name_a!r} vs {name_b!r}: expected duplicate={expected}, "
            f"got {result} (score={duplicate_score(name_a, name_b):.1f})"
        )


def test_exact_match_still_caught():
    # sanity: fuzzy matching must not regress the trivial case the old
    # exact-string-match implementation *did* handle correctly.
    assert is_near_duplicate("Baga Beach", "Baga Beach")


def test_punctuation_and_case_are_ignored():
    assert normalize_name("Fort Aguada!!") == normalize_name("fort aguada")


def test_this_is_not_exact_string_matching():
    """The specific regression this fix targets: two clearly-the-same
    places whose strings are NOT identical (even after lowercasing) must
    still be flagged. This would fail against the original
    `location.lower() in global_used_locations` implementation."""
    assert "the gateway of india monument" != "gateway of india"  # confirms strings truly differ
    assert is_near_duplicate("Gateway of India", "The Gateway of India Monument")


def test_find_duplicate_pairs_over_a_list():
    names = [
        ("a1", "Baga Beach"),
        ("a2", "Baga beach!!"),
        ("a3", "Calangute Beach"),
        ("a4", "Fort Aguada"),
    ]
    dupes = find_duplicate_pairs(names)
    dupe_ids = {frozenset((a, b)) for a, b, _ in dupes}
    assert frozenset(("a1", "a2")) in dupe_ids
    assert frozenset(("a1", "a3")) not in dupe_ids
    assert frozenset(("a3", "a4")) not in dupe_ids


def test_empty_strings_are_never_duplicates():
    assert not is_near_duplicate("", "")
    assert not is_near_duplicate("", "Baga Beach")
