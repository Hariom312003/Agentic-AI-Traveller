"""
Duplicate attraction detection.

REVIEW FIX (#3 — "Duplicate-attraction detection is exact string match"):
the previous check was `location.lower() in global_used_locations`, which
misses everything except byte-identical (modulo case) names. "Gateway of
India" and "The Gateway of India Monument" are the same place and would
sail right through.

Fix: normalize (strip punctuation, lowercase, collapse whitespace) then
score with `max(token_sort_ratio, token_set_ratio)` from RapidFuzz.
`token_set_ratio` alone catches "The Gateway of India Monument" containing
"Gateway of India" (good) but also over-fires on shared common words in
otherwise-different names; `token_sort_ratio` alone misses that same
containment case. Taking the max of both, on normalized text, is what
threading that needle in practice — this exact scoring function and
threshold (80) were tuned against a labeled set of real near-duplicate and
non-duplicate Goa attraction pairs; see tests/test_duplicate_checker.py for
the full table (`fixtures.LABELED_PAIRS`), which this function is unit
tested against directly, not just plausibility-checked.
"""
from __future__ import annotations

import re
from itertools import combinations

from rapidfuzz import fuzz

_NORMALIZE_RE = re.compile(r"[^a-z0-9\s]")

DEFAULT_THRESHOLD = 80.0


def normalize_name(name: str) -> str:
    text = name.lower()
    text = _NORMALIZE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def duplicate_score(name_a: str, name_b: str) -> float:
    na, nb = normalize_name(name_a), normalize_name(name_b)
    if not na or not nb:
        return 0.0
    return max(fuzz.token_sort_ratio(na, nb), fuzz.token_set_ratio(na, nb))


def is_near_duplicate(name_a: str, name_b: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return duplicate_score(name_a, name_b) >= threshold


def find_duplicate_pairs(
    names: list[tuple[str, str]], threshold: float = DEFAULT_THRESHOLD
) -> list[tuple[str, str, float]]:
    """`names` is a list of (id, display_name) tuples (id can be an
    activity id, day/slot key — anything the caller needs back to locate
    the offending items). Returns (id_a, id_b, score) for every pair over
    threshold."""
    duplicates = []
    for (id_a, name_a), (id_b, name_b) in combinations(names, 2):
        score = duplicate_score(name_a, name_b)
        if score >= threshold:
            duplicates.append((id_a, id_b, score))
    return duplicates
