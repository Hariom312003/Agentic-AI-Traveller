"""
Structural validation rules beyond duplicate detection.

Each `check_*` function is pure: (itinerary, context) -> list[ValidationIssue].
`run_all_checks` composes them. Kept as small pure functions rather than one
monolithic validator method because each one gets its own focused unit test
in tests/test_validation_rules.py, and because the Validator Agent can
selectively re-run a subset of checks after a refinement pass without
re-running everything.
"""
from __future__ import annotations

from src.models.itinerary import Activity, Itinerary
from src.models.validation import IssueCategory, IssueSeverity, ValidationIssue
from src.validation.duplicate_checker import find_duplicate_pairs

# Rough opening-hours heuristic: activities tagged nightlife shouldn't be
# scheduled in the morning slot, attractions with "temple"/"museum" style
# categories shouldn't be the sole content of a night slot. This is a
# *heuristic* safety net, not a live opening-hours API — see docs/architecture.md
# for why a live-hours integration is a documented extension point rather
# than something faked here.
_SLOT_CATEGORY_CONFLICTS: dict[str, set[str]] = {
    "morning": {"nightlife"},
    "night": {"culture"},  # most museums/temples are closed at night
}


def check_duplicates(itinerary: Itinerary, threshold: float) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    id_to_day = {a.id: d.day_number for d in itinerary.days for a in d.all_activities()}
    named = [(a.id, a.title) for a in itinerary.all_activities()]
    for id_a, id_b, score in find_duplicate_pairs(named, threshold=threshold):
        issues.append(
            ValidationIssue(
                category=IssueCategory.DUPLICATE_ACTIVITY,
                severity=IssueSeverity.WARNING,
                message=f"Possible duplicate activities ({score:.0f}% similar): {id_a} and {id_b}",
                day_number=id_to_day.get(id_a),
                affected_activity_ids=[id_a, id_b],
            )
        )
    return issues


def check_budget(itinerary: Itinerary, budget_total: float | None, actual_total: float) -> list[ValidationIssue]:
    if budget_total is None or budget_total <= 0:
        return []
    overflow_ratio = actual_total / budget_total
    if overflow_ratio > 1.15:
        return [ValidationIssue(
            category=IssueCategory.BUDGET_OVERFLOW,
            severity=IssueSeverity.CRITICAL,
            message=f"Estimated cost {actual_total:,.0f} exceeds budget {budget_total:,.0f} by {(overflow_ratio - 1) * 100:.0f}%",
        )]
    if overflow_ratio > 1.0:
        return [ValidationIssue(
            category=IssueCategory.BUDGET_OVERFLOW,
            severity=IssueSeverity.WARNING,
            message=f"Estimated cost {actual_total:,.0f} slightly exceeds budget {budget_total:,.0f}",
        )]
    return []


def check_day_balance(itinerary: Itinerary) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for day in itinerary.days:
        count = len(day.all_activities())
        if count == 0:
            issues.append(ValidationIssue(
                category=IssueCategory.EMPTY_SLOT,
                severity=IssueSeverity.CRITICAL,
                message=f"Day {day.day_number} has no activities scheduled at all",
                day_number=day.day_number,
            ))
        elif count > 7:
            issues.append(ValidationIssue(
                category=IssueCategory.DAY_IMBALANCE,
                severity=IssueSeverity.WARNING,
                message=f"Day {day.day_number} is overloaded with {count} activities — consider a lighter pace",
                day_number=day.day_number,
            ))
    return issues


def check_slot_category_conflicts(itinerary: Itinerary) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for day in itinerary.days:
        for slot_name, forbidden_categories in _SLOT_CATEGORY_CONFLICTS.items():
            for activity in day.slot(slot_name):
                if activity.category.value in forbidden_categories:
                    issues.append(ValidationIssue(
                        category=IssueCategory.TRAVEL_TIME_CONFLICT,
                        severity=IssueSeverity.WARNING,
                        message=f"'{activity.title}' ({activity.category.value}) looks unusual in the {slot_name} slot on Day {day.day_number}",
                        day_number=day.day_number,
                        affected_activity_ids=[activity.id],
                    ))
    return issues


def check_groundedness(itinerary: Itinerary) -> tuple[list[ValidationIssue], float]:
    activities = itinerary.all_activities()
    if not activities:
        return [], 0.0
    grounded = sum(1 for a in activities if a.source.value == "knowledge_base")
    ratio = grounded / len(activities)
    issues: list[ValidationIssue] = []
    if ratio < 0.3:
        issues.append(ValidationIssue(
            category=IssueCategory.UNGROUNDED_CONTENT,
            severity=IssueSeverity.INFO,
            message=(
                f"Only {ratio * 100:.0f}% of activities are grounded in the curated knowledge base "
                f"for {itinerary.destination}; the rest rely on the model's general knowledge and "
                f"are labeled accordingly."
            ),
        ))
    return issues, ratio


def route_efficiency_score(itinerary: Itinerary) -> float:
    """A simple, explainable proxy for "did we bounce around too much":
    penalizes days that mix many distinct location strings across slots,
    since that usually means excess transit. Returns 0..1, higher=better.
    This is intentionally not a real routing-distance calculation (that
    needs geocoding + a maps API — a documented extension point, not faked
    here with placeholder coordinates)."""
    if not itinerary.days:
        return 1.0
    day_scores = []
    for day in itinerary.days:
        activities = day.all_activities()
        if len(activities) <= 1:
            day_scores.append(1.0)
            continue
        distinct_locations = len({a.location for a in activities if a.location})
        # more distinct locations per activity => more hopping around
        ratio = distinct_locations / max(len(activities), 1)
        day_scores.append(max(0.0, 1.0 - max(0.0, ratio - 0.5)))
    return round(sum(day_scores) / len(day_scores), 3)


def run_all_checks(
    itinerary: Itinerary,
    budget_total: float | None,
    actual_total: float,
    duplicate_threshold: float,
) -> tuple[list[ValidationIssue], float, float]:
    """Returns (issues, grounded_ratio, route_score)."""
    issues: list[ValidationIssue] = []
    issues += check_duplicates(itinerary, duplicate_threshold)
    issues += check_budget(itinerary, budget_total, actual_total)
    issues += check_day_balance(itinerary)
    issues += check_slot_category_conflicts(itinerary)
    groundedness_issues, grounded_ratio = check_groundedness(itinerary)
    issues += groundedness_issues
    route_score = route_efficiency_score(itinerary)
    return issues, grounded_ratio, route_score
