from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.itinerary import Activity, ActivityCategory, ActivitySource, Day, Itinerary
from src.validation.rules import (
    check_budget,
    check_day_balance,
    check_duplicates,
    check_groundedness,
    check_slot_category_conflicts,
    route_efficiency_score,
)


def _activity(id_, title, category=ActivityCategory.ATTRACTION, source=ActivitySource.KNOWLEDGE_BASE, location=None):
    return Activity(id=id_, title=title, category=category, source=source, location=location)


def test_check_duplicates_flags_near_duplicate_titles():
    itinerary = Itinerary(itinerary_id="t1", destination="Goa", duration_days=1, days=[
        Day(day_number=1, morning=[_activity("a1", "Baga Beach")], afternoon=[_activity("a2", "Baga beach!!")])
    ])
    issues = check_duplicates(itinerary, threshold=80.0)
    assert len(issues) == 1
    assert issues[0].category.value == "duplicate_activity"


def test_check_duplicates_no_false_positive_for_distinct_places():
    itinerary = Itinerary(itinerary_id="t1", destination="Goa", duration_days=1, days=[
        Day(day_number=1, morning=[_activity("a1", "Baga Beach")], afternoon=[_activity("a2", "Fort Aguada")])
    ])
    assert check_duplicates(itinerary, threshold=80.0) == []


def test_check_budget_flags_critical_overflow():
    issues = check_budget(itinerary=None, budget_total=1000.0, actual_total=1300.0)
    assert len(issues) == 1
    assert issues[0].severity.value == "critical"


def test_check_budget_flags_warning_for_slight_overflow():
    issues = check_budget(itinerary=None, budget_total=1000.0, actual_total=1050.0)
    assert len(issues) == 1
    assert issues[0].severity.value == "warning"


def test_check_budget_ok_within_budget():
    assert check_budget(itinerary=None, budget_total=1000.0, actual_total=900.0) == []


def test_check_budget_skips_when_no_budget_given():
    assert check_budget(itinerary=None, budget_total=None, actual_total=5000.0) == []


def test_check_day_balance_flags_empty_day():
    itinerary = Itinerary(itinerary_id="t1", destination="Goa", duration_days=2, days=[
        Day(day_number=1, morning=[_activity("a1", "Something")]),
        Day(day_number=2),
    ])
    issues = check_day_balance(itinerary)
    assert len(issues) == 1
    assert issues[0].severity.value == "critical"
    assert issues[0].day_number == 2


def test_check_day_balance_flags_overloaded_day():
    activities = [_activity(f"a{i}", f"Place {i}") for i in range(8)]
    itinerary = Itinerary(itinerary_id="t1", destination="Goa", duration_days=1, days=[
        Day(day_number=1, morning=activities[:4], afternoon=activities[4:])
    ])
    issues = check_day_balance(itinerary)
    assert len(issues) == 1
    assert issues[0].severity.value == "warning"


def test_check_slot_category_conflicts_flags_nightlife_in_morning():
    itinerary = Itinerary(itinerary_id="t1", destination="Goa", duration_days=1, days=[
        Day(day_number=1, morning=[_activity("a1", "Club X", category=ActivityCategory.NIGHTLIFE)])
    ])
    issues = check_slot_category_conflicts(itinerary)
    assert len(issues) == 1


def test_check_groundedness_ratio_and_low_ratio_issue():
    itinerary = Itinerary(itinerary_id="t1", destination="Goa", duration_days=1, days=[
        Day(day_number=1, morning=[
            _activity("a1", "Place 1", source=ActivitySource.KNOWLEDGE_BASE),
            _activity("a2", "Place 2", source=ActivitySource.MODEL_KNOWLEDGE),
            _activity("a3", "Place 3", source=ActivitySource.MODEL_KNOWLEDGE),
            _activity("a4", "Place 4", source=ActivitySource.MODEL_KNOWLEDGE),
        ])
    ])
    issues, ratio = check_groundedness(itinerary)
    assert ratio == 0.25
    assert len(issues) == 1
    assert issues[0].severity.value == "info"


def test_check_groundedness_no_issue_when_mostly_grounded():
    itinerary = Itinerary(itinerary_id="t1", destination="Goa", duration_days=1, days=[
        Day(day_number=1, morning=[
            _activity("a1", "Place 1", source=ActivitySource.KNOWLEDGE_BASE),
            _activity("a2", "Place 2", source=ActivitySource.KNOWLEDGE_BASE),
        ])
    ])
    issues, ratio = check_groundedness(itinerary)
    assert ratio == 1.0
    assert issues == []


def test_route_efficiency_score_penalizes_scattered_locations():
    scattered = Itinerary(itinerary_id="t1", destination="Goa", duration_days=1, days=[
        Day(day_number=1, morning=[
            _activity("a1", "P1", location="Loc A"), _activity("a2", "P2", location="Loc B"),
            _activity("a3", "P3", location="Loc C"), _activity("a4", "P4", location="Loc D"),
        ])
    ])
    focused = Itinerary(itinerary_id="t2", destination="Goa", duration_days=1, days=[
        Day(day_number=1, morning=[
            _activity("a1", "P1", location="Loc A"), _activity("a2", "P2", location="Loc A"),
            _activity("a3", "P3", location="Loc A"), _activity("a4", "P4", location="Loc A"),
        ])
    ])
    assert route_efficiency_score(scattered) < route_efficiency_score(focused)


def test_route_efficiency_score_handles_empty_itinerary():
    empty = Itinerary(itinerary_id="t1", destination="Goa", duration_days=0, days=[])
    assert route_efficiency_score(empty) == 1.0
