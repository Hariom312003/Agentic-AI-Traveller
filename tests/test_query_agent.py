"""
Query Agent fallback extraction tests.

`test_keyword_fallback_extracts_known_destination_from_free_text` covers a
real bug caught via the full Streamlit smoke test (scripts/smoke_test_streamlit.py):
with zero LLM keys configured, a user typing only the free-text description
(not separately filling a "Destination" field) got "your destination" as a
placeholder everywhere downstream, because the original fallback extractor
only recovered duration/budget/style, never destination. Destination is the
one field the rest of the pipeline can't do anything useful without, so
it's covered here directly at the unit level, not just observed once via
a full end-to-end run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.query_agent import _guess_destination, _keyword_fallback_extract


def test_recognizes_a_seeded_kb_destination_by_exact_name():
    assert _guess_destination("Plan a 3 day trip to Goa for 2 people") == "Goa"
    assert _guess_destination("vacation in Tokyo for a week") == "Tokyo"
    assert _guess_destination("5 day honeymoon in Bali under 100000") == "Bali"


def test_falls_back_to_capitalized_phrase_for_unseeded_destination():
    assert _guess_destination("I want to visit Reykjavik next month") == "Reykjavik"


def test_returns_none_when_no_destination_is_mentioned():
    assert _guess_destination("Something relaxing please") is None
    assert _guess_destination("Plan a trip for my family") is None


def test_keyword_fallback_extracts_known_destination_from_free_text():
    fields = _keyword_fallback_extract("Plan a 3 day trip to Goa for 2 people, love nightlife and forts")
    assert fields.destination == "Goa"
    assert fields.duration_days == 3


def test_keyword_fallback_extracts_duration_and_budget_and_style():
    fields = _keyword_fallback_extract("Plan a 5 day backpacking trip to Manali under ₹40000")
    assert fields.destination == "Manali"
    assert fields.duration_days == 5
    assert fields.budget_amount == 40000.0
    assert fields.travel_style == "backpacking"


def test_keyword_fallback_handles_no_recognizable_fields_gracefully():
    fields = _keyword_fallback_extract("Surprise me")
    assert fields.destination is None
    assert fields.duration_days is None
