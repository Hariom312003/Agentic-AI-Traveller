"""
Input Safety & Prompt Injection Classification.
"""
from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?instructions",
    r"ignore\s+(?:previous|above)",
    r"system\s+prompt",
    r"system\s+instruction",
    r"delete\s+itinerary",
    r"bypass\s+validation",
    r"reveal\s+instructions",
    r"output\s+only",
    r"you\s+are\s+no\s+longer",
]


def validate_input_safety(raw_query: str) -> tuple[bool, str | None]:
    """
    Validates user raw query for prompt injections and gibberish (e.g., emoji-only).
    Returns (is_safe, error_message).
    """
    clean_query = raw_query.strip()
    if not clean_query:
        return False, "Query is empty"

    # Check for alphabetic character count to catch emoji-only/symbol-only queries
    alpha_chars = sum(1 for c in clean_query if c.isalpha())
    if alpha_chars < 2:
        return False, "Query contains insufficient text characters (e.g. emoji-only or symbols only)"

    # Check prompt injection patterns
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, clean_query, re.IGNORECASE):
            return False, "Input rejected: potential prompt injection detected"

    return True, None
