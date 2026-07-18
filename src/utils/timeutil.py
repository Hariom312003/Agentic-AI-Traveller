"""Tiny shared time helper — one place to get a UTC timestamp, so every
model uses the same (timezone-aware, non-deprecated) call instead of each
file independently choosing between `datetime.utcnow()` (deprecated as of
Python 3.12) and `datetime.now(timezone.utc)`."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
