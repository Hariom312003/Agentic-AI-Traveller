"""Validation domain models — the Validator Agent's output contract."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"  # forces a re-plan loop if under retry budget


class IssueCategory(str, Enum):
    DUPLICATE_ACTIVITY = "duplicate_activity"
    BUDGET_OVERFLOW = "budget_overflow"
    DAY_IMBALANCE = "day_imbalance"
    EMPTY_SLOT = "empty_slot"
    TRAVEL_TIME_CONFLICT = "travel_time_conflict"
    LOCK_VIOLATION = "lock_violation"
    UNGROUNDED_CONTENT = "ungrounded_content"
    SCHEMA_ERROR = "schema_error"


class ValidationIssue(BaseModel):
    category: IssueCategory
    severity: IssueSeverity
    message: str
    day_number: int | None = None
    affected_activity_ids: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    duplicate_count: int = 0
    budget_status: str = "within_budget"  # within_budget | over_budget | unknown
    route_efficiency_score: float | None = None  # 0..1, higher is better
    grounded_ratio: float | None = None  # fraction of activities sourced from knowledge base

    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    def needs_replan(self) -> bool:
        return len(self.critical_issues()) > 0
