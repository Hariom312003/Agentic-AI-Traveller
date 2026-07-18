"""
Explainability & telemetry.

Every agent execution is recorded as an `AgentExecutionRecord` and appended
to `state["execution_trace"]` via a LangGraph reducer (see
src/models/state.py). This is what powers the "Explainability" feature
requested in the spec: which agent ran, in what order, how long it took,
which LLM (if any) answered, what was retrieved, and a short reasoning
summary — all inspectable per-request, not just in a local log file.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from pydantic import BaseModel, Field
from src.utils.timeutil import utc_now


class AgentExecutionRecord(BaseModel):
    agent_name: str
    started_at: datetime
    finished_at: datetime | None = None
    latency_ms: float | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    retry_count: int = 0
    success: bool = True
    error: str | None = None
    reasoning_summary: str | None = None
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class _RunHandle:
    """Small mutable box returned by `track_agent` so the caller can attach
    extra fields (provider used, doc ids, summary) before the block exits."""

    def __init__(self, record: AgentExecutionRecord):
        self.record = record


@contextmanager
def track_agent(agent_name: str) -> Iterator[_RunHandle]:
    """Context manager that times an agent's execution and yields a handle
    the agent body can annotate. On exception, records failure and re-raises
    so the graph's own error handling still applies — telemetry never
    swallows errors."""
    record = AgentExecutionRecord(agent_name=agent_name, started_at=utc_now())
    handle = _RunHandle(record)
    start = time.perf_counter()
    try:
        yield handle
    except Exception as exc:  # noqa: BLE001 - intentionally broad, re-raised below
        record.success = False
        record.error = str(exc)
        raise
    finally:
        record.finished_at = utc_now()
        record.latency_ms = round((time.perf_counter() - start) * 1000, 2)
