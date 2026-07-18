"""API-only response shapes — thin wrappers around domain models for
endpoints that don't return a full trip response (health, checkpoint
listings, provider status)."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    environment: str
    configured_providers: list[str]
    vector_store_documents: int


class CheckpointSummary(BaseModel):
    checkpoint_id: str
    created_at: str | None = None
    next_nodes: list[str]
    has_itinerary: bool
    itinerary_version: int | None = None


class CheckpointListResponse(BaseModel):
    session_id: str
    checkpoints: list[CheckpointSummary]


class ProviderStatusResponse(BaseModel):
    providers: dict[str, dict]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
