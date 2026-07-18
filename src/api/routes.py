"""
API routes.

Kept thin on purpose: every route does request validation (via pydantic,
automatic through FastAPI) + orchestration (build state, invoke the right
graph, shape the response) and delegates all actual logic to
`src.graph.workflow` / `src.agents.*`. No business logic lives here.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from src.agents.rag_agent import get_retriever
from src.agents.summary_agent import build_final_response
from src.api.schemas import (
    CheckpointListResponse,
    CheckpointSummary,
    HealthResponse,
    ProviderStatusResponse,
)
from src.config import get_settings
from src.graph.workflow import get_graph_manager
from src.llm.health import get_health_registry
from src.memory.store import get_memory_store
from src.models.request import RefinementRequest, RollbackRequest, TripRequest
from src.models.state import new_state
from src.monitoring.logging_config import read_recent_logs

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    try:
        doc_count = get_retriever().vector_store.count()
    except Exception:
        doc_count = 0
    configured = [p for p in settings.provider_priority_list() if settings.key_for(p)]
    return HealthResponse(
        status="ok", environment=settings.environment,
        configured_providers=configured, vector_store_documents=doc_count,
    )


@router.post("/plan")
def plan(request: TripRequest) -> dict:
    session_id = request.session_id or f"session-{uuid.uuid4().hex[:12]}"
    request.session_id = session_id
    manager = get_graph_manager()
    state = new_state(mode="plan", trip_request=request, user_id=request.user_id, session_id=session_id)

    try:
        result = manager.plan_graph.invoke(state, config=manager.config_for(session_id))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Planning failed: {exc}") from exc

    return {"session_id": session_id, **result["final_response"]}


@router.post("/refine")
def refine(request: RefinementRequest) -> dict:
    manager = get_graph_manager()
    config = manager.config_for(request.session_id)
    snapshot = manager.get_state(request.session_id)

    if not snapshot.values or not snapshot.values.get("itinerary"):
        raise HTTPException(
            status_code=404,
            detail=f"No existing itinerary found for session '{request.session_id}'. Call /plan first.",
        )

    refine_state = dict(snapshot.values)
    refine_state.update({
        "mode": "refine",
        "refinement_instruction": request.instruction,
        "target_days": request.target_days,
        "planner_attempts": 0,
        "validation_report": None,
        "execution_trace": [],
        "errors": [],
    })

    try:
        result = manager.refine_graph.invoke(refine_state, config=config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Refinement failed: {exc}") from exc

    return {"session_id": request.session_id, **result["final_response"]}


@router.post("/rollback")
def rollback(request: RollbackRequest) -> dict:
    manager = get_graph_manager()

    if request.checkpoint_id is None:
        history = manager.history(request.session_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"No history for session '{request.session_id}'")
        summaries = [
            CheckpointSummary(
                checkpoint_id=snap.config["configurable"]["checkpoint_id"],
                created_at=snap.created_at,
                next_nodes=list(snap.next),
                has_itinerary=bool(snap.values.get("itinerary")),
                itinerary_version=snap.values["itinerary"].version if snap.values.get("itinerary") else None,
            )
            for snap in history
        ]
        return CheckpointListResponse(session_id=request.session_id, checkpoints=summaries).model_dump()

    rollback_cfg = manager.rollback_config(request.session_id, request.checkpoint_id)
    if rollback_cfg is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{request.checkpoint_id}' not found")

    snapshot = manager.plan_graph.get_state(rollback_cfg)
    # Re-commit the historical values as a new checkpoint on top of the
    # thread's history, so this rollback becomes the new "tip" for any
    # subsequent /refine call on this session_id (see
    # CompiledStateGraph.update_state — this is LangGraph's supported
    # time-travel/fork mechanism, not a workaround).
    manager.plan_graph.update_state(manager.config_for(request.session_id), snapshot.values)

    return {"session_id": request.session_id, "rolled_back_to": request.checkpoint_id, **build_final_response(snapshot.values)}


@router.get("/memory/{user_id}")
def get_memory(user_id: str) -> dict:
    profile = get_memory_store().get_profile(user_id)
    return profile.model_dump(mode="json")


@router.get("/providers/status", response_model=ProviderStatusResponse)
def provider_status() -> ProviderStatusResponse:
    return ProviderStatusResponse(providers=get_health_registry().snapshot())


@router.get("/logs/recent")
def recent_logs(limit: int = 200) -> list[dict]:
    return read_recent_logs(limit=limit)
