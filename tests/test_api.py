"""
API integration tests. Like the graph tests, these run with ZERO LLM
provider keys configured — every endpoint must still return a coherent,
well-formed response via the rule-based fallback paths. A fresh `TestClient`
(and fresh underlying app + all module singletons reset) is built per test
so tests never leak vector-store/checkpoint state into one another.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma_db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    get_settings.cache_clear()

    import src.agents.rag_agent as rag_agent_module
    import src.graph.workflow as workflow_module
    import src.memory.store as memory_store_module
    import src.llm.health as health_module

    rag_agent_module._retriever_singleton = None
    workflow_module._manager_singleton = None
    memory_store_module._store_singleton = None
    health_module._registry_instance = None

    from src.api.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_store_documents"] > 100  # the 5-destination seed KB


def test_plan_endpoint_returns_full_itinerary(client):
    response = client.post("/plan", json={
        "raw_query": "3 day budget trip to Goa with nightlife and culture",
        "destination": "Goa", "duration_days": 3, "travelers_count": 2,
        "interests": ["nightlife", "culture"],
    })
    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body
    assert body["itinerary"] is not None
    assert len(body["itinerary"]["days"]) == 3
    assert body["budget"]["total"] > 0
    assert body["explainability"]["total_agents_run"] >= 7


def test_plan_endpoint_rejects_empty_query(client):
    response = client.post("/plan", json={"raw_query": "   "})
    assert response.status_code == 422  # pydantic validation error


def test_refine_endpoint_requires_existing_session(client):
    response = client.post("/refine", json={"session_id": "does-not-exist", "instruction": "Replace day 1"})
    assert response.status_code == 404


def test_plan_then_refine_preserves_locked_days(client):
    plan_response = client.post("/plan", json={
        "raw_query": "3 day trip to Goa", "destination": "Goa", "duration_days": 3,
    })
    session_id = plan_response.json()["session_id"]
    original_days = plan_response.json()["itinerary"]["days"]

    refine_response = client.post("/refine", json={
        "session_id": session_id, "instruction": "Replace day 2 with more adventure activities", "target_days": [2],
    })
    assert refine_response.status_code == 200
    new_days = refine_response.json()["itinerary"]["days"]

    assert new_days[0]["morning"] == original_days[0]["morning"]
    assert new_days[2]["morning"] == original_days[2]["morning"]


def test_rollback_lists_checkpoints_when_no_id_given(client):
    plan_response = client.post("/plan", json={"raw_query": "2 day trip to Goa", "destination": "Goa", "duration_days": 2})
    session_id = plan_response.json()["session_id"]

    response = client.post("/rollback", json={"session_id": session_id})
    assert response.status_code == 200
    body = response.json()
    assert len(body["checkpoints"]) > 1


def test_rollback_to_specific_checkpoint(client):
    plan_response = client.post("/plan", json={"raw_query": "2 day trip to Manali", "destination": "Manali", "duration_days": 2})
    session_id = plan_response.json()["session_id"]

    listing = client.post("/rollback", json={"session_id": session_id}).json()
    earliest_checkpoint_id = listing["checkpoints"][-1]["checkpoint_id"]

    response = client.post("/rollback", json={"session_id": session_id, "checkpoint_id": earliest_checkpoint_id})
    assert response.status_code == 200
    assert response.json()["rolled_back_to"] == earliest_checkpoint_id


def test_rollback_unknown_checkpoint_returns_404(client):
    plan_response = client.post("/plan", json={"raw_query": "2 day trip to Goa", "destination": "Goa", "duration_days": 2})
    session_id = plan_response.json()["session_id"]
    response = client.post("/rollback", json={"session_id": session_id, "checkpoint_id": "bogus-id"})
    assert response.status_code == 404


def test_memory_endpoint_returns_profile_for_new_user(client):
    response = client.get("/memory/brand-new-user")
    assert response.status_code == 200
    assert response.json()["user_id"] == "brand-new-user"


def test_provider_status_endpoint(client):
    response = client.get("/providers/status")
    assert response.status_code == 200
    assert "providers" in response.json()
