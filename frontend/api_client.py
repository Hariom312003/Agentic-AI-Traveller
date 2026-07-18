"""
API client for the Streamlit frontend.

Deliberately thin: every function is a direct 1:1 wrapper over one backend
endpoint, returning parsed JSON so views never construct URLs or handle raw
`requests` objects themselves. Every function raises exactly one exception
type — `APIError` — whether the failure was the backend returning a 4xx/5xx
(server reachable, request rejected) or the backend being completely
unreachable (connection refused, DNS failure, timeout). Views only ever
need to catch `APIError` once; they don't need to know or care which of
those two very different failure modes actually happened.
"""
from __future__ import annotations

import os

import requests

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 120  # itinerary generation can involve several LLM calls + retries


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


def _request(method: str, path: str, **kwargs) -> dict:
    try:
        response = requests.request(method, f"{BASE_URL}{path}", **kwargs)
    except requests.exceptions.RequestException as exc:
        raise APIError(0, f"Could not reach the backend at {BASE_URL} ({exc.__class__.__name__}). "
                           f"Is it running? Start it with ./run_api.sh.") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(response.status_code, detail)
    return response.json()


def health() -> dict:
    return _request("GET", "/health", timeout=10)


def plan_trip(payload: dict) -> dict:
    return _request("POST", "/plan", json=payload, timeout=TIMEOUT_SECONDS)


def refine_trip(session_id: str, instruction: str, target_days: list[int] | None) -> dict:
    payload = {"session_id": session_id, "instruction": instruction, "target_days": target_days}
    return _request("POST", "/refine", json=payload, timeout=TIMEOUT_SECONDS)


def list_checkpoints(session_id: str) -> dict:
    return _request("POST", "/rollback", json={"session_id": session_id}, timeout=15)


def rollback_to(session_id: str, checkpoint_id: str) -> dict:
    payload = {"session_id": session_id, "checkpoint_id": checkpoint_id}
    return _request("POST", "/rollback", json=payload, timeout=30)


def get_memory(user_id: str) -> dict:
    return _request("GET", f"/memory/{user_id}", timeout=10)


def provider_status() -> dict:
    return _request("GET", "/providers/status", timeout=10)


def recent_logs(limit: int = 200) -> list[dict]:
    try:
        return _request("GET", "/logs/recent", params={"limit": limit}, timeout=10)
    except APIError:
        return []  # the log console is a nice-to-have; never let it break the rest of the page


def is_backend_reachable() -> bool:
    try:
        health()
        return True
    except APIError:
        return False
