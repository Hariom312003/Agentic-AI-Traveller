# API Reference

Base URL: `http://localhost:8000` (configurable via `API_BASE_URL`/`API_PORT`).
Interactive Swagger UI: `http://localhost:8000/docs` — every schema below is also
available there, generated directly from the pydantic models, so it can't drift out of
sync with the code.

All request/response bodies are JSON.

## `GET /health`

Liveness + basic environment info. Used by Docker healthchecks and the Streamlit
sidebar's "Backend connected" indicator.

**Response**
```json
{
  "status": "ok",
  "environment": "development",
  "configured_providers": ["gemini"],
  "vector_store_documents": 140
}
```

## `POST /plan`

Runs the full planning graph from scratch. This is the only endpoint that creates a
new session.

**Request** (`TripRequest` — see `src/models/request.py`; only `raw_query` is
required, everything else is optional and either explicitly provided or extracted by
the Query Agent from `raw_query`)
```json
{
  "raw_query": "5 day trip to Goa under 60000 for 2 people, love nightlife and forts",
  "user_id": "demo-traveler",
  "destination": "Goa",
  "duration_days": 5,
  "travelers_count": 2,
  "travel_style": "budget",
  "budget_amount": 60000,
  "budget_currency": "INR",
  "interests": ["nightlife", "culture"],
  "food_preferences": ["vegetarian"],
  "special_requests": "want to see Dudhsagar Falls"
}
```

**Response** — `200 OK`
```json
{
  "session_id": "session-a1b2c3d4e5f6",
  "itinerary": { "itinerary_id": "...", "destination": "Goa", "days": [ ... ], "version": 1 },
  "budget": { "flights": 12000, "hotels": 9000, "...": "...", "total": 34500, "currency": "INR" },
  "rewards": { "recommendations": [ ... ], "disclaimer": "..." },
  "validation": { "is_valid": true, "issues": [ ... ], "grounded_ratio": 0.83 },
  "explainability": {
    "execution_trace": [ { "agent_name": "query_agent", "latency_ms": 812.3, "...": "..." } ],
    "total_agents_run": 9,
    "total_latency_ms": 5230.1,
    "providers_used": ["gemini"]
  },
  "warnings": []
}
```

**Errors:** `422` on an empty/invalid request body (pydantic validation); `500` if the
graph raises unexpectedly (should be rare — most failure modes degrade to a fallback
path rather than raising).

## `POST /refine`

Surgically edits specific day(s) of an existing session's itinerary. Requires `/plan`
to have already been called for this `session_id`.

**Request** (`RefinementRequest`)
```json
{
  "session_id": "session-a1b2c3d4e5f6",
  "instruction": "Replace Day 2 with more relaxed, nightlife-focused activities",
  "target_days": [2]
}
```

`target_days` is the reliable way to scope an edit (a day-picker in the UI). If
omitted, the instruction text is parsed for day mentions ("Day 2", "days 2-4", "days 2,
3 and 5"); if none are found either, the instruction is treated as applying to every
day. See `src/refinement/locking.py::resolve_target_days`.

**Response** — same shape as `/plan`'s response. `itinerary.version` is incremented;
every day *not* in `target_days` is guaranteed byte-identical to before (`locked:
true` in its JSON) — see [`docs/architecture.md`](architecture.md#2--locked-day-editing-wasnt-enforced).

**Errors:** `404` if `session_id` has no existing itinerary (call `/plan` first); `500`
on an unexpected graph error.

## `POST /rollback`

Two modes depending on whether `checkpoint_id` is provided.

**List available checkpoints** — request:
```json
{ "session_id": "session-a1b2c3d4e5f6" }
```
Response:
```json
{
  "session_id": "session-a1b2c3d4e5f6",
  "checkpoints": [
    { "checkpoint_id": "1ef...", "created_at": "2026-07-17T05:30:00Z",
      "next_nodes": [], "has_itinerary": true, "itinerary_version": 2 },
    { "checkpoint_id": "1ef...", "created_at": "2026-07-17T05:29:40Z",
      "next_nodes": ["summary_agent"], "has_itinerary": true, "itinerary_version": 1 }
  ]
}
```
Newest first.

**Roll back to a specific checkpoint** — request:
```json
{ "session_id": "session-a1b2c3d4e5f6", "checkpoint_id": "1ef..." }
```
Response: same shape as `/plan`'s response, plus `"rolled_back_to": "<checkpoint_id>"`.
This also commits the restored state as the new tip, so a subsequent `/refine` call
continues from the rolled-back point.

**Errors:** `404` if the session has no history, or the given `checkpoint_id` doesn't
exist for it.

## `GET /memory/{user_id}`

Returns the destination-agnostic behavioral profile for a user (`UserProfile` — see
`src/models/user.py`). Returns a fresh, empty profile (not a 404) for a user_id that's
never been seen before.

## `GET /providers/status`

Returns the live `ProviderHealthRegistry` snapshot — circuit state, success/failure
counts, rate-limit hit count, last error per configured provider. Powers the Agent
Monitor page's provider health table.

## `GET /logs/recent?limit=200`

Returns the last `limit` structured log lines (JSON objects) from the backend's log
file. Powers the Agent Monitor page's log console. Returns `[]` (not an error) if no
log file exists yet.

## Error shape

Unhandled exceptions return:
```json
{ "error": "internal_server_error", "detail": "<message>" }
```
Validation errors (422) use FastAPI/pydantic's standard error format.
