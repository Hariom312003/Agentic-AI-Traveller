# Deployment Guide

## Contents
- [Local installation](#local-installation)
- [Docker deployment](#docker-deployment)
- [Configuration reference](#configuration-reference)
- [Production considerations](#production-considerations)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## Local installation

Requires Python 3.11+ (developed and tested against 3.12).

```bash
git clone <this-repo>
cd ai_traveller
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add whichever LLM provider key(s) you have — none are required to run
the app; see the root [README](../README.md#quickstart) for what changes with zero
keys configured. Then:

```bash
./run.sh          # runs both API and UI, Ctrl+C stops both
```

or individually (two terminals):
```bash
./run_api.sh       # http://localhost:8000  (Swagger docs at /docs)
./run_ui.sh         # http://localhost:8501
```

Both scripts create `./venv` automatically on first run if one doesn't already exist,
and copy `.env.example` → `.env` if you skip that step.

### Opening in VS Code / any IDE

This is a standard Python project — no special IDE configuration is required. Point
your IDE's Python interpreter at `./venv/bin/python` (or `venv\Scripts\python.exe` on
Windows) after running `pip install -r requirements.txt`, and both `src/` and
`frontend/` will resolve correctly since `PYTHONPATH` handling is done via each
entrypoint script (`run_api.sh`, `run_ui.sh`) and `sys.path.insert(0, ...)` at the top
of `app.py` — you don't need to install the project as a package.

## Docker deployment

```bash
cp .env.example .env      # add API keys
docker compose up --build
```

This builds two images from the same multi-stage `Dockerfile` (`api` and `ui` targets)
and runs them as separate services (`docker-compose.yml`), with:
- Named volumes for the vector store, SQLite databases, and logs (survive container
  restarts/rebuilds).
- A healthcheck on the API service; the UI service waits for it to be healthy before
  starting (`depends_on: condition: service_healthy`).
- The UI's `API_BASE_URL` pointed at the API container's internal Docker network
  hostname (`http://api:8000`), not `localhost`.

To run just the API (e.g. behind your own frontend):
```bash
docker build --target api -t ai-traveller-api .
docker run -p 8000:8000 --env-file .env -v ai_traveller_data:/app/data -v ai_traveller_chroma:/app/chroma_db ai-traveller-api
```

## Configuration reference

Every setting is documented inline in `.env.example` and typed/validated in
`src/config.py`. The ones most worth knowing about:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER_PRIORITY` | Comma-separated failover order | `gemini,groq,openrouter,gemini_backup,groq_backup,openai,anthropic` |
| `EMBEDDING_PROVIDER` | `gemini` (production) or `local` (offline fallback, used automatically if no Gemini key) | `gemini` |
| `CIRCUIT_FAILURE_THRESHOLD` | Consecutive failures before a provider's circuit opens | `3` |
| `CIRCUIT_RECOVERY_SECONDS` | Cooldown before a half-open retry probe | `60` |
| `MAX_PLANNER_REPAIR_ATTEMPTS`* | Replan-loop cap on critical validation issues | `2` |
| `DUPLICATE_FUZZY_THRESHOLD`* | RapidFuzz score threshold for flagging duplicates | `80` |

*(These two are set in `src/config.py`'s `Settings` defaults rather than `.env.example`,
since they're tuning parameters more than deployment config — override via environment
variable of the same name in SCREAMING_SNAKE_CASE if needed, pydantic-settings picks
them up automatically.)

## Production considerations

Read this alongside [`docs/architecture.md`'s "Extension points"](architecture.md#extension-points)
section — it covers the same ground from the "what would need to change" angle.

- **Checkpoint/memory storage is SQLite**, chosen deliberately for zero-ops local/small
  deployments. A single `sqlite3.Connection(..., check_same_thread=False)` is shared
  process-wide; SQLite serializes writes internally, which is fine for one process but
  **does not scale across multiple API worker processes** (e.g. `uvicorn --workers 4`)
  without moving to a networked checkpointer. LangGraph ships a Postgres checkpointer
  (`langgraph-checkpoint-postgres`) that's a drop-in replacement for `SqliteSaver` in
  `src/graph/workflow.py::GraphManager.__init__` if you need to scale horizontally.
- **The vector store (ChromaDB) is also local-disk (`PersistentClient`)**. For a
  multi-instance deployment, either mount a shared volume (works, but not built for
  concurrent writers) or move to Chroma's client/server mode or a hosted vector DB.
- **Rate limits / API costs**: `CIRCUIT_FAILURE_THRESHOLD` and
  `LLM_MAX_RETRIES_PER_PROVIDER` directly affect how aggressively the app burns through
  provider quota during an outage. The defaults are conservative (fail over fast rather
  than hammer a struggling provider) — tune based on your actual provider quotas.
- **CORS** (`CORS_ORIGINS`) defaults to localhost origins. Set this to your real
  frontend origin(s) before exposing the API beyond local development.
- **Logs** are written to `logs/app.log` as JSON lines with rotation
  (`RotatingFileHandler`, 5MB × 5 backups). For a real production deployment, ship
  these to your log aggregator of choice instead of relying on the Agent Monitor page's
  in-app console (which is a convenience for local/small deployments, not a substitute
  for real observability tooling at scale).

## Troubleshooting

**`pip install -r requirements.txt` fails with a dependency conflict.**
Shouldn't happen — every pin in `requirements.txt` was verified against a clean
install during development specifically to avoid this (see the comments inline in the
file for the three conflicts that were caught and fixed: `langchain-core` version
pinning, `httpx` range, `tenacity` range). If you hit a new one, it's likely a newer
transitive dependency shifted — try `pip install -r requirements.txt --upgrade` or
relax the specific pin that conflicts.

**The app runs but every itinerary looks generic / says "OFFLINE FALLBACK".**
No LLM provider key is configured (or all configured ones are failing). Check
`GET /health`'s `configured_providers` list, and `GET /providers/status` for circuit
state and `last_error` per provider. Add at least one working API key to `.env` and
restart the API.

**"No curated knowledge base entries" for a destination I expect to be covered.**
Only Goa, Manali, Tokyo, Paris, and Bali ship with curated data
(`data/destinations/*.json`). Anywhere else, the Planner Agent uses the LLM's general
knowledge (if a provider is configured) and labels those activities `model_knowledge`
— this is correct, documented behavior, not a bug. Add a new destination file and run
`python scripts/ingest_data.py` to extend the curated set.

**The trip map is empty / shows "couldn't geocode any stops".**
The map uses OpenStreetMap's free Nominatim geocoder, which needs outbound internet
access from wherever the app is *running* (not from wherever you're viewing it) and is
rate-limited. This is expected to sometimes fail or be slow — the day-timeline view
covers the same itinerary without needing geocoding at all.

**`docker compose up` fails looking for `.env`.**
Run `cp .env.example .env` first — `docker-compose.yml` expects it to exist (even with
no keys filled in).

**Checkpoint DB / vector store seem stuck on old data after changing `data/destinations/`.**
Run `python scripts/ingest_data.py --rebuild` to wipe and re-embed the vector store
from scratch. The checkpoint DB (session history) and vector store (destination
knowledge) are independent — clearing one doesn't affect the other.

## Roadmap

Documented gaps, in the order they'd likely matter most for a real production
deployment (see also each item's cross-reference in
[`docs/architecture.md`'s Extension points](architecture.md#extension-points)):

1. **Live external data integrations** — a real Maps/Places API for geocoding, opening
   hours, and route distances (replacing the current heuristics); a live flight/hotel
   pricing API (replacing the Budget Agent's tier-based heuristic estimates); a real
   card-comparison API (replacing the illustrative rewards catalogue).
2. **Postgres-backed checkpointing and a hosted vector store** for true multi-process/
   multi-instance horizontal scaling (see "Production considerations" above).
3. **Auth** — there is currently no authentication layer; `user_id` is a client-
   supplied string used only to partition memory. A real deployment needs to bind
   `user_id` to an authenticated session before this is safe to expose beyond trusted
   users.
4. **Load testing** — the test suite validates correctness end-to-end but doesn't
   include load/stress testing. A `locust` or `k6` script against `/plan` and `/refine`
   would be the natural next addition once there's a real deployment target to tune
   against.
5. **More destinations and richer per-destination data** (opening hours, price ranges
   with actual currency conversion, multilingual content).
