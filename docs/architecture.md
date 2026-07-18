# Architecture

## Contents
- [Design principles](#design-principles)
- [System diagram](#system-diagram)
- [Module-by-module](#module-by-module)
- [The four review fixes](#the-four-review-fixes)
- [RAG pipeline in depth](#rag-pipeline-in-depth)
- [Multi-LLM routing in depth](#multi-llm-routing-in-depth)
- [Extension points](#extension-points)

## Design principles

Four rules shaped every module in this codebase, because they're the difference between
"looks production-grade" and actually being closer to it:

1. **Verify, don't trust.** The model's self-reported claims (which knowledge-base
   entries it used, whether a day was left unchanged) are never taken at face value.
   Where a claim matters, there's code that independently checks it — fuzzy-matching a
   returned activity title against what was actually retrieved
   (`src/planning_engine/conversion.py::verify_source`), hashing a day's content before
   and after a refinement (`src/refinement/locking.py`).
2. **Degrade honestly, not silently.** Every fallback path — no LLM keys, an
   ungrounded destination, a geocoding failure — is labeled as what it is
   (`source: rule_based_fallback`, `grounded: false`, "map unavailable") rather than
   dressed up to look like the primary path succeeded.
3. **No unguarded shared mutable state.** The bug that motivated fix #4 (a bare module
   dict mutated from multiple threads) is a category, not a one-off — every piece of
   process-wide state in this codebase (`ProviderHealthRegistry`, `MemoryStore`,
   `GraphManager`'s SQLite connection) is owned by a class with an explicit lock or a
   database's own concurrency guarantees, never a bare module-level dict or list.
4. **Real integration tests over unit tests that only prove the mocks work.** Most of
   `tests/` exercises the actual compiled LangGraph graphs, the actual FastAPI app via
   `TestClient`, and the actual Streamlit app via `AppTest` — end to end, with zero API
   keys configured, so the fallback paths are exercised by CI on every commit, not just
   asserted about in a docstring.

## System diagram

```
                                   ┌────────────────────┐
                                   │  Streamlit Frontend │  (app.py + frontend/)
                                   └──────────┬─────────┘
                                              │ REST (requests)
                                              ▼
                                   ┌────────────────────┐
                                   │   FastAPI Backend   │  (src/api/)
                                   │  /plan /refine       │
                                   │  /rollback /memory    │
                                   └──────────┬─────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │      LangGraph Workflow         │  (src/graph/workflow.py)
                              │  plan_graph  /  refine_graph      │
                              │  both share one SqliteSaver          │
                              │  checkpointer (real rollback)          │
                              └───────────────┬───────────────────────┘
                                              │
      ┌───────────┬───────────┬──────────────┼───────────────┬────────────┬─────────────┐
      ▼           ▼           ▼               ▼                ▼            ▼             ▼
   Query      Memory        RAG            Planner           Budget       Rewards      Validator
   Agent       Agent       Agent             Agent             Agent        Agent         Agent
      │           │           │               │                                              │
      │           │           ▼               ▼                                              │
      │           │    ┌─────────────┐  ┌──────────────┐                                     │
      │           │    │ Hybrid       │  │ LLM Router    │◄────────────────────────────────────┘
      │           │    │ Retriever    │  │ (5 providers, │        (loops back to Planner on
      │           │    │ BM25+vector  │  │  circuit       │         critical validation issues,
      │           │    │ +ChromaDB    │  │  breaker,       │         up to max_planner_repair_attempts)
      │           │    └─────────────┘  │  thread-safe     │
      │           │                     │  health registry)  │
      │           ▼                     └──────────────┘
      │    ┌─────────────┐
      │    │ MemoryStore  │  (SQLite, destination-agnostic behavioral profile)
      │    │ (SQLite)      │
      │    └─────────────┘
      ▼
 Rule-based keyword
 fallback (no LLM needed)
```

Refinement follows a separate, shorter graph (`refinement_agent → validator_agent →
summary_agent`) that shares the same checkpointer/thread_id space, so a session's
history is continuous across an initial `/plan` and any number of later `/refine`
calls — see [`docs/workflow.md`](workflow.md) for the full node/edge/state-transition
breakdown.

## Module-by-module

### `src/config.py`
Single `Settings` object (pydantic-settings) loaded once as a cached singleton. Every
other module reads configuration through `get_settings()` — nothing reads
`os.environ` directly outside this file. This is what makes "which LLM provider is
configured", "what's the duplicate-match threshold", "where's the checkpoint DB" each a
one-line change instead of a grep-and-replace across the codebase.

**Inputs:** `.env` file / process environment. **Outputs:** typed, validated settings.
**Failure mode:** a malformed `.env` fails at the first `get_settings()` call (usually
API startup), not partway through a request.

### `src/models/`
Pydantic domain models (`itinerary.py`, `request.py`, `user.py`, `budget.py`,
`validation.py`) plus the LangGraph `TripState` TypedDict (`state.py`). `Day` carries
its own `content_hash()` — the single source of truth for "did this day's content
change", used by both the locking mechanism and tests.

### `src/llm/`
- `base.py` — the `LLMProvider` interface + a `ProviderError` hierarchy
  (`RateLimitError` / `TransientProviderError` / `NonRetryableProviderError`) that lets
  the router make retry/failover decisions without string-matching error messages.
- `providers.py` — one class per vendor SDK (Gemini via `google-genai`, Groq, OpenRouter
  via an OpenAI-compatible client, OpenAI, Anthropic). This is the only file allowed to
  import a vendor SDK directly.
- `health.py` — `ProviderHealthRegistry`, the thread-safe circuit breaker state (see
  [fix #4](#the-four-review-fixes) below).
- `router.py` — walks the configured provider priority list, skips unavailable
  (circuit-open) providers, retries transient errors locally with exponential backoff
  (`tenacity`), fails over immediately on rate limits, and raises
  `AllProvidersExhaustedError` only once every configured provider has been tried.

**Failure mode:** if zero providers are configured or all are exhausted, callers (the
agents) catch `AllProvidersExhaustedError` and fall back to rule-based logic — the
router itself never silently returns a fake response.

### `src/rag/`
See [RAG pipeline in depth](#rag-pipeline-in-depth).

### `src/memory/`
- `store.py` — `MemoryStore`, a SQLite-backed key-value-ish store for `UserProfile`
  (one row per user, upserted).
- `behavioral.py` — the destination-agnostic learning logic: keyword-based style
  inference (`infer_style_signals`), exponentially-smoothed interest weights, and
  `scrub_place_names` (explicit known-name scrub + a capitalized-proper-noun-run
  heuristic for names it wasn't explicitly told about — see the module docstring for
  why both passes exist).

### `src/agents/`
Nine node functions, one per agent, each returning a partial `TripState` update (the
LangGraph convention). Every agent wraps its body in
`src.monitoring.telemetry.track_agent(name)`, which times it and lets it attach
provider/model/retrieved-doc-id/reasoning-summary metadata — this is the entire
implementation of the "Explainability" feature; nothing bespoke per agent.

| Agent | Reads | Writes | LLM call? |
|---|---|---|---|
| `query_agent` | `trip_request.raw_query` | merged, structured `trip_request` | yes (falls back to regex) |
| `memory_agent` | `user_id` | `user_profile` | no |
| `rag_agent` | `trip_request` | `retrieved_context`, `grounded_ratio` | no (retrieval only; optional LLM rerank) |
| `planner_agent` | RAG context, preferences | `itinerary` | yes (falls back to rule-based) |
| `budget_agent` | `itinerary` | `budget_breakdown` | no (heuristic) |
| `rewards_agent` | `budget_breakdown` | `rewards_summary` | no (static catalogue) |
| `validator_agent` | `itinerary`, `budget_breakdown` | `validation_report` | no |
| `refinement_agent` | `itinerary`, instruction | new `itinerary` (target days only) | yes per target day (falls back to rule-based per day) |
| `memory_update_agent` | this trip's signals | updated `user_profile` | no |
| `summary_agent` | everything | `final_response` | no |

### `src/validation/`
`duplicate_checker.py` (fix #3) and `rules.py` (budget overflow, day balance, empty
slots, slot/category conflicts, groundedness ratio, a route-efficiency heuristic).

### `src/planning_engine/`
`scheduler.py` (time-slot assignment heuristics), `fallback_planner.py` (the zero-LLM
emergency planner — see its module docstring for why it refuses to invent plausible-
sounding fake landmark names for ungrounded destinations), `conversion.py` (the one
place that converts an LLM's JSON output into domain `Day`/`Activity` objects and
independently verifies groundedness — shared by both the Planner and Refinement
agents so that check can't drift out of sync between them).

### `src/refinement/locking.py`
See [fix #2](#the-four-review-fixes).

### `src/graph/workflow.py`
Builds both compiled graphs, wires the conditional replan edge, and owns the
`GraphManager` (the checkpointer + both graphs + rollback helpers). See
[fix #1](#the-four-review-fixes) and [`docs/workflow.md`](workflow.md).

### `src/monitoring/`
`telemetry.py` (the `AgentExecutionRecord` + `track_agent` context manager) and
`logging_config.py` (structured JSON file logging + a Rich console handler; the Agent
Monitor page's log console reads the JSON log file directly via `read_recent_logs`).

### `src/api/`
Thin FastAPI routes (`routes.py`) over a `main.py` app factory with a `lifespan` that
warms the vector store and graph manager at startup (so a config problem surfaces at
boot, not on someone's first request) and closes the checkpoint DB connection at
shutdown.

### `frontend/`
`api_client.py` (every backend call funnelled through one exception type — `APIError`
— whether the failure was an HTTP error response or the backend being completely
unreachable), `theme.py` (the visual design system), `geo_map.py` (best-effort
geocoding + Leaflet map), `pdf_export.py` (fpdf2-based export with Unicode-to-Latin-1
sanitization, since LLM output routinely contains smart quotes/em-dashes that the core
PDF fonts can't render), and `views/` (one module per page).

## The four review fixes

### #1 — No checkpointing or rollback

**Before:** `graph.compile()` with no checkpointer — every run was stateless.

**Now:** both `plan_graph` and `refine_graph` are compiled with a shared
`langgraph.checkpoint.sqlite.SqliteSaver`, keyed by `session_id` as LangGraph's
`thread_id`. This gets automatic per-superstep checkpointing, `GraphManager.history()`
(lists every checkpoint), and `GraphManager.rollback_config()` (finds a historical
checkpoint's config so it can be used to fork/restore from that point) essentially for
free from LangGraph itself. The `/rollback` API endpoint (no `checkpoint_id` → list
checkpoints; with one → restore) calls `CompiledStateGraph.update_state()` to commit
the restored values as the new tip, so a subsequent `/refine` call continues from the
rolled-back point.

One implementation detail worth calling out: LangGraph's checkpoint serializer will,
in a future version, refuse to deserialize types it wasn't explicitly told about
(currently a warning, moving to an error). Rather than leave that as a ticking
time bomb or set the permissive `allowed_msgpack_modules=True` (which would happily
deserialize *any* importable class from a checkpoint file — a real, if narrow,
security surface), `src/graph/workflow.py` builds an explicit allow-list from every
domain type that can appear in `TripState`.

**Proof:** `tests/test_graph_checkpointing.py` — runs the real compiled graphs,
asserts multiple checkpoints exist after one run, and actually rolls back and checks
the restored itinerary version.

### #2 — "Locked" day editing wasn't enforced

**Before:** the whole itinerary JSON was regenerated by the LLM on every refinement,
with a system-prompt instruction to "preserve unaffected days as much as possible".
`locked_days` was computed *after the fact* by diffing which days happened not to
change — a report, not a guarantee.

**Now** (`src/refinement/locking.py`): the refinement agent never sends an untouched
day to the LLM and never lets the LLM's output populate it. `apply_refinement` builds
the result itinerary by deep-copying every non-target `Day` object directly from the
original — the model is structurally incapable of altering a day it was never shown.
`verify_locks_held` is a defense-in-depth assertion that recomputes each non-target
day's `content_hash()` post-merge and confirms it's byte-identical to the pre-
refinement hash; this should be mathematically impossible to fail given the
construction above.

**Proof:** `tests/test_refinement_locking.py::test_verify_locks_held_actually_detects_a_violation`
is the test that matters most here — it deliberately constructs the *exact* broken
state the old system could produce (a non-target day silently altered) and confirms
the checker catches it, so the "verification" isn't a tautology that can never fail.

### #3 — Duplicate detection was exact string match

**Before:** `location.lower() in global_used_locations` — misses everything except
byte-identical (modulo case) names.

**Now** (`src/validation/duplicate_checker.py`): normalize (strip punctuation,
lowercase) then score with `max(token_sort_ratio, token_set_ratio)` from RapidFuzz,
threshold 80. This exact scoring function and threshold were chosen empirically
against a labeled set of real near-duplicate and non-duplicate destination attraction
pairs (`tests/test_duplicate_checker.py::LABELED_PAIRS`) — not picked arbitrarily.

**Proof:** `tests/test_duplicate_checker.py`, including a test that reproduces the
exact case the old exact-match implementation would have missed ("Gateway of India"
vs. "The Gateway of India Monument").

### #4 — Global mutable provider-health state, no locking

**Before:** a bare module-level `dict` mutated with unguarded `+=` and multi-statement
check-then-act sequences from every request thread — a textbook lost-update race under
concurrent requests.

**Now** (`src/llm/health.py`): all state lives inside `ProviderHealthRegistry`, every
read-modify-write happens under a single `threading.RLock`, and there is no code path
outside that class that touches the underlying state.

**Proof:** `tests/test_llm_router.py::test_concurrent_health_updates_are_not_lost`
spins up 50 real `threading.Thread`s hammering `record_success`/`record_failure`
simultaneously (synchronized with a `Barrier` to maximize actual contention) and
asserts the final counters are exactly right — a real concurrency bug needs a real
concurrent test, not a single-threaded one that can't ever observe the race.

## RAG pipeline in depth

1. **Ingestion** (`src/rag/chunking.py`): source documents are structured JSON per
   destination (`data/destinations/<city>.json`), not raw prose — travel knowledge is
   naturally tabular, and structure-preserving chunking means one attraction/
   restaurant/tip = one chunk, never half of one place and half of another. Chunks are
   tagged `is_schedulable: true/false` so informational sections (safety tips, visa
   info) ride along as context without ever being offered to the Planner as a literal
   bookable "place".
2. **Embeddings** (`src/rag/embeddings.py`): pluggable. `GeminiEmbeddingProvider` for
   production (needs `GEMINI_API_KEY` + network); `LocalHashingEmbeddingProvider` (a
   deterministic `HashingVectorizer` over word n-grams) as a zero-download, fully
   offline fallback — this is what lets the entire RAG pipeline be unit-tested with no
   network access at all, and it's also a legitimate degraded-mode production fallback
   if no embedding key is configured.
3. **Storage**: ChromaDB (`vector_store.py`) for dense vectors (always given
   precomputed embeddings — never Chroma's own default embedding function, which
   downloads a model on first use and would silently break offline), and an in-memory
   BM25 index (`lexical_store.py`, `rank_bm25`) for lexical search.
4. **Hybrid retrieval** (`hybrid_retriever.py`): both legs are queried, then fused with
   Reciprocal Rank Fusion — fused on *rank*, not raw score, because BM25 scores and
   cosine similarities live on incomparable scales and naively averaging them lets
   whichever has larger numbers silently dominate. **Destination isolation is a hard
   filter** (`where={"destination": ...}` on the vector query, a metadata filter on
   BM25), not just a rerank penalty — this was a strength the original code review
   specifically called out, and it needed to hold even against the weaker offline
   hashing embedder used in tests (`tests/test_rag.py::test_strict_destination_isolation_no_cross_contamination`).
5. **Reranking**: a cheap, deterministic metadata-aware pass runs on every query (boosts
   destination/interest-category matches); an optional LLM-based rerank pass exists for
   callers willing to spend an extra model call, with a graceful fallback to the
   metadata-reranked order if that call fails.
6. **Context building** (`context_builder.py`): dedupes near-identical retrieved chunks
   (reusing the same fuzzy matcher as fix #3), compresses to a character budget, and
   builds the `allowed_places` list the Planner is instructed to use for factual
   content — **this is the hallucination-prevention mechanism**: not a hope that the
   model stays grounded, but the Planner's output being independently checked against
   this exact list afterward (`verify_source` in `conversion.py`).

## Multi-LLM routing in depth

Priority order, retry, and circuit-breaking are covered above and in the
[`src/llm/`](#srcllm) section. Two things worth calling out that aren't obvious from
reading one file in isolation:

- **Rate limits fail over immediately** (no local retry) — retrying a 429 against the
  same key just burns more quota for no benefit. **Transient errors** (5xx, timeouts,
  connection resets) get a bounded local retry with exponential backoff before failing
  over, since those often really do resolve on retry.
- **Schema-validated generation** (`src/agents/common.py::generate_structured`) is
  layered on top of the router: every LLM-calling agent asks for JSON, parses it, and
  validates it against a pydantic schema, retrying (with the validation error fed back
  into the prompt) up to `max_json_parse_retries` times before the caller falls back to
  its rule-based path. This is the "output schema verification" strength the code
  review noted, centralized in one function instead of re-implemented per agent.

## Extension points

Documented here rather than half-implemented with placeholder data, per the project's
own stated preference for honest scoping over fake completeness:

- **Live opening-hours / traffic / route-distance data**: `src/validation/rules.py`'s
  `route_efficiency_score` and `src/planning_engine/scheduler.py`'s slot heuristics are
  a documented approximation. Wiring a real Google Places/Maps (or OSRM) integration
  would replace the location-string-overlap heuristic with real geocoded
  distances/durations.
- **Live rewards/card data**: `src/agents/rewards_agent.py` reads
  `data/rewards_catalogue.json` structurally — point it at a real card-comparison API
  or your own curated data and the agent logic doesn't need to change.
- **More destinations**: drop a new `data/destinations/<city>.json` following the
  existing schema (see any existing file, or `src/rag/chunking.py`'s section lists) and
  run `python scripts/ingest_data.py`.
- **Horizontal scaling**: the checkpoint/memory stores are SQLite, chosen for
  zero-ops portability. Both LangGraph and this project's `MemoryStore` interface are
  small enough to swap for a Postgres-backed checkpointer/store for a true
  multi-process deployment — see `docs/deployment.md`.
