# Workflow

## The two graphs

`src/graph/workflow.py` builds two separate `StateGraph`s that share the same node
functions where applicable and, critically, **the same checkpointer and thread_id
space** — so a session's checkpoint history is continuous across an initial `/plan`
call and any number of later `/refine` calls.

### `plan_graph`

```
START
  │
  ▼
query_agent ───────────► parses raw_query into a structured TripRequest
  │                       (LLM extraction; falls back to regex/keyword parsing)
  ▼
memory_agent ──────────► loads the user's destination-agnostic behavioral profile
  │
  ▼
rag_agent ─────────────► hybrid retrieval (BM25 + vector) for the destination,
  │                       builds the grounded prompt context + allowed_places list
  ▼
planner_agent ─────────► generates the day-by-day itinerary
  │                       (LLM, schema-validated; falls back to rule-based planner)
  ▼
budget_agent ──────────► estimates a full cost breakdown from the itinerary
  │
  ▼
rewards_agent ─────────► matches budget categories to illustrative reward ideas
  │
  ▼
validator_agent ───────► runs all structural checks (duplicates, budget overflow,
  │                       day balance, groundedness, ...)
  │
  ├── critical issues AND attempts < max_planner_repair_attempts ──► back to planner_agent
  │
  ▼ (otherwise)
memory_update_agent ───► folds this trip's signals into the persisted profile
  │
  ▼
summary_agent ─────────► assembles the final API/UI response (incl. full execution trace)
  │
  ▼
 END
```

### `refine_graph`

```
START
  │
  ▼
refinement_agent ──────► resolves target day(s), regenerates ONLY those days
  │                       (LLM per day; falls back to rule-based per day),
  │                       merges via apply_refinement (untouched days copied verbatim),
  │                       verifies locks held (hash check)
  ▼
validator_agent ───────► re-validates the updated itinerary
  │
  ├── critical issues AND attempts < max_planner_repair_attempts ──► back to refinement_agent
  │
  ▼ (otherwise)
summary_agent ─────────► assembles the response
  │
  ▼
 END
```

`refine_graph` intentionally has no `memory_update_agent` step — a single refinement
instruction ("reduce the budget", "add nightlife") is a much weaker behavioral signal
than a full trip request, and folding it into the persisted profile with the same
weight risked noisy, premature preference updates from what might just be
one-off experimentation.

## State (`TripState`)

`src/models/state.py` defines the shared state as a `TypedDict`. Most fields are
overwrite-on-return (a node's partial return dict is merged key-by-key into state); two
use LangGraph `Annotated[..., operator.add]` reducers so every agent can append to them
without knowing the current contents:

- `execution_trace: list[AgentExecutionRecord]` — every agent's telemetry record
- `errors: list[str]` — non-fatal warnings (e.g. "day 2 refinement used the rule-based
  fallback because no LLM provider was reachable")

## Sequence: first plan, then a refinement, then a rollback

```
User          Streamlit           FastAPI              plan_graph / refine_graph        SqliteSaver
 │                │                   │                          │                          │
 │  fill form      │                   │                          │                          │
 │────────────────►│                   │                          │                          │
 │                │  POST /plan        │                          │                          │
 │                │──────────────────►│                          │                          │
 │                │                   │  invoke(state, thread_id) │                          │
 │                │                   │─────────────────────────►│                          │
 │                │                   │                          │  checkpoint after each     │
 │                │                   │                          │  node ─────────────────────►│
 │                │                   │◄─────────────────────────│                          │
 │                │◄──────────────────│  {session_id, itinerary, │                          │
 │◄────────────────│  render timeline   │   budget, ...}            │                          │
 │                │                   │                          │                          │
 │  "replace Day 2" │                   │                          │                          │
 │────────────────►│                   │                          │                          │
 │                │  POST /refine      │                          │                          │
 │                │──────────────────►│  get_state(thread_id) ───►│                          │
 │                │                   │◄─────────────────────────│  (loads latest checkpoint) │
 │                │                   │  invoke(refine_state,     │                          │
 │                │                   │         same thread_id) ─►│                          │
 │                │                   │                          │  more checkpoints ─────────►│
 │                │◄──────────────────│◄─────────────────────────│                          │
 │◄────────────────│  Day 1 & 3 byte-   │                          │                          │
 │                │  identical, Day 2   │                          │                          │
 │                │  changed, locked=   │                          │                          │
 │                │  true on 1 & 3      │                          │                          │
 │                │                   │                          │                          │
 │  "undo that"     │                   │                          │                          │
 │────────────────►│                   │                          │                          │
 │                │  POST /rollback     │                          │                          │
 │                │  {session_id}        │                          │                          │
 │                │──────────────────►│  history(thread_id) ─────►│                          │
 │                │                   │◄─────────────────────────│  get_state_history()  ─────►│
 │                │◄──────────────────│  [list of checkpoints]     │                          │
 │◄────────────────│  pick one           │                          │                          │
 │  POST /rollback  │                   │                          │                          │
 │  {checkpoint_id}  │                   │                          │                          │
 │────────────────►│──────────────────►│  rollback_config() ──────►│                          │
 │                │                   │  get_state(that config)   │                          │
 │                │                   │  update_state(tip, values)│  new checkpoint on top ────►│
 │                │◄──────────────────│◄─────────────────────────│  of history (new tip)      │
 │◄────────────────│  restored version   │                          │                          │
```

## Why a conditional replan loop instead of always accepting the first plan

`validator_agent`'s output feeds a conditional edge
(`_should_replan` / `_should_replan_refine` in `workflow.py`). Only **critical**
severity issues (empty days, budget overflow beyond 15%) trigger a loop back to the
planner — warnings (a possible duplicate, an unusual slot placement) are surfaced to
the user but don't block. This is capped by `settings.max_planner_repair_attempts`
(default 2) so a persistently broken destination or provider outage can't loop
forever; after the cap, the best available itinerary proceeds with its issues visible
in the response rather than the request simply failing.

## Where LLM calls happen (and where they don't)

| Node | LLM call | Fallback |
|---|---|---|
| `query_agent` | yes, structured extraction | regex/keyword extraction |
| `memory_agent` | no | — |
| `rag_agent` | no (retrieval); optional LLM rerank | metadata-only rerank |
| `planner_agent` | yes, structured itinerary generation | rule-based fallback planner |
| `budget_agent` | no (heuristic estimation) | — |
| `rewards_agent` | no (static catalogue lookup) | — |
| `validator_agent` | no (deterministic checks) | — |
| `refinement_agent` | yes, per target day | rule-based per-day fallback |
| `memory_update_agent` | no | — |
| `summary_agent` | no | — |

This is why the system produces a complete, usable response with **zero LLM provider
keys configured at all** — every LLM-calling node has a real, non-degenerate fallback,
not just an error path.
