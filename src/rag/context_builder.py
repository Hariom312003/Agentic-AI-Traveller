"""
Context construction for the Planner Agent's prompt.

Three jobs:
1. De-duplicate retrieved chunks that describe the same real-world place
   (reuses the same fuzzy-matching logic as the Validator's duplicate
   checker — one definition of "same place", not two that can disagree).
2. Compress: cap total context length so the prompt stays small and cheap,
   keeping the highest-fused-score chunks.
3. Render a prompt block AND a structured "allowed_places" list. The
   Planner is instructed to only use places from `allowed_places` for this
   destination — that's the concrete hallucination-prevention mechanism
   (grounding by constraint, not just by hope), and the Validator later
   checks the itinerary's `source_doc_ids` against what was actually
   offered here.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.rag.hybrid_retriever import RetrievedChunk
from src.validation.duplicate_checker import is_near_duplicate


@dataclass
class BuiltContext:
    prompt_block: str
    allowed_places: list[dict]
    source_chunk_ids: list[str]
    grounded: bool


def dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        name = str(chunk.metadata.get("name", chunk.chunk_id))
        if any(is_near_duplicate(name, str(k.metadata.get("name", k.chunk_id))) for k in kept):
            continue
        kept.append(chunk)
    return kept


def compress_context(chunks: list[RetrievedChunk], max_chars: int = 6000) -> list[RetrievedChunk]:
    kept: list[RetrievedChunk] = []
    running = 0
    for chunk in chunks:  # already sorted best-first by the retriever
        if running + len(chunk.text) > max_chars and kept:
            break
        kept.append(chunk)
        running += len(chunk.text)
    return kept


def build_planner_context(chunks: list[RetrievedChunk], destination: str, max_chars: int = 6000) -> BuiltContext:
    deduped = dedupe_chunks(chunks)
    compressed = compress_context(deduped, max_chars=max_chars)

    # Only *places* (attractions, restaurants, shopping, etc.) are offered as
    # schedulable candidates. Informational chunks (safety, local tips,
    # visa info, culture notes) still ride along in the prompt for context,
    # but must never show up as something the planner could literally
    # schedule as a "visit Safety" activity.
    schedulable = [c for c in compressed if c.metadata.get("is_schedulable", True)]
    informational = [c for c in compressed if not c.metadata.get("is_schedulable", True)]

    allowed_places = [
        {
            "name": c.metadata.get("name", c.chunk_id),
            "category": c.metadata.get("category", "attraction"),
            "chunk_id": c.chunk_id,
            "recommended_duration": c.metadata.get("recommended_duration"),
            "budget_category": c.metadata.get("budget_category"),
            "latitude": c.metadata.get("latitude"),
            "longitude": c.metadata.get("longitude"),
            "address": c.metadata.get("address"),
            "map_link": c.metadata.get("map_link"),
        }
        for c in schedulable
    ]

    if compressed:
        lines = [f"Verified knowledge base for {destination} (use ONLY these for factual place names):"]
        for c in schedulable:
            lines.append(f"- [{c.chunk_id}] {c.text}")
        if informational:
            lines.append("\nGeneral destination context (background only — these are NOT bookable activities):")
            for c in informational:
                lines.append(f"- {c.text}")
        prompt_block = "\n".join(lines)
        grounded = True
    else:
        prompt_block = (
            f"No curated knowledge base entries were found for '{destination}'. "
            f"You may use your general world knowledge, but every generated activity's "
            f"`source` field MUST be set to 'model_knowledge' (not 'knowledge_base') so "
            f"the response is transparently labeled as unverified."
        )
        grounded = False

    return BuiltContext(
        prompt_block=prompt_block,
        allowed_places=allowed_places,
        source_chunk_ids=[c.chunk_id for c in compressed],
        grounded=grounded,
    )
