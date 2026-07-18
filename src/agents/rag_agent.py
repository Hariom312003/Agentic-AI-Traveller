"""
RAG Agent (Destination Knowledge Agent).

Runs hybrid retrieval for the requested destination and interests, builds
the grounded prompt context for the Planner, and records `grounded_ratio`
in state so downstream validation and the API response can be honest about
how much of the plan is verified vs. general-knowledge.
"""
from __future__ import annotations

from src.rag.context_builder import build_planner_context
from src.rag.embeddings import get_embedding_provider
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.lexical_store import LexicalStore
from src.rag.vector_store import VectorStore
from src.models.state import TripState
from src.monitoring.telemetry import track_agent

_retriever_singleton: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Lazily built singleton — loading + embedding the full destination
    corpus on every request would be wasteful; the lexical/vector indices
    are immutable within a process lifetime (ingestion is a separate
    offline step via scripts/ingest_data.py)."""
    global _retriever_singleton
    if _retriever_singleton is None:
        from src.config import get_settings
        from src.rag.chunking import load_all_destinations

        settings = get_settings()
        embedder = get_embedding_provider()
        vector_store = VectorStore(settings.chroma_db_path, embedder)
        if vector_store.is_empty():
            # First-run convenience: ingest automatically instead of forcing
            # a manual step before the API can answer anything. Production
            # deployments still run scripts/ingest_data.py explicitly at
            # build/deploy time (see Dockerfile) — this is a safety net for
            # local dev, not the primary ingestion path.
            chunks = load_all_destinations(settings.destinations_data_path)
            if chunks:
                vector_store.upsert_chunks(chunks)
        chunks = load_all_destinations(settings.destinations_data_path)
        lexical_store = LexicalStore(chunks)
        _retriever_singleton = HybridRetriever(vector_store, lexical_store)
    return _retriever_singleton


def run_rag_agent(state: TripState) -> dict:
    with track_agent("rag_agent") as handle:
        trip_request = state["trip_request"]
        destination = trip_request.destination or "unspecified destination"

        # Check file cache and trigger dynamic knowledge acquisition if missing
        if destination != "unspecified destination":
            from src.rag.acquisition import is_destination_cached, acquire_destination_knowledge
            if not is_destination_cached(destination):
                try:
                    acquire_destination_knowledge(destination)
                    global _retriever_singleton
                    _retriever_singleton = None
                except Exception as exc:
                    # Log failure and allow planner to try general-knowledge/offline fallbacks
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Failed to dynamically acquire knowledge for '{destination}': {exc}"
                    )

        prompt_block = ""
        allowed_places = []
        source_chunk_ids = []
        grounded = False

        if destination != "unspecified destination":
            from src.planning_engine.optimizer import build_optimized_context_block
            try:
                prompt_block, allowed_places = build_optimized_context_block(
                    destination,
                    trip_request.interests or [],
                    trip_request.travel_style,
                    trip_request.duration_days or 4,
                    trip_request.season
                )
                if prompt_block:
                    grounded = True
                    source_chunk_ids = [p["chunk_id"] for p in allowed_places]
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(f"Error building optimized skeleton context: {exc}")

        # Fallback to standard context builder if optimizer returned empty
        if not grounded:
            query_parts = [trip_request.raw_query]
            if trip_request.interests:
                query_parts.append("Interests: " + ", ".join(trip_request.interests))
            if trip_request.travel_style:
                query_parts.append(f"Style: {trip_request.travel_style}")
            query = " | ".join(query_parts)

            retriever = get_retriever()
            chunks = retriever.retrieve(
                query=query,
                destination=destination,
                interests=trip_request.interests,
                top_k=max(12, trip_request.duration_days * 4 if trip_request.duration_days else 12),
            )
            from src.rag.context_builder import build_planner_context
            context = build_planner_context(chunks, destination)
            prompt_block = context.prompt_block
            allowed_places = context.allowed_places
            source_chunk_ids = context.source_chunk_ids
            grounded = context.grounded

        handle.record.retrieved_doc_ids = source_chunk_ids
        handle.record.reasoning_summary = (
            f"Retrieved {len(allowed_places)} grounded place(s) for {destination}"
            if grounded
            else f"No curated knowledge base entries for '{destination}' — planner will use general knowledge, labeled accordingly"
        )
        handle.record.extra["grounded"] = grounded

        return {
            "retrieved_context": [{"prompt_block": prompt_block, "allowed_places": allowed_places,
                                    "grounded": grounded, "source_chunk_ids": source_chunk_ids}],
            "grounded_ratio": 1.0 if grounded else 0.0,
            "execution_trace": [handle.record],
        }
