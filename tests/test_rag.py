"""
RAG pipeline tests. Runs entirely on `LocalHashingEmbeddingProvider` (no
network, no API key, no model download) so this suite works in any
environment, including CI — see src/rag/embeddings.py for why that provider
exists at all.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.chunking import load_all_destinations, known_destinations
from src.rag.context_builder import build_planner_context, dedupe_chunks
from src.rag.embeddings import LocalHashingEmbeddingProvider
from src.rag.hybrid_retriever import HybridRetriever, RetrievedChunk, reciprocal_rank_fusion
from src.rag.lexical_store import LexicalStore
from src.rag.vector_store import VectorStore

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "destinations")


@pytest.fixture(scope="module")
def retriever():
    chunks = load_all_destinations(DATA_DIR)
    tmp_dir = tempfile.mkdtemp(prefix="ai_traveller_test_chroma_")
    embedder = LocalHashingEmbeddingProvider(dimensions=256)
    vector_store = VectorStore(tmp_dir, embedder, collection_name="test")
    vector_store.upsert_chunks(chunks)
    lexical_store = LexicalStore(chunks)
    yield HybridRetriever(vector_store, lexical_store)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_seed_data_loads_for_all_five_destinations():
    dests = known_destinations(DATA_DIR)
    expected = {"Goa", "Manali", "Tokyo", "Paris", "Bali"}
    assert expected.issubset(set(dests))


def test_chunks_are_nonempty_and_tagged():
    chunks = load_all_destinations(DATA_DIR)
    assert len(chunks) > 100
    assert all(c.metadata.get("destination") for c in chunks)
    assert any(c.metadata.get("is_schedulable") is True for c in chunks)
    assert any(c.metadata.get("is_schedulable") is False for c in chunks)


def test_hybrid_retrieval_returns_relevant_results(retriever):
    hits = retriever.retrieve("temples and spiritual sites", destination="Bali", interests=["culture"], top_k=5)
    names = {h.metadata.get("name") for h in hits}
    assert names & {"Tanah Lot Temple", "Uluwatu Temple", "Tirta Empul Temple"}


def test_strict_destination_isolation_no_cross_contamination(retriever):
    """Regression test for the review-praised 'no Jaipur landmarks on a
    Manali trip' behavior — must hold even though the offline hashing
    embedder is far weaker than a real semantic model."""
    hits = retriever.retrieve("sunset viewpoints", destination="Bali", interests=[], top_k=10)
    assert hits, "expected at least some hits"
    assert all(h.metadata.get("destination") == "Bali" for h in hits)


def test_unknown_destination_returns_no_hits(retriever):
    hits = retriever.retrieve("best beaches", destination="Nowhereland", interests=[], top_k=5)
    assert hits == []


def test_context_builder_marks_ungrounded_when_no_kb_match(retriever):
    hits = retriever.retrieve("best beaches", destination="Nowhereland", interests=[], top_k=5)
    ctx = build_planner_context(hits, "Nowhereland")
    assert ctx.grounded is False
    assert ctx.allowed_places == []
    assert "model_knowledge" in ctx.prompt_block


def test_context_builder_grounded_for_seeded_destination(retriever):
    hits = retriever.retrieve("beaches and nightlife", destination="Goa", interests=["nightlife"], top_k=8)
    ctx = build_planner_context(hits, "Goa")
    assert ctx.grounded is True
    assert len(ctx.allowed_places) > 0


def test_context_builder_excludes_informational_sections_from_allowed_places(retriever):
    """'Safety' / 'Local Tips' chunks are useful context but must never be
    offered to the planner as a literal bookable place."""
    hits = retriever.retrieve("safety tips local advice", destination="Goa", interests=[], top_k=10)
    ctx = build_planner_context(hits, "Goa")
    place_names = {p["name"].lower() for p in ctx.allowed_places}
    assert "safety" not in place_names
    assert "local tips" not in place_names


def test_reciprocal_rank_fusion_prefers_agreement():
    lexical = [{"chunk_id": "a", "text": "x", "metadata": {}}, {"chunk_id": "b", "text": "y", "metadata": {}}]
    vector = [{"chunk_id": "b", "text": "y", "metadata": {}}, {"chunk_id": "c", "text": "z", "metadata": {}}]
    fused = reciprocal_rank_fusion(lexical, vector)
    # "b" appears in both lists (rank 1 lexical, rank 0 vector) so should
    # score at least as high as "a" (lexical-only, rank 0).
    scores = {c.chunk_id: c.fused_score for c in fused}
    assert scores["b"] >= scores["a"]
    assert fused[0].chunk_id in ("a", "b")


def test_dedupe_chunks_collapses_near_duplicate_names():
    chunks = [
        RetrievedChunk("id1", "text1", {"name": "Baga Beach"}, 0.9, True, True),
        RetrievedChunk("id2", "text2", {"name": "Baga beach!!"}, 0.8, True, False),
        RetrievedChunk("id3", "text3", {"name": "Calangute Beach"}, 0.7, True, True),
    ]
    deduped = dedupe_chunks(chunks)
    assert len(deduped) == 2
    assert {c.chunk_id for c in deduped} == {"id1", "id3"}
