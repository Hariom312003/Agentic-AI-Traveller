"""
Hybrid retrieval: fuse BM25 + vector search, then rerank.

Fusion strategy: Reciprocal Rank Fusion (RRF). We deliberately fuse on
*rank*, not raw score — BM25 scores and cosine-similarity scores live on
different, incomparable scales, and naively averaging them means whichever
happens to have larger numbers silently dominates. RRF sidesteps that by
only ever asking "where did this chunk rank in each list", which is scale-free.

Reranking: a lightweight metadata-aware reranker runs by default (boosts
exact category/interest matches and destination match — cheap, deterministic,
no extra LLM call). An optional LLM-based reranker (`rerank_with_llm`) is
available for callers that want an extra semantic pass and are willing to
spend a model call on it; the Destination Knowledge/RAG Agent uses the
metadata reranker by default and can opt into the LLM pass for high-stakes
queries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.rag.lexical_store import LexicalStore
from src.rag.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    fused_score: float
    matched_lexical: bool
    matched_vector: bool


def reciprocal_rank_fusion(
    lexical_hits: list[dict], vector_hits: list[dict], k: int = 60
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    matched: dict[str, dict[str, bool]] = {}

    for rank, hit in enumerate(lexical_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        payload[cid] = hit
        matched.setdefault(cid, {"lexical": False, "vector": False})["lexical"] = True

    for rank, hit in enumerate(vector_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        payload.setdefault(cid, hit)
        matched.setdefault(cid, {"lexical": False, "vector": False})["vector"] = True

    fused = [
        RetrievedChunk(
            chunk_id=cid,
            text=payload[cid]["text"],
            metadata=payload[cid]["metadata"],
            fused_score=score,
            matched_lexical=matched[cid]["lexical"],
            matched_vector=matched[cid]["vector"],
        )
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda c: c.fused_score, reverse=True)
    return fused


def metadata_rerank(
    chunks: list[RetrievedChunk], destination: str | None, interests: list[str], top_k: int
) -> list[RetrievedChunk]:
    """Cheap, deterministic rerank pass: boosts chunks whose metadata
    matches the requested destination and interests. Runs with every query
    — no extra network call, no extra latency worth mentioning."""
    interests_lower = {i.lower() for i in interests}

    def boosted_score(c: RetrievedChunk) -> float:
        score = c.fused_score
        meta_dest = str(c.metadata.get("destination", "")).lower()
        if destination and meta_dest == destination.lower():
            score *= 1.25
        elif destination and meta_dest and meta_dest != destination.lower():
            score *= 0.4  # cross-destination leakage guard, not a hard filter
        category = str(c.metadata.get("category", "")).lower()
        if category in interests_lower:
            score *= 1.15
        if c.matched_lexical and c.matched_vector:
            score *= 1.1  # agreement between both retrieval signals
        return score

    reranked = sorted(chunks, key=boosted_score, reverse=True)
    return reranked[:top_k]


def rerank_with_llm(chunks: list[RetrievedChunk], query: str, top_k: int) -> list[RetrievedChunk]:
    """Optional second-pass semantic rerank using the configured LLM router.
    Asks the model to order candidate chunk ids by relevance and return
    JSON; falls back to the input order (already metadata-reranked) if the
    model call fails or returns something unparseable — reranking is an
    enhancement, never a hard dependency."""
    if len(chunks) <= top_k:
        return chunks
    from src.llm.router import get_router

    listing = "\n".join(f"{i}. [{c.chunk_id}] {c.text[:180]}" for i, c in enumerate(chunks))
    prompt = (
        f"Query: {query}\n\nCandidate passages:\n{listing}\n\n"
        f"Return ONLY a JSON array of the {top_k} most relevant chunk_id strings, most relevant first."
    )
    try:
        response = get_router().generate(prompt=prompt, system_prompt="You are a precise passage reranker.", temperature=0.0)
        ids = json.loads(response.text[response.text.find("[") : response.text.rfind("]") + 1])
        by_id = {c.chunk_id: c for c in chunks}
        ordered = [by_id[i] for i in ids if i in by_id]
        remaining = [c for c in chunks if c.chunk_id not in ids]
        return (ordered + remaining)[:top_k]
    except Exception:
        return chunks[:top_k]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, lexical_store: LexicalStore):
        self.vector_store = vector_store
        self.lexical_store = lexical_store

    def retrieve(
        self,
        query: str,
        destination: str | None = None,
        interests: list[str] | None = None,
        top_k: int = 8,
        candidate_pool: int = 25,
        use_llm_rerank: bool = False,
        strict_destination: bool = True,
    ) -> list[RetrievedChunk]:
        """`strict_destination=True` (default) hard-filters both retrieval
        legs to the requested destination before fusion — this is what
        prevents "recommending landmarks from Jaipur on a trip to Manali"
        (a strength the code review specifically called out) regardless of
        how the embedding provider happens to score cross-destination
        similarity. Set False only for deliberately destination-agnostic
        queries (there are none in the current agents, but the option is
        here rather than baked in as an assumption)."""
        interests = interests or []
        where = {"destination": destination} if (destination and strict_destination) else None
        lex_destination = destination if strict_destination else None

        lexical_hits = self.lexical_store.query(query, n_results=candidate_pool, destination=lex_destination)
        vector_hits = self.vector_store.query(query, n_results=candidate_pool, where=where)
        fused = reciprocal_rank_fusion(lexical_hits, vector_hits)
        reranked = metadata_rerank(fused, destination, interests, top_k=candidate_pool)
        if use_llm_rerank:
            reranked = rerank_with_llm(reranked, query, top_k=top_k)
        return reranked[:top_k]
