"""
Lexical retrieval via BM25.

Dense embeddings are good at "what is semantically similar" and bad at
exact-term precision (a query for "Fort Aguada" can drift toward "forts in
general" once it's a 384-dim vector). BM25 is the opposite: excellent at
"this document contains these exact words", weak on paraphrase. Combining
both (see `hybrid_retriever.py`) covers more of the query space than either
alone — this is what "hybrid retrieval", not "fake RAG", means concretely.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.rag.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if chunks else None

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def query(self, text: str, n_results: int = 10, destination: str | None = None) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
        hits = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            if destination and str(chunk.metadata.get("destination", "")).lower() != destination.lower():
                continue
            hits.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": float(scores[idx] / max_score),  # normalized to [0, 1] for fusion
            })
            if len(hits) >= n_results:
                break
        return hits
