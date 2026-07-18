"""
Vector store wrapper around ChromaDB.

We always pass precomputed embeddings (from `src/rag/embeddings.py`) rather
than letting Chroma manage its own embedding function. Two reasons: (1) it
keeps embedding provider selection in one place instead of split across two
config systems, and (2) Chroma's default embedding function downloads a
model from the internet on first use, which silently breaks in any offline
or restricted-egress environment — precomputed embeddings sidestep that
entirely.
"""
from __future__ import annotations

from typing import Any

import chromadb

from src.rag.chunking import Chunk
from src.rag.embeddings import EmbeddingProvider


class VectorStore:
    def __init__(self, persist_path: str, embedding_provider: EmbeddingProvider, collection_name: str = "destinations"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedder = embedding_provider
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = self.embedder.embed([c.text for c in batch])
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )
            total += len(batch)
        return total

    def query(self, text: str, n_results: int = 10, where: dict[str, Any] | None = None) -> list[dict]:
        if self.is_empty():
            return []
        query_embedding = self.embedder.embed_one(text)
        # Guard against Chroma's count() including entries excluded by `where`:
        # request generously, Chroma clips internally to what actually matches.
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, max(self.collection.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append({
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": meta,
                # cosine distance -> similarity in [0, 1], clipped for safety
                "score": max(0.0, 1.0 - dist / 2.0),
            })
        return hits

    def count(self) -> int:
        return self.collection.count()
