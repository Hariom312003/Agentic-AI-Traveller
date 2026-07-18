"""
Data ingestion CLI.

Run this once (or whenever data/destinations/*.json changes) to (re)build
the Chroma vector store. Also invoked automatically at container startup
(see Dockerfile) so a fresh deployment isn't serving an empty knowledge base.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --rebuild   # wipe and re-embed everything
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.rag.chunking import load_all_destinations
from src.rag.embeddings import get_embedding_provider
from src.rag.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest destination knowledge base into the vector store")
    parser.add_argument("--rebuild", action="store_true", help="Delete the existing collection and re-embed from scratch")
    args = parser.parse_args()

    settings = get_settings()

    if args.rebuild:
        path = Path(settings.chroma_db_path)
        if path.exists():
            print(f"Removing existing vector store at {path}")
            shutil.rmtree(path)

    print(f"Loading source documents from {settings.destinations_data_path} ...")
    chunks = load_all_destinations(settings.destinations_data_path)
    if not chunks:
        print("No destination JSON files found — nothing to ingest.")
        return
    print(f"Loaded {len(chunks)} chunks across "
          f"{len({c.metadata['destination'] for c in chunks})} destinations.")

    embedder = get_embedding_provider()
    print(f"Using embedding provider: {type(embedder).__name__}")

    store = VectorStore(settings.chroma_db_path, embedder)
    start = time.time()
    n = store.upsert_chunks(chunks)
    elapsed = time.time() - start
    print(f"Ingested {n} chunks in {elapsed:.1f}s. Collection now has {store.count()} vectors "
          f"at {settings.chroma_db_path}.")


if __name__ == "__main__":
    main()
