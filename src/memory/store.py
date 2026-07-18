"""
Persistent memory store.

SQLite instead of ChromaDB-for-everything: user profiles are small,
structured, and queried by exact `user_id` — a relational lookup, not a
similarity search. Reserving the vector store for what it's actually good
at (semantic retrieval over destination knowledge) keeps both systems
simpler. One connection per call (SQLite handles concurrent readers fine;
writes are serialized by SQLite itself) avoids any shared-connection
threading issues.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.models.user import BehavioralPreferences, PastTrip, UserProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_profile(self, user_id: str) -> UserProfile:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return UserProfile(user_id=user_id)
        data = json.loads(row["payload"])
        return UserProfile.model_validate(data)

    def save_profile(self, profile: UserProfile) -> None:
        payload = profile.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
                """,
                (profile.user_id, payload, profile.created_at.isoformat(), profile.updated_at.isoformat()),
            )

    def all_user_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id FROM user_profiles").fetchall()
        return [r["user_id"] for r in rows]

    def delete_profile(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))


_store_singleton: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _store_singleton
    if _store_singleton is None:
        from src.config import get_settings

        _store_singleton = MemoryStore(get_settings().memory_db_path)
    return _store_singleton
