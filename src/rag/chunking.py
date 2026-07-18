"""
Ingestion & chunking.

Source documents live as structured JSON per destination
(data/destinations/<city>.json) rather than raw prose, because travel
knowledge is naturally tabular (a list of attractions, each with a category,
cost band, hours) and structure-preserving chunking beats naive
character-window chunking for this kind of content: each attraction /
restaurant / tip becomes exactly one chunk, so retrieval never returns half
of one place and half of another.

`load_destination_documents` is the one function anything outside this file
should call — it returns fully-formed `Chunk` objects ready to embed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _entity_to_chunk(destination: str, section: str, entity: dict[str, Any], idx: int) -> Chunk:
    """One knowledge-base entity (an attraction, a restaurant, a safety tip)
    becomes one chunk. The rendered `text` is what gets embedded/searched;
    `metadata` is what the planner/validator read back out structurally."""
    name = _clean(entity.get("name") or entity.get("title") or f"{section}-{idx}")
    parts = [f"{name} ({destination}, category: {entity.get('category', section)})."]
    if entity.get("description"):
        parts.append(_clean(entity["description"]))
    for key in ("recommended_duration", "budget_category", "best_time", "transport_tips", "tags"):
        if entity.get(key):
            val = entity[key]
            val = ", ".join(val) if isinstance(val, list) else val
            parts.append(f"{key.replace('_', ' ').title()}: {val}.")

    text = " ".join(p for p in parts if p)
    chunk_id = f"{destination.lower().replace(' ', '_')}::{section}::{idx}::{name.lower().replace(' ', '_')[:40]}"
    metadata = {
        "destination": destination,
        "section": section,
        "name": name,
        "category": entity.get("category", section),
        "is_schedulable": True,
        **{k: v for k, v in entity.items() if k not in ("name", "description") and isinstance(v, (str, int, float, bool))},
    }
    return Chunk(chunk_id=chunk_id, text=text, metadata=metadata)


def load_destination_file(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    destination = data.get("destination") or path.stem.replace("_", " ").title()
    chunks: list[Chunk] = []

    # sections that are lists of structured entities
    list_sections = [
        "attractions", "restaurants", "hidden_gems", "nightlife", "shopping",
        "festivals", "adventure_activities",
    ]
    for section in list_sections:
        for idx, entity in enumerate(data.get(section, [])):
            chunks.append(_entity_to_chunk(destination, section, entity, idx))

    # sections that are free-form key -> text/blocks (safety, culture, visa, transport, emergency)
    freeform_sections = ["local_tips", "safety", "culture", "transport", "visa_information", "emergency_contacts", "weather"]
    for section in freeform_sections:
        block = data.get(section)
        if not block:
            continue
        if isinstance(block, dict):
            text = " ".join(f"{k.replace('_', ' ').title()}: {v}." for k, v in block.items() if v)
        elif isinstance(block, list):
            text = " ".join(_clean(x) for x in block)
        else:
            text = _clean(block)
        if text:
            chunk_id = f"{destination.lower().replace(' ', '_')}::{section}::0"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=f"{destination} — {section.replace('_', ' ').title()}: {text}",
                metadata={"destination": destination, "section": section, "category": section, "name": section.replace("_", " ").title(), "is_schedulable": False},
            ))

    return chunks


def load_all_destinations(data_dir: str) -> list[Chunk]:
    directory = Path(data_dir)
    if not directory.exists():
        return []
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.json")):
        chunks.extend(load_destination_file(path))
    return chunks


def known_destinations(data_dir: str) -> list[str]:
    directory = Path(data_dir)
    if not directory.exists():
        return []
    names = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            names.append(data.get("destination") or path.stem.replace("_", " ").title())
        except (json.JSONDecodeError, OSError):
            continue
    return names
