from __future__ import annotations

import json
from pathlib import Path
import httpx

from src.agents.common import generate_structured
from src.config import get_settings
from src.models.destination_knowledge import DestinationKnowledge
from src.rag.chunking import Chunk, load_destination_file, known_destinations
from src.rag.embeddings import get_embedding_provider
from src.rag.vector_store import VectorStore
from src.monitoring.logging_config import get_logger

logger = get_logger(__name__)


def _sanitize_name(destination: str) -> str:
    return destination.lower().strip().replace(" ", "_")


def is_destination_cached(destination: str) -> bool:
    settings = get_settings()
    known = known_destinations(settings.destinations_data_path)
    dest_lower = destination.lower().strip()
    for k in known:
        if k.lower().strip() == dest_lower:
            return True
    return False


def _fetch_page_text(title: str, endpoint: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "exlimit": 1,
        "titles": title,
        "explaintext": 1,
        "format": "json",
        "redirects": 1,
    }
    headers = {"User-Agent": "AITraveller/1.0 (contact@example.com)"}
    try:
        response = httpx.get(endpoint, headers=headers, params=params, timeout=30.0)
        if response.status_code != 200:
            return ""
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return ""
        page = list(pages.values())[0]
        if "missing" in page:
            return ""
        return page.get("extract", "").strip()
    except Exception as exc:
        logger.warning(f"Error fetching page '{title}' from endpoint '{endpoint}': {exc}")
        return ""


def acquire_destination_knowledge(destination: str) -> list[Chunk]:
    settings = get_settings()
    sanitized = _sanitize_name(destination)
    dest_dir = Path(settings.destinations_data_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filepath = dest_dir / f"{sanitized}.json"

    # 1. Check file cache
    if filepath.exists():
        logger.info(f"Loading '{destination}' from file cache: {filepath}")
        return load_destination_file(filepath)

    logger.info(f"Acquiring fresh travel knowledge for '{destination}'...")

    # 2. Fetch raw guide text
    extract = ""
    # Try Wikivoyage first
    wikivoyage_endpoint = "https://en.wikivoyage.org/w/api.php"
    extract = _fetch_page_text(destination, wikivoyage_endpoint)

    # Try Wikipedia as fallback
    if not extract:
        logger.info("Wikivoyage page not found. Falling back to Wikipedia.")
        wikipedia_endpoint = "https://en.wikipedia.org/w/api.php"
        extract = _fetch_page_text(destination, wikipedia_endpoint)

    if not extract:
        logger.warning(f"No online guide found for '{destination}'. Relying fully on LLM world knowledge.")
        extract = f"Synthesize travel guide details for {destination} using your general world knowledge."

    # 3. Call LLM to parse raw text into structured schema
    system_prompt = (
        "You are an expert travel coordinator agent. Your job is to structure raw tourist guides and descriptions into clean, structured JSON guides.\n"
        "Rules:\n"
        "- Generate real, specific, and famous attractions (between 8 and 12 places), restaurants (4 to 6), hidden gems, nightlife, and shopping areas.\n"
        "- Do NOT use generic placeholders like 'visit a market' or 'walk downtown'. Every landmark name must be an actual, real-world spot.\n"
        "- Estimate coordinates (latitude, longitude) as float values and search map links (OpenStreetMap queries) for every landmark using your internal world knowledge.\n"
        "- Populate the exact Pydantic schema provided."
    )
    user_prompt = (
        f"Please extract and synthesize a travel guide for '{destination}' based on this source text:\n\n"
        f"{extract[:15000]}\n\n"  # Capped to avoid token bloat
        f"Generate the detailed JSON guide with coordinates, map links, recommended durations, and other fields.\n"
        f"IMPORTANT: The root of the JSON response MUST be a dictionary object (not a list). All attractions must be nested under the 'attractions' key."
    )

    try:
        model_out, _ = generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=DestinationKnowledge,
            temperature=0.3,
        )
    except Exception as exc:
        from src.agents.common import SchemaGenerationError, extract_json
        if isinstance(exc, SchemaGenerationError) and exc.last_raw_response:
            try:
                parsed = extract_json(exc.last_raw_response)
                # Rescue if the LLM outputted a list of items directly
                if isinstance(parsed, list):
                    logger.info("Rescued guide list response, converting to DestinationKnowledge dictionary.")
                    from src.models.destination_knowledge import KnowledgeAttraction
                    attractions = []
                    for item in parsed:
                        if isinstance(item, dict):
                            try:
                                attractions.append(KnowledgeAttraction.model_validate(item))
                            except Exception:
                                pass
                    model_out = DestinationKnowledge(destination=destination, attractions=attractions)
                elif isinstance(parsed, dict) and "attractions" in parsed:
                    model_out = DestinationKnowledge.model_validate(parsed)
                else:
                    raise exc
            except Exception as inner_exc:
                logger.error(f"Fallback parse failed for '{destination}': {inner_exc}")
                raise exc
        else:
            logger.error(f"Structured guide generation failed for '{destination}': {exc}")
            raise exc

    # Force destination field consistency
    model_out.destination = destination

    # 4. Save JSON file to cache
    filepath.write_text(model_out.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Saved cache file for '{destination}' to: {filepath}")

    # 5. Load and split into chunks
    chunks = load_destination_file(filepath)

    # 6. Index into Vector Database (ChromaDB)
    embedder = get_embedding_provider()
    vector_store = VectorStore(settings.chroma_db_path, embedder)
    if chunks:
        vector_store.upsert_chunks(chunks)
        logger.info(f"Successfully indexed {len(chunks)} chunks for '{destination}' into vector store.")

    return chunks
