from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.monitoring.logging_config import get_logger

logger = get_logger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def score_place(place: dict, interests: list[str], travel_style: str | None, season: str | None = None) -> float:
    """Score a candidate place based on interest tags, travel style, and seasonality."""
    score = 100.0
    category = str(place.get("category", "")).lower()

    # Interest matching
    for interest in interests:
        interest_lower = interest.lower().strip()
        if interest_lower in category:
            score += 30.0
        if interest_lower in [t.lower() for t in place.get("tags", [])]:
            score += 20.0

    # Travel style / Budget alignment
    budget_cat = str(place.get("budget_category", "")).lower()
    if travel_style:
        style_lower = travel_style.lower().strip()
        if style_lower == "budget" and ("free" in budget_cat or "low" in budget_cat):
            score += 25.0
        elif style_lower == "luxury" and ("high" in budget_cat or "premium" in budget_cat):
            score += 25.0

    # Seasonality check
    if season:
        season_lower = season.lower().strip()
        if season_lower in [t.lower() for t in place.get("tags", [])] or season_lower in str(place.get("best_time", "")).lower():
            score += 20.0

    # Hidden gem boost for uniqueness
    if place.get("section") == "hidden_gems":
        score += 15.0

    return score


def build_candidate_pool(destination: str, interests: list[str], travel_style: str | None, season: str | None = None) -> list[dict]:
    """Retrieve all places for a destination from cache and sort them by relevance score."""
    settings = get_settings()
    sanitized = destination.lower().strip().replace(" ", "_")
    filepath = Path(settings.destinations_data_path) / f"{sanitized}.json"

    if not filepath.exists():
        return []

    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to read guide JSON file at {filepath}: {exc}")
        return []

    candidates = []
    sections = [
        "attractions",
        "restaurants",
        "hidden_gems",
        "nightlife",
        "shopping",
        "adventure_activities",
    ]
    for sec in sections:
        for idx, entity in enumerate(data.get(sec, [])):
            entity = dict(entity)
            entity["section"] = sec
            # Create a unique chunk id if not present
            entity["chunk_id"] = entity.get(
                "chunk_id",
                f"{sanitized}::{sec}::{idx}::{entity.get('name','').lower().replace(' ', '_')[:20]}",
            )
            candidates.append(entity)

    # Score and sort candidates
    scored = [(score_place(c, interests, travel_style, season), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    return [x[1] for x in scored]


def cluster_places_by_day(places: list[dict], num_days: int) -> list[list[dict]]:
    """Cluster valid coordinates using K-Means into `num_days` sets, distributing unmapped ones."""
    if not places:
        return [[] for _ in range(num_days)]

    valid_places = []
    invalid_places = []
    for p in places:
        if p.get("latitude") is not None and p.get("longitude") is not None:
            valid_places.append(p)
        else:
            invalid_places.append(p)

    num_days = min(num_days, len(valid_places) or num_days)
    if not valid_places:
        clusters = [[] for _ in range(num_days)]
        for idx, p in enumerate(places):
            clusters[idx % num_days].append(p)
        return clusters

    # Centroid initialization
    centroids = []
    for i in range(num_days):
        idx = int(i * len(valid_places) / num_days)
        centroids.append(
            {"latitude": valid_places[idx]["latitude"], "longitude": valid_places[idx]["longitude"]}
        )

    # 5 iterations of K-Means
    clusters = [[] for _ in range(num_days)]
    for _ in range(5):
        clusters = [[] for _ in range(num_days)]
        for p in valid_places:
            min_dist = float("inf")
            best_c = 0
            for c_idx, c in enumerate(centroids):
                dist = haversine_distance(p["latitude"], p["longitude"], c["latitude"], c["longitude"])
                if dist < min_dist:
                    min_dist = dist
                    best_c = c_idx
            clusters[best_c].append(p)

        # Update centroids
        for c_idx in range(num_days):
            c_places = clusters[c_idx]
            if c_places:
                avg_lat = sum(p["latitude"] for p in c_places) / len(c_places)
                avg_lng = sum(p["longitude"] for p in c_places) / len(c_places)
                centroids[c_idx] = {"latitude": avg_lat, "longitude": avg_lng}

    # Distribute invalid places
    for idx, p in enumerate(invalid_places):
        clusters[idx % num_days].append(p)

    return clusters


def optimize_route(places: list[dict]) -> list[dict]:
    """Order places using a greedy Travelling Salesperson algorithm starting from the first item."""
    if len(places) <= 2:
        return places

    # Calculate centroid of valid coordinates to use as fallback for places missing coordinates
    valid_lats = [p["latitude"] for p in places if p.get("latitude") is not None]
    valid_lngs = [p["longitude"] for p in places if p.get("longitude") is not None]
    centroid_lat = sum(valid_lats) / len(valid_lats) if valid_lats else 0.0
    centroid_lng = sum(valid_lngs) / len(valid_lngs) if valid_lngs else 0.0

    ordered = [places[0]]
    remaining = list(places[1:])

    while remaining:
        last = ordered[-1]
        best_idx = 0
        min_dist = float("inf")

        last_lat = last.get("latitude") if last.get("latitude") is not None else centroid_lat
        last_lng = last.get("longitude") if last.get("longitude") is not None else centroid_lng

        for idx, p in enumerate(remaining):
            p_lat = p.get("latitude") if p.get("latitude") is not None else centroid_lat
            p_lng = p.get("longitude") if p.get("longitude") is not None else centroid_lng
            
            dist = haversine_distance(last_lat, last_lng, p_lat, p_lng)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        ordered.append(remaining.pop(best_idx))

    return ordered


def assign_slots_for_day(day_places: list[dict]) -> dict[str, list[dict]]:
    """Assign sorted day places into standard Morning, Afternoon, Evening, and Night buckets."""
    slots = {"morning": [], "afternoon": [], "evening": [], "night": []}

    attractions = []
    food = []
    nightlife = []
    shopping = []

    for p in day_places:
        sec = p.get("section", "attractions")
        cat = str(p.get("category", "")).lower()
        if sec == "nightlife" or "night" in cat:
            nightlife.append(p)
        elif sec == "restaurants" or "food" in cat or "eat" in cat:
            food.append(p)
        elif sec == "shopping" or "shop" in cat:
            shopping.append(p)
        else:
            attractions.append(p)

    # 1. Morning slot (must be attraction)
    if attractions:
        slots["morning"].append(attractions.pop(0))
    elif shopping:
        slots["morning"].append(shopping.pop(0))

    # 2. Afternoon slot (Attraction / Lunch)
    if food:
        slots["afternoon"].append(food.pop(0))
    if attractions:
        slots["afternoon"].append(attractions.pop(0))

    # 3. Evening slot (Shopping / Attraction)
    if shopping:
        slots["evening"].append(shopping.pop(0))
    elif attractions:
        slots["evening"].append(attractions.pop(0))

    # 4. Night slot (Dinner / Nightlife)
    if food:
        slots["night"].append(food.pop(0))
    if nightlife:
        slots["night"].append(nightlife.pop(0))

    # Distribute leftovers to even out the slots
    remaining = attractions + food + nightlife + shopping
    for p in remaining:
        min_slot = min(slots.keys(), key=lambda k: len(slots[k]))
        slots[min_slot].append(p)

    return slots


def build_optimized_context_block(destination: str, interests: list[str], travel_style: str | None, num_days: int, season: str | None = None) -> tuple[str, list[dict]]:
    """Build the geographically optimized prompt text skeleton and candidate list for the Planner Agent."""
    candidates = build_candidate_pool(destination, interests, travel_style, season)
    if not candidates:
        return "", []

    # Limit to top 100 candidates, score/rank them, and select the top 25 high-relevance candidates
    candidates = candidates[:100]
    top_candidates = candidates[:25]

    # Cluster by day
    day_clusters = cluster_places_by_day(top_candidates, num_days)

    lines = [
        f"Verified Curated Knowledge Base Skeleton for {destination} (You MUST schedule these pre-routed spots exactly as outlined below):"
    ]

    all_allowed = []

    for d_idx, day_places in enumerate(day_clusters):
        day_num = d_idx + 1
        # Route optimization within day
        routed_places = optimize_route(day_places)
        scheduled_slots = assign_slots_for_day(routed_places)

        lines.append(f"\n--- DAY {day_num} ---")
        for slot in ("morning", "afternoon", "evening", "night"):
            slot_places = scheduled_slots[slot]
            if slot_places:
                lines.append(f"Slot: {slot.capitalize()}")
                for p in slot_places:
                    name = p["name"]
                    chunk_id = p["chunk_id"]
                    lat = p.get("latitude")
                    lng = p.get("longitude")
                    addr = p.get("address")
                    mlink = p.get("map_link")
                    duration = p.get("recommended_duration") or "2 hours"
                    cost = p.get("budget_category") or "Varies"
                    desc = p.get("description", "")
                    
                    lines.append(
                        f"  - [{chunk_id}] {name} ({destination})\n"
                        f"    Category: {p.get('category', 'attraction')}\n"
                        f"    Description: {desc}\n"
                        f"    Recommended Duration: {duration}\n"
                        f"    Budget Category: {cost}\n"
                        f"    Latitude: {lat}\n"
                        f"    Longitude: {lng}\n"
                        f"    Address: {addr}\n"
                        f"    Map Link: {mlink}"
                    )
                    
                    all_allowed.append({
                        "name": name,
                        "category": p.get("category", "attraction"),
                        "chunk_id": chunk_id,
                        "recommended_duration": duration,
                        "budget_category": cost,
                        "latitude": lat,
                        "longitude": lng,
                        "address": addr,
                        "map_link": mlink
                    })

    prompt_block = "\n".join(lines)
    return prompt_block, all_allowed
