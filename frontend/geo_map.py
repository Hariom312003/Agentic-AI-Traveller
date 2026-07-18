"""
Trip map — best-effort geocoding via Nominatim (OpenStreetMap), rendered
with folium (Leaflet.js) directly as HTML, so no `streamlit-folium`
dependency is needed: `folium.Map._repr_html_()` already returns a
self-contained HTML snippet that `st.components.v1.html()` can render.

Deliberately NOT a hard dependency of the rest of the app: Nominatim is a
free, rate-limited public service (max ~1 request/second, per their usage
policy — hence the `RateLimiter` below) and requires outbound internet
access from wherever this app is actually run. If a place fails to
geocode (offline environment, rate limited, name too ambiguous), that pin
is silently skipped rather than crashing the whole map — a partial map
beats no map, and the day timeline view (this module's sibling) already
covers the same information without needing geocoding at all.
"""
from __future__ import annotations

import folium
from geopy.exc import GeopyError
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from frontend.theme import COLORS

_geolocator = Nominatim(user_agent="ai_traveller_app (educational project)")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1.1, max_retries=1, swallow_exceptions=True)
_cache: dict[str, tuple[float, float] | None] = {}

# Nominatim's usage policy caps requests to ~1/second, and this call is
# synchronous/blocking in the Streamlit render path. Even at a perfect 1
# req/sec, a 5-day itinerary's ~20 activities would take 20+ seconds
# geocoded one at a time — capping the pin count keeps "Load map" bounded
# and responsive rather than scaling linearly with trip length.
MAX_PINS = 10


def geocode_place(name: str, destination: str) -> tuple[float, float] | None:
    """Cached, rate-limited, never raises. Returns None (not an exception)
    on any failure so callers can just skip the pin."""
    cache_key = f"{name}|{destination}"
    if cache_key in _cache:
        return _cache[cache_key]
    try:
        query = f"{name}, {destination}"
        location = _geocode(query, timeout=5)
        result = (location.latitude, location.longitude) if location else None
    except GeopyError:
        result = None
    except Exception:
        result = None
    _cache[cache_key] = result
    return result


_DAY_MARKER_COLORS = ["#2F6E5E", "#C08A28", "#16213A", "#B3452F", "#3A4562", "#7A8B6F"]


def build_trip_map_html(itinerary: dict) -> str | None:
    """Returns folium's self-contained map HTML, or None if nothing could
    be geocoded (caller should show the day-timeline view instead)."""
    destination = itinerary["destination"]
    points: list[tuple[float, float, str, int]] = []
    geocode_count = 0

    for day in itinerary["days"]:
        for slot in ("morning", "afternoon", "evening", "night"):
            for activity in day.get(slot, []):
                title = activity.get("title", "")
                lat = activity.get("latitude")
                lng = activity.get("longitude")
                
                if lat is not None and lng is not None:
                    try:
                        points.append((float(lat), float(lng), title, day["day_number"]))
                    except (ValueError, TypeError):
                        pass
                else:
                    if geocode_count < MAX_PINS:
                        coords = geocode_place(title, destination)
                        if coords:
                            points.append((*coords, title, day["day_number"]))
                            geocode_count += 1

    if not points:
        return None

    avg_lat = sum(p[0] for p in points) / len(points)
    avg_lng = sum(p[1] for p in points) / len(points)
    fmap = folium.Map(location=[avg_lat, avg_lng], zoom_start=12, tiles="CartoDB positron")

    for lat, lng, title, day_number in points:
        color = _DAY_MARKER_COLORS[(day_number - 1) % len(_DAY_MARKER_COLORS)]
        folium.CircleMarker(
            location=[lat, lng], radius=8, color=color, fill=True, fill_color=color, fill_opacity=0.85,
            tooltip=f"Day {day_number}: {title}",
        ).add_to(fmap)

    return fmap._repr_html_()
