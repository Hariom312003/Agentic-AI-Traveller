"""Dashboard — landing page: quick status, current trip summary if one
exists, and shortcuts. Deliberately light; the detailed views live on
their own pages."""
from __future__ import annotations

import streamlit as st

from frontend import api_client


def render() -> None:
    st.title("Dashboard")

    result = st.session_state.get("trip_result")

    if not result:
        st.markdown(
            "Welcome! This is a multi-agent AI travel planner: a **Query Agent** parses your request, "
            "a **RAG Agent** retrieves verified destination knowledge, a **Planner Agent** builds the "
            "day-by-day itinerary, and **Budget**, **Rewards**, and **Validator** agents check and cost it "
            "out — with every step logged for the Agent Monitor page."
        )
        st.info("No active trip yet — head to **Plan a Trip** in the sidebar to generate one.")

        try:
            health = api_client.health()
            cols = st.columns(3)
            cols[0].metric("Knowledge base documents", health["vector_store_documents"])
            cols[1].metric("Configured LLM providers", len(health["configured_providers"]))
            cols[2].metric("Environment", health["environment"])
            if not health["configured_providers"]:
                st.warning(
                    "No LLM provider API keys are configured yet — the system will still work end-to-end "
                    "using the rule-based fallback planner, but for AI-generated itineraries add at least "
                    "one key to your `.env` file (see `.env.example`)."
                )
        except Exception:
            st.error("Could not reach the backend API. Start it with `./run_api.sh`.")
        return

    itinerary = result["itinerary"]
    budget = result.get("budget")
    st.success(f"Active trip: **{itinerary['destination']}**, {itinerary['duration_days']} days (version {itinerary['version']})")

    cols = st.columns(4)
    cols[0].metric("Destination", itinerary["destination"])
    cols[1].metric("Days", itinerary["duration_days"])
    if budget:
        cols[2].metric("Est. total", f"{budget['total']:,.0f} {budget['currency']}")
    trace = result.get("explainability", {}).get("execution_trace", [])
    cols[3].metric("Agents run", len(trace))

    st.markdown("Use the sidebar to open **Plan a Trip** for the full timeline, **Budget & Rewards** for the "
                "cost breakdown, or **Refine & Rollback** to make surgical edits.")
