"""Memory Viewer — shows what the Memory Agent has learned about this
traveler. Deliberately shows only style signals, never raw place names
(see src/memory/behavioral.py) — that's not a UI choice, it's what's
actually stored."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import api_client
from frontend.theme import COLORS


def render() -> None:
    st.title("Memory")
    st.caption(
        "Behavioral preferences learned across trips — destination-agnostic by design, so a preference "
        "learned on one trip usefully applies to the next one anywhere in the world."
    )

    try:
        profile = api_client.get_memory(st.session_state.user_id)
    except api_client.APIError as exc:
        st.error(f"Could not load profile: {exc.detail}")
        return

    prefs = profile.get("behavioral_preferences", {})
    past_trips = profile.get("past_trips", [])

    if not past_trips and not prefs.get("interest_weights"):
        st.info(f"No memory on file yet for **{st.session_state.user_id}** — plan a trip to start building a profile.")
        return

    cols = st.columns(3)
    cols[0].metric("Pace preference", prefs.get("pace") or "—")
    cols[1].metric("Budget tier leaning", prefs.get("budget_tier") or "—")
    cols[2].metric("Trips on file", len(past_trips))

    interest_weights = prefs.get("interest_weights", {})
    if interest_weights:
        st.markdown("### Interest weights")
        df = pd.DataFrame(sorted(interest_weights.items(), key=lambda kv: kv[1], reverse=True), columns=["Interest", "Weight"])
        fig = go.Figure(go.Bar(
            x=df["Weight"], y=df["Interest"], orientation="h", marker_color=COLORS["teal"],
        ))
        fig.update_layout(height=max(220, 40 * len(df)), margin=dict(t=10, b=10, l=10, r=10),
                           xaxis=dict(range=[0, 1]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    if prefs.get("food_preferences"):
        st.markdown("### Food preferences")
        st.write(", ".join(prefs["food_preferences"]))

    if prefs.get("notes"):
        st.markdown("### Style notes (place-names automatically scrubbed)")
        for note in prefs["notes"]:
            st.markdown(f"- {note}")

    if past_trips:
        st.markdown("### Trip history")
        df = pd.DataFrame(past_trips)
        st.dataframe(df, use_container_width=True, hide_index=True)
