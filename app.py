"""
AI Traveller — Streamlit frontend entry point.

Run with: streamlit run app.py  (or `./run_ui.sh`)

This file owns only navigation and shared session state; every page's
actual content lives in frontend/views/*.py so this stays readable as the
page count grows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frontend import api_client
from frontend.theme import inject_theme
from frontend.views import (
    agent_monitor,
    budget_view,
    dashboard,
    export_view,
    memory_view,
    refinement_view,
    trip_planner,
)

st.set_page_config(
    page_title="AI Traveller",
    page_icon="\U0001F9ED",  # compass emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

DEFAULTS = {
    "user_id": "demo-traveler",
    "session_id": None,
    "trip_result": None,
    "current_page": "Dashboard",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

PAGES = {
    "Dashboard": dashboard,
    "Plan a Trip": trip_planner,
    "Budget & Rewards": budget_view,
    "Refine & Rollback": refinement_view,
    "Memory": memory_view,
    "Agent Monitor": agent_monitor,
    "Export": export_view,
}

with st.sidebar:
    st.markdown("## \U0001F9ED AI Traveller")
    st.caption("Multi-agent travel planning")

    st.session_state.user_id = st.text_input(
        "Traveler name / ID", value=st.session_state.user_id,
        help="Used to load and save your behavioral preferences across trips.",
    )

    page_name = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown("---")
    if api_client.is_backend_reachable():
        st.success("Backend connected", icon="\u2705")
    else:
        st.error("Backend unreachable", icon="\u26a0\ufe0f")
        st.caption(f"Expected at `{api_client.BASE_URL}` — start it with `./run_api.sh`.")

    if st.session_state.session_id:
        st.caption(f"Active session: `{st.session_state.session_id}`")

PAGES[page_name].render()
