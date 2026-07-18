"""Refine & Rollback — the surgical-editing panel (explicit day picker +
instruction, backed by the hash-verified locking in
src/refinement/locking.py) and checkpoint history / rollback."""
from __future__ import annotations

import streamlit as st

from frontend import api_client
from frontend.theme import COLORS
from frontend.views.trip_planner import render_day_timeline

QUICK_INSTRUCTIONS = [
    "Make this day more relaxed with fewer activities",
    "Add more nightlife to this day",
    "Replace with adventure/outdoor activities",
    "Make this day more budget-friendly",
    "Add more cultural/historical sites",
    "Focus this day on food and local cuisine",
]


def _refine_panel(result: dict) -> None:
    itinerary = result["itinerary"]
    day_numbers = [d["day_number"] for d in itinerary["days"]]

    st.markdown("### Edit specific days")
    st.caption(
        "Pick the day(s) you want changed. Every other day is guaranteed byte-identical afterward — "
        "not just 'mostly preserved' — because the app never sends untouched days to the model at all."
    )

    target_days = st.multiselect("Day(s) to change", day_numbers, default=[day_numbers[0]] if day_numbers else [])
    quick_pick = st.selectbox("Quick instruction (optional)", [""] + QUICK_INSTRUCTIONS)
    instruction = st.text_area(
        "Instruction", value=quick_pick,
        placeholder="e.g. Replace Day 2 with more relaxed, nightlife-focused activities",
    )

    if st.button("Apply refinement", type="primary", disabled=not target_days or not instruction.strip(), key="apply_refinement_button"):
        with st.spinner(f"Regenerating day(s) {target_days}..."):
            try:
                new_result = api_client.refine_trip(st.session_state.session_id, instruction.strip(), target_days)
            except api_client.APIError as exc:
                st.error(f"Refinement failed: {exc.detail}")
                return
        st.session_state.trip_result = new_result
        st.success(f"Updated day(s) {target_days}. New version: {new_result['itinerary']['version']}.")
        if new_result.get("warnings"):
            for w in new_result["warnings"]:
                st.warning(w)
        st.rerun()


def _rollback_panel() -> None:
    st.markdown("### Checkpoint history & rollback")
    st.caption("Every plan and refinement step is checkpointed. You can revert to any earlier point.")

    if st.button("Load checkpoint history", key="load_checkpoint_history_button"):
        try:
            listing = api_client.list_checkpoints(st.session_state.session_id)
        except api_client.APIError as exc:
            st.error(f"Could not load history: {exc.detail}")
            return
        st.session_state["_checkpoint_listing"] = listing["checkpoints"]

    checkpoints = st.session_state.get("_checkpoint_listing")
    if not checkpoints:
        return

    for cp in checkpoints:
        cols = st.columns([3, 2, 2, 2])
        cols[0].markdown(f"<span class='at-mono'>{cp['checkpoint_id'][:18]}...</span>", unsafe_allow_html=True)
        cols[1].write(cp.get("created_at", "")[:19].replace("T", " "))
        version = f"v{cp['itinerary_version']}" if cp.get("itinerary_version") else "—"
        cols[2].write(f"Itinerary {version}" if cp.get("has_itinerary") else "No itinerary yet")
        if cols[3].button("Restore", key=f"restore-{cp['checkpoint_id']}"):
            try:
                restored = api_client.rollback_to(st.session_state.session_id, cp["checkpoint_id"])
            except api_client.APIError as exc:
                st.error(f"Rollback failed: {exc.detail}")
                return
            st.session_state.trip_result = restored
            st.success(f"Rolled back to checkpoint {cp['checkpoint_id'][:12]}...")
            st.rerun()


def render() -> None:
    st.title("Refine & Rollback")
    result = st.session_state.get("trip_result")

    if not result:
        st.info("Plan a trip first — refinement and rollback both operate on an existing session.")
        return

    tab1, tab2 = st.tabs(["Surgical refinement", "Checkpoint history"])
    with tab1:
        _refine_panel(result)
        st.markdown("---")
        st.markdown("### Current itinerary")
        render_day_timeline(result["itinerary"])
    with tab2:
        _rollback_panel()
