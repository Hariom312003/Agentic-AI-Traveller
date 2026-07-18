"""Export — download the finished itinerary as PDF or raw JSON."""
from __future__ import annotations

import json

import streamlit as st

from frontend.pdf_export import build_itinerary_pdf


def render() -> None:
    st.title("Export")
    result = st.session_state.get("trip_result")

    if not result:
        st.info("Plan a trip first — there's nothing to export yet.")
        return

    itinerary = result["itinerary"]
    st.write(f"Exporting **{itinerary['destination']}**, {itinerary['duration_days']} days (version {itinerary['version']}).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### PDF")
        st.caption("A formatted document with the day-by-day plan, budget breakdown, and reward ideas.")
        try:
            pdf_bytes = build_itinerary_pdf(itinerary, result.get("budget"), result.get("rewards"))
            st.download_button(
                "Download PDF", data=pdf_bytes,
                file_name=f"{itinerary['destination'].lower().replace(' ', '_')}_itinerary.pdf",
                mime="application/pdf", use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not generate PDF: {exc}")

    with col2:
        st.markdown("#### JSON")
        st.caption("The complete raw response — itinerary, budget, rewards, validation, and the full execution trace.")
        json_bytes = json.dumps(result, indent=2, default=str).encode("utf-8")
        st.download_button(
            "Download JSON", data=json_bytes,
            file_name=f"{itinerary['destination'].lower().replace(' ', '_')}_trip.json",
            mime="application/json", use_container_width=True,
        )

    with st.expander("Preview raw JSON"):
        st.json(result)
