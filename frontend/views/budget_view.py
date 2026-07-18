"""Budget & Rewards — cost breakdown chart plus illustrative reward/savings
matches for the active trip."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.theme import COLORS

_CHART_COLORS = [COLORS["teal"], COLORS["brass"], "#7A8B6F", "#3A4562", COLORS["brick"], "#8FB8AC", "#D9C08A", "#5C6B8A"]


def _budget_chart(budget: dict) -> go.Figure:
    labels = ["Flights", "Hotels", "Food", "Activities", "Shopping", "Local Transport", "Emergency Buffer", "Taxes & Fees"]
    keys = ["flights", "hotels", "food", "activities", "shopping", "local_transport", "emergency_buffer", "taxes_and_fees"]
    values = [budget.get(k, 0) for k in keys]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=_CHART_COLORS),
        textinfo="label+percent", textfont=dict(family="Work Sans, sans-serif", size=12),
    )])
    fig.update_layout(
        showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=380,
        annotations=[dict(text=f"{budget['total']:,.0f}<br>{budget['currency']}", x=0.5, y=0.5, font_size=18, showarrow=False)],
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render() -> None:
    st.title("Budget & Rewards")
    result = st.session_state.get("trip_result")

    if not result or not result.get("budget"):
        st.info("No active trip with a budget yet — plan a trip first.")
        return

    budget = result["budget"]
    itinerary = result["itinerary"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(_budget_chart(budget), use_container_width=True)
    with col2:
        st.metric("Total estimated cost", f"{budget['total']:,.0f} {budget['currency']}")
        st.metric("Per day", f"{budget['total'] / max(itinerary['duration_days'], 1):,.0f} {budget['currency']}")
        st.metric("Per traveler", f"{budget['total'] / max(itinerary['travelers_count'], 1):,.0f} {budget['currency']}")

        validation = result.get("validation")
        if validation:
            status_labels = {"within_budget": "✅ Within budget", "over_budget": "⚠️ Over budget", "unknown": "No budget target set"}
            st.caption(status_labels.get(validation.get("budget_status"), ""))

    st.markdown("### Line-item breakdown")
    df = pd.DataFrame([
        {"Category": k.replace("_", " ").title(), "Amount": v}
        for k, v in budget.items() if isinstance(v, (int, float)) and k not in ("total",)
    ]).sort_values("Amount", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True, column_config={
        "Amount": st.column_config.NumberColumn(format="%.0f " + budget["currency"])
    })

    rewards = result.get("rewards")
    if rewards and rewards.get("recommendations"):
        st.markdown("### Reward & savings ideas")
        st.caption(rewards.get("disclaimer", ""))
        for rec in rewards["recommendations"]:
            with st.container():
                st.markdown(
                    f"""<div class="at-card">
                        <b>{rec['category']}</b> — {rec['instrument']}<br/>
                        <span style="color:{COLORS['ink_soft']}; font-size:0.9rem;">{rec['reason']}</span><br/>
                        <span class="at-mono" style="color:{COLORS['teal_dark']};">
                            ~{rec['estimated_savings']:,.0f} {rec['currency']} potential savings
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
