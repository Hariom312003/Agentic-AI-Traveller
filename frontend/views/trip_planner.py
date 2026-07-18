"""Trip Planner — the main page: request form, then the day-by-day
itinerary rendered as a transit-map-style timeline (the app's signature
visual — see frontend/theme.py's module docstring for the design
rationale)."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from frontend import api_client
from frontend.theme import COLORS, badge_for_source

INTEREST_OPTIONS = [
    "nightlife", "culture", "adventure", "nature", "food", "shopping",
    "history", "relaxation", "photography", "wildlife",
]
FOOD_OPTIONS = ["vegetarian", "vegan", "non-vegetarian", "halal", "no seafood", "no restrictions"]


def _render_form() -> None:
    st.title("Plan a trip")
    st.caption("Describe the trip in your own words, or fill in the fields below — both feed the same Query Agent.")

    with st.form("plan_trip_form"):
        raw_query = st.text_area(
            "Describe your trip",
            placeholder="e.g. Plan a 5 day trip to Goa under ₹60000 for 2 people, we love nightlife and old forts",
            height=90,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            destination = st.text_input("Destination (optional if described above)")
            duration_days = st.number_input("Duration (days)", min_value=1, max_value=30, value=4)
        with col2:
            travelers_count = st.number_input("Travelers", min_value=1, max_value=20, value=2)
            travel_style = st.selectbox(
                "Travel style", ["", "budget", "mid-range", "luxury", "backpacking", "honeymoon", "family", "business", "adventure"],
            )
        with col3:
            budget_amount = st.number_input("Budget (optional)", min_value=0, value=0, step=1000)
            budget_currency = st.selectbox("Currency", ["INR", "USD", "EUR", "GBP"])

        interests = st.multiselect("Interests", INTEREST_OPTIONS)
        food_preferences = st.multiselect("Food preferences", FOOD_OPTIONS)
        special_requests = st.text_input("Anything specific to include? (optional)")

        submitted = st.form_submit_button("Generate itinerary", use_container_width=True)

    if submitted:
        if not raw_query.strip() and not destination.strip():
            st.error("Please describe your trip or at least enter a destination.")
            return

        payload = {
            "raw_query": raw_query.strip() or f"Plan a {duration_days} day trip to {destination}",
            "user_id": st.session_state.user_id,
            "destination": destination.strip() or None,
            "duration_days": int(duration_days),
            "travelers_count": int(travelers_count),
            "travel_style": travel_style or None,
            "budget_amount": float(budget_amount) if budget_amount else None,
            "budget_currency": budget_currency,
            "interests": interests,
            "food_preferences": food_preferences,
            "special_requests": special_requests or None,
        }
        with st.spinner("Running the multi-agent pipeline (query \u2192 memory \u2192 RAG \u2192 planner \u2192 budget \u2192 rewards \u2192 validator)..."):
            try:
                result = api_client.plan_trip(payload)
            except api_client.APIError as exc:
                st.error(f"Planning failed: {exc.detail}")
                return
        st.session_state.session_id = result["session_id"]
        st.session_state.trip_result = result
        st.rerun()


def _slot_icon(slot: str) -> str:
    return {"morning": "\u2600\ufe0f", "afternoon": "\U0001F324\ufe0f", "evening": "\U0001F307", "night": "\U0001F303"}.get(slot, "")


def render_day_timeline(itinerary: dict) -> None:
    """The signature visual: each day as a vertical transit-line timeline
    with ticket-stub stops, and a passport-stamp badge on days the
    Refinement Agent marked `locked` — a direct visual callback to the
    hash-verified locking guarantee, not just decoration."""
    for day in itinerary["days"]:
        header_cols = st.columns([5, 1])
        with header_cols[0]:
            title = f"Day {day['day_number']}"
            if day.get("theme"):
                title += f" — {day['theme']}"
            st.markdown(f"### {title}")
        with header_cols[1]:
            if day.get("locked"):
                st.markdown(
                    f'<div class="at-stamp">\U0001F510 LOCKED</div>', unsafe_allow_html=True,
                )

        for slot in ("morning", "afternoon", "evening", "night"):
            activities = day.get(slot, [])
            if not activities:
                continue
            st.markdown(f"**{_slot_icon(slot)} {slot.capitalize()}**")
            for activity in activities:
                badge = badge_for_source(activity.get("source", "model_knowledge"))
                cost = f"~{activity['estimated_cost']:.0f} {activity.get('currency','INR')}" if activity.get("estimated_cost") else ""
                duration = f"{activity['duration_minutes']} min" if activity.get("duration_minutes") else ""
                meta = "  \u00b7  ".join(x for x in [activity.get("start_time", ""), cost, duration] if x)
                st.markdown(
                    f"""
                    <div class="at-ticket">
                      <div class="at-ticket-time">{meta}</div>
                      <b>{activity['title']}</b> {badge}
                      <div style="margin-top:0.3rem; color:{COLORS['ink_soft']}; font-size:0.9rem;">
                        {activity.get('description','')}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)


def render_trip_overview(overview: dict) -> None:
    st.markdown("### ✈️ Trip Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Destination:** {overview.get('destination')}")
        st.markdown(f"**Duration:** {overview.get('duration_days')} Days")
        st.markdown(f"**Travel Style:** {overview.get('travel_style')}")
        st.markdown(f"**Estimated Budget:** {overview.get('estimated_budget')}")
    with col2:
        st.markdown(f"**Best Time to Visit:** {overview.get('best_time_to_visit')}")
        st.markdown(f"**Total Attractions:** {overview.get('total_attractions')}")
        st.markdown(f"**Cities Covered:** {', '.join(overview.get('cities_covered', []))}")
        st.markdown(f"**Total Walking Distance:** {overview.get('total_walking_distance')}")
    with col3:
        st.markdown(f"**Estimated Daily Travel:** {overview.get('estimated_daily_travel')}")
        st.markdown(f"**Recommended Transport:** {overview.get('recommended_transport')}")
        st.markdown(f"**Weather:** {overview.get('weather')}")
        st.markdown(f"**Difficulty:** {overview.get('difficulty')} | **Rating:** {overview.get('overall_trip_rating')}")


def render_ai_trip_summary(ai_summary: str) -> None:
    st.markdown("### 📝 AI Trip Summary")
    st.info(ai_summary)


def render_trip_highlights(h: dict) -> None:
    st.markdown("### 🎯 Trip Highlights")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("**Top Attractions:**")
        for item in h.get("top_attractions", []):
            st.markdown(f"- {item}")
        st.markdown("**Hidden Gems:**")
        for item in h.get("hidden_gems", []):
            st.markdown(f"- {item}")
        st.markdown("**Best Restaurants:**")
        for item in h.get("best_restaurants", []):
            st.markdown(f"- {item}")
    with col_h2:
        st.markdown("**Signature Local Foods:**")
        for item in h.get("signature_local_foods", []):
            st.markdown(f"- {item}")
        if h.get("best_sunset_spot"):
            st.markdown(f"**Best Sunset Spot:** {h.get('best_sunset_spot')}")
        st.markdown("**Must-Try Experiences:**")
        for item in h.get("must_try_experiences", []):
            st.markdown(f"- {item}")


def render_budget_dashboard(b: dict) -> None:
    st.markdown("### 💰 Budget Summary Dashboard")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown(f"- **Flights:** {b.get('flights')}")
        st.markdown(f"- **Hotels:** {b.get('hotels')}")
        st.markdown(f"- **Food:** {b.get('food')}")
        st.markdown(f"- **Transport:** {b.get('transport')}")
        st.markdown(f"- **Activities:** {b.get('activities')}")
    with col_b2:
        st.markdown(f"- **Shopping:** {b.get('shopping')}")
        st.markdown(f"- **Emergency Buffer:** {b.get('emergency_buffer')}")
        st.markdown(f"- **Taxes:** {b.get('taxes')}")
        st.markdown(f"**Total Cost:** **{b.get('total_cost')}**")
        st.markdown(f"**Budget Utilization:** {b.get('budget_utilization')}")
    if b.get("savings_suggestions"):
        with st.expander("Savings Suggestions"):
            for s in b.get("savings_suggestions", []):
                st.markdown(f"- {s}")


def render_weather_overview(w: dict) -> None:
    st.markdown("### 🌤 Weather Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Average Temperature:** {w.get('average_temperature')}")
        st.markdown(f"**Rain Probability:** {w.get('rain_probability')}")
    with col2:
        st.markdown(f"**Clothing:** {w.get('clothing_recommendations')}")
        if w.get("weather_warnings"):
            st.warning(f"Warning: {w.get('weather_warnings')}")
    if w.get("packing_suggestions"):
        st.markdown("**Packing Suggestions:** " + ", ".join(w.get("packing_suggestions", [])))


def render_food_recommendations(food_recs: dict) -> None:
    st.markdown("### 🍽 Food Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Vegetarian Options:** {food_recs.get('vegetarian_options')}")
        st.markdown("**Must-Try Local Foods:**")
        st.markdown(", ".join(food_recs.get("must_try_local_foods", [])))
    with col2:
        st.markdown("**Street Food Recommendations:**")
        st.markdown(", ".join(food_recs.get("street_food_recommendations", [])))
        st.markdown("**Famous Restaurants:**")
        st.markdown(", ".join(food_recs.get("famous_restaurants", [])))


def render_transportation_summary(t: dict) -> None:
    st.markdown("### 🚖 Transportation Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Airport Transfer:** {t.get('airport_transfer')}")
        st.markdown(f"**Local Transport:** {t.get('local_transport')}")
        st.markdown(f"**Metro:** {t.get('metro')}")
    with col2:
        st.markdown(f"**Taxi:** {t.get('taxi')}")
        st.markdown(f"**Ride Sharing:** {t.get('ride_sharing')}")
        st.markdown(f"**Walking:** {t.get('walking')}")
        st.markdown(f"**Estimated Daily Travel Time:** {t.get('estimated_daily_travel_time')}")


def render_important_travel_tips(tips: dict) -> None:
    st.markdown("### ⚠️ Important Travel Tips")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Local Customs:** {tips.get('local_customs')}")
        st.markdown(f"**Safety Advice:** {tips.get('safety_advice')}")
        st.markdown(f"**Currency:** {tips.get('currency')}")
        st.markdown(f"**Common Scams:** {tips.get('common_scams')}")
        st.markdown(f"**Tipping Etiquette:** {tips.get('tipping_etiquette')}")
    with col2:
        st.markdown(f"**Emergency Contacts:** {tips.get('emergency_contacts')}")
        st.markdown(f"**Internet Availability:** {tips.get('internet_availability')}")
        st.markdown(f"**SIM Card Suggestions:** {tips.get('sim_card_suggestions')}")
        st.markdown(f"**Local Laws:** {tips.get('local_laws')}")
        st.markdown(f"**Cultural Etiquette:** {tips.get('cultural_etiquette')}")


def _budget_chart(budget: dict) -> go.Figure:
    labels = ["Flights", "Hotels", "Food", "Activities", "Shopping", "Local Transport", "Emergency Buffer", "Taxes & Fees"]
    keys = ["flights", "hotels", "food", "activities", "shopping", "local_transport", "emergency_buffer", "taxes_and_fees"]
    values = [budget.get(k, 0) for k in keys]
    chart_colors = [COLORS["teal"], COLORS["brass"], "#7A8B6F", "#3A4562", COLORS["brick"], "#8FB8AC", "#D9C08A", "#5C6B8A"]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=chart_colors),
        textinfo="label+percent", textfont=dict(family="Work Sans, sans-serif", size=12),
    )])
    fig.update_layout(
        showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=380,
        annotations=[dict(text=f"{budget['total']:,.0f}<br>{budget['currency']}", x=0.5, y=0.5, font_size=18, showarrow=False)],
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_budget_details(budget: dict, itinerary: dict) -> None:
    st.markdown("### 📊 Detailed Budget Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(_budget_chart(budget), use_container_width=True)
    with col2:
        st.metric("Total Estimated Cost", f"{budget['total']:,.0f} {budget['currency']}")
        st.metric("Per Day Cost", f"{budget['total'] / max(itinerary['duration_days'], 1):,.0f} {budget['currency']}")
        st.metric("Per Traveler Cost", f"{budget['total'] / max(itinerary['travelers_count'], 1):,.0f} {budget['currency']}")

    st.markdown("#### Line-item breakdown")
    df = pd.DataFrame([
        {"Category": k.replace("_", " ").title(), "Amount": v}
        for k, v in budget.items() if isinstance(v, (int, float)) and k not in ("total",)
    ]).sort_values("Amount", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True, column_config={
        "Amount": st.column_config.NumberColumn(format="%.0f " + budget["currency"])
    })


def _execution_timeline(trace: list[dict]) -> go.Figure:
    fig = go.Figure()
    colors = {True: COLORS["teal"], False: COLORS["brick"]}
    for record in trace:
        fig.add_trace(go.Bar(
            y=[record["agent_name"]], x=[record.get("latency_ms") or 0], orientation="h",
            marker_color=colors[record.get("success", True)],
            text=f"{record.get('latency_ms', 0):.0f} ms", textposition="outside",
            showlegend=False,
        ))
    fig.update_layout(
        height=max(220, 42 * len(trace)), margin=dict(t=10, b=10, l=10, r=40),
        xaxis_title="Latency (ms)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_agent_execution(explainability: dict) -> None:
    st.markdown("### 🤖 Agent Execution Timeline")
    exp = explainability
    cols = st.columns(3)
    cols[0].metric("Agents Run", exp["total_agents_run"])
    cols[1].metric("Total Pipeline Latency", f"{exp['total_latency_ms']:,.0f} ms")
    cols[2].metric("Providers Utilized", ", ".join(exp["providers_used"]) or "none (offline fallback)")

    trace = exp["execution_trace"]
    st.plotly_chart(_execution_timeline(trace), use_container_width=True)

    with st.expander("Show detailed agent execution trace"):
        for record in trace:
            status_icon = "✅" if record.get("success", True) else "❌"
            st.markdown(f"**{status_icon} {record['agent_name']}** ({record.get('latency_ms', 0):.0f} ms)")
            st.markdown(
                f"- **LLM Provider:** {record.get('llm_provider') or '—'}  |  "
                f"**LLM Model:** {record.get('llm_model') or '—'}  |  "
                f"**Retries:** {record.get('retry_count', 0)}"
            )
            if record.get("reasoning_summary"):
                st.markdown(f"- **Summary:** {record['reasoning_summary']}")
            if record.get("error"):
                st.error(record["error"])
            st.markdown("---")


def render_sources_and_confidence(itinerary: dict, validation: dict) -> None:
    st.markdown("### 📋 Sources and Grounding Confidence")
    grounded_pct = validation.get("grounded_ratio", 0.0)
    
    st.markdown(f"**Overall Verification Confidence:** {grounded_pct * 100:.0f}% Grounded in Curated Knowledge Base")
    st.progress(grounded_pct)
    
    kb_places = []
    model_places = []
    fallback_places = []
    
    for day in itinerary.get("days", []):
        for slot in ("morning", "afternoon", "evening", "night"):
            for act in day.get(slot, []):
                source = act.get("source", "model_knowledge")
                title_with_day = f"{act['title']} (Day {day['day_number']} {slot.capitalize()})"
                if source == "knowledge_base":
                    kb_places.append((title_with_day, act.get("source_doc_ids", [])))
                elif source == "model_knowledge":
                    model_places.append(title_with_day)
                elif source == "rule_based_fallback":
                    fallback_places.append(title_with_day)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🛡 Verified Knowledge Base Stops**")
        if kb_places:
            for place, doc_ids in kb_places:
                docs_str = f" [Chunks: {', '.join(doc_ids)}]" if doc_ids else ""
                st.markdown(f"- ✅ {place}{docs_str}")
        else:
            st.markdown("*No verified knowledge base stops were scheduled.*")
            
    with col2:
        st.markdown("**🧠 Model Knowledge & Fallback Stops**")
        if model_places or fallback_places:
            for place in model_places:
                st.markdown(f"- 💡 {place} (Model General Knowledge)")
            for place in fallback_places:
                st.markdown(f"- ⚠️ {place} (Offline Rule-Based Fallback)")
        else:
            st.markdown("*All stops are fully verified from the knowledge base.*")


def _render_result(result: dict) -> None:
    itinerary = result["itinerary"]
    budget = result.get("budget")
    summary = result.get("trip_summary")
    validation = result.get("validation", {})
    explainability = result.get("explainability")

    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)

    if validation and not validation.get("is_valid", True):
        st.error("This itinerary has unresolved critical validation issues — consider regenerating or refining it.")
    elif validation and validation.get("issues"):
        with st.expander(f"ℹ️ {len(validation['issues'])} validation note(s)"):
            for issue in validation["issues"]:
                st.write(f"**[{issue['severity']}]** {issue['message']}")

    st.markdown("---")

    if summary:
        render_trip_overview(summary.get("overview", {}))
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)
        render_ai_trip_summary(summary.get("ai_summary", ""))
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)
        render_trip_highlights(summary.get("highlights", {}))
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)
        render_budget_dashboard(summary.get("budget", {}))
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)
        render_weather_overview(summary.get("weather", {}))
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)
        render_food_recommendations(summary.get("food", {}))
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)
        render_transportation_summary(summary.get("transport", {}))
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)
        render_important_travel_tips(summary.get("tips", {}))
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)
        if summary.get("ai_reasoning"):
            st.markdown("### 🤖 AI Planning Reasoning")
            st.success(summary.get("ai_reasoning", ""))
            st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)

    # 9. Interactive Map Summary
    st.markdown("### 🗺 Interactive Map Summary")
    st.caption(
        "Best-effort geocoding via OpenStreetMap — needs outbound internet access "
        "and can take a few seconds. Not loaded automatically."
    )
    if st.button("Load map", key="load_trip_map_button"):
        from frontend.geo_map import build_trip_map_html
        with st.spinner("Locating stops on the map..."):
            try:
                map_html = build_trip_map_html(itinerary)
            except Exception:
                map_html = None
        if map_html:
            st.components.v1.html(map_html, height=480)
        else:
            st.info(
                "Couldn't geocode any stops right now (this needs outbound internet access to "
                "OpenStreetMap's geocoder). The day-by-day timeline below covers the same itinerary."
            )
    st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)

    # 10. Day-by-Day Itinerary
    st.markdown("### 📅 Day-by-Day Itinerary")
    render_day_timeline(itinerary)
    st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)

    # 11. Budget Details
    if budget:
        render_budget_details(budget, itinerary)
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)

    # 12. Agent Execution Timeline
    if explainability:
        render_agent_execution(explainability)
        st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)

    # 13. Sources and Confidence
    render_sources_and_confidence(itinerary, validation)
    st.markdown('<div class="at-divider"></div>', unsafe_allow_html=True)

    if st.button("Plan a new trip", key="plan_new_trip_button"):
        st.session_state.session_id = None
        st.session_state.trip_result = None
        st.rerun()


def render() -> None:
    if st.session_state.get("trip_result"):
        _render_result(st.session_state.trip_result)
    else:
        _render_form()
