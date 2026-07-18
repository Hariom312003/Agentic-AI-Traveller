"""Agent Monitor — the explainability surface: which agent ran, in what
order, how long each took, which LLM (if any) answered, plus live provider
health and a tailing log console."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import api_client
from frontend.theme import COLORS


def _execution_timeline(trace: list[dict]) -> go.Figure:
    fig = go.Figure()
    colors = {True: COLORS["teal"], False: COLORS["brick"]}
    for i, record in enumerate(trace):
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


def _provider_health_section() -> None:
    st.markdown("### LLM provider health")
    try:
        status = api_client.provider_status()["providers"]
    except api_client.APIError as exc:
        st.error(f"Could not load provider status: {exc.detail}")
        return

    if not status:
        st.caption("No provider calls recorded yet this session.")
        return

    rows = []
    for name, health in status.items():
        rows.append({
            "Provider": name,
            "Circuit state": health["circuit_state"],
            "Successes": health["total_successes"],
            "Failures": health["total_failures"],
            "Rate limit hits": health["rate_limit_hits"],
            "Last error": (health.get("last_error") or "")[:80],
        })
    df = pd.DataFrame(rows)

    def _highlight(row):
        color = {"closed": "#E4EEE9", "open": "#F5E4DE", "half_open": "#F1EADB"}.get(row["Circuit state"], "")
        return [f"background-color: {color}"] * len(row)

    st.dataframe(df.style.apply(_highlight, axis=1), use_container_width=True, hide_index=True)


def render() -> None:
    st.title("Agent Monitor")
    st.caption("Explainability console: execution order, latency, LLM provider used, and reasoning per agent.")

    result = st.session_state.get("trip_result")
    if result and result.get("explainability"):
        exp = result["explainability"]
        cols = st.columns(3)
        cols[0].metric("Agents run", exp["total_agents_run"])
        cols[1].metric("Total latency", f"{exp['total_latency_ms']:,.0f} ms")
        cols[2].metric("Providers used", ", ".join(exp["providers_used"]) or "none (fallback paths only)")

        trace = exp["execution_trace"]
        st.markdown("### Execution timeline")
        st.plotly_chart(_execution_timeline(trace), use_container_width=True)

        st.markdown("### Per-agent detail")
        for record in trace:
            status_icon = "✅" if record.get("success", True) else "❌"
            with st.expander(f"{status_icon} {record['agent_name']} — {record.get('latency_ms', 0):.0f} ms"):
                c1, c2 = st.columns(2)
                c1.write(f"**LLM provider:** {record.get('llm_provider') or '—'}")
                c1.write(f"**LLM model:** {record.get('llm_model') or '—'}")
                c1.write(f"**Retries:** {record.get('retry_count', 0)}")
                c2.write(f"**Started:** {record.get('started_at', '')[:19].replace('T',' ')}")
                c2.write(f"**Retrieved docs:** {len(record.get('retrieved_doc_ids', []))}")
                if record.get("reasoning_summary"):
                    st.write(f"**Summary:** {record['reasoning_summary']}")
                if record.get("error"):
                    st.error(record["error"])
    else:
        st.info("No trip planned yet this session — plan a trip to see its execution trace here.")

    st.markdown("---")
    _provider_health_section()

    st.markdown("---")
    st.markdown("### Recent log console")
    limit = st.slider("Lines to show", 20, 500, 100)
    logs = api_client.recent_logs(limit=limit)
    if logs:
        log_text = "\n".join(
            f"[{l.get('timestamp','')}] {l.get('level',''):8s} {l.get('logger','')}: {l.get('message','')}"
            for l in logs
        )
        st.code(log_text, language="log")
    else:
        st.caption("No logs available yet (or the backend log file hasn't been created).")
