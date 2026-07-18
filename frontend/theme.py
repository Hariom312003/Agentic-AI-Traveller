"""
Visual design system for the Streamlit frontend.

Direction: "field cartographer's notebook" rather than a generic SaaS
dashboard — the app is about journeys, so the visual language leans on
maps, tickets, and passport stamps instead of a default blue-gradient AI
look. Concretely:

- Palette: deep navy ink + a cool sage-stone neutral (not the ubiquitous
  warm cream) + forest-teal as the primary accent + brass-gold as a
  secondary accent for badges/highlights. A muted brick-red exists ONLY
  for warnings/validation issues, so it never competes with the teal CTA
  color.
- Type: "Fraunces" (a characterful serif with old travel-poster energy)
  for headings/destination names, "Work Sans" for body copy, and
  "JetBrains Mono" for anything resembling data — times, prices, agent
  latencies — deliberately styled like an airport departures board, which
  ties the typographic choice back to the subject matter instead of being
  an arbitrary "code font for numbers" default.
- Signature element: the day timeline (see `render_day_timeline` in
  trip_planner.py) styled as a transit-map line with circular "stops", and
  a passport-stamp badge on any day marked `locked` — a deliberate visual
  callback to the actual hash-verified locking guarantee described in
  src/refinement/locking.py, not just a decorative badge.
"""
from __future__ import annotations

import streamlit as st

COLORS = {
    "ink": "#16213A",          # deep navy — headers, primary text on light bg
    "ink_soft": "#3A4562",     # secondary text
    "stone": "#EEF1EC",        # page background — cool sage-stone, not cream
    "stone_dark": "#DDE3D8",   # card borders / dividers
    "paper": "#FFFFFF",        # card surfaces
    "teal": "#2F6E5E",         # primary accent — buttons, links, active states
    "teal_dark": "#204C40",
    "brass": "#C08A28",        # secondary accent — badges, highlights, stamps
    "brick": "#B3452F",        # reserved for warnings/critical validation issues ONLY
    "brick_soft": "#F5E4DE",
}

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700"
    "&family=Work+Sans:wght@400;500;600"
    "&family=JetBrains+Mono:wght@400;600&display=swap');"
)


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        {FONT_IMPORT}

        html, body, [class*="css"] {{
            font-family: 'Work Sans', sans-serif;
            color: {COLORS['ink']};
        }}

        .stApp {{
            background: {COLORS['stone']};
        }}

        h1, h2, h3 {{
            font-family: 'Fraunces', serif;
            color: {COLORS['ink']};
            font-weight: 600;
            letter-spacing: -0.01em;
        }}

        [data-testid="stSidebar"] {{
            background: {COLORS['ink']};
        }}
        [data-testid="stSidebar"] * {{
            color: {COLORS['stone']} !important;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            font-family: 'Work Sans', sans-serif;
        }}

        .stButton > button {{
            background: {COLORS['teal']};
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            padding: 0.5rem 1.25rem;
            transition: background 0.15s ease;
        }}
        .stButton > button:hover {{
            background: {COLORS['teal_dark']};
            color: white;
        }}

        .at-card {{
            background: {COLORS['paper']};
            border: 1px solid {COLORS['stone_dark']};
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
        }}

        .at-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}

        .at-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            letter-spacing: 0.03em;
        }}
        .at-badge-grounded {{ background: #E4EEE9; color: {COLORS['teal_dark']}; }}
        .at-badge-model {{ background: #F1EADB; color: {COLORS['brass']}; }}
        .at-badge-fallback {{ background: {COLORS['brick_soft']}; color: {COLORS['brick']}; }}

        .at-stamp {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 600;
            color: {COLORS['brass']};
            border: 1.5px solid {COLORS['brass']};
            border-radius: 999px;
            padding: 0.15rem 0.55rem;
            transform: rotate(-3deg);
        }}

        .at-ticket {{
            background: {COLORS['paper']};
            border: 1px solid {COLORS['stone_dark']};
            border-left: 5px solid {COLORS['teal']};
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.7rem;
            position: relative;
        }}
        .at-ticket-time {{
            font-family: 'JetBrains Mono', monospace;
            color: {COLORS['ink_soft']};
            font-size: 0.82rem;
        }}

        .at-divider {{
            border-top: 1px dashed {COLORS['stone_dark']};
            margin: 0.6rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_for_source(source: str) -> str:
    labels = {
        "knowledge_base": ("VERIFIED", "at-badge-grounded"),
        "model_knowledge": ("MODEL KNOWLEDGE", "at-badge-model"),
        "rule_based_fallback": ("OFFLINE FALLBACK", "at-badge-fallback"),
        "user_specified": ("YOUR REQUEST", "at-badge-grounded"),
    }
    text, css_class = labels.get(source, ("UNKNOWN", "at-badge-model"))
    return f'<span class="at-badge {css_class}">{text}</span>'
