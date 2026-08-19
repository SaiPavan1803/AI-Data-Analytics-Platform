"""
app.py  –  AI Data Analyst Agent  –  Premium UI Redesign
========================================================
Streamlit dashboard with dark-mode glassmorphism design inspired by
ChatGPT / Perplexity / Notion AI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import (
    analyzer,
    anomaly_detector,
    chart_generator,
    chat_agent,
    data_loader,
    forecasting,
    insight_engine,
    report_generator,
)

# --------------------------------------------------------------------------- #
# Page config                                                                 #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Premium CSS                                                                 #
# --------------------------------------------------------------------------- #
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e4e7eb;
}
.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #0d1321 40%, #111827 100%);
}
[data-testid="stHeader"] {
    background: rgba(10, 14, 23, 0.8);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(13,19,33,0.97) 0%, rgba(17,24,39,0.99) 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(24px);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: #94a3b8;
    font-weight: 500;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    padding: 0.55rem 0.9rem;
    border-radius: 10px;
    margin-bottom: 2px;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    background: rgba(99, 102, 241, 0.08);
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label p {
    font-size: 0.88rem;
    font-weight: 500;
}

/* ── Typography ─────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #f1f5f9 !important;
}
h1 { font-size: 2rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.15rem !important; }
p, span, li, label { color: #cbd5e1; }

/* ── Glassmorphism cards ────────────────────────────────── */
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
}
.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.2);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    transform: translateY(-2px);
}

/* ── KPI Cards ──────────────────────────────────────────── */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1.2rem 0;
}
.kpi-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(30,41,59,0.4) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.25);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}
.kpi-card .kpi-icon {
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
    display: inline-block;
}
.kpi-card .kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.3rem;
}
.kpi-card .kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.2;
}
.kpi-card .kpi-trend {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.4rem;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 20px;
}
.kpi-card .trend-up {
    color: #34d399;
    background: rgba(52, 211, 153, 0.1);
}
.kpi-card .trend-down {
    color: #f87171;
    background: rgba(248, 113, 113, 0.1);
}
.kpi-card .trend-neutral {
    color: #fbbf24;
    background: rgba(251, 191, 36, 0.1);
}
.kpi-card.kpi-revenue::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-card.kpi-profit::before { background: linear-gradient(90deg, #34d399, #10b981); }
.kpi-card.kpi-growth::before { background: linear-gradient(90deg, #f59e0b, #f97316); }
.kpi-card.kpi-quality::before { background: linear-gradient(90deg, #06b6d4, #3b82f6); }

/* ── Insight cards ──────────────────────────────────────── */
.insight-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #6366f1;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.6rem 0;
    transition: all 0.25s ease;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #cbd5e1;
}
.insight-card:hover {
    background: rgba(30, 41, 59, 0.6);
    border-left-color: #818cf8;
    transform: translateX(4px);
}

/* ── Activity panel ─────────────────────────────────────── */
.activity-panel {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.activity-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    animation: fadeSlideIn 0.4s ease forwards;
    opacity: 0;
}
.activity-item:last-child { border-bottom: none; }
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
.activity-check {
    width: 24px; height: 24px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem;
    flex-shrink: 0;
}
.activity-check.done {
    background: rgba(52, 211, 153, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.activity-check.pending {
    background: rgba(251, 191, 36, 0.1);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.2);
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.5; }
}
.activity-text {
    font-size: 0.88rem;
    font-weight: 500;
    color: #e2e8f0;
}
.activity-time {
    margin-left: auto;
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 500;
}

/* ── Chat styling ───────────────────────────────────────── */
.chat-container {
    max-width: 900px;
    margin: 0 auto;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: #fff;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25);
    float: right;
    clear: both;
}
.chat-bubble-assistant {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e2e8f0;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 0;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.6;
    float: left;
    clear: both;
}
.chat-suggestion {
    display: inline-block;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
    padding: 0.45rem 1rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.3rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.chat-suggestion:hover {
    background: rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-1px);
}

/* ── Typing indicator ───────────────────────────────────── */
.typing-indicator {
    display: flex;
    gap: 4px;
    padding: 0.8rem 1.2rem;
    align-items: center;
}
.typing-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6366f1;
    animation: typingBounce 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

/* ── Quality gauge ──────────────────────────────────────── */
.quality-gauge {
    text-align: center;
    padding: 1.5rem;
}
.quality-score {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.quality-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
}
.score-excellent { color: #34d399; }
.score-good { color: #6366f1; }
.score-fair { color: #fbbf24; }
.score-poor { color: #f87171; }

/* ── Section headers ────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.section-header .section-icon {
    width: 40px; height: 40px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(99,102,241,0.15);
}
.section-header h2 {
    margin: 0 !important;
    font-size: 1.4rem !important;
}
.section-header .section-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 2px;
}

/* ── Executive summary panel ────────────────────────────── */
.exec-summary {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.05) 100%);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.exec-summary::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.exec-summary h3 {
    color: #a5b4fc !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem !important;
}
.exec-summary p, .exec-summary li {
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* ── Upload zone ────────────────────────────────────────── */
.upload-zone {
    background: rgba(30, 41, 59, 0.3);
    border: 2px dashed rgba(99, 102, 241, 0.25);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    margin: 1rem 0;
}
.upload-zone:hover {
    border-color: rgba(99, 102, 241, 0.5);
    background: rgba(30, 41, 59, 0.5);
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
    background: linear-gradient(135deg, #5b52f0, #7577f5) !important;
}

/* ── Dataframe ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ── Metric cards (Streamlit native) ────────────────────── */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}

/* ── Expander ───────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(30, 41, 59, 0.3) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}

/* ── Spinner ────────────────────────────────────────────── */
.stSpinner > div {
    color: #6366f1 !important;
}

/* ── Tabs ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 20px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #a5b4fc !important;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.5); }

/* ── Animations ─────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeInUp 0.5s ease forwards;
}

/* ── Hide Streamlit branding ────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; height: 0; }
.stDeployButton { display: none !important; }

/* ── Responsive ─────────────────────────────────────────── */
@media (max-width: 768px) {
    .kpi-container { grid-template-columns: repeat(2, 1fr); }
    .kpi-card .kpi-value { font-size: 1.3rem; }
}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Plotly dark theme helper                                                    #
# --------------------------------------------------------------------------- #
PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
    title=dict(font=dict(size=16, color="#f1f5f9", weight='bold'), x=0.02),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11, color="#94a3b8"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11, color="#94a3b8"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#94a3b8"),
        bordercolor="rgba(255,255,255,0.06)",
        borderwidth=1,
    ),
    margin=dict(l=50, r=30, t=50, b=50),
    colorway=[
        "#6366f1",
        "#8b5cf6",
        "#06b6d4",
        "#34d399",
        "#f59e0b",
        "#f87171",
        "#ec4899",
    ],
    hoverlabel=dict(
        bgcolor="rgba(30,41,59,0.95)",
        bordercolor="rgba(99,102,241,0.3)",
        font=dict(family="Inter", size=12, color="#e2e8f0"),
    ),
)

COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "success": "#34d399",
    "warning": "#f59e0b",
    "danger": "#f87171",
    "pink": "#ec4899",
}


def style_fig(fig: go.Figure) -> go.Figure:
    """Apply the premium dark theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# --------------------------------------------------------------------------- #
# Session state                                                               #
# --------------------------------------------------------------------------- #
def _init_state():
    defaults = {
        "df": None,
        "dataset_name": None,
        "overview": None,
        "explanation": None,
        "insights": None,
        "figures": None,
        "anomalies": None,
        "anomaly_notes": None,
        "forecast": None,
        "chat_history": [],
        "model_name": analyzer.DEFAULT_MODEL,
        "activity_log": [],
        "dashboard_loaded": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()

# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:1.2rem 0 0.6rem;">'
        '<div style="font-size:2rem;margin-bottom:0.3rem;">⚡</div>'
        '<h2 style="margin:0;font-size:1.25rem;background:linear-gradient(135deg,#a5b4fc,#818cf8);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        "AI Data Analyst</h2>"
        '<p style="color:#64748b;font-size:0.78rem;margin:0.2rem 0 0;">'
        "Powered by Ollama · Local LLM</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.session_state.model_name = st.text_input(
        "Model",
        value=st.session_state.model_name,
        help="Examples: llama3, qwen3, mistral. Pull with `ollama pull <name>`.",
    )

    st.markdown(
        '<p style="color:#475569;font-size:0.7rem;text-transform:uppercase;'
        'letter-spacing:0.1em;font-weight:600;margin:0.8rem 0 0.4rem;">Navigation</p>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Go to",
        [
            "📤  Upload Dataset",
            "📊  Dashboard",
            "💡  AI Insights",
            "💬  Chat with Data",
            "🔍  Data Quality",
            "🚨  Anomaly Detection",
            "📈  Forecasting",
            "📄  Report Generation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0;">'
        '<p style="color:#475569;font-size:0.72rem;">'
        "M.Tech AI Project · All inference runs locally</p></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _require_df():
    if st.session_state.df is None:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:3rem;">'
            '<div style="font-size:3rem;margin-bottom:1rem;">📤</div>'
            '<h3 style="margin-bottom:0.5rem;">No dataset loaded</h3>'
            '<p style="color:#94a3b8;">Upload a CSV from the <b>Upload Dataset</b> page to get started.</p>'
            "</div>",
            unsafe_allow_html=True,
        )
        return False
    return True


def _section_header(icon: str, title: str, subtitle: str = ""):
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-header">'
        f'<div class="section-icon">{icon}</div>'
        f"<div><h2>{title}</h2>{sub_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _kpi_html(
    label: str,
    value: str,
    trend: str = "",
    trend_dir: str = "up",
    css_class: str = "kpi-revenue",
):
    trend_html = ""
    if trend:
        arrow = "↑" if trend_dir == "up" else ("↓" if trend_dir == "down" else "→")
        trend_html = f'<div class="kpi-trend trend-{trend_dir}">{arrow} {trend}</div>'
    return (
        f'<div class="kpi-card {css_class}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{trend_html}"
        f"</div>"
    )


def _render_kpi_row(kpis: list):
    """Render a row of KPI cards. kpis = [(label, value, trend, trend_dir, css_class), ...]"""
    cols = st.columns(len(kpis))
    for col, (label, value, trend, tdir, cls) in zip(cols, kpis):
        col.markdown(_kpi_html(label, value, trend, tdir, cls), unsafe_allow_html=True)


def _sparkline_fig(values: list, color: str = "#6366f1") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            y=values,
            mode="lines",
            line=dict(color=color, width=2, shape="spline"),
            fill="tozeroy",
            fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
        )
    )
    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _add_activity(text: str, status: str = "done"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.append({"text": text, "status": status, "time": ts})


def _render_activity_panel():
    if not st.session_state.activity_log:
        return
    items_html = ""
    for i, item in enumerate(st.session_state.activity_log):
        check_cls = "done" if item["status"] == "done" else "pending"
        check_icon = "✓" if item["status"] == "done" else "⟳"
        delay = i * 0.1
        items_html += (
            f'<div class="activity-item" style="animation-delay:{delay}s">'
            f'<div class="activity-check {check_cls}">{check_icon}</div>'
            f'<span class="activity-text">{item["text"]}</span>'
            f'<span class="activity-time">{item["time"]}</span>'
            f"</div>"
        )
    st.markdown(
        f'<div class="activity-panel">'
        f'<div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:#64748b;margin-bottom:0.8rem;">'
        f"⚡ Agent Activity</div>"
        f"{items_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Page: Upload                                                                #
# --------------------------------------------------------------------------- #
def page_upload():
    _section_header("📤", "Upload Dataset", "Load a CSV to begin analysis")

    st.markdown(
        '<div class="upload-zone">',
        unsafe_allow_html=True,
    )
    col_l, col_r = st.columns(2)
    with col_l:
        file = st.file_uploader("Choose a CSV file", type=["csv"])
    with col_r:
        use_sample = st.checkbox("Use bundled sample sales dataset")
    st.markdown("</div>", unsafe_allow_html=True)

    loaded = False
    if use_sample:
        sample_path = os.path.join("data", "sample_sales.csv")
        if os.path.exists(sample_path):
            df = data_loader.load_csv(sample_path)
            st.session_state.df = df
            st.session_state.dataset_name = "sample_sales.csv"
            loaded = True
        else:
            st.error(f"Sample file not found at {sample_path}")
    elif file is not None:
        df = data_loader.load_csv(file)
        st.session_state.df = df
        st.session_state.dataset_name = file.name
        loaded = True

    if loaded:
        df = st.session_state.df
        st.session_state.overview = data_loader.basic_overview(df)
        # reset downstream
        for key in (
            "explanation",
            "insights",
            "figures",
            "anomalies",
            "anomaly_notes",
            "forecast",
            "activity_log",
            "dashboard_loaded",
        ):
            st.session_state[key] = None if key != "activity_log" else []
            if key == "dashboard_loaded":
                st.session_state[key] = False

        _add_activity("Dataset Loaded", "done")
        _add_activity("Schema Analyzed", "done")
        _add_activity("Data Quality Checked", "done")

        ov = st.session_state.overview
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        _render_kpi_row(
            [
                ("Rows", f"{ov['rows']:,}", "", "neutral", "kpi-revenue"),
                ("Columns", str(ov["cols"]), "", "neutral", "kpi-profit"),
                (
                    "Numeric Cols",
                    str(len(ov["numeric_columns"])),
                    "",
                    "neutral",
                    "kpi-growth",
                ),
                (
                    "Date Cols",
                    str(len(ov["datetime_columns"])),
                    "",
                    "neutral",
                    "kpi-quality",
                ),
            ]
        )
        st.markdown("</div>", unsafe_allow_html=True)

        _render_activity_panel()

        with st.expander("Data Preview", expanded=True):
            st.dataframe(data_loader.preview(df, 10), use_container_width=True)

        with st.expander("Column Types"):
            st.json(
                {
                    "numeric": ov["numeric_columns"],
                    "categorical": ov["categorical_columns"],
                    "datetime": ov["datetime_columns"],
                }
            )


# --------------------------------------------------------------------------- #
# Page: Dashboard (KPIs + AI Summary + Charts)                               #
# --------------------------------------------------------------------------- #
def page_dashboard():
    _section_header("📊", "Dashboard", "Overview of your dataset")
    if not _require_df():
        return
    df = st.session_state.df
    ov = st.session_state.overview

    # ── Compute KPIs from data ────────────────────────────
    numeric = df.select_dtypes(include="number")
    rev_col = next((c for c in numeric.columns if "revenue" in c.lower()), None)
    profit_col = next((c for c in numeric.columns if "profit" in c.lower()), None)
    date_col = next(
        (c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None
    )

    total_revenue = df[rev_col].sum() if rev_col else 0
    total_profit = df[profit_col].sum() if profit_col else 0
    profit_margin = (total_profit / total_revenue * 100) if total_revenue else 0

    # Growth: compare last 25% to first 25% of data
    growth_pct = 0.0
    if rev_col and date_col:
        ts = df.sort_values(date_col)
        n = len(ts)
        q1 = ts[rev_col].iloc[: n // 4].mean()
        q4 = ts[rev_col].iloc[-(n // 4) :].mean()
        if q1 > 0:
            growth_pct = ((q4 - q1) / q1) * 100

    # Data quality score
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    dupes = df.duplicated().sum()
    quality_score = max(
        0, round((1 - (missing_cells + dupes) / max(total_cells, 1)) * 100, 1)
    )

    # Sparkline data
    rev_spark = []
    prof_spark = []
    if rev_col and date_col:
        ts = df.sort_values(date_col)
        rev_spark = (
            ts.groupby(pd.Grouper(key=date_col, freq="M"))[rev_col].sum().tolist()
        )
    if profit_col and date_col:
        ts = df.sort_values(date_col)
        prof_spark = (
            ts.groupby(pd.Grouper(key=date_col, freq="M"))[profit_col].sum().tolist()
        )

    # ── KPI row ───────────────────────────────────────────
    st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
    _render_kpi_row(
        [
            (
                "Total Revenue",
                f"${total_revenue:,.0f}",
                f"{growth_pct:+.1f}%",
                "up" if growth_pct >= 0 else "down",
                "kpi-revenue",
            ),
            (
                "Total Profit",
                f"${total_profit:,.0f}",
                f"{profit_margin:.1f}% margin",
                "up" if profit_margin > 10 else "down",
                "kpi-profit",
            ),
            (
                "Growth",
                f"{growth_pct:+.1f}%",
                "trending up" if growth_pct >= 0 else "trending down",
                "up" if growth_pct >= 0 else "down",
                "kpi-growth",
            ),
            (
                "Data Quality",
                f"{quality_score}%",
                "excellent" if quality_score > 95 else "needs review",
                "up" if quality_score > 90 else "down",
                "kpi-quality",
            ),
        ]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sparklines row ────────────────────────────────────
    if rev_spark or prof_spark:
        sc1, sc2 = st.columns(2)
        if rev_spark:
            with sc1:
                st.markdown(
                    '<div style="color:#94a3b8;font-size:0.75rem;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">'
                    "Revenue Trend</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _sparkline_fig(rev_spark, COLORS["primary"]),
                    use_container_width=True,
                )
        if prof_spark:
            with sc2:
                st.markdown(
                    '<div style="color:#94a3b8;font-size:0.75rem;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem;">'
                    "Profit Trend</div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _sparkline_fig(prof_spark, COLORS["success"]),
                    use_container_width=True,
                )

    # ── AI Executive Summary ──────────────────────────────
    st.markdown("")
    if st.session_state.insights is None:
        st.session_state.insights = insight_engine.generate_insights(df)
        _add_activity("Insights Generated", "done")

    with st.container():
        st.markdown(
            '<div class="exec-summary"><h3>⚡ AI Executive Summary</h3></div>',
            unsafe_allow_html=True,
        )
        for ins in st.session_state.insights:
            st.markdown(
                f'<div class="insight-card">{ins}</div>',
                unsafe_allow_html=True,
            )

    if st.button("Generate AI Narration", key="dash_narrate"):
        with st.spinner("Composing executive summary…"):
            narration = insight_engine.narrate_insights(
                df,
                st.session_state.insights,
                model=st.session_state.model_name,
            )
        st.markdown(
            f'<div class="exec-summary"><h3>🤖 AI Narration</h3>'
            f"<p>{narration}</p></div>",
            unsafe_allow_html=True,
        )

    # ── Interactive Charts ────────────────────────────────
    st.markdown("")
    st.markdown(
        '<div style="font-size:0.78rem;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.08em;color:#64748b;margin:1.5rem 0 0.8rem;">'
        "📊 Interactive Visual Analytics</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.figures is None:
        st.session_state.figures = chart_generator.recommend_charts(df)
        _add_activity("Charts Created", "done")

    chart_cols = 2
    for i in range(0, len(st.session_state.figures), chart_cols):
        cols = st.columns(chart_cols)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(st.session_state.figures):
                title, fig = st.session_state.figures[idx]
                with col:
                    st.markdown(
                        f'<div class="glass-card" style="padding:0.8rem;">'
                        f'<div style="font-size:0.82rem;font-weight:600;color:#a5b4fc;'
                        f'margin-bottom:0.5rem;">{title}</div></div>',
                        unsafe_allow_html=True,
                    )
                    style_fig(fig)
                    st.plotly_chart(fig, use_container_width=True)

    _add_activity("Dashboard Rendered", "done")
    _render_activity_panel()


# --------------------------------------------------------------------------- #
# Page: AI Insights                                                           #
# --------------------------------------------------------------------------- #
def page_insights():
    _section_header("💡", "AI Insights", "Automated business intelligence")
    if not _require_df():
        return
    df = st.session_state.df

    if st.session_state.insights is None:
        st.session_state.insights = insight_engine.generate_insights(df)

    # Insight icons
    icons = ["📈", "📉", "🔗", "⚠️", "📊", "💰", "🎯", "🔍"]
    for i, ins in enumerate(st.session_state.insights):
        icon = icons[i % len(icons)]
        st.markdown(
            f'<div class="insight-card" style="border-left-color:{COLORS["primary"]};">'
            f'<span style="margin-right:8px;">{icon}</span>{ins}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    if st.button("Narrate Insights with Ollama", key="insight_narrate"):
        with st.spinner("Composing executive summary…"):
            narration = insight_engine.narrate_insights(
                df,
                st.session_state.insights,
                model=st.session_state.model_name,
            )
        st.markdown(
            f'<div class="exec-summary"><h3>🤖 Executive Summary</h3>'
            f"<p>{narration}</p></div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Page: Chat with Data                                                        #
# --------------------------------------------------------------------------- #
def page_chat():
    _section_header("💬", "Chat with Data", "Ask anything about your dataset")
    if not _require_df():
        return
    df = st.session_state.df

    # ── Suggested prompts ─────────────────────────────────
    numeric = df.select_dtypes(include="number").columns.tolist()
    cats = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    suggestions = [
        f"Which {cats[0]} performs best?" if cats else "Show top performers",
        "Why did profit drop?",
        "Forecast next quarter revenue.",
        "Find anomalies in the data.",
    ]
    if cats and numeric:
        suggestions.append(f"Top 5 {cats[0]} by {numeric[0]}")
    if len(numeric) >= 2:
        suggestions.append(f"Compare {numeric[0]} and {numeric[1]}")

    st.markdown(
        '<div style="margin-bottom:1rem;">'
        '<span style="color:#64748b;font-size:0.78rem;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.08em;">Suggested</span></div>',
        unsafe_allow_html=True,
    )
    sug_cols = st.columns(min(len(suggestions), 3))
    for i, sug in enumerate(suggestions):
        with sug_cols[i % len(sug_cols)]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state._chat_prompt = sug

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("")

    # ── Chat history ──────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{msg["content"]}</div>'
                '<div style="clear:both;"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-assistant">{msg["content"]}</div>'
                '<div style="clear:both;"></div>',
                unsafe_allow_html=True,
            )
            if msg.get("table") is not None:
                st.dataframe(msg["table"], use_container_width=True)
            if msg.get("figure") is not None:
                style_fig(msg["figure"])
                st.plotly_chart(msg["figure"], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────
    prompt = st.chat_input("Ask anything about your data…")
    if prompt is None and hasattr(st.session_state, "_chat_prompt"):
        prompt = st.session_state._chat_prompt
        del st.session_state._chat_prompt

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # Typing indicator
            with st.spinner("Analysing…"):
                res = chat_agent.answer(
                    df,
                    prompt,
                    model=st.session_state.model_name,
                )
            st.markdown(res.text)
            if res.table is not None:
                st.dataframe(res.table, use_container_width=True)
            if res.figure is not None:
                style_fig(res.figure)
                st.plotly_chart(res.figure, use_container_width=True)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": res.text,
                "table": res.table,
                "figure": res.figure,
            }
        )
        st.rerun()


# --------------------------------------------------------------------------- #
# Page: Data Quality Intelligence                                             #
# --------------------------------------------------------------------------- #
def page_data_quality():
    _section_header(
        "🔍", "Data Quality Intelligence", "Automated data health assessment"
    )
    if not _require_df():
        return
    df = st.session_state.df
    ov = st.session_state.overview

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())
    quality_score = max(
        0, round((1 - (missing_cells + dupes) / max(total_cells, 1)) * 100, 1)
    )

    # Quality score gauge
    score_class = (
        "score-excellent"
        if quality_score > 95
        else "score-good"
        if quality_score > 85
        else "score-fair"
        if quality_score > 70
        else "score-poor"
    )
    score_label = (
        "Excellent"
        if quality_score > 95
        else "Good"
        if quality_score > 85
        else "Fair"
        if quality_score > 70
        else "Poor"
    )

    gauge_col, info_col = st.columns([1, 2])

    with gauge_col:
        # Gauge chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                number={
                    "suffix": "%",
                    "font": {"size": 40, "color": "#f1f5f9", "family": "Inter"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#64748b",
                        "tickfont": {"color": "#94a3b8"},
                    },
                    "bar": {"color": COLORS["primary"]},
                    "bgcolor": "rgba(15,23,42,0.6)",
                    "bordercolor": "rgba(255,255,255,0.06)",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(248,113,113,0.15)"},
                        {"range": [50, 70], "color": "rgba(251,191,36,0.15)"},
                        {"range": [70, 85], "color": "rgba(99,102,241,0.15)"},
                        {"range": [85, 100], "color": "rgba(52,211,153,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": COLORS["primary"], "width": 3},
                        "thickness": 0.8,
                        "value": quality_score,
                    },
                },
                title={
                    "text": f"Data Quality: {score_label}",
                    "font": {"color": "#94a3b8", "size": 14, "family": "Inter"},
                },
            )
        )
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with info_col:
        st.markdown(
            '<div class="kpi-container" style="grid-template-columns:repeat(2,1fr);">',
            unsafe_allow_html=True,
        )
        _render_kpi_row(
            [
                (
                    "Missing Values",
                    f"{missing_cells:,}",
                    f"{missing_cells / max(total_cells, 1) * 100:.1f}% of cells",
                    "down" if missing_cells > 0 else "up",
                    "kpi-revenue",
                ),
                (
                    "Duplicate Rows",
                    f"{dupes:,}",
                    f"{dupes / max(df.shape[0], 1) * 100:.1f}% of rows",
                    "down" if dupes > 0 else "up",
                    "kpi-profit",
                ),
            ]
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Missing values breakdown
        missing_per_col = df.isna().sum()
        missing_per_col = missing_per_col[missing_per_col > 0].sort_values(
            ascending=False
        )
        if not missing_per_col.empty:
            st.markdown(
                '<div style="color:#94a3b8;font-size:0.78rem;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.5rem;">'
                "Missing Values by Column</div>",
                unsafe_allow_html=True,
            )
            fig_miss = go.Figure(
                go.Bar(
                    x=missing_per_col.values,
                    y=missing_per_col.index.tolist(),
                    orientation="h",
                    marker=dict(
                        color=missing_per_col.values,
                        colorscale=[[0, COLORS["primary"]], [1, COLORS["danger"]]],
                        showscale=False,
                    ),
                    text=missing_per_col.values,
                    textposition="outside",
                    textfont=dict(color="#94a3b8"),
                )
            )
            fig_miss.update_layout(
                height=max(200, len(missing_per_col) * 40),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(title="Missing Count"),
                margin=dict(l=120, r=40, t=10, b=30),
            )
            style_fig(fig_miss)
            st.plotly_chart(fig_miss, use_container_width=True)

        # Outlier count
        num_cols = df.select_dtypes(include="number").columns.tolist()
        outlier_count = 0
        if num_cols:
            for c in num_cols[:5]:
                s = df[c].dropna()
                if len(s) > 10:
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    outlier_count += int(
                        ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                    )

        st.markdown("")
        _render_kpi_row(
            [
                (
                    "Outliers Detected",
                    f"{outlier_count:,}",
                    "across numeric columns",
                    "neutral",
                    "kpi-growth",
                ),
                (
                    "Total Rows",
                    f"{df.shape[0]:,}",
                    f"{df.shape[1]} columns",
                    "neutral",
                    "kpi-quality",
                ),
            ]
        )

    # Recommendations
    st.markdown("")
    st.markdown(
        '<div style="color:#94a3b8;font-size:0.78rem;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.08em;margin:0.5rem 0;">'
        "🔧 Recommended Fixes</div>",
        unsafe_allow_html=True,
    )
    recs = []
    if missing_cells > 0:
        cols_with_missing = (
            missing_per_col.index.tolist()[:3] if not missing_per_col.empty else []
        )
        recs.append(f"Fill or impute missing values in: {', '.join(cols_with_missing)}")
    if dupes > 0:
        recs.append(f"Remove {dupes} duplicate rows to improve data integrity.")
    if outlier_count > 0:
        recs.append(f"Review {outlier_count} potential outliers using IQR method.")
    if quality_score < 90:
        recs.append(
            "Overall data quality needs attention — consider data cleaning pipeline."
        )
    if not recs:
        recs.append("Data quality is excellent. No immediate fixes needed.")

    for rec in recs:
        st.markdown(
            f'<div class="insight-card" style="border-left-color:{COLORS["accent"]};">'
            f"🔧 {rec}</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Page: Anomaly Detection                                                     #
# --------------------------------------------------------------------------- #
def page_anomalies():
    _section_header(
        "🚨", "Anomaly Detection", "IsolationForest-based outlier identification"
    )
    if not _require_df():
        return
    df = st.session_state.df

    st.markdown(
        '<div class="glass-card" style="margin-bottom:1.2rem;">',
        unsafe_allow_html=True,
    )
    contam = st.slider(
        "Contamination (expected % outliers)",
        0.01,
        0.20,
        0.03,
        0.01,
    )
    if st.button("Detect Anomalies"):
        with st.spinner("Running IsolationForest…"):
            anomalies = anomaly_detector.detect_anomalies(df, contamination=contam)
            notes = anomaly_detector.explain_anomalies(df, anomalies)
        st.session_state.anomalies = anomalies
        st.session_state.anomaly_notes = notes
        _add_activity("Outliers Detected", "done")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.anomalies is not None:
        if st.session_state.anomalies.empty:
            st.markdown(
                '<div class="glass-card" style="text-align:center;padding:2rem;">'
                '<div style="font-size:2.5rem;margin-bottom:0.5rem;">✅</div>'
                "<h3>No anomalies detected</h3>"
                '<p style="color:#94a3b8;">Your data looks clean with the current settings.</p>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            n_anomalies = len(st.session_state.anomalies)
            _render_kpi_row(
                [
                    (
                        "Anomalies Found",
                        str(n_anomalies),
                        f"{contam * 100:.0f}% contamination",
                        "neutral",
                        "kpi-revenue",
                    ),
                    (
                        "Avg Anomaly Score",
                        f"{st.session_state.anomalies['anomaly_score'].mean():.3f}",
                        "IsolationForest",
                        "neutral",
                        "kpi-growth",
                    ),
                ]
            )

            st.markdown("")
            st.markdown(
                '<div style="color:#94a3b8;font-size:0.78rem;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.5rem;">'
                "Top Anomalous Rows</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(st.session_state.anomalies, use_container_width=True)

            # Anomaly score distribution
            num = df.select_dtypes(include="number").dropna(axis=1, how="all")
            if not num.empty:
                from sklearn.ensemble import IsolationForest

                model = IsolationForest(
                    n_estimators=200, contamination=contam, random_state=42
                )
                scores = -model.fit_predict(
                    num.iloc[:, :8].fillna(num.median(numeric_only=True))
                )
                score_dist = num.iloc[:, :8].apply(lambda c: c.abs().mean())

                fig_dist = go.Figure(
                    go.Histogram(
                        x=[
                            -s
                            for s in model.score_samples(
                                num.iloc[:, :8].fillna(num.median(numeric_only=True))
                            )
                        ],
                        nbinsx=30,
                        marker=dict(color=COLORS["primary"], opacity=0.7),
                    )
                )
                fig_dist.update_layout(
                    title="Anomaly Score Distribution",
                    xaxis_title="Anomaly Score",
                    yaxis_title="Count",
                )
                style_fig(fig_dist)
                st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown(
                '<div style="color:#94a3b8;font-size:0.78rem;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.5rem;">'
                "Explanations</div>",
                unsafe_allow_html=True,
            )
            for n in st.session_state.anomaly_notes:
                st.markdown(
                    f'<div class="insight-card" style="border-left-color:{COLORS["danger"]};">'
                    f"⚠️ {n}</div>",
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------- #
# Page: Forecasting                                                           #
# --------------------------------------------------------------------------- #
def page_forecast():
    _section_header("📈", "Forecasting Dashboard", "ML-powered time-series predictions")
    if not _require_df():
        return
    df = st.session_state.df
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not date_cols or not num_cols:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:2rem;">'
            '<div style="font-size:2.5rem;margin-bottom:0.5rem;">⚠️</div>'
            '<p style="color:#94a3b8;">Need at least one date column and one numeric column.</p>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="glass-card" style="margin-bottom:1.2rem;">', unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox("Date column", date_cols)
    value_col = c2.selectbox("Value column", num_cols)
    freq = c3.selectbox("Aggregation", ["D", "W", "M", "Q"], index=2)
    periods = c4.number_input("Periods ahead", 1, 36, 6)
    model_name = st.radio("Model", ["RandomForest", "Linear"], horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Forecast"):
        try:
            with st.spinner("Training model…"):
                res = forecasting.forecast(
                    df,
                    date_col,
                    value_col,
                    periods=int(periods),
                    freq=freq,
                    model_name=model_name,
                )
            st.session_state.forecast = res
            _add_activity("Forecast Generated", "done")
        except Exception as e:
            st.error(str(e))

    res = st.session_state.forecast
    if res is not None:
        # KPI row
        _render_kpi_row(
            [
                (
                    f"{res.model_name} MAE",
                    f"{res.metric_mae:,.2f}",
                    "lower is better",
                    "neutral",
                    "kpi-revenue",
                ),
                (
                    "R² (test)",
                    f"{res.metric_r2:.3f}",
                    "higher is better",
                    "up" if res.metric_r2 > 0.5 else "down",
                    "kpi-profit",
                ),
                (
                    "Forecast Periods",
                    str(len(res.forecast)),
                    f"freq: {freq}",
                    "neutral",
                    "kpi-growth",
                ),
                (
                    "Trend",
                    "Upward"
                    if res.forecast["value"].iloc[-1] > res.history["value"].iloc[-1]
                    else "Downward",
                    "projected",
                    "neutral",
                    "kpi-quality",
                ),
            ]
        )

        # Forecast chart
        st.markdown("")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=res.history["date"],
                y=res.history["value"],
                name="History",
                mode="lines",
                line=dict(color=COLORS["primary"], width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=res.forecast["date"],
                y=res.forecast["value"],
                name="Forecast",
                mode="lines+markers",
                line=dict(color=COLORS["success"], width=2.5, dash="dash"),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(res.forecast["date"]) + list(res.forecast["date"])[::-1],
                y=list(res.forecast["upper"]) + list(res.forecast["lower"])[::-1],
                fill="toself",
                fillcolor="rgba(99,102,241,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% Confidence",
                showlegend=True,
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            title=f"Forecast: {value_col}",
            xaxis_title="Date",
            yaxis_title=value_col,
            height=450,
        )
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        # AI summary
        last_hist = res.history["value"].iloc[-1]
        last_fc = res.forecast["value"].iloc[-1]
        fc_change = ((last_fc - last_hist) / max(abs(last_hist), 1e-9)) * 100
        direction = "increase" if fc_change >= 0 else "decrease"

        st.markdown(
            f'<div class="exec-summary">'
            f"<h3>🤖 AI Forecast Summary</h3>"
            f"<p>The {res.model_name} model projects {value_col} to "
            f"<b>{direction}</b> by <b>{abs(fc_change):.1f}%</b> over the next "
            f"{len(res.forecast)} periods. "
            f"MAE on the test set is <b>{res.metric_mae:,.2f}</b> with "
            f"R² = <b>{res.metric_r2:.3f}</b>. "
            f"{'The model shows good predictive power.' if res.metric_r2 > 0.5 else 'Consider collecting more data for improved accuracy.'}"
            f"</p></div>",
            unsafe_allow_html=True,
        )

        with st.expander("Forecast Table"):
            st.dataframe(res.forecast, use_container_width=True)


# --------------------------------------------------------------------------- #
# Page: Report Generation                                                     #
# --------------------------------------------------------------------------- #
def page_report():
    _section_header("📄", "Report Generation", "Compile a professional PDF report")
    if not _require_df():
        return
    df = st.session_state.df
    ov = st.session_state.overview

    st.markdown(
        '<div class="glass-card" style="margin-bottom:1.2rem;">'
        '<p style="color:#cbd5e1;">This will compile a PDF report from all analysis '
        "produced across the dashboard sections.</p>",
        unsafe_allow_html=True,
    )

    recs = st.text_area(
        "Recommendations (one per line)",
        "Focus marketing spend on the top-performing category.\n"
        "Investigate flagged anomalies for data-quality issues.\n"
        "Re-train forecasts monthly as new data lands.",
    ).splitlines()
    st.markdown("</div>", unsafe_allow_html=True)

    # Report sections preview
    sections = [
        ("📊", "Dataset Overview", f"{ov['rows']:,} rows × {ov['cols']} columns"),
        (
            "💡",
            "AI Insights",
            f"{len(st.session_state.insights or [])} insights generated",
        ),
        ("📈", "Charts", f"{len(st.session_state.figures or [])} visualizations"),
        (
            "🚨",
            "Anomalies",
            f"{len(st.session_state.anomalies) if st.session_state.anomalies is not None else 0} detected",
        ),
        (
            "📈",
            "Forecast",
            "Included" if st.session_state.forecast else "Not generated",
        ),
    ]
    st.markdown(
        '<div style="color:#94a3b8;font-size:0.78rem;font-weight:600;'
        'text-transform:uppercase;letter-spacing:0.08em;margin:1rem 0 0.5rem;">'
        "Report Sections</div>",
        unsafe_allow_html=True,
    )
    for icon, title, detail in sections:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;padding:0.6rem 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="font-size:1.2rem;">{icon}</span>'
            f'<span style="font-weight:600;color:#e2e8f0;flex:1;">{title}</span>'
            f'<span style="color:#64748b;font-size:0.82rem;">{detail}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    if st.button("Generate PDF Report"):
        explanation = (
            st.session_state.explanation or "Auto-generated explanation not requested."
        )
        insights = st.session_state.insights or insight_engine.generate_insights(df)
        figures = st.session_state.figures or chart_generator.recommend_charts(df)
        anomalies = st.session_state.anomalies
        anomaly_notes = st.session_state.anomaly_notes
        fc = st.session_state.forecast

        fc_table, fc_fig = None, None
        if fc is not None:
            fc_table = fc.forecast.copy()
            fc_fig = go.Figure()
            fc_fig.add_trace(
                go.Scatter(x=fc.history["date"], y=fc.history["value"], name="History")
            )
            fc_fig.add_trace(
                go.Scatter(
                    x=fc.forecast["date"], y=fc.forecast["value"], name="Forecast"
                )
            )

        os.makedirs("reports", exist_ok=True)
        path = os.path.join(
            "reports",
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        )
        with st.spinner("Building PDF…"):
            report_generator.build_report(
                output_path=path,
                dataset_name=st.session_state.dataset_name or "dataset.csv",
                overview=ov,
                explanation=explanation,
                insights=insights,
                figures=figures,
                anomaly_table=anomalies,
                anomaly_notes=anomaly_notes,
                forecast_table=fc_table,
                forecast_figure=fc_fig,
                recommendations=[r for r in recs if r.strip()],
            )
        with open(path, "rb") as f:
            st.markdown(
                '<div class="glass-card" style="text-align:center;padding:2rem;">'
                '<div style="font-size:2.5rem;margin-bottom:0.5rem;">✅</div>'
                "<h3>Report Generated Successfully</h3>"
                '<p style="color:#94a3b8;">Your PDF report is ready for download.</p>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇  Download Report",
                f,
                file_name=os.path.basename(path),
                mime="application/pdf",
            )
            _add_activity("Report Generated", "done")


# --------------------------------------------------------------------------- #
# Router                                                                      #
# --------------------------------------------------------------------------- #
ROUTES = {
    "📤  Upload Dataset": page_upload,
    "📊  Dashboard": page_dashboard,
    "💡  AI Insights": page_insights,
    "💬  Chat with Data": page_chat,
    "🔍  Data Quality": page_data_quality,
    "🚨  Anomaly Detection": page_anomalies,
    "📈  Forecasting": page_forecast,
    "📄  Report Generation": page_report,
}
ROUTES[page]()
