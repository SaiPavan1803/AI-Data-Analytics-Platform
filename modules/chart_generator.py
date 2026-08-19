"""
chart_generator.py
------------------
Heuristic chart recommendation engine. Produces a list of Plotly figures
based on the columns present in the dataset. All charts use a premium
dark theme.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


MAX_CATEGORIES = 25
TOP_N_BAR = 10

# Premium dark theme defaults
_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
    title=dict(font=dict(size=15, color="#f1f5f9"), x=0.02),
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


def _style(fig: go.Figure) -> go.Figure:
    """Apply premium dark styling to a figure."""
    fig.update_layout(**_DARK_LAYOUT)
    return fig


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def recommend_charts(df: pd.DataFrame) -> List[Tuple[str, go.Figure]]:
    """Return a list of (title, plotly_figure) suggested for this dataset."""
    figs: List[Tuple[str, go.Figure]] = []
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not _is_datetime(df[c])
        and df[c].nunique(dropna=True) <= MAX_CATEGORIES
    ]
    datetime_cols = [c for c in df.columns if _is_datetime(df[c])]

    # 1. Histogram of the first numeric column
    if numeric:
        col = numeric[0]
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        figs.append((f"Distribution of {col}", _style(fig)))

    # 2. Bar chart: top-N of first categorical against first numeric
    if categorical and numeric:
        cat, num = categorical[0], numeric[0]
        agg = (
            df.groupby(cat, dropna=False)[num]
            .sum()
            .sort_values(ascending=False)
            .head(TOP_N_BAR)
            .reset_index()
        )
        fig = px.bar(
            agg,
            x=cat,
            y=num,
            title=f"{num} by {cat} (top {TOP_N_BAR})",
            color=num,
            color_continuous_scale=[[0, "#6366f1"], [1, "#8b5cf6"]],
        )
        fig.update_layout(coloraxis_showscale=False)
        figs.append((f"{num} by {cat}", _style(fig)))

    # 3. Time-series line chart
    if datetime_cols and numeric:
        d, num = datetime_cols[0], numeric[0]
        ts = df[[d, num]].dropna().sort_values(d)
        ts = ts.groupby(pd.Grouper(key=d, freq="D"))[num].sum().reset_index()
        fig = px.line(ts, x=d, y=num, title=f"{num} over time")
        fig.update_traces(line=dict(color="#6366f1", width=2.5))
        fig.update_traces(fill="tozeroy", fillcolor="rgba(99,102,241,0.08)")
        figs.append((f"{num} over time", _style(fig)))

    # 4. Pie chart of a second categorical, if available
    if len(categorical) >= 2:
        cat = categorical[1]
        vc = df[cat].value_counts().head(8).reset_index()
        vc.columns = [cat, "count"]
        fig = px.pie(vc, names=cat, values="count", title=f"Share of {cat}", hole=0.4)
        fig.update_traces(
            marker=dict(
                colors=[
                    "#6366f1",
                    "#8b5cf6",
                    "#06b6d4",
                    "#34d399",
                    "#f59e0b",
                    "#f87171",
                    "#ec4899",
                    "#a78bfa",
                ]
            ),
            textfont=dict(color="#e2e8f0"),
        )
        figs.append((f"Share of {cat}", _style(fig)))

    # 5. Correlation heatmap if >=2 numeric columns
    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(
            xaxis=dict(tickfont=dict(size=10, color="#94a3b8")),
            yaxis=dict(tickfont=dict(size=10, color="#94a3b8")),
        )
        figs.append(("Correlation Heatmap", _style(fig)))

    # 6. Scatter of two numeric columns
    if len(numeric) >= 2:
        x, y = numeric[0], numeric[1]
        color = categorical[0] if categorical else None
        fig = px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}", opacity=0.7)
        figs.append((f"{y} vs {x}", _style(fig)))

    return figs


def bar_top_n(df: pd.DataFrame, category: str, value: str, n: int = 10) -> go.Figure:
    agg = (
        df.groupby(category, dropna=False)[value]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    fig = px.bar(agg, x=category, y=value, title=f"Top {n} {category} by {value}")
    return _style(fig)


def line_over_time(
    df: pd.DataFrame, date_col: str, value: str, freq: str = "M"
) -> go.Figure:
    ts = (
        df[[date_col, value]]
        .dropna()
        .groupby(pd.Grouper(key=date_col, freq=freq))[value]
        .sum()
        .reset_index()
    )
    fig = px.line(ts, x=date_col, y=value, title=f"{value} over time ({freq})")
    fig.update_traces(line=dict(color="#6366f1", width=2.5))
    return _style(fig)