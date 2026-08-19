"""
chat_agent.py
-------------
Conversational analytics. The agent first answers a small set of common
intents through pandas (deterministic, fast). For everything else it asks
the local Ollama LLM, passing it a compact schema of the dataframe.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .analyzer import ask_ollama, DEFAULT_MODEL


@dataclass
class ChatResponse:
    text: str
    table: Optional[pd.DataFrame] = None
    figure: Optional[go.Figure] = None


# --------------------------------------------------------------------------- #
# Deterministic intent handlers                                               #
# --------------------------------------------------------------------------- #
def _numeric_cols(df):  return df.select_dtypes(include="number").columns.tolist()
def _cat_cols(df):
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]
def _date_cols(df):
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _find_column(df: pd.DataFrame, query: str) -> Optional[str]:
    q = query.lower()
    # exact (case-insensitive)
    for c in df.columns:
        if c.lower() == q:
            return c
    # substring
    for c in df.columns:
        if c.lower() in q or q in c.lower():
            return c
    return None


def _try_top_n(df: pd.DataFrame, query: str) -> Optional[ChatResponse]:
    m = re.search(r"top\s+(\d+)?\s*([\w\s]+?)(?:\s+by\s+([\w\s]+))?[\?\.]?$",
                  query.lower())
    if not m:
        return None
    n = int(m.group(1) or 10)
    cat_hint, val_hint = m.group(2).strip(), (m.group(3) or "").strip()
    cat_col = _find_column(df, cat_hint) or (_cat_cols(df)[0] if _cat_cols(df) else None)
    val_col = _find_column(df, val_hint) if val_hint else None
    if val_col is None and _numeric_cols(df):
        val_col = _numeric_cols(df)[0]
    if not (cat_col and val_col):
        return None
    agg = (df.groupby(cat_col)[val_col].sum()
             .sort_values(ascending=False).head(n).reset_index())
    fig = px.bar(agg, x=cat_col, y=val_col,
                 title=f"Top {n} {cat_col} by {val_col}")
    return ChatResponse(
        text=f"Top {n} {cat_col} by total {val_col}:",
        table=agg, figure=fig,
    )


def _try_highest(df: pd.DataFrame, query: str) -> Optional[ChatResponse]:
    if not re.search(r"\b(highest|max|maximum|best)\b", query.lower()):
        return None
    val_col = None
    for c in _numeric_cols(df):
        if c.lower() in query.lower():
            val_col = c; break
    if val_col is None and _numeric_cols(df):
        val_col = _numeric_cols(df)[0]
    cat_col = None
    for c in _cat_cols(df):
        if c.lower() in query.lower():
            cat_col = c; break
    if not (val_col and cat_col):
        return None
    agg = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
    return ChatResponse(
        text=f"**{agg.index[0]}** has the highest total {val_col} "
             f"({agg.iloc[0]:,.2f}).",
        table=agg.head(5).reset_index(),
    )


def _try_monthly_peak(df: pd.DataFrame, query: str) -> Optional[ChatResponse]:
    if "month" not in query.lower():
        return None
    dates = _date_cols(df); nums = _numeric_cols(df)
    if not (dates and nums):
        return None
    d, v = dates[0], nums[0]
    ts = (df[[d, v]].dropna()
            .groupby(pd.Grouper(key=d, freq="M"))[v].sum())
    if ts.empty:
        return None
    peak = ts.idxmax()
    fig = px.line(ts.reset_index(), x=d, y=v, title=f"{v} by month")
    return ChatResponse(
        text=f"The highest {v} was in **{peak.strftime('%B %Y')}** "
             f"({ts.max():,.2f}).",
        figure=fig,
    )


def _try_compare(df: pd.DataFrame, query: str) -> Optional[ChatResponse]:
    m = re.search(r"compare\s+([\w\s]+?)\s+(?:and|vs|versus|with)\s+([\w\s]+)",
                  query.lower())
    if not m:
        return None
    a = _find_column(df, m.group(1).strip())
    b = _find_column(df, m.group(2).strip())
    if not (a and b and a in _numeric_cols(df) and b in _numeric_cols(df)):
        return None
    fig = px.scatter(df, x=a, y=b, trendline=None,
                     title=f"{a} vs {b}", opacity=0.7)
    corr = df[[a, b]].corr().iloc[0, 1]
    return ChatResponse(
        text=f"Correlation between **{a}** and **{b}** is {corr:+.2f}.",
        figure=fig,
    )


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def answer(df: pd.DataFrame, question: str,
           model: str = DEFAULT_MODEL) -> ChatResponse:
    """Try deterministic handlers first, then fall back to the LLM."""
    for handler in (_try_top_n, _try_highest, _try_monthly_peak, _try_compare):
        try:
            res = handler(df, question)
        except Exception:
            res = None
        if res is not None:
            return res

    # LLM fallback — pass a compact schema, not the data
    schema = {
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "row_count": int(df.shape[0]),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }
    prompt = (
        "You are a data analyst answering a question about a pandas DataFrame. "
        "You will NOT execute code. Provide a concise (2-4 sentence) natural-"
        "language answer using only the metadata supplied. If the question "
        "cannot be answered from the metadata alone, say so and suggest the "
        "specific aggregation the user could run.\n\n"
        f"Schema: {json.dumps(schema, default=str)[:2500]}\n\n"
        f"Question: {question}"
    )
    return ChatResponse(text=ask_ollama(prompt, model=model))
