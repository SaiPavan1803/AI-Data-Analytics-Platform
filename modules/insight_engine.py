"""
insight_engine.py
-----------------
Generates business-level insights from a tabular dataset using pandas
heuristics, then optionally polishes them through an LLM call.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .analyzer import ask_ollama, DEFAULT_MODEL


def _pick_value_column(df: pd.DataFrame) -> Optional[str]:
    """Choose the most likely 'business value' numeric column."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        return None
    priority = ["revenue", "sales", "profit", "amount", "total", "price", "value"]
    for key in priority:
        for c in numeric:
            if key in c.lower():
                return c
    return numeric[0]


def _pick_category_column(df: pd.DataFrame) -> Optional[str]:
    cats = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and 2 <= df[c].nunique(dropna=True) <= 50
    ]
    return cats[0] if cats else None


def _pick_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def generate_insights(df: pd.DataFrame) -> List[str]:
    """Produce a list of plain-English insight strings."""
    insights: List[str] = []

    val_col = _pick_value_column(df)
    cat_col = _pick_category_column(df)
    date_col = _pick_date_column(df)

    # 1. Top / bottom category
    if val_col and cat_col:
        agg = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
        if len(agg) >= 2:
            insights.append(
                f"Top performing **{cat_col}** by total {val_col}: "
                f"**{agg.index[0]}** ({agg.iloc[0]:,.2f}). "
                f"Lowest: **{agg.index[-1]}** ({agg.iloc[-1]:,.2f})."
            )

    # 2. Time trend
    if val_col and date_col:
        ts = (df[[date_col, val_col]].dropna()
                .groupby(pd.Grouper(key=date_col, freq="M"))[val_col]
                .sum())
        if len(ts) >= 2:
            change = (ts.iloc[-1] - ts.iloc[0]) / max(abs(ts.iloc[0]), 1e-9) * 100
            direction = "increased" if change >= 0 else "decreased"
            insights.append(
                f"Monthly **{val_col}** has {direction} by "
                f"**{change:+.1f}%** from {ts.index[0].date()} to "
                f"{ts.index[-1].date()}."
            )
            best_month = ts.idxmax()
            insights.append(
                f"Best month: **{best_month.strftime('%B %Y')}** "
                f"with {val_col} of {ts.max():,.2f}."
            )

    # 3. Strong correlations
    num = df.select_dtypes(include="number")
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, 0)
        pair = corr.unstack().sort_values(ascending=False)
        if not pair.empty and pair.iloc[0] >= 0.6:
            a, b = pair.index[0]
            insights.append(
                f"Strong correlation detected between **{a}** and "
                f"**{b}** (|r| = {pair.iloc[0]:.2f})."
            )

    # 4. Missing data flag
    miss = df.isna().mean()
    high_miss = miss[miss > 0.2]
    if not high_miss.empty:
        insights.append(
            "Columns with >20% missing values: "
            + ", ".join(f"**{c}** ({p:.0%})" for c, p in high_miss.items())
            + "."
        )

    # 5. Skew flag
    for c in num.columns:
        s = num[c].dropna()
        if len(s) > 10 and abs(s.skew()) > 2:
            insights.append(
                f"**{c}** is highly skewed (skewness={s.skew():.2f}); consider "
                "log-transform or outlier review."
            )
            break

    if not insights:
        insights.append("No strong patterns detected with the default heuristics.")
    return insights


def narrate_insights(df: pd.DataFrame, insights: List[str],
                     model: str = DEFAULT_MODEL) -> str:
    """Use the LLM to weave the insights into a short executive summary."""
    bullet = "\n".join(f"- {i}" for i in insights)
    prompt = (
        "Rewrite these data insights as a concise executive summary (max 6 "
        "sentences) for a business stakeholder. Keep numbers exact, drop "
        "markdown bold, and end with one actionable recommendation.\n\n"
        f"Insights:\n{bullet}"
    )
    return ask_ollama(prompt, model=model)
