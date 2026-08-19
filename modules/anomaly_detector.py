"""
anomaly_detector.py
-------------------
Detects anomalies in numeric or time-series data using IsolationForest and
a z-score fallback. Returns both the anomalous rows and short textual
explanations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03,
                     max_features: int = 8) -> pd.DataFrame:
    """Return a copy of df restricted to anomalous rows with a score column."""
    num = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if num.empty:
        return pd.DataFrame()
    num = num.iloc[:, :max_features].fillna(num.median(numeric_only=True))
    if num.shape[0] < 10:
        return pd.DataFrame()

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    labels = model.fit_predict(num)
    scores = -model.score_samples(num)  # higher = more anomalous

    out = df.loc[num.index].copy()
    out["anomaly_score"] = scores
    out["is_anomaly"] = labels == -1
    return (out[out["is_anomaly"]]
              .sort_values("anomaly_score", ascending=False)
              .head(25))


def explain_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame,
                      value_col: Optional[str] = None) -> List[str]:
    """Produce short natural-language explanations for each anomaly row."""
    if anomalies.empty:
        return ["No significant anomalies detected."]
    numeric = df.select_dtypes(include="number").columns.tolist()
    if value_col is None and numeric:
        value_col = numeric[0]

    explanations: List[str] = []
    if value_col and value_col in df.columns:
        avg = df[value_col].mean()
        std = df[value_col].std(ddof=0) or 1.0
        for idx, row in anomalies.head(10).iterrows():
            v = row[value_col]
            z = (v - avg) / std
            pct = (v - avg) / max(abs(avg), 1e-9) * 100
            direction = "higher" if v > avg else "lower"
            explanations.append(
                f"Row {idx}: {value_col} = {v:,.2f} is {abs(pct):.1f}% "
                f"{direction} than the mean ({avg:,.2f}), z-score "
                f"{z:+.2f}. Possible promotional event, data-entry "
                f"error, or genuine outlier."
            )
    else:
        for idx in anomalies.head(10).index:
            explanations.append(
                f"Row {idx}: multivariate outlier flagged by IsolationForest."
            )
    return explanations


def zscore_flags(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Simple univariate fallback returning a boolean mask of outliers."""
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=bool)
    z = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    return z.abs() > threshold
