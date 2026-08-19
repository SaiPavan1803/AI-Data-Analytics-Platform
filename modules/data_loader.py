"""
data_loader.py
---------------
Handles CSV loading, dataset preview, type inference and basic cleanup.
"""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_csv(file_like) -> pd.DataFrame:
    """Load a CSV file from a path or a file-like object (Streamlit uploader)."""
    if hasattr(file_like, "read"):
        raw = file_like.read()
        if isinstance(raw, bytes):
            buf = io.BytesIO(raw)
        else:
            buf = io.StringIO(raw)
        df = pd.read_csv(buf)
    else:
        df = pd.read_csv(file_like)
    df = _coerce_dates(df)
    return df


def _coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to auto-convert string columns that look like dates."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(20)
            if sample.empty:
                continue
            # Heuristic: presence of '-' or '/' and parseable
            if sample.str.contains(r"[-/]").mean() > 0.6:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
                    if parsed.notna().mean() > 0.8:
                        df[col] = parsed
                except Exception:
                    pass
    return df


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify columns into numeric, categorical and datetime buckets."""
    numeric_cols, categorical_cols, datetime_cols = [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def basic_overview(df: pd.DataFrame) -> Dict:
    """Return a small dictionary describing the dataset shape and quality."""
    types = detect_column_types(df)
    missing = df.isna().sum()
    missing = missing[missing > 0].to_dict()
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {k: int(v) for k, v in missing.items()},
        "numeric_columns": types["numeric"],
        "categorical_columns": types["categorical"],
        "datetime_columns": types["datetime"],
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


def preview(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.head(n)
