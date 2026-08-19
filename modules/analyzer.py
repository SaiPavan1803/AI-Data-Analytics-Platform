"""
analyzer.py
-----------
Dataset summarisation utilities and a thin wrapper around the local Ollama
HTTP API. The wrapper degrades gracefully when Ollama is not running so the
rest of the Streamlit app still works.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"  # change to "qwen3" if you prefer


# --------------------------------------------------------------------------- #
# Ollama helper                                                               #
# --------------------------------------------------------------------------- #
def ask_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> str:
    """Send a single prompt to Ollama and return the textual response.

    Returns a friendly fallback string if Ollama is not reachable so the
    UI never crashes.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as exc:  # pragma: no cover
        return (
            "[Ollama unavailable] Could not reach a local model at "
            f"{OLLAMA_URL}. Start Ollama with `ollama serve` and pull a "
            f"model (`ollama pull {model}`). Underlying error: {exc}"
        )


# --------------------------------------------------------------------------- #
# Dataset profiling                                                           #
# --------------------------------------------------------------------------- #
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    return num.describe().T.round(3)


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    cat = df.select_dtypes(exclude=["number", "datetime"])
    for c in cat.columns:
        vc = df[c].value_counts(dropna=False).head(top_n)
        out[c] = {
            "unique": int(df[c].nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in vc.items()},
        }
    return out


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(numeric_only=True).round(3)


def dataset_profile(df: pd.DataFrame) -> Dict:
    return {
        "shape": df.shape,
        "numeric_summary": numeric_summary(df).to_dict(),
        "categorical_summary": categorical_summary(df),
        "correlation": correlation_matrix(df).to_dict(),
    }


# --------------------------------------------------------------------------- #
# Natural-language dataset explanation                                        #
# --------------------------------------------------------------------------- #
def explain_dataset(df: pd.DataFrame, overview: Dict, model: str = DEFAULT_MODEL) -> str:
    """Ask the LLM to produce a short business-style explanation of the data."""
    sample = df.head(5).to_dict(orient="records")
    prompt = (
        "You are a senior data analyst. Given the metadata and a small sample "
        "of a tabular dataset, write a 4-6 sentence explanation aimed at a "
        "business stakeholder. Identify what the dataset likely tracks, the "
        "probable primary metric, and one or two questions it could answer. "
        "Avoid markdown headings.\n\n"
        f"Metadata:\n{json.dumps(overview, default=str)[:2500]}\n\n"
        f"Sample rows:\n{json.dumps(sample, default=str)[:1500]}"
    )
    return ask_ollama(prompt, model=model)
