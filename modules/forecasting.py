"""
forecasting.py
--------------
Time-series style forecasting using simple ML regressors. Works on any
(date, value) pair the user picks. Two models supported: LinearRegression
and RandomForestRegressor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass
class ForecastResult:
    history: pd.DataFrame          # columns: date, value
    forecast: pd.DataFrame         # columns: date, value, lower, upper
    metric_mae: float
    metric_r2: float
    model_name: str


def _prepare_series(df: pd.DataFrame, date_col: str, value_col: str,
                    freq: str = "M") -> pd.DataFrame:
    ts = (df[[date_col, value_col]].dropna()
            .groupby(pd.Grouper(key=date_col, freq=freq))[value_col]
            .sum().reset_index())
    ts.columns = ["date", "value"]
    return ts


def _build_features(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.copy()
    ts["t"] = np.arange(len(ts))
    ts["month"] = ts["date"].dt.month
    ts["quarter"] = ts["date"].dt.quarter
    ts["year"] = ts["date"].dt.year
    return ts


def forecast(df: pd.DataFrame, date_col: str, value_col: str,
             periods: int = 6, freq: str = "M",
             model_name: str = "RandomForest") -> ForecastResult:
    ts = _prepare_series(df, date_col, value_col, freq=freq)
    if len(ts) < 6:
        raise ValueError("Need at least 6 aggregated periods to forecast.")

    feats = _build_features(ts)
    X = feats[["t", "month", "quarter", "year"]].values
    y = feats["value"].values

    # 80/20 train-test split
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_name == "Linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    if len(y_test):
        pred_test = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred_test))
        r2 = float(r2_score(y_test, pred_test)) if len(y_test) >= 2 else float("nan")
    else:
        mae, r2 = float("nan"), float("nan")

    # Refit on full data and forecast future periods
    model.fit(X, y)
    last_date = ts["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=periods, freq=freq,
    )
    future = pd.DataFrame({"date": future_dates})
    future["t"] = np.arange(len(ts), len(ts) + periods)
    future["month"] = future["date"].dt.month
    future["quarter"] = future["date"].dt.quarter
    future["year"] = future["date"].dt.year

    preds = model.predict(future[["t", "month", "quarter", "year"]].values)
    residual_std = float(np.std(y - model.predict(X))) or 1.0

    future_out = pd.DataFrame({
        "date": future_dates,
        "value": preds,
        "lower": preds - 1.96 * residual_std,
        "upper": preds + 1.96 * residual_std,
    })

    return ForecastResult(
        history=ts,
        forecast=future_out,
        metric_mae=mae,
        metric_r2=r2,
        model_name=model_name,
    )
