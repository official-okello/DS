import pandas as pd
import numpy as np
from typing import Tuple


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags=(1, 3, 7, 14, 30),
    groupby_col: str = "station_id",
    inplace: bool = False
) -> pd.DataFrame:
    
    if not inplace:
        df = df.copy()
    if groupby_col not in df.columns:
        raise KeyError(f"Grouping column '{groupby_col}' not found in DataFrame")
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df.groupby(groupby_col)[target].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    windows=(3, 7, 14, 30),
    groupby_col: str = "station_id",
    quantiles=(0.25, 0.5, 0.75),
    inplace: bool = False
) -> pd.DataFrame:
    
    if not inplace:
        df = df.copy()
    if groupby_col not in df.columns:
        raise KeyError(f"Grouping column '{groupby_col}' not found in DataFrame")
    for w in windows:
        grp = df.groupby(groupby_col)[target]
        shifted = grp.shift(1)
        df[f"{target}_roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"{target}_roll_std_{w}"] = shifted.rolling(w).std()
        df[f"{target}_roll_med_{w}"] = shifted.rolling(w).median()
        # quantiles
        for q in quantiles:
            q_name = int(q * 100)
            try:
                df[f"{target}_roll_q{q_name}_{w}"] = shifted.rolling(w).quantile(q)
            except Exception:
                # fallback to apply as older pandas might not support rolling.quantile
                df[f"{target}_roll_q{q_name}_{w}"] = shifted.rolling(w).apply(lambda x: x.quantile(q) if len(x) > 0 else pd.NA)
    return df


def add_cyclic_time_features(df: pd.DataFrame, date_col: str = "Sample Date") -> pd.DataFrame:
    """Add cyclic encodings for month and day-of-year to capture seasonality."""
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df["dayofyear"] = df[date_col].dt.dayofyear
        df["month"] = df[date_col].dt.month
        # dayofyear cyclical
        doy_rad = 2 * np.pi * df["dayofyear"] / 365.0
        df["sin_doy"] = np.sin(doy_rad)
        df["cos_doy"] = np.cos(doy_rad)
        # month cyclical
        month_rad = 2 * np.pi * (df["month"] - 1) / 12.0
        df["sin_month"] = np.sin(month_rad)
        df["cos_month"] = np.cos(month_rad)
    return df


def finalize_features(df: pd.DataFrame, target: str, dropna: bool = True, exclude_columns: list | None = None) -> Tuple[pd.DataFrame, pd.Series]:

    df = df.copy()
    # Drop any columns that should be excluded (e.g., other target columns)
    if exclude_columns:
        cols_to_drop = [c for c in exclude_columns if c != target]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors='ignore')

    if dropna:
        df = df.dropna()

    y = df[target]
    X = df.drop(columns=[target])
    # Keep only numeric columns for model training/prediction
    X = X.select_dtypes(include=["number"]).copy()
    return X, y