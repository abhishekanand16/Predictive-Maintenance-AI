"""
Feature engineering module for Predictive Maintenance AI.

Creates time-series features grouped by engine_id to avoid cross-engine leakage.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from typing import List, Optional
from scipy import stats

import config


def rolling_mean(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = config.ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Calculate rolling mean for specified columns, grouped by engine_id.
    
    Important: Rolling windows reset per engine to avoid cross-engine leakage.
    
    Args:
        df: DataFrame with engine_id and sensor columns.
        columns: List of column names to compute rolling mean for.
        windows: List of window sizes.
        
    Returns:
        DataFrame with added rolling mean columns.
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            df[f"{col}_rolling_mean_{window}"] = (
                df.groupby("engine_id")[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    
    return df


def rolling_std(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = config.ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Calculate rolling standard deviation, grouped by engine_id.
    
    Args:
        df: DataFrame with engine_id and sensor columns.
        columns: List of column names to compute rolling std for.
        windows: List of window sizes.
        
    Returns:
        DataFrame with added rolling std columns.
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            df[f"{col}_rolling_std_{window}"] = (
                df.groupby("engine_id")[col]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)  # Fill NaN for first row of each engine
                .reset_index(level=0, drop=True)
            )
    
    return df


def rolling_min_max(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = config.ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Calculate rolling min and max, grouped by engine_id.
    
    Args:
        df: DataFrame with engine_id and sensor columns.
        columns: List of column names.
        windows: List of window sizes.
        
    Returns:
        DataFrame with added rolling min/max columns.
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            df[f"{col}_rolling_min_{window}"] = (
                df.groupby("engine_id")[col]
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(level=0, drop=True)
            )
            
            df[f"{col}_rolling_max_{window}"] = (
                df.groupby("engine_id")[col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )
    
    return df


def rolling_slope(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = config.ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    Calculate linear slope over rolling windows, grouped by engine_id.
    
    Uses linear regression to compute trend slope.
    
    Args:
        df: DataFrame with engine_id and sensor columns.
        columns: List of column names.
        windows: List of window sizes.
        
    Returns:
        DataFrame with added slope columns.
    """
    df = df.copy()
    
    def compute_slope(series: pd.Series, window: int) -> pd.Series:
        """Compute slope for each rolling window."""
        slopes = []
        for i in range(len(series)):
            start_idx = max(0, i - window + 1)
            window_data = series.iloc[start_idx:i+1]
            if len(window_data) >= 2:
                x = np.arange(len(window_data))
                slope, _ = np.polyfit(x, window_data.values, 1)
                slopes.append(slope)
            else:
                slopes.append(0.0)
        return pd.Series(slopes, index=series.index)
    
    for col in columns:
        for window in windows:
            df[f"{col}_rolling_slope_{window}"] = (
                df.groupby("engine_id")[col]
                .apply(lambda x: compute_slope(x, window))
                .reset_index(level=0, drop=True)
            )
    
    return df


def first_order_diff(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Calculate first-order difference: sensor[t] - sensor[t-1].
    
    Grouped by engine_id, so first row of each engine is set to 0.
    
    Args:
        df: DataFrame with engine_id and sensor columns.
        columns: List of column names.
        
    Returns:
        DataFrame with added diff columns.
    """
    df = df.copy()
    
    for col in columns:
        df[f"{col}_diff"] = (
            df.groupby("engine_id")[col]
            .diff()
            .fillna(0)  # First row of each engine becomes 0
        )
    
    return df


def create_all_features(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    op_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create all time-series features for the dataset.
    
    Features include:
    - Rolling mean (windows: 5, 10, 20)
    - Rolling std (windows: 5, 10, 20)
    - Rolling min/max (windows: 5, 10, 20)
    - Rolling slope (windows: 5, 10, 20)
    - First-order difference
    
    All features are computed per engine_id to avoid leakage.
    
    Args:
        df: Input DataFrame with engine_id, cycle, sensors, and op_settings.
        sensor_cols: List of sensor column names. If None, auto-detected.
        op_cols: List of operational setting column names. If None, auto-detected.
        
    Returns:
        DataFrame with all engineered features.
    """
    df = df.copy()
    
    # Auto-detect columns if not provided
    if sensor_cols is None:
        sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    if op_cols is None:
        op_cols = [col for col in df.columns if col.startswith("op_setting_")]
    
    feature_cols = sensor_cols + op_cols
    
    print(f"Creating features for {len(feature_cols)} columns...")
    
    print("  - Rolling mean...")
    df = rolling_mean(df, feature_cols)
    
    print("  - Rolling std...")
    df = rolling_std(df, feature_cols)
    
    print("  - Rolling min/max...")
    df = rolling_min_max(df, feature_cols)
    
    print("  - Rolling slope...")
    df = rolling_slope(df, feature_cols)
    
    print("  - First-order difference...")
    df = first_order_diff(df, feature_cols)
    
    print(f"Feature engineering complete. Shape: {df.shape}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding metadata columns).
    
    Args:
        df: DataFrame with features.
        
    Returns:
        List of feature column names.
    """
    exclude_cols = ["engine_id", "cycle", "RUL", "failure"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    from preprocess import load_train_data, calculate_rul_labels, create_failure_labels
    
    print("Loading data...")
    df = load_train_data()
    df = calculate_rul_labels(df)
    df = create_failure_labels(df)
    
    # Use a subset for testing
    test_engines = df["engine_id"].unique()[:5]
    df_test = df[df["engine_id"].isin(test_engines)].copy()
    
    print(f"\nTesting feature engineering on {len(test_engines)} engines...")
    df_features = create_all_features(df_test)
    
    print(f"\nOriginal columns: {len(df_test.columns)}")
    print(f"Feature columns: {len(df_features.columns)}")
    print(f"\nSample feature columns:")
    feature_cols = get_feature_columns(df_features)
    print(feature_cols[:10])
