"""
Data preprocessing module for Predictive Maintenance AI.

Handles loading, cleaning, normalization, and train/validation splitting
of the NASA C-MAPSS dataset.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Tuple, Optional

import config


def load_train_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load training data from C-MAPSS dataset.
    
    Args:
        file_path: Path to train_FD001.txt. If None, uses config path.
        
    Returns:
        DataFrame with columns assigned from config.
    """
    if file_path is None:
        file_path = config.DATA_RAW_DIR / config.TRAIN_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {file_path}. "
            f"Please place {config.TRAIN_FILE} in {config.DATA_RAW_DIR}"
        )
    
    df = pd.read_csv(file_path, sep=" ", header=None)
    # Remove trailing spaces/empty columns
    df = df.iloc[:, :-2]  # Last two columns are typically empty
    df.columns = config.COLUMN_NAMES
    
    return df


def load_test_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load test data from C-MAPSS dataset.
    
    Note: This is for final evaluation only, not for training/validation.
    
    Args:
        file_path: Path to test_FD001.txt. If None, uses config path.
        
    Returns:
        DataFrame with columns assigned from config.
    """
    if file_path is None:
        file_path = config.DATA_RAW_DIR / config.TEST_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {file_path}. "
            f"Please place {config.TEST_FILE} in {config.DATA_RAW_DIR}"
        )
    
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.iloc[:, :-2]
    df.columns = config.COLUMN_NAMES
    
    return df


def load_rul_labels(file_path: Optional[Path] = None) -> pd.Series:
    """
    Load ground truth RUL labels for test set.
    
    Args:
        file_path: Path to RUL_FD001.txt. If None, uses config path.
        
    Returns:
        Series with RUL values indexed by engine_id.
    """
    if file_path is None:
        file_path = config.DATA_RAW_DIR / config.RUL_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"RUL labels not found at {file_path}. "
            f"Please place {config.RUL_FILE} in {config.DATA_RAW_DIR}"
        )
    
    rul = pd.read_csv(file_path, sep=" ", header=None)
    rul = rul.iloc[:, 0]  # First column contains RUL values
    
    # Create engine_id index (1-indexed)
    rul.index = range(1, len(rul) + 1)
    rul.index.name = "engine_id"
    rul.name = "RUL"
    
    return rul


def remove_constant_sensors(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Remove sensors with constant or near-constant values.
    
    Args:
        df: Input DataFrame with sensor columns.
        
    Returns:
        Tuple of (cleaned DataFrame, list of removed sensor names).
    """
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    removed_sensors = []
    
    for col in sensor_cols:
        variance = df[col].var()
        if variance < config.MIN_VARIANCE_THRESHOLD:
            removed_sensors.append(col)
            df = df.drop(columns=[col])
    
    return df, removed_sensors


def calculate_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each cycle.
    
    RUL = max_cycle_per_engine - current_cycle
    
    Args:
        df: DataFrame with engine_id and cycle columns.
        
    Returns:
        DataFrame with added 'RUL' column.
    """
    df = df.copy()
    
    # Calculate max cycle per engine
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df = df.merge(
        max_cycles.rename("max_cycle"),
        left_on="engine_id",
        right_index=True,
        how="left"
    )
    
    # Calculate RUL
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df = df.drop(columns=["max_cycle"])
    
    return df


def create_failure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary failure labels.
    
    failure = 1 if RUL <= FAILURE_THRESHOLD, else 0
    
    Args:
        df: DataFrame with 'RUL' column.
        
    Returns:
        DataFrame with added 'failure' column.
    """
    df = df.copy()
    df["failure"] = (df["RUL"] <= config.FAILURE_THRESHOLD).astype(int)
    return df


def split_by_engine(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_VAL_SPLIT_RATIO,
    random_seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by engine_id to prevent data leakage.
    
    Args:
        df: Input DataFrame with engine_id column.
        train_ratio: Proportion of engines for training.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train DataFrame, validation DataFrame).
    """
    np.random.seed(random_seed)
    
    unique_engines = df["engine_id"].unique()
    n_train = int(len(unique_engines) * train_ratio)
    
    train_engines = np.random.choice(
        unique_engines,
        size=n_train,
        replace=False
    )
    
    train_df = df[df["engine_id"].isin(train_engines)].copy()
    val_df = df[~df["engine_id"].isin(train_engines)].copy()
    
    return train_df, val_df


def normalize_sensors(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], MinMaxScaler]:
    """
    Normalize sensor readings using MinMaxScaler.
    
    Fits scaler on training data only, then transforms train/val/test.
    
    Args:
        train_df: Training DataFrame.
        val_df: Optional validation DataFrame.
        test_df: Optional test DataFrame.
        scaler: Optional pre-fitted scaler. If None, fits on train_df.
        
    Returns:
        Tuple of (normalized train_df, normalized val_df, normalized test_df, fitted scaler).
    """
    sensor_cols = [col for col in train_df.columns if col.startswith("sensor_")]
    op_cols = [col for col in train_df.columns if col.startswith("op_setting_")]
    feature_cols = sensor_cols + op_cols
    
    train_df = train_df.copy()
    if val_df is not None:
        val_df = val_df.copy()
    if test_df is not None:
        test_df = test_df.copy()
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(train_df[feature_cols])
    
    # Transform
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    if val_df is not None:
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    if test_df is not None:
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, val_df, test_df, scaler


def save_scaler(scaler: MinMaxScaler, filepath: Path) -> None:
    """Save fitted scaler to disk."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)


def load_scaler(filepath: Path) -> MinMaxScaler:
    """Load fitted scaler from disk."""
    return joblib.load(filepath)


def preprocess_pipeline(
    save_processed: bool = True,
    save_scaler_file: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Complete preprocessing pipeline.
    
    Steps:
    1. Load training data
    2. Remove constant sensors
    3. Calculate RUL labels
    4. Create failure labels
    5. Split by engine_id into train/validation
    6. Normalize sensors
    
    Args:
        save_processed: Whether to save processed DataFrames.
        save_scaler_file: Whether to save the fitted scaler.
        
    Returns:
        Tuple of (train_df, val_df, fitted_scaler).
    """
    print("Loading training data...")
    df = load_train_data()
    
    print("Removing constant sensors...")
    df, removed = remove_constant_sensors(df)
    print(f"Removed {len(removed)} constant sensors: {removed}")
    
    print("Calculating RUL labels...")
    df = calculate_rul_labels(df)
    
    print("Creating failure labels...")
    df = create_failure_labels(df)
    
    print("Splitting by engine_id...")
    train_df, val_df = split_by_engine(df)
    print(f"Train engines: {train_df['engine_id'].nunique()}")
    print(f"Validation engines: {val_df['engine_id'].nunique()}")
    
    print("Normalizing sensors...")
    train_df, val_df, _, scaler = normalize_sensors(train_df, val_df=val_df)
    
    if save_processed:
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_pickle(config.DATA_PROCESSED_DIR / "train_processed.pkl")
        val_df.to_pickle(config.DATA_PROCESSED_DIR / "val_processed.pkl")
        print(f"Saved processed data to {config.DATA_PROCESSED_DIR}")
    
    if save_scaler_file:
        scaler_path = config.MODELS_DIR / "scaler.joblib"
        save_scaler(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
    
    return train_df, val_df, scaler


if __name__ == "__main__":
    # Run preprocessing pipeline
    train_df, val_df, scaler = preprocess_pipeline()
    print("\nPreprocessing complete!")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"\nFailure class distribution (train):")
    print(train_df["failure"].value_counts())
    print(f"\nFailure class distribution (validation):")
    print(val_df["failure"].value_counts())
