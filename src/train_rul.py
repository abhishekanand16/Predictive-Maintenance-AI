"""
Training script for RUL prediction (regression).

Implements Linear Regression and XGBoost Regressor.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from typing import Tuple
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from preprocess import load_scaler
from features import get_feature_columns, create_all_features
from evaluate import evaluate_regression, print_regression_report, save_metrics
import config


def prepare_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare features and RUL labels for training.
    
    Args:
        train_df: Training DataFrame with features and 'RUL' column.
        val_df: Validation DataFrame with features and 'RUL' column.
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    feature_cols = get_feature_columns(train_df)
    
    X_train = train_df[feature_cols].values
    y_train = train_df["RUL"].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df["RUL"].values
    
    return X_train, y_train, X_val, y_val


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_model: bool = True
) -> Tuple[LinearRegression, dict]:
    """
    Train Linear Regression baseline model.
    
    Args:
        X_train: Training features.
        y_train: Training RUL values.
        X_val: Validation features.
        y_val: Validation RUL values.
        save_model: Whether to save the model.
        
    Returns:
        Tuple of (trained model, metrics dictionary).
    """
    print("\n" + "="*50)
    print("Training Linear Regression (Baseline)")
    print("="*50)
    
    model = LinearRegression(**config.LINEAR_REGRESSOR_PARAMS)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    save_dir = config.RESULTS_DIR / "linear_regression" if save_model else None
    metrics = evaluate_regression(y_val, y_pred, save_dir=save_dir)
    print_regression_report(metrics)
    
    if save_model:
        model_path = config.MODELS_DIR / "linear_regression.joblib"
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
    
    return model, metrics


def train_xgboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_model: bool = True,
    use_mlflow: bool = True
) -> Tuple[XGBRegressor, dict]:
    """
    Train XGBoost regressor (final model).
    
    Args:
        X_train: Training features.
        y_train: Training RUL values.
        X_val: Validation features.
        y_val: Validation RUL values.
        save_model: Whether to save the model.
        use_mlflow: Whether to log to MLflow.
        
    Returns:
        Tuple of (trained model, metrics dictionary).
    """
    print("\n" + "="*50)
    print("Training XGBoost Regressor (Final Model)")
    print("="*50)
    
    model = XGBRegressor(**config.XGB_REGRESSOR_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    
    save_dir = config.RESULTS_DIR / "xgboost_regressor" if save_model else None
    metrics = evaluate_regression(y_val, y_pred, save_dir=save_dir)
    print_regression_report(metrics)
    
    if use_mlflow:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name="xgboost_regressor"):
            mlflow.log_params(config.XGB_REGRESSOR_PARAMS)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, "model")
            print("\nLogged to MLflow")
    
    if save_model:
        model_path = config.MODELS_DIR / "xgboost_regressor.joblib"
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
    
    return model, metrics


def main():
    """Main training pipeline for RUL prediction."""
    print("="*70)
    print("RUL PREDICTION MODEL TRAINING")
    print("="*70)
    
    # Load processed data
    print("\nLoading processed data...")
    train_df = pd.read_pickle(config.DATA_PROCESSED_DIR / "train_processed.pkl")
    val_df = pd.read_pickle(config.DATA_PROCESSED_DIR / "val_processed.pkl")
    
    # Load or create features
    features_train_path = config.DATA_PROCESSED_DIR / "train_features.pkl"
    features_val_path = config.DATA_PROCESSED_DIR / "val_features.pkl"
    
    if features_train_path.exists() and features_val_path.exists():
        print("Loading feature-engineered data...")
        train_df = pd.read_pickle(features_train_path)
        val_df = pd.read_pickle(features_val_path)
    else:
        print("Creating features...")
        train_df = create_all_features(train_df)
        val_df = create_all_features(val_df)
        # Save for future use
        train_df.to_pickle(features_train_path)
        val_df.to_pickle(features_val_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"\nRUL statistics (train):")
    print(train_df["RUL"].describe())
    
    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, X_val, y_val = prepare_data(train_df, val_df)
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Train models
    results = {}
    
    # Baseline: Linear Regression
    lr_model, lr_metrics = train_linear_regression(
        X_train, y_train, X_val, y_val
    )
    results["linear_regression"] = lr_metrics
    
    # Final: XGBoost
    xgb_model, xgb_metrics = train_xgboost_regressor(
        X_train, y_train, X_val, y_val, use_mlflow=True
    )
    results["xgboost"] = xgb_metrics
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_path = config.RESULTS_DIR / "regression_results.csv"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path)
    print(f"\nResults summary saved to {results_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
