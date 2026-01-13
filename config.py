"""
Configuration module for Predictive Maintenance AI System.

All hyperparameters, paths, and model settings are centralized here.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Dataset file names
TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"

# Column definitions
COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Feature engineering parameters
ROLLING_WINDOWS = [5, 10, 20]  # Window sizes for rolling statistics

# Failure prediction threshold
FAILURE_THRESHOLD = 30  # N cycles - failure occurs if RUL <= N

# Train/validation split
TRAIN_VAL_SPLIT_RATIO = 0.8  # 80% engines for training, 20% for validation
RANDOM_SEED = 42

# Model hyperparameters
# XGBoost Classifier
XGB_CLASSIFIER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
}

# XGBoost Regressor
XGB_REGRESSOR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "eval_metric": "rmse",
}

# Random Forest Classifier
RF_CLASSIFIER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

# Logistic Regression
LR_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

# Linear Regression
LINEAR_REGRESSOR_PARAMS = {}

# Preprocessing
MIN_VARIANCE_THRESHOLD = 1e-6  # Remove sensors with variance below this threshold
SCALER_TYPE = "MinMaxScaler"  # Options: "MinMaxScaler", "StandardScaler"

# MLflow settings
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR.absolute()}"
MLFLOW_EXPERIMENT_NAME = "predictive_maintenance"

# SHAP settings
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP summary plots

# Dashboard settings
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "Predictive Maintenance AI Dashboard"
