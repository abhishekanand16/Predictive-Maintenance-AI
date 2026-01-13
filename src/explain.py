"""
SHAP explainability module for Predictive Maintenance AI.

Provides model interpretability using SHAP values.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Optional
import joblib

import config


def explain_classifier(
    model,
    X_sample: np.ndarray,
    feature_names: list,
    save_dir: Optional[Path] = None
) -> shap.Explanation:
    """
    Generate SHAP explanations for classification model.
    
    Args:
        model: Trained classifier (XGBoost).
        X_sample: Sample data for explanation (can be subset).
        feature_names: List of feature names.
        save_dir: Directory to save plots.
        
    Returns:
        SHAP Explanation object.
    """
    print("Generating SHAP explanations for classifier...")
    
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Limit sample size for performance
    if len(X_sample) > config.SHAP_SAMPLE_SIZE:
        sample_indices = np.random.choice(
            len(X_sample),
            size=config.SHAP_SAMPLE_SIZE,
            replace=False
        )
        X_sample = X_sample[sample_indices]
    
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample,
        feature_names=feature_names
    )
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_summary_classifier.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Bar plot (mean absolute SHAP values)
        plt.figure(figsize=(10, 8))
        shap.plots.bar(explanation, show=False)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_bar_classifier.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"SHAP plots saved to {save_dir}")
    
    return explanation


def explain_regressor(
    model,
    X_sample: np.ndarray,
    feature_names: list,
    save_dir: Optional[Path] = None
) -> shap.Explanation:
    """
    Generate SHAP explanations for regression model.
    
    Args:
        model: Trained regressor (XGBoost).
        X_sample: Sample data for explanation (can be subset).
        feature_names: List of feature names.
        save_dir: Directory to save plots.
        
    Returns:
        SHAP Explanation object.
    """
    print("Generating SHAP explanations for regressor...")
    
    explainer = shap.TreeExplainer(model)
    
    if len(X_sample) > config.SHAP_SAMPLE_SIZE:
        sample_indices = np.random.choice(
            len(X_sample),
            size=config.SHAP_SAMPLE_SIZE,
            replace=False
        )
        X_sample = X_sample[sample_indices]
    
    shap_values = explainer.shap_values(X_sample)
    
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample,
        feature_names=feature_names
    )
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_summary_regressor.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(explanation, show=False)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_bar_regressor.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"SHAP plots saved to {save_dir}")
    
    return explanation


def explain_single_prediction(
    model,
    X_instance: np.ndarray,
    feature_names: list,
    task_type: str = "classification"
) -> shap.Explanation:
    """
    Generate SHAP waterfall plot for a single prediction.
    
    Args:
        model: Trained model.
        X_instance: Single instance (1D array or 2D with one row).
        feature_names: List of feature names.
        task_type: "classification" or "regression".
        
    Returns:
        SHAP Explanation object for single instance.
    """
    if X_instance.ndim == 1:
        X_instance = X_instance.reshape(1, -1)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_instance[0],
        feature_names=feature_names
    )
    
    return explanation


def get_top_features(
    explanation: shap.Explanation,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Get top K features by absolute SHAP value.
    
    Args:
        explanation: SHAP Explanation object.
        top_k: Number of top features to return.
        
    Returns:
        DataFrame with feature names and mean absolute SHAP values.
    """
    if explanation.values.ndim == 1:
        # Single instance
        abs_shap = np.abs(explanation.values)
    else:
        # Multiple instances
        abs_shap = np.abs(explanation.values).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        "feature": explanation.feature_names,
        "mean_abs_shap": abs_shap
    }).sort_values("mean_abs_shap", ascending=False).head(top_k)
    
    return feature_importance


if __name__ == "__main__":
    # Example usage
    from train_failure import prepare_data
    from features import get_feature_columns, create_all_features
    
    print("Loading data and model...")
    
    # Load feature-engineered data (same as training script)
    features_train_path = config.DATA_PROCESSED_DIR / "train_features.pkl"
    features_val_path = config.DATA_PROCESSED_DIR / "val_features.pkl"
    
    if features_train_path.exists() and features_val_path.exists():
        print("Loading feature-engineered data...")
        train_df = pd.read_pickle(features_train_path)
        val_df = pd.read_pickle(features_val_path)
    else:
        print("Feature-engineered data not found. Creating features...")
        train_df = pd.read_pickle(config.DATA_PROCESSED_DIR / "train_processed.pkl")
        val_df = pd.read_pickle(config.DATA_PROCESSED_DIR / "val_processed.pkl")
        train_df = create_all_features(train_df)
        val_df = create_all_features(val_df)
        # Save for future use
        train_df.to_pickle(features_train_path)
        val_df.to_pickle(features_val_path)
        print("Feature engineering complete.")
    
    X_train, y_train, X_val, y_val = prepare_data(train_df, val_df)
    feature_cols = get_feature_columns(train_df)
    
    # Load model
    model_path = config.MODELS_DIR / "xgboost_classifier.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        print("Generating SHAP explanations...")
        explanation = explain_classifier(
            model,
            X_val[:100],  # Use subset
            feature_cols,
            save_dir=config.RESULTS_DIR / "shap"
        )
        
        top_features = get_top_features(explanation, top_k=15)
        print("\nTop 15 features by SHAP importance:")
        print(top_features)
    else:
        print(f"Model not found at {model_path}. Please train models first.")
