"""
Evaluation module for Predictive Maintenance AI.

Provides unified evaluation functions for both classification and regression tasks.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional

import config


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None
) -> dict:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities (optional, for PR-AUC).
        save_dir: Directory to save plots. If None, plots are not saved.
        
    Returns:
        Dictionary of metrics.
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            # Handle case where only one class is present
            metrics["roc_auc"] = 0.0
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Precision-Recall curve
        if y_proba is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve (AUC = {metrics['pr_auc']:.3f})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "pr_curve.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Optional[Path] = None
) -> dict:
    """
    Evaluate regression model performance.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        save_dir: Directory to save plots. If None, plots are not saved.
        
    Returns:
        Dictionary of metrics.
    """
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Scatter plot: predicted vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                "r--", linewidth=2, label="Perfect Prediction")
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title(f"RUL Prediction (RMSE = {rmse:.2f}, MAE = {mae:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "rul_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Residual plot
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
        plt.xlabel("Predicted RUL")
        plt.ylabel("Residuals (True - Predicted)")
        plt.title("Residual Plot")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "residual_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict
) -> None:
    """Print formatted classification report."""
    print("\n" + "="*50)
    print("Classification Results")
    print("="*50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if "pr_auc" in metrics:
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def print_regression_report(metrics: dict) -> None:
    """Print formatted regression report."""
    print("\n" + "="*50)
    print("Regression Results")
    print("="*50)
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print("="*50)


def save_metrics(metrics: dict, filepath: Path) -> None:
    """Save metrics to CSV file."""
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")
