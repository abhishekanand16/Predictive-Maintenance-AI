"""
Pipeline runner script for Predictive Maintenance AI.

Runs the complete pipeline: preprocessing -> feature engineering -> training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.preprocess import preprocess_pipeline
from src.features import create_all_features
import config


def main():
    """Run the complete pipeline."""
    print("="*70)
    print("PREDICTIVE MAINTENANCE AI - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Preprocessing
    print("\n[1/4] Data Preprocessing...")
    print("-" * 70)
    train_df, val_df, scaler = preprocess_pipeline()
    
    # Step 2: Feature Engineering
    print("\n[2/4] Feature Engineering...")
    print("-" * 70)
    train_features = create_all_features(train_df)
    val_features = create_all_features(val_df)
    
    # Save feature-engineered data
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_features.to_pickle(config.DATA_PROCESSED_DIR / "train_features.pkl")
    val_features.to_pickle(config.DATA_PROCESSED_DIR / "val_features.pkl")
    print("Feature-engineered data saved.")
    
    # Step 3: Train Classification Model
    print("\n[3/4] Training Failure Prediction Model...")
    print("-" * 70)
    from src.train_failure import main as train_failure_main
    train_failure_main()
    
    # Step 4: Train Regression Model
    print("\n[4/4] Training RUL Prediction Model...")
    print("-" * 70)
    from src.train_rul import main as train_rul_main
    train_rul_main()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. View results in results/ directory")
    print("2. Launch dashboard: streamlit run dashboard/app.py")
    print("3. View MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()
