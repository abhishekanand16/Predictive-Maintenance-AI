# Predictive Maintenance AI System

An end-to-end machine learning system for predicting machine failures and estimating Remaining Useful Life (RUL) using industrial sensor data from NASA's C-MAPSS Turbofan Engine Degradation Dataset.

## 🎯 Problem Statement

Unexpected machine failures cause significant operational downtime and financial losses in manufacturing industries. Traditional preventive maintenance strategies are inefficient, either reacting too late or servicing equipment prematurely. This project addresses this challenge by building an AI-powered predictive maintenance system that:

1. **Predicts imminent failures** (classification) - Identifies if a machine will fail within the next N cycles
2. **Estimates Remaining Useful Life** (regression) - Predicts how many operational cycles remain before failure

## 💼 Business Value

- **Reduce unplanned downtime** - Proactive maintenance scheduling
- **Enable condition-based maintenance** - Service equipment only when needed
- **Improve asset utilization** - Maximize equipment lifespan
- **Provide interpretable AI predictions** - Engineers can understand and trust model decisions

## 📊 Dataset

### NASA C-MAPSS Turbofan Engine Degradation Dataset

Each engine starts in a healthy state and degrades over time until failure.

**Data Structure:**
- `engine_id`: Unique identifier for each engine
- `cycle`: Time step in the engine's lifecycle
- `op_setting_1-3`: 3 operational settings
- `sensor_1-21`: 21 sensor readings monitoring engine health

**Target Variables:**
- **RUL (Remaining Useful Life)**: Number of cycles until failure
- **Failure Label**: Binary indicator (1 if RUL ≤ 30 cycles, else 0)

**Data Split Strategy:**
- Training data (`train_FD001.txt`) is split by `engine_id` into train/validation sets (80/20)
- Test data (`test_FD001.txt`) is reserved for **final evaluation only** (no leakage)

## 🏗️ Project Structure

```
predictive-maintenance-ai/
├── data/
│   ├── raw/                 # Place C-MAPSS files here
│   └── processed/           # Processed and feature-engineered data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_engineering.ipynb
├── src/
│   ├── preprocess.py        # Data loading and preprocessing
│   ├── features.py          # Time-series feature engineering
│   ├── train_failure.py     # Classification model training
│   ├── train_rul.py         # Regression model training
│   ├── evaluate.py          # Evaluation utilities
│   └── explain.py           # SHAP explainability
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── models/                  # Saved model artifacts
├── results/                 # Evaluation results and plots
├── mlruns/                  # MLflow experiment tracking
├── config.py                # Centralized configuration
├── requirements.txt
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/abhishekanand16/Predictive-Maintenance-AI
cd Predictive-Maintenance-AI
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download and place dataset:**
   - Download NASA C-MAPSS dataset (FD001 subset)
   - Place the following files in `data/raw/`:
     - `train_FD001.txt`
     - `test_FD001.txt`
     - `RUL_FD001.txt`

## 📝 Usage

### 1. Data Preprocessing

```bash
python src/preprocess.py
```

This will:
- Load and clean the training data
- Remove constant sensors
- Calculate RUL and failure labels
- Split data by engine_id (train/validation)
- Normalize sensors and save scaler

### 2. Feature Engineering

Run the feature engineering notebook or use the module:

```python
from src.features import create_all_features
from src.preprocess import load_train_data

df = load_train_data()
df_features = create_all_features(df)
```

Or run the notebook:
```bash
jupyter notebook notebooks/02_feature_engineering.ipynb
```

### 3. Model Training

**Train failure prediction model:**
```bash
python src/train_failure.py
```

**Train RUL prediction model:**
```bash
python src/train_rul.py
```

Both scripts will:
- Load processed and feature-engineered data
- Train baseline and final models
- Evaluate on validation set
- Save models and log to MLflow

### 4. Model Explainability

Generate SHAP explanations:
```python
from src.explain import explain_classifier, explain_regressor
# See src/explain.py for usage examples
```

### 5. Interactive Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

The dashboard allows you to:
- Upload new sensor data
- View failure probability predictions
- View RUL estimates
- Visualize sensor degradation trends
- Explore SHAP explanations

## 🧠 Model Architecture

### Classification Model (Failure Prediction)

**Models Implemented:**
1. **Logistic Regression** (baseline)
2. **Random Forest Classifier** (intermediate)
3. **XGBoost Classifier** (final)

**Class Imbalance Handling:**
- Primary: Class weights (`scale_pos_weight` in XGBoost)
- Optional: SMOTE (experimental, not recommended for time-series)

**Evaluation Metrics:**
- Precision, Recall, F1-Score
- Precision-Recall AUC (PR-AUC)
- ROC-AUC

### Regression Model (RUL Estimation)

**Models Implemented:**
1. **Linear Regression** (baseline)
2. **XGBoost Regressor** (final)

**Evaluation Metrics:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### Feature Engineering

Time-series features created per engine (grouped by `engine_id` to avoid leakage):

| Feature Type | Description | Window Sizes |
|--------------|-------------|--------------|
| Rolling Mean | Average over rolling window | 5, 10, 20 cycles |
| Rolling Std | Standard deviation over window | 5, 10, 20 cycles |
| Rolling Min/Max | Min and max over window | 5, 10, 20 cycles |
| Rolling Slope | Linear trend over window | 5, 10, 20 cycles |
| First-order Diff | sensor[t] - sensor[t-1] | - |

**Important:** All rolling features are computed using `groupby(engine_id)` to prevent cross-engine data leakage.

## ⚙️ Configuration

All hyperparameters and settings are centralized in `config.py`:

- Data paths and file names
- Model hyperparameters (XGBoost, Random Forest, Logistic Regression)
- Feature engineering parameters (rolling windows)
- Failure threshold (N=30 cycles)
- Train/validation split ratio
- Random seed for reproducibility

See `config.py` for all configurable parameters.

## 📈 Results

### Model Performance

Results are saved to `results/` directory:
- `classification_results.csv` - Classification metrics for all models
- `regression_results.csv` - Regression metrics for all models
- Evaluation plots (confusion matrices, PR curves, scatter plots)

### Experiment Tracking

All experiments are logged to MLflow:
- Model parameters
- Evaluation metrics
- Model artifacts
- SHAP plots

View results:
```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view the MLflow UI.

## 🔍 Explainability

The system uses SHAP (SHapley Additive exPlanations) to explain model predictions:

- **Global explanations**: Feature importance across all predictions
- **Local explanations**: Waterfall plots for individual predictions
- **Top features**: Identifies which sensors contribute most to predictions

SHAP explanations help engineers understand:
- Which sensors indicate degradation
- How sensor changes influence failure probability
- Why a specific engine is predicted to fail

## 🎨 Dashboard Features

The Streamlit dashboard provides:

1. **Predictions Page**
   - Upload sensor data CSV
   - Select engine ID
   - View failure probability (gauge chart)
   - View predicted RUL
   - SHAP waterfall plot for explanation
   - Sensor degradation trends

2. **Sensor Analysis Page**
   - Visualize individual sensor trends
   - Compare sensors across engines
   - Statistical summaries

3. **Model Info Page**
   - Model architecture details
   - Feature descriptions
   - Configuration summary

## 🧪 Reproducibility

- Fixed random seed (42) for all random operations
- Deterministic data splitting by engine_id
- Saved scalers for consistent preprocessing
- MLflow tracking for experiment reproducibility

## 🚧 Future Enhancements

- [ ] Real-time sensor ingestion pipeline
- [ ] Alerting system for high-risk engines
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Multi-asset scaling (handle multiple engine types)
- [ ] Advanced deep learning models (LSTM, Transformer)
- [ ] Automated hyperparameter tuning
- [ ] Model versioning and A/B testing

## 📚 References

- [NASA C-MAPSS Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## 📄 License

This project is for educational and portfolio purposes.

## 👤 Author

Built as a comprehensive machine learning portfolio project demonstrating:
- End-to-end ML pipeline development
- Time-series feature engineering
- Handling class imbalance
- Model explainability
- Production-ready code structure
- Interactive dashboard development

---

**Note:** This project follows industry best practices including proper train/validation splitting, avoiding data leakage, and maintaining reproducibility. The test set (`test_FD001.txt`) is reserved for final evaluation only and is never used during training or validation.
