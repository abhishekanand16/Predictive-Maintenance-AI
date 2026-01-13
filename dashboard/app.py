"""
Streamlit Dashboard for Predictive Maintenance AI.

Interactive dashboard for failure prediction and RUL estimation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import joblib
import shap

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.preprocess import (
    load_scaler,
    normalize_sensors,
    calculate_rul_labels,
    create_failure_labels
)
from src.features import create_all_features, get_feature_columns
from src.explain import explain_single_prediction, get_top_features

# Page configuration
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and scaler."""
    try:
        scaler = load_scaler(config.MODELS_DIR / "scaler.joblib")
        clf_model = joblib.load(config.MODELS_DIR / "xgboost_classifier.joblib")
        reg_model = joblib.load(config.MODELS_DIR / "xgboost_regressor.joblib")
        return scaler, clf_model, reg_model
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please train models first. Error: {e}")
        return None, None, None


def preprocess_uploaded_data(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Preprocess uploaded data for prediction."""
    # Ensure required columns exist
    required_cols = config.COLUMN_NAMES
    if not all(col in df.columns for col in required_cols):
        st.error(f"Uploaded data must contain columns: {required_cols}")
        return None
    
    # Calculate RUL if not present
    if "RUL" not in df.columns:
        df = calculate_rul_labels(df)
    
    # Normalize
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    op_cols = [col for col in df.columns if col.startswith("op_setting_")]
    feature_cols = sensor_cols + op_cols
    
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler.transform(df[feature_cols])
    
    return df_normalized


def main():
    """Main dashboard application."""
    st.markdown('<h1 class="main-header">🔧 Predictive Maintenance AI Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load models
    scaler, clf_model, reg_model = load_models()
    
    if scaler is None or clf_model is None or reg_model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Predictions", "Sensor Analysis", "Model Info"]
    )
    
    if page == "Predictions":
        predictions_page(scaler, clf_model, reg_model)
    elif page == "Sensor Analysis":
        sensor_analysis_page(scaler, clf_model, reg_model)
    elif page == "Model Info":
        model_info_page()


def predictions_page(scaler, clf_model, reg_model):
    """Main predictions page."""
    st.header("🚀 Failure Prediction & RUL Estimation")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Sensor Data (CSV)",
        type=["csv"],
        help="Upload a CSV file with sensor data. Must include engine_id, cycle, and sensor columns."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Preprocess
            df_processed = preprocess_uploaded_data(df, scaler)
            if df_processed is None:
                st.stop()
            
            # Create features
            with st.spinner("Creating features..."):
                df_features = create_all_features(df_processed)
            
            # Engine selector
            engine_ids = sorted(df_features["engine_id"].unique())
            selected_engine = st.selectbox("Select Engine ID", engine_ids)
            
            # Get latest cycle for selected engine
            engine_data = df_features[df_features["engine_id"] == selected_engine].sort_values("cycle")
            if len(engine_data) == 0:
                st.warning(f"No data found for engine {selected_engine}")
                st.stop()
            
            latest_cycle = engine_data.iloc[-1]
            
            # Prepare features for prediction
            feature_cols = get_feature_columns(df_features)
            X_pred = latest_cycle[feature_cols].values.reshape(1, -1)
            
            # Make predictions
            failure_proba = clf_model.predict_proba(X_pred)[0, 1]
            rul_pred = max(0, reg_model.predict(X_pred)[0])
            
            # Display predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Failure Probability", f"{failure_proba:.1%}")
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=failure_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Failure Risk"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.metric("Predicted RUL", f"{rul_pred:.1f} cycles")
                # RUL bar
                max_rul = max(100, rul_pred * 1.2)
                fig_rul = go.Figure(go.Bar(
                    x=[rul_pred],
                    y=["RUL"],
                    orientation='h',
                    marker_color='blue' if rul_pred > config.FAILURE_THRESHOLD else 'red',
                    text=[f"{rul_pred:.1f}"],
                    textposition='auto'
                ))
                fig_rul.update_layout(
                    xaxis_title="Remaining Useful Life (Cycles)",
                    height=200,
                    xaxis_range=[0, max_rul]
                )
                fig_rul.add_vline(
                    x=config.FAILURE_THRESHOLD,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({config.FAILURE_THRESHOLD})"
                )
                st.plotly_chart(fig_rul, use_container_width=True)
            
            with col3:
                current_cycle = latest_cycle["cycle"]
                st.metric("Current Cycle", f"{current_cycle}")
                st.metric("Engine Status", 
                         "⚠️ At Risk" if failure_proba > 0.5 else "✅ Healthy")
            
            # SHAP Explanation
            st.subheader("🔍 Model Explanation (SHAP)")
            with st.spinner("Generating SHAP explanation..."):
                try:
                    shap_explanation = explain_single_prediction(
                        clf_model,
                        X_pred[0],
                        feature_cols,
                        task_type="classification"
                    )
                    
                    # Get top features
                    top_features = get_top_features(shap_explanation, top_k=10)
                    
                    # Waterfall plot
                    fig_waterfall = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=["relative"] * len(top_features) + ["total"],
                        x=top_features["feature"].tolist() + ["Prediction"],
                        textposition="outside",
                        text=[f"{val:.3f}" for val in top_features["mean_abs_shap"].values] + [f"{failure_proba:.3f}"],
                        y=top_features["mean_abs_shap"].values.tolist() + [failure_proba],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    fig_waterfall.update_layout(
                        title="Top 10 Features Contributing to Prediction",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                    
                    # Feature importance table
                    st.dataframe(top_features, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate SHAP explanation: {e}")
            
            # Sensor trends
            st.subheader("📈 Sensor Degradation Trends")
            sensor_cols = [col for col in engine_data.columns if col.startswith("sensor_")]
            selected_sensors = st.multiselect(
                "Select Sensors to Display",
                sensor_cols[:10],  # Limit to first 10 for performance
                default=sensor_cols[:3]
            )
            
            if selected_sensors:
                fig_trends = go.Figure()
                for sensor in selected_sensors:
                    fig_trends.add_trace(go.Scatter(
                        x=engine_data["cycle"],
                        y=engine_data[sensor],
                        mode='lines',
                        name=sensor,
                        line=dict(width=2)
                    ))
                fig_trends.update_layout(
                    title="Sensor Values Over Time",
                    xaxis_title="Cycle",
                    yaxis_title="Sensor Value",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_trends, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            st.exception(e)
    
    else:
        st.info("👆 Please upload a CSV file to get started.")


def sensor_analysis_page(scaler, clf_model, reg_model):
    """Sensor analysis page."""
    st.header("📊 Sensor Analysis")
    
    # Try to load sample data
    try:
        df = pd.read_pickle(config.DATA_PROCESSED_DIR / "train_processed.pkl")
        st.success("✅ Loaded sample training data")
        
        # Engine selector
        engine_ids = sorted(df["engine_id"].unique()[:20])  # Limit for performance
        selected_engine = st.selectbox("Select Engine ID", engine_ids)
        
        engine_data = df[df["engine_id"] == selected_engine].sort_values("cycle")
        
        # Sensor selector
        sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
        selected_sensor = st.selectbox("Select Sensor", sensor_cols)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=engine_data["cycle"],
            y=engine_data[selected_sensor],
            mode='lines+markers',
            name=selected_sensor,
            line=dict(width=2, color='blue')
        ))
        
        # Add RUL overlay if available
        if "RUL" in engine_data.columns:
            fig.add_trace(go.Scatter(
                x=engine_data["cycle"],
                y=engine_data["RUL"] * (engine_data[selected_sensor].max() / engine_data["RUL"].max()),
                mode='lines',
                name="RUL (scaled)",
                line=dict(width=2, color='red', dash='dash'),
                yaxis='y2'
            ))
            fig.update_layout(yaxis2=dict(
                title="RUL (scaled)",
                overlaying='y',
                side='right'
            ))
        
        fig.update_layout(
            title=f"{selected_sensor} Over Time - Engine {selected_engine}",
            xaxis_title="Cycle",
            yaxis_title="Sensor Value",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{engine_data[selected_sensor].mean():.2f}")
        with col2:
            st.metric("Std", f"{engine_data[selected_sensor].std():.2f}")
        with col3:
            st.metric("Min", f"{engine_data[selected_sensor].min():.2f}")
        with col4:
            st.metric("Max", f"{engine_data[selected_sensor].max():.2f}")
    
    except FileNotFoundError:
        st.warning("Training data not found. Please run preprocessing first.")


def model_info_page():
    """Model information page."""
    st.header("ℹ️ Model Information")
    
    st.subheader("Model Architecture")
    st.markdown("""
    ### Classification Model (Failure Prediction)
    - **Algorithm**: XGBoost Classifier
    - **Objective**: Predict if engine will fail within 30 cycles
    - **Metrics**: Precision, Recall, F1-Score, PR-AUC
    
    ### Regression Model (RUL Estimation)
    - **Algorithm**: XGBoost Regressor
    - **Objective**: Estimate Remaining Useful Life in cycles
    - **Metrics**: RMSE, MAE
    """)
    
    st.subheader("Features")
    st.markdown("""
    The models use engineered time-series features:
    - Rolling statistics (mean, std, min, max) with windows [5, 10, 20]
    - Rolling slope (trend)
    - First-order differences
    - All features computed per engine to avoid data leakage
    """)
    
    st.subheader("Configuration")
    st.code(f"""
    Failure Threshold: {config.FAILURE_THRESHOLD} cycles
    Rolling Windows: {config.ROLLING_WINDOWS}
    Train/Val Split: {config.TRAIN_VAL_SPLIT_RATIO*100:.0f}% / {(1-config.TRAIN_VAL_SPLIT_RATIO)*100:.0f}%
    Random Seed: {config.RANDOM_SEED}
    """, language="python")


if __name__ == "__main__":
    main()
