import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objs as go
import pandas as pd
import json

# Configure the page structure
st.set_page_config(
    page_title="Chennai Water AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("💧 AI-Powered Water Demand & Storage Forecasting")
st.markdown("### Case Study: Chennai Poondi Reservoir System")
MODEL_H5 = 'poondi_multi_variate_lstm_model.h5'
SCALER_PKL = 'multi_variate_scaler.pkl'
METRICS_JSON = 'model_metrics.json'

try:
    model = tf.keras.models.load_model(MODEL_H5)
    scaler = joblib.load(SCALER_PKL)
    with open(METRICS_JSON, 'r') as f:
        metrics = json.load(f)
    model_loaded = True
except Exception as e:
    st.error(f"Error loading files: {e}. Please run train_model.py first.")
    model_loaded = False
if model_loaded:
    look_back = metrics['look_back']
    n_features = metrics['n_features']

    # --- Sidebar: Scenario Inputs ---
    st.sidebar.header("Scenario Settings")
    
    # Slider 1: Current Water Level (Default: 2000 MCFT)
    current_level = st.sidebar.slider("Current Water Level (MCFT)", 0, 3231, 2000)
    
    # Slider 2: Rainfall Forecast (Default: 15 mm - moderate rain)
    rainfall_forecast = st.sidebar.slider("Rainfall Forecast (mm/day)", 0, 100, 15)
    
    # Slider 3: Consumption Rate (Default: 750 MLD - typical Chennai usage)
    consumption_rate = st.sidebar.slider("City Consumption (MLD)", 500, 900, 750)
    # --- AI Prediction Engine ---
    
    # 1. Create Synthetic History (Sequence of 60 days)
    level_history = np.full(look_back, current_level)
    rain_history = np.full(look_back, rainfall_forecast)

    # 2. Structure Data for Scaler (Columns: POONDI, TOTAL_RAIN)
    input_seq_df = pd.DataFrame({'POONDI': level_history, 'TOTAL_RAIN': rain_history})
    input_seq_scaled = scaler.transform(input_seq_df.values)
    
    # 3. Reshape for LSTM (1 Sample, 60 Days, 2 Features)
    input_seq_reshaped = input_seq_scaled[-look_back:].reshape(1, look_back, n_features)

    # 4. Generate Prediction
    prediction_scaled = model.predict(input_seq_reshaped)

    # 5. Inverse Transform (Convert 0-1 back to MCFT)
    dummy_array = np.zeros((1, n_features))
    dummy_array[:, 0] = prediction_scaled.flatten()
    prediction_inverse = scaler.inverse_transform(dummy_array)[:, 0]
    
    predicted_next_day_level = prediction_inverse[0]
    # --- Visualization Logic ---
    
    # Calculate 30-Day Projection
    days = list(range(1, 31))
    daily_change = predicted_next_day_level - current_level
    # Adjust drop rate based on consumption slider (Simplified Physics)
    consumption_impact = (consumption_rate - 700) * 0.05
    
    trajectory = []
    val = current_level
    for _ in days:
        val = max(0, val + daily_change - consumption_impact)
        trajectory.append(val)
    
    capacity = 3231 

    # Create Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=trajectory, mode='lines', name='Forecast', line=dict(color='#00CC96', width=3)))
    fig.add_hline(y=capacity * 0.1, line_dash="dot", line_color="red", annotation_text="Critical Level")
    fig.add_hline(y=capacity * 0.5, line_dash="dot", line_color="orange", annotation_text="Warning Level")

    fig.update_layout(title="30-Day Reservoir Forecast", xaxis_title="Days from Now", yaxis_title="Water Level (MCFT)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- Risk & Stats Section ---
    col1, col2, col3 = st.columns(3)
    
    final_level_pct = (trajectory[-1] / capacity) * 100
    
    with col1:
        st.metric("Projected Level (30 Days)", f"{trajectory[-1]:.0f} MCFT")
    with col2:
        if final_level_pct < 10:
            st.error(f"Risk: CRITICAL ({final_level_pct:.1f}%)")
        elif final_level_pct < 50:
            st.warning(f"Risk: MODERATE ({final_level_pct:.1f}%)")
        else:
            st.success(f"Risk: SAFE ({final_level_pct:.1f}%)")
    with col3:
        st.metric("AI Prediction (Next Day)", f"{predicted_next_day_level:.0f} MCFT")

    st.markdown("---")
    st.markdown("### 🔬 Model Performance Metrics (Research)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test RMSE", f"{metrics['test_rmse']} MCFT")
    c2.metric("Test MAE", f"{metrics['test_mae']} MCFT")
    c3.metric("Model Features", f"{metrics['n_features']} (Multi-Variate)")