import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objs as go
import pandas as pd
import json

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Water Demand AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title (Generic & Professional)
st.title("💧 AI-Powered Water Demand Forecasting")
st.markdown("### Intelligent Reservoir Management System")

# --- 2. Load the AI Model & Artifacts ---
# Note: We still load the specific files you trained, but the UI will hide the names.
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
    st.error(f"System Error: Model artifacts not found. ({e})")
    model_loaded = False

# --- 3. Sidebar Controls ---
if model_loaded:
    look_back = metrics['look_back']
    n_features = metrics['n_features']

    st.sidebar.header("Forecasting Parameters")
    
    # Renamed generic labels
    current_level = st.sidebar.slider("Current Reservoir Level (MCFT)", 0, 3231, 2000)
    rainfall_forecast = st.sidebar.slider("Predicted Rainfall (mm/day)", 0, 100, 15)
    consumption_rate = st.sidebar.slider("Projected Demand (MLD)", 500, 900, 750)
    
    # --- 4. AI Prediction Engine ---
    
    # Create Synthetic History based on current sliders
    level_history = np.full(look_back, current_level)
    rain_history = np.full(look_back, rainfall_forecast)

    # Structure Data for Scaler
    input_seq_df = pd.DataFrame({'POONDI': level_history, 'TOTAL_RAIN': rain_history})
    input_seq_scaled = scaler.transform(input_seq_df.values)
    
    # Reshape for LSTM
    input_seq_reshaped = input_seq_scaled[-look_back:].reshape(1, look_back, n_features)

    # Generate Prediction
    prediction_scaled = model.predict(input_seq_reshaped)

    # Inverse Transform to get MCFT
    dummy_array = np.zeros((1, n_features))
    dummy_array[:, 0] = prediction_scaled.flatten()
    prediction_inverse = scaler.inverse_transform(dummy_array)[:, 0]
    
    predicted_next_day_level = prediction_inverse[0]

    # --- 5. Visualization & Decision Support ---
    
    # Calculate 30-Day Projection Path
    days = list(range(1, 31))
    
    # Demand/Supply Logic
    daily_change = predicted_next_day_level - current_level
    # Adjusting forecast based on the 'Projected Demand' slider
    consumption_impact = (consumption_rate - 700) * 0.05
    
    trajectory = []
    val = current_level
    for _ in days:
        val = max(0, val + daily_change - consumption_impact)
        trajectory.append(val)
    
    # Define Thresholds (Generic Capacity)
    capacity = 3231 

    # Create Plotly Chart with Proper Legend
    fig = go.Figure()

    # The Forecast Trace
    fig.add_trace(go.Scatter(
        x=days, 
        y=trajectory, 
        mode='lines', 
        name='AI Forecast Path', 
        line=dict(color='#00CC96', width=4)
    ))

    # The Warning Trace (Legend entry)
    fig.add_trace(go.Scatter(
        x=[days[0], days[-1]], 
        y=[capacity * 0.5, capacity * 0.5], 
        mode='lines', 
        name='Warning Threshold (50%)', 
        line=dict(color='orange', width=2, dash='dot')
    ))

    # The Critical Trace (Legend entry)
    fig.add_trace(go.Scatter(
        x=[days[0], days[-1]], 
        y=[capacity * 0.1, capacity * 0.1], 
        mode='lines', 
        name='Critical Threshold (10%)', 
        line=dict(color='red', width=2, dash='dot')
    ))

    fig.update_layout(
        title="30-Day Water Storage Forecast", 
        xaxis_title="Forecast Horizon (Days)", 
        yaxis_title="Water Volume (MCFT)", 
        height=450,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Key Metrics Section ---
    col1, col2, col3 = st.columns(3)
    
    final_level_pct = (trajectory[-1] / capacity) * 100
    
    with col1:
        st.metric("Projected Storage (30 Days)", f"{trajectory[-1]:.0f} MCFT")
    
    with col2:
        if final_level_pct < 10:
            st.error(f"Status: CRITICAL SHORTAGE ({final_level_pct:.1f}%)")
        elif final_level_pct < 50:
            st.warning(f"Status: LOW SUPPLY ({final_level_pct:.1f}%)")
        else:
            st.success(f"Status: ADEQUATE SUPPLY ({final_level_pct:.1f}%)")
            
    with col3:
        st.metric("AI Next-Day Prediction", f"{predicted_next_day_level:.0f} MCFT")