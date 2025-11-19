import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from tensorflow.keras.models import load_model

# --- Configuration ---
st.set_page_config(page_title="Water Demand AI", layout="wide")

# --- 1. Load Data & Models ---
@st.cache_resource
def load_artifacts():
    model = load_model('water_demand_lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('Aquifer_Petrignano.csv')
    df = df.rename(columns={
        'Date': 'DATE',
        'Rainfall_Bastia_Umbra': 'RAIN',
        'Depth_to_Groundwater_P25': 'DEPTH',
        'Temperature_Bastia_Umbra': 'TEMP',
        'Volume_C10_Petrignano': 'DEMAND'
    })
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
    df = df.sort_values('DATE')
    # Handle negative volume (extraction) as positive demand
    df['DEMAND'] = df['DEMAND'].abs()
    df = df.dropna(subset=['DEMAND'])
    df = df.interpolate(method='linear', limit_direction='forward')
    df = df.fillna(0)
    df = df.set_index('DATE')
    return df[['DEMAND', 'RAIN', 'TEMP', 'DEPTH']]

try:
    model, scaler = load_artifacts()
    df = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- 2. Sidebar: Controls ---
st.sidebar.header("⚙️ Simulation Controls")

st.sidebar.subheader("Forecast Parameters")
days_to_predict = st.sidebar.slider("Days to Forecast", 7, 90, 30)

st.sidebar.subheader("Weather Scenarios")
future_rain = st.sidebar.slider("Simulated Rainfall (mm)", 0.0, 20.0, float(df['RAIN'].mean()))
future_temp = st.sidebar.slider("Simulated Temp (°C)", -5.0, 40.0, float(df['TEMP'].mean()))

st.sidebar.subheader("Infrastructure Specs")
# Set default capacity to 5 Million m3 (approx 6 months of supply)
reservoir_capacity = st.sidebar.number_input("Max Reservoir Capacity (m³)", value=5000000, step=100000)
# Default current level to 80% of capacity
initial_level = st.sidebar.number_input("Current Storage Level (m³)", value=int(reservoir_capacity * 0.8), step=100000)

# --- 3. Main Dashboard ---
st.title("💧 AI-Powered Water Demand Forecasting")

# Metrics
current_demand = df['DEMAND'].iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Last Recorded Demand", f"{current_demand:,.0f} m³")
col2.metric("Aquifer Depth", f"{df['DEPTH'].iloc[-1]:.2f} m")
col3.metric("Simulated Rainfall", f"{future_rain:.1f} mm")

# --- 4. Forecasting Logic ---
def make_forecast(model, scaler, last_sequence, days, rain, temp):
    forecast = []
    curr_seq = last_sequence.copy()
    last_depth = df['DEPTH'].iloc[-1]
    
    # Get min/max from original df for scaling manual inputs
    rain_min, rain_max = df['RAIN'].min(), df['RAIN'].max()
    temp_min, temp_max = df['TEMP'].min(), df['TEMP'].max()
    depth_min, depth_max = df['DEPTH'].min(), df['DEPTH'].max()

    for _ in range(days):
        input_data = curr_seq.reshape(1, 60, 4)
        pred_scaled = model.predict(input_data, verbose=0)[0][0]
        
        # Scale user inputs
        s_rain = (rain - rain_min) / (rain_max - rain_min)
        s_temp = (temp - temp_min) / (temp_max - temp_min)
        s_depth = (last_depth - depth_min) / (depth_max - depth_min)
        
        next_step = np.array([pred_scaled, s_rain, s_temp, s_depth])
        forecast.append(next_step)
        curr_seq = np.append(curr_seq[1:], [next_step], axis=0)
        
    return np.array(forecast)

# Prepare Data
scaled_dataset = scaler.transform(df.values)
last_60_days = scaled_dataset[-60:]

with st.spinner('Running AI Model...'):
    forecast_scaled = make_forecast(model, scaler, last_60_days, days_to_predict, future_rain, future_temp)

# Invert scaling
forecast_values = scaler.inverse_transform(forecast_scaled)
predicted_demand = forecast_values[:, 0]

# --- 5. Visualization ---
fig = go.Figure()
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1)[1:]

fig.add_trace(go.Scatter(
    x=df.index[-90:], y=df['DEMAND'].iloc[-90:],
    mode='lines', name='Historical', line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=future_dates, y=predicted_demand,
    mode='lines', name='Forecast', line=dict(color='red', dash='dash')
))
fig.update_layout(title='Demand Forecast', xaxis_title='Date', yaxis_title='Volume (m³)', height=400)
st.plotly_chart(fig, use_container_width=True)

# --- 6. Storage Analysis ---
st.subheader("⚠️ Storage Risk Analysis")

total_demand = np.sum(predicted_demand)
# Calculate Net Storage (Assuming 0 inflow for worst-case scenario)
final_storage = initial_level - total_demand
percent_full = (final_storage / reservoir_capacity) * 100

col_a, col_b = st.columns([1, 2])

with col_a:
    st.metric("Total Forecasted Demand", f"{total_demand:,.0f} m³")
    if final_storage < 0:
        st.metric("Storage Deficit", f"{abs(final_storage):,.0f} m³", delta_color="inverse")
    else:
        st.metric("Ending Storage", f"{final_storage:,.0f} m³")

with col_b:
    st.write(f"**Reservoir Status ({days_to_predict} Days)**")
    if final_storage < 0:
        st.error(f"🚨 CRITICAL: Water supply depleted! Deficit of {abs(final_storage):,.0f} m³ predicted.")
        st.progress(0)
    else:
        st.progress(min(1.0, max(0.0, final_storage / reservoir_capacity)))
        st.write(f"Reservoir will be **{percent_full:.1f}%** full.")
        
        if percent_full < 20:
            st.warning("Warning: Storage levels approaching critical low (<20%).")
        else:
            st.success("Storage levels adequate for forecasted demand.")