import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from tensorflow.keras.models import load_model

# --- Page Config ---
st.set_page_config(page_title="Water AI", layout="wide")

# --- 1. Load Artifacts ---
@st.cache_resource
def load_brain():
    # Load the trained model and the scaler used for normalization
    model = load_model('water_demand_lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# --- 2. Load & Prep Data ---
@st.cache_data
def get_data():
    df = pd.read_csv('Aquifer_Petrignano.csv')
    
    # REPLICATE SENSOR FUSION FROM TRAINING
    df['TEMP_AVG'] = df[['Temperature_Bastia_Umbra', 'Temperature_Petrignano']].mean(axis=1)
    
    df = df.rename(columns={
        'Date': 'DATE',
        'Rainfall_Bastia_Umbra': 'RAIN',
        'Depth_to_Groundwater_P25': 'DEPTH',
        'Volume_C10_Petrignano': 'DEMAND'
    })
    
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
    df = df.sort_values('DATE')
    df['DEMAND'] = df['DEMAND'].abs()
    
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(0)
    df = df.set_index('DATE')
    
    # Return exactly the columns model expects: [Target, Rain, Temp, Depth]
    return df[['DEMAND', 'RAIN', 'TEMP_AVG', 'DEPTH']]

try:
    model, scaler = load_brain()
    df = get_data()
except Exception as e:
    st.error(f"🚨 Error: {e}. Please run the training notebook first!")
    st.stop()

# --- 3. Sidebar Controls ---
st.sidebar.title("⚙️ Control Panel")

days = st.sidebar.slider("Forecast Days", 7, 90, 30)

st.sidebar.subheader("Scenario Simulation")
avg_rain = df['RAIN'].mean()
avg_temp = df['TEMP_AVG'].mean()

sim_rain = st.sidebar.slider("Rainfall (mm)", 0.0, 20.0, float(avg_rain))
sim_temp = st.sidebar.slider("Temperature (°C)", -5.0, 40.0, float(avg_temp))

st.sidebar.subheader("Reservoir")
# Realistic capacity for 30k demand/day = ~5-6M m3
capacity = st.sidebar.number_input("Reservoir Capacity (m³)", value=5000000, step=500000)
current_lvl = st.sidebar.number_input("Current Level (m³)", value=3500000, step=100000)

# --- 4. Forecast Engine ---
def forecast_future(model, scaler, last_sequence, n_days, rain, temp):
    predictions = []
    curr_seq = last_sequence.copy()
    last_depth = df['DEPTH'].iloc[-1]
    
    # Get scale ranges to normalize user input manually
    # scaler.data_min_ stores [min_demand, min_rain, min_temp, min_depth]
    min_vals = scaler.data_min_
    max_vals = scaler.data_max_
    
    for _ in range(n_days):
        # 1. Predict Scaled Demand
        input_seq = curr_seq.reshape(1, 60, 4)
        pred_demand_scaled = model.predict(input_seq, verbose=0)[0][0]
        
        # 2. Scale User Inputs (Rain/Temp) 
        # Formula: (Value - Min) / (Max - Min)
        s_rain = (rain - min_vals[1]) / (max_vals[1] - min_vals[1])
        s_temp = (temp - min_vals[2]) / (max_vals[2] - min_vals[2])
        s_depth = (last_depth - min_vals[3]) / (max_vals[3] - min_vals[3]) # Constant depth assumption
        
        # 3. Create New Row [Demand, Rain, Temp, Depth]
        new_row = np.array([pred_demand_scaled, s_rain, s_temp, s_depth])
        
        # 4. Update Sequence
        predictions.append(new_row)
        curr_seq = np.append(curr_seq[1:], [new_row], axis=0)
        
    return np.array(predictions)

# Run Forecast
last_60_days_scaled = scaler.transform(df.values)[-60:]
with st.spinner("🤖 AI is thinking..."):
    forecast_scaled = forecast_future(model, scaler, last_60_days_scaled, days, sim_rain, sim_temp)

# Inverse Transform to get real values
forecast_final = scaler.inverse_transform(forecast_scaled)[:, 0] # Column 0 is Demand

# --- 5. Dashboard UI ---
st.title("💧 Water Demand Forecasting System")

# KPI Row
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Current Demand", f"{df['DEMAND'].iloc[-1]:,.0f} m³")
kpi2.metric("Avg Temp (Fusion)", f"{df['TEMP_AVG'].iloc[-1]:.1f} °C")
kpi3.metric("Aquifer Depth", f"{df['DEPTH'].iloc[-1]:.2f} m")

# Chart
dates = pd.date_range(start=df.index[-1], periods=days + 1)[1:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-180:], y=df['DEMAND'].iloc[-180:], name='Historical', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=dates, y=forecast_final, name='AI Forecast', line=dict(color='red', dash='dash')))
fig.update_layout(title="Demand Projections", xaxis_title="Date", yaxis_title="Volume (m³)", height=450)
st.plotly_chart(fig, use_container_width=True)

# Storage Logic
total_draw = np.sum(forecast_final)
end_level = current_lvl - total_draw
pct = (end_level / capacity) * 100

st.subheader("⚠️ Storage Impact Analysis")
c1, c2 = st.columns([1, 2])

c1.metric("Forecasted Drawdown", f"{total_draw:,.0f} m³", delta=f"-{total_draw/current_lvl:.1%}")

if end_level < 0:
    c2.error(f"🚨 CRITICAL FAILURE: Water runout in {days} days. Deficit: {abs(end_level):,.0f} m³")
    c2.progress(0)
else:
    c2.info(f"Projected Storage: {end_level:,.0f} / {capacity:,.0f} m³ ({pct:.1f}%)")
    c2.progress(min(1.0, pct/100))