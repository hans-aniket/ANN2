import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from tensorflow.keras.models import load_model

# --- Page Config ---
st.set_page_config(page_title="Water Demand AI", layout="wide")

# --- 1. Load Artifacts ---
@st.cache_resource
def load_brain():
    model = load_model('water_demand_lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# --- 2. Load & Prep Data ---
@st.cache_data
def get_data():
    df = pd.read_csv('Aquifer_Petrignano.csv')
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
    return df[['DEMAND', 'RAIN', 'TEMP_AVG', 'DEPTH']]

try:
    model, scaler = load_brain()
    df = get_data()
except Exception as e:
    st.error(f"🚨 Error: {e}")
    st.stop()

# --- 3. Sidebar Controls ---
st.sidebar.title("⚙️ Simulation Controls")

days = st.sidebar.slider("Forecast Horizon", 7, 90, 30)

st.sidebar.subheader("Weather Scenario")
# Defaults based on recent history
recent_rain = float(df['RAIN'].tail(30).mean())
recent_temp = float(df['TEMP_AVG'].tail(30).mean())

sim_rain = st.sidebar.slider("Avg Rainfall (mm)", 0.0, 20.0, recent_rain)
sim_temp = st.sidebar.slider("Avg Temperature (°C)", -5.0, 40.0, recent_temp)

st.sidebar.subheader("Infrastructure")
capacity = st.sidebar.number_input("Reservoir Capacity (m³)", value=5000000, step=500000)
current_lvl = st.sidebar.number_input("Current Level (m³)", value=3500000, step=100000)

# --- 4. Forecast Logic (Calculated BEFORE UI) ---
def forecast_future(model, scaler, last_sequence, n_days, rain, temp):
    predictions = []
    curr_seq = last_sequence.copy()
    last_depth = df['DEPTH'].iloc[-1]
    
    min_vals = scaler.data_min_
    max_vals = scaler.data_max_
    
    for _ in range(n_days):
        input_seq = curr_seq.reshape(1, 60, 4)
        pred_demand_scaled = model.predict(input_seq, verbose=0)[0][0]
        
        s_rain = (rain - min_vals[1]) / (max_vals[1] - min_vals[1])
        s_temp = (temp - min_vals[2]) / (max_vals[2] - min_vals[2])
        s_depth = (last_depth - min_vals[3]) / (max_vals[3] - min_vals[3])
        
        new_row = np.array([pred_demand_scaled, s_rain, s_temp, s_depth])
        predictions.append(new_row)
        curr_seq = np.append(curr_seq[1:], [new_row], axis=0)
        
    return np.array(predictions)

# Run Forecast Immediately
last_60_days_scaled = scaler.transform(df.values)[-60:]
forecast_scaled = forecast_future(model, scaler, last_60_days_scaled, days, sim_rain, sim_temp)
forecast_final = scaler.inverse_transform(forecast_scaled)[:, 0] # Extract Demand

# --- 5. Dashboard UI (Now using Dynamic Data) ---
st.title("💧 Water Demand Forecasting System")

# Calculate Dynamic Metrics based on the Forecast we just ran
avg_pred_demand = np.mean(forecast_final)
total_pred_demand = np.sum(forecast_final)
peak_pred_demand = np.max(forecast_final)
historical_avg = df['DEMAND'].mean()

# Dynamic KPI Row
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Forecasted Avg Demand", 
    f"{avg_pred_demand:,.0f} m³", 
    delta=f"{((avg_pred_demand - historical_avg)/historical_avg)*100:.1f}% vs Hist",
    delta_color="inverse" # Red if demand goes UP
)

col2.metric(
    "Peak Demand Day", 
    f"{peak_pred_demand:,.0f} m³",
    "Max stress point"
)

col3.metric(
    "Total Water Required", 
    f"{total_pred_demand:,.0f} m³",
    f"Over next {days} days"
)

col4.metric(
    "Scenario Inputs",
    f"{sim_temp}°C / {sim_rain}mm",
    "Simulation Settings"
)

# --- 6. Visualization ---
today = pd.Timestamp.now().normalize()
future_dates = pd.date_range(start=today, periods=days + 1)[1:]
shift_delta = today - df.index[-1]
shifted_history_index = df.index[-180:] + shift_delta

fig = go.Figure()
fig.add_trace(go.Scatter(x=shifted_history_index, y=df['DEMAND'].iloc[-180:], name='Historical Context', line=dict(color='rgba(31, 119, 180, 0.5)')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_final, name='Simulated Forecast', line=dict(color='#ff7f0e', width=3)))

fig.update_layout(title="Demand Simulation", xaxis_title="Date", yaxis_title="Volume (m³)", height=450, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- 7. Storage Logic ---
end_level = current_lvl - total_pred_demand
pct = (end_level / capacity) * 100

st.subheader("Reservoir Status (End of Period)")
c1, c2 = st.columns([1, 3])

if end_level < 0:
    c1.metric("Final Status", "DEPLETED", delta="CRITICAL", delta_color="inverse")
    c2.error(f"🚨 STORAGE DEPLETED: Deficit of {abs(end_level):,.0f} m³ predicted under these conditions.")
    c2.progress(0)
else:
    c1.metric("Final Storage", f"{end_level:,.0f} m³", delta=f"{pct:.1f}% Full")
    c2.progress(min(1.0, pct/100))
    if pct < 20:
        c2.warning("⚠️ Warning: Low storage levels predicted.")
    else:
        c2.success("✅ Storage levels adequate.")