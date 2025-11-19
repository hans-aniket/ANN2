import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

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
    # Reusing cleaning logic from notebook for consistency
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
    st.error(f"Error loading files. Make sure 'water_demand_lstm_model.h5', 'scaler.pkl', and the CSV are in the directory. Error: {e}")
    st.stop()

# --- 2. Sidebar: Scenario Controls ---
st.sidebar.header("⚙️ Forecasting Scenarios")
st.sidebar.markdown("Simulate future conditions to predict water demand.")

days_to_predict = st.sidebar.slider("Days to Forecast", 7, 90, 30)
future_rain = st.sidebar.slider("Simulated Avg Rainfall (mm)", 0.0, 20.0, float(df['RAIN'].mean()))
future_temp = st.sidebar.slider("Simulated Avg Temp (°C)", -5.0, 40.0, float(df['TEMP'].mean()))
# Note: Depth usually reacts slowly, so we might assume a static or trend-based change, 
# but here we let the user adjust the starting point or assume constant.
future_depth_change = st.sidebar.slider("Groundwater Level Change (%)", -10, 10, 0)

# --- 3. Main Dashboard ---
st.title("💧 AI-Powered Water Demand Forecasting")
st.markdown("### Reservoir Storage Management System")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
current_demand = df['DEMAND'].iloc[-1]
avg_demand = df['DEMAND'].mean()

col1.metric("Last Recorded Demand", f"{current_demand:.2f} m³")
col2.metric("Avg Historical Demand", f"{avg_demand:.2f} m³")
col3.metric("Last Groundwater Depth", f"{df['DEPTH'].iloc[-1]:.2f} m")
col4.metric("Data Points", len(df))

# --- 4. Forecasting Logic ---
def make_forecast(model, scaler, last_sequence, days, rain, temp, depth_change):
    forecast = []
    # Copy sequence to avoid modifying original
    curr_seq = last_sequence.copy() 
    
    # Calculate new depth based on percentage change spread over the period
    last_depth = df['DEPTH'].iloc[-1]
    target_depth = last_depth * (1 + depth_change/100)
    depth_step = (target_depth - last_depth) / days

    for i in range(days):
        # Predict next step
        # Reshape for model (1, 60, 4)
        input_data = curr_seq.reshape(1, 60, 4)
        pred_scaled = model.predict(input_data, verbose=0)[0][0]
        
        # Prepare inputs for next step simulation
        # Features order: DEMAND, RAIN, TEMP, DEPTH
        # We use the predicted demand, and user-simulated weather
        current_depth = last_depth + (depth_step * (i+1))
        
        # We need to create a row with scaled values to append to sequence
        # Create a dummy row to scale the inputs correctly
        dummy_row = np.array([[pred_scaled, rain, temp, current_depth]])
        
        # Note: The model output is already scaled (DEMAND). 
        # But RAIN, TEMP, DEPTH need scaling relative to original data range.
        # To do this strictly correctly without fitting a new scaler, we rely on the 
        # previously fitted scaler. Since it fits on 4 cols, we construct a row:
        
        # Inverse transform to get actual demand for storage
        # But we need scaled values to feed back into LSTM.
        # Let's construct the next row in scaled space.
        
        # Hack: Construct a full unscaled row to use the scaler transform
        # Since we don't have the unscaled predicted demand yet (it is scaled),
        # we assume pred_scaled is X. 
        # Actually, let's simplify: The model output is 0-1. 
        # We need to scale user inputs (Rain/Temp) to 0-1 based on original data min/max.
        
        # Get MinMax values from original data for manual scaling
        rain_min, rain_max = df['RAIN'].min(), df['RAIN'].max()
        temp_min, temp_max = df['TEMP'].min(), df['TEMP'].max()
        depth_min, depth_max = df['DEPTH'].min(), df['DEPTH'].max()
        
        scaled_rain = (rain - rain_min) / (rain_max - rain_min)
        scaled_temp = (temp - temp_min) / (temp_max - temp_min)
        scaled_depth = (current_depth - depth_min) / (depth_max - depth_min)
        
        next_step = np.array([pred_scaled, scaled_rain, scaled_temp, scaled_depth])
        
        # Store prediction (we will inverse transform all later)
        forecast.append(next_step)
        
        # Update sequence: remove first, add new at end
        curr_seq = np.append(curr_seq[1:], [next_step], axis=0)
        
    return np.array(forecast)

# Get last 60 days of scaled data
look_back = 60
# We need to scale the whole dataset to extract the last sequence correctly
scaled_dataset = scaler.transform(df.values)
last_60_days = scaled_dataset[-look_back:]

with st.spinner('Calculating forecasts...'):
    forecast_scaled = make_forecast(model, scaler, last_60_days, days_to_predict, future_rain, future_temp, future_depth_change)

# Inverse Transform Predictions
# The scaler expects 4 columns. forecast_scaled has 4 columns.
forecast_values = scaler.inverse_transform(forecast_scaled)
predicted_demand = forecast_values[:, 0] # First column is DEMAND

# --- 5. Visualization ---

# Create Future Dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1)[1:]

# Plot
fig = go.Figure()

# Historical Data (Last 180 days for clarity)
fig.add_trace(go.Scatter(
    x=df.index[-180:], 
    y=df['DEMAND'].iloc[-180:],
    mode='lines',
    name='Historical Demand',
    line=dict(color='blue')
))

# Forecast Data
fig.add_trace(go.Scatter(
    x=future_dates, 
    y=predicted_demand,
    mode='lines+markers',
    name='Predicted Demand',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='Water Demand Forecast',
    xaxis_title='Date',
    yaxis_title='Volume (m³)',
    template='plotly_white',
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# --- 6. Risk Analysis & Decision Support ---
st.markdown("### ⚠️ Risk & Policy Analysis")

total_predicted_demand = np.sum(predicted_demand)
st.write(f"**Total Cumulative Demand for next {days_to_predict} days:** {total_predicted_demand:,.2f} m³")

col_a, col_b = st.columns(2)

with col_a:
    st.info("💧 **Reservoir Status:**")
    reservoir_capacity = 500000 # Arbitrary capacity for demo
    current_storage = 300000 # Arbitrary current level
    
    new_storage = current_storage - total_predicted_demand
    st.progress(min(1.0, max(0.0, new_storage / reservoir_capacity)))
    st.write(f"Remaining Storage: {new_storage:,.0f} / {reservoir_capacity:,.0f} m³")
    
    if new_storage < (reservoir_capacity * 0.2):
        st.error("CRITICAL WARNING: Storage levels projected to drop below 20%!")

with col_b:
    st.success("📊 **Scenario Insights:**")
    if future_rain < 2.0 and future_temp > 25:
        st.warning("Drought conditions simulated. Demand is expected to rise significantly.")
    elif future_rain > 10.0:
        st.write("High rainfall simulated. Demand likely to stabilize or decrease (agricultural offset).")
    else:
        st.write("Conditions appear within normal operating ranges.")

st.markdown("---")
st.caption("Data Source: Aquifer Petrignano | Model: LSTM (TensorFlow/Keras)")