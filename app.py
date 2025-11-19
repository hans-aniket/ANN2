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