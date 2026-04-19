# =====================================================
# 🌐 CLOUD-BASED FLOOD RESPONSE DASHBOARD (FINAL FIXED)
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Flood System", layout="wide")

st.title("🌊 Smart City Flood Disaster Response System")
st.markdown("Real-Time IoT + ML Based Flood Prediction Dashboard")

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# =====================================================
# SESSION STATE (CRITICAL FIX)
# =====================================================

if "rain_series" not in st.session_state:
    st.session_state.rain_series = []

if "water_series" not in st.session_state:
    st.session_state.water_series = []

if "risk_series" not in st.session_state:
    st.session_state.risk_series = []

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("📡 IoT Sensor Controls")

rainfall = st.sidebar.slider("Rainfall", 0.0, 1.0, 0.5)
water_level = st.sidebar.slider("Water Level", 0.0, 1.0, 0.5)
drainage = st.sidebar.slider("Drainage Condition", 0.0, 1.0, 0.5)
population = st.sidebar.slider("Population Density", 0.0, 1.0, 0.5)

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict(rainfall, water_level, drainage, population):

    data = pd.DataFrame([[rainfall, water_level, drainage, population]],
                        columns=features)

    scaled = scaler.transform(data)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    return pred, prob

# =====================================================
# MAIN PREDICTION
# =====================================================

pred, prob = predict(rainfall, water_level, drainage, population)
risk_value = max(prob)

labels = ["LOW", "MEDIUM", "HIGH"]

# =====================================================
# DISPLAY RESULT
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚨 Flood Risk Level")

    if labels[pred] == "HIGH":
        st.error("🔴 HIGH RISK")
    elif labels[pred] == "MEDIUM":
        st.warning("🟠 MEDIUM RISK")
    else:
        st.success("🟢 LOW RISK")

    st.write("Probability:", prob)

# =====================================================
# GAUGE CHART
# =====================================================

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value * 100,
        title={'text': "Flood Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# LIVE TIME SERIES GRAPHS (FIXED)
# =====================================================

st.markdown("## 📊 Live Flood Monitoring System")

# Safe fallback
rain_series = st.session_state.rain_series
water_series = st.session_state.water_series
risk_series = st.session_state.risk_series

if len(rain_series) == 0:
    rain_series = [0]
    water_series = [0]
    risk_series = [0]

col3, col4 = st.columns(2)

with col3:
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(y=rain_series, name="Rainfall"))
    fig1.add_trace(go.Scatter(y=water_series, name="Water Level"))

    fig1.update_layout(title="🌧️ Sensor Trends")

    st.plotly_chart(fig1, use_container_width=True)

with col4:
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(y=risk_series, name="Flood Risk"))

    fig2.update_layout(title="🚨 Risk Trend Over Time")

    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# LIVE SIMULATION
# =====================================================

st.markdown("---")

if st.button("🌊 Start Live IoT Simulation"):

    placeholder = st.empty()

    for i in range(30):

        r = np.random.rand()
        w = np.random.rand()
        d = np.random.rand()
        p = np.random.rand()

        data = pd.DataFrame([[r, w, d, p]], columns=features)
        scaled = scaler.transform(data)

        pred = model.predict(scaled)[0]
        prob = max(model.predict_proba(scaled)[0])

        # SAVE TO SESSION STATE (CRITICAL FIX)
        st.session_state.rain_series.append(r)
        st.session_state.water_series.append(w)
        st.session_state.risk_series.append(prob)

        placeholder.write(f"""
        ### 🔴 Live IoT Stream {i+1}

        - 🌧 Rainfall: {r:.2f}
        - 🌊 Water Level: {w:.2f}
        - 🚧 Drainage: {d:.2f}
        - 👥 Population: {p:.2f}

        ### 🚨 Prediction: {labels[pred]}
        ### 📊 Risk Score: {prob:.2f}
        """)

        time.sleep(0.6)