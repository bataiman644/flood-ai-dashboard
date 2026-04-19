# =====================================================
# 🌊 IoT + ML LIVE PREDICTION SYSTEM (FIXED STEP 1)
# =====================================================

import joblib
import numpy as np
import time
import pandas as pd
from iot_simulator import get_sensor_data

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

print("✅ Model Loaded Successfully")

# =====================================================
# PREDICTION FUNCTION (FIXED)
# =====================================================

def predict_flood(data):

    # Convert dictionary → DataFrame (IMPORTANT FIX)
    input_df = pd.DataFrame([data], columns=features)

    # Scale correctly
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    return labels[prediction], prob

# =====================================================
# REAL-TIME LOOP
# =====================================================

print("\n📡 Starting Real-Time Flood Prediction System...\n")

while True:

    sensor_data = get_sensor_data()

    print("📊 Sensor Data:")
    for k, v in sensor_data.items():
        print(f"{k}: {v}")

    label, prob = predict_flood(sensor_data)

    print("\n🚨 Flood Risk Prediction:", label)
    print("📈 Probabilities:", prob)

    if label == "HIGH":
        print("🔴 ALERT: HIGH FLOOD RISK!")
    elif label == "MEDIUM":
        print("🟠 WARNING: Moderate Risk")
    else:
        print("🟢 Safe Conditions")

    print("=" * 50)

    time.sleep(3)