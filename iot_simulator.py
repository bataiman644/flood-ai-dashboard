# =====================================================
# 🌊 IoT SENSOR SIMULATOR (FIXED FOR REALISTIC PREDICTIONS)
# =====================================================

import numpy as np

# =====================================================
# RISK-BASED SENSOR GENERATION
# =====================================================

def get_sensor_data():

    # 🌧️ Rainfall (0–1 normalized)
    rainfall = np.random.uniform(0, 1)

    # 🌊 Water level depends strongly on rainfall
    water_level = rainfall * np.random.uniform(0.8, 1.8)

    # 🏙️ Drainage (bad drainage increases risk)
    drainage = np.random.uniform(0, 1)

    # 👥 Population density
    population = np.random.uniform(0, 1)

    # 🔥 RISK BOOST LOGIC (IMPORTANT FIX)
    risk_boost = rainfall + water_level - drainage

    # Inject extreme scenarios sometimes
    if np.random.rand() < 0.15:  # 15% extreme events
        rainfall = np.random.uniform(0.7, 1)
        water_level = np.random.uniform(0.6, 1)
        drainage = np.random.uniform(0, 0.3)

    return {
        "rainfall": rainfall,
        "water_level": water_level,
        "drainage": drainage,
        "population": population
    }