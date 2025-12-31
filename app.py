import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Smart Water Usage Spike Detection", layout="centered")
st.title("💧 Smart Water Usage Spike Detection")
st.info("Enter inputs and click **Predict Water Usage**")

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("water_usage.csv")

data = load_data()

# ------------------------------
# Preprocessing
# ------------------------------
X = data.drop("WaterUsage", axis=1)
y = data["WaterUsage"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# Models
# ------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

bagging = BaggingRegressor(n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)

# ------------------------------
# User Inputs
# ------------------------------
st.subheader("🔢 Input Parameters")

temperature = st.slider("🌡️ Temperature (°C)", 10, 50, 30)
day_type = st.selectbox("📅 Day Type", ["Weekday", "Weekend"])
household_size = st.slider("🏠 Household Size", 1, 10, 4)
season = st.selectbox("🌦️ Season", ["Summer", "Winter", "Monsoon"])
festival = st.selectbox("🎉 Festival Day", ["No", "Yes"])
previous_usage = st.number_input("💦 Previous Day Usage (litres)", 50, 1000, 300)

# Encode categorical values
day_type = 1 if day_type == "Weekend" else 0
festival = 1 if festival == "Yes" else 0

season_map = {"Summer": 0, "Winter": 1, "Monsoon": 2}
season = season_map[season]

# Create input dataframe
input_data = pd.DataFrame([[
    temperature,
    day_type,
    household_size,
    season,
    festival,
    previous_usage
]], columns=X.columns)

input_scaled = scaler.transform(input_data)

# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔮 Predict Water Usage"):

    ridge_pred = ridge.predict(input_scaled)[0]
    ensemble_pred = bagging.predict(input_scaled)[0]

    average_usage = y.mean()
    overuse = ((ensemble_pred - average_usage) / average_usage) * 100

    if overuse > 20:
        category = "High Usage"
        alert = "⚠️ Spike Detected"
    elif overuse > 5:
        category = "Medium Usage"
        alert = "⚠️ Possible Spike"
    else:
        category = "Low Usage"
        alert = "✅ Normal Usage"

    # ------------------------------
    # Output
    # ------------------------------
    st.subheader("📊 Prediction Results")
    st.metric("Ridge Prediction", f"{int(ridge_pred)} litres/day")
    st.metric("Ensemble Prediction", f"{int(ensemble_pred)} litres/day")

    st.write(f"**Usage Category:** {category}")
    st.write(f"**Overuse Percentage:** {overuse:.2f}%")

    st.subheader("🚨 Spike Alert")
    if "Spike Detected" in alert:
        st.error(alert)
    elif "Possible Spike" in alert:
        st.warning(alert)
    else:
        st.success(alert)
