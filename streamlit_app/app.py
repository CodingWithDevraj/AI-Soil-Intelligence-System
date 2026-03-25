import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# LOAD MODELS
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(BASE_DIR, "models")

crop_model = joblib.load(os.path.join(model_dir, "crop_model.pkl"))
soil_model = joblib.load(os.path.join(model_dir, "soil_model.pkl"))
fert_model = joblib.load(os.path.join(model_dir, "fertilizer_model.pkl"))

# -----------------------------
# UI
# -----------------------------
st.title("🌱 AI Soil Analysis System")

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")

temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")

rainfall = st.number_input("Rainfall", value=100)
moisture = st.number_input("Moisture", value=50)

organic_carbon = st.number_input("Organic Carbon", value=0.5)
ec = st.number_input("Electrical Conductivity", value=1.0)

if st.button("Predict"):

    crop = crop_model.predict([[N, P, K, temp, humidity, ph, rainfall]])[0]

    soil = soil_model.predict([[temp, humidity, moisture, ph, organic_carbon, ec, N, P, K]])[0]

    fert = fert_model.predict([[temp, humidity, moisture, N, P, K]])[0]

    st.success("Prediction Complete")

    st.write(f"🌾 Crop: {crop}")
    st.write(f"🌍 Soil: {soil}")
    st.write(f"🧪 Fertilizer: {fert}")