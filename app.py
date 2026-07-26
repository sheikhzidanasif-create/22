import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(page_title="Car Sales Prediction", layout="centered")

st.title("🚗 Car Sales & Price Prediction App")
st.write("Apni gadi ki details enter karein aur estimated price dekhein.")

# Model Load karne ka function
@st.cache_resource
def load_model():
    with open('car_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()

    st.subheader("Vehicle Details")

    # User Inputs
    year = st.number_input("Manufacture Year", min_value=2000, max_value=2026, value=2018)
    engine_size = st.slider("Engine Size (in Litres)", min_value=0.8, max_value=5.0, value=1.5, step=0.1)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=45000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

    # Categorical Feature Encoding (Fuel Type)
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_petrol = 1 if fuel_type == "Petrol" else 0

    # Prediction Button
    if st.button("Predict Selling Price"):
        input_data = np.array([[year, engine_size, kms_driven, fuel_diesel, fuel_petrol]])
        prediction = model.predict(input_data)[0]
        
        st.success(f"Estimated Price: **₹ {prediction:.2f} Lakhs**")

except FileNotFoundError:
    st.error("Error: `car_model.pkl` file nahi mili. Kripya pehle model training wali file run karein.")
