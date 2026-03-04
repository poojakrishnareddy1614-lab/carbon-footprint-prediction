import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
import streamlit as st
import numpy as np

st.title("Carbon Emission Prediction & Scenario Simulator")

year = st.number_input("Year", value=2025)
population = st.number_input("Population", value=1.4e9)
gdp = st.number_input("GDP", value=3e12)

if st.button("Predict CO2"):
    input_data = np.array([[year, population, gdp]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.write("Predicted CO2 Emission:", prediction)