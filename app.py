import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Bento Motors Vehicle Price Predictor", layout="centered")

@st.cache_resource
def load_model():
    with open("best_bento_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Bento Motors Vehicle Price Predictor")
st.write("Enter vehicle details below to estimate the price.")

mileage = st.number_input("Mileage", min_value=0.0, value=25000.0)
reg_code = st.text_input("Registration code", value="20")
standard_colour = st.text_input("Colour", value="Black")
standard_make = st.text_input("Make", value="Ford")
standard_model = st.text_input("Model", value="Focus")
vehicle_condition = st.selectbox("Vehicle condition", ["USED", "NEW"])
year_of_registration = st.number_input("Year of registration", min_value=1950, max_value=2025, value=2020)
body_type = st.text_input("Body type", value="Hatchback")
crossover_car_and_van = st.selectbox("Crossover car and van", [False, True])
fuel_type = st.text_input("Fuel type", value="Petrol")

vehicle_age = 2020 - year_of_registration
reg_code_numeric = pd.to_numeric(pd.Series([reg_code]), errors="coerce")[0]
mileage_per_year = mileage / vehicle_age if vehicle_age > 0 else mileage
is_new = 1 if vehicle_condition == "NEW" else 0

input_df = pd.DataFrame({
    "mileage": [mileage],
    "reg_code": [reg_code],
    "standard_colour": [standard_colour],
    "standard_make": [standard_make],
    "standard_model": [standard_model],
    "vehicle_condition": [vehicle_condition],
    "year_of_registration": [year_of_registration],
    "body_type": [body_type],
    "crossover_car_and_van": [crossover_car_and_van],
    "fuel_type": [fuel_type],
    "vehicle_age": [vehicle_age],
    "is_new": [is_new],
    "reg_code_numeric": [reg_code_numeric],
    "mileage_per_year": [mileage_per_year]
})

if st.button("Predict price"):
    pred = model.predict(input_df)[0]

    # Use this if your model was trained on log(price)
    pred_price = np.expm1(pred)

    st.subheader("Predicted price")
    st.success(f"£{pred_price:,.2f}")