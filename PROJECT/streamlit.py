import streamlit as st
import numpy as np
import joblib

# Load the trained model and preprocessing tools
model = joblib.load("Train.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("le.pkl")     # For Warehouse Block
le1 = joblib.load("le1.pkl")   # For Shipment Mode
le2 = joblib.load("le2.pkl")   # For Product Importance
le3 = joblib.load("le3.pkl")   # For Gender

st.title("E-Commerce Delivery Prediction")

st.subheader("Enter Relevant Details")

# Gender options from label encoder
gender_options = list(le3.classes_)
gender = st.selectbox("Gender", gender_options)

# Warehouse Block options
warehouse_options = list(le.classes_)
warehouse = st.selectbox("Warehouse Block", warehouse_options)

# Mode of Shipment options
shipment_options = list(le1.classes_)
shipment = st.selectbox("Mode of Shipment", shipment_options)

# Product Importance options
pi_options = list(le2.classes_)
pi = st.selectbox("Product Importance", pi_options)

# Encode inputs using LabelEncoders
gender_encoded = le3.transform([gender])[0]
warehouse_encoded = le.transform([warehouse])[0]
shipment_encoded = le1.transform([shipment])[0]
pi_encoded = le2.transform([pi])[0]

# Prepare input array
input_data = np.array([[gender_encoded, warehouse_encoded, shipment_encoded, pi_encoded]])

# Apply scaling
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Prediction"):
    prediction = model.predict(input_scaled)[0]  # Use .predict instead of .prediction
    result = "Delivered On Time" if prediction == 1 else "Not Delivered On Time"
    st.success(f"Prediction: {result}")
