import streamlit as st
import joblib
import numpy as np

# Load the saved ML model
model = joblib.load("model.pkl")

# Streamlit App Title
st.title("📊 Student Performance Predictor")
st.write("Enter the details below to predict student performance.")

# Input Fields with Range Constraints
hours_studied = st.slider("Hours Studied", 1, 9, 5)
previous_scores = st.slider("Previous Scores", 40, 99, 70)
extracurricular = st.radio("Extracurricular Activities", [0, 1])
sleep_hours = st.slider("Sleep Hours", 4, 9, 7)
sample_papers = st.slider("Sample Question Papers Practiced", 10, 100, 50)

# Predict Button
if st.button("Predict Performance"):
    # Prepare input data for prediction
    input_data = np.array([[hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers]])
    prediction = model.predict(input_data)
    
    # Display Prediction
    st.success(f"Predicted Performance Score: {prediction[0]:.2f}")


