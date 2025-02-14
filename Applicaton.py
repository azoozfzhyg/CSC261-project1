import streamlit as st
import numpy as np
import joblib
# Load the trained model

model = joblib.load("student_performance_predictor.pkl")

# Streamlit App UI
st.title("📚 Student Performance Predictor")

st.markdown("### Enter Student Details:")

# Input fields
hours_studied = st.slider("Hours Studied", 1, 9, 5)
previous_scores = st.slider("Previous Scores", 40, 99, 70)
extracurricular = st.radio("Extracurricular Activities (0 = No, 1 = Yes)", [0, 1])
sleep_hours = st.slider("Sleep Hours", 4, 9, 6)
sample_papers = st.slider("Sample Question Papers Practiced", 10, 100, 50)

# Prediction
if st.button("Predict Performance"):
    input_data = np.array([[hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers]])
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Performance: {prediction:.2f}")

st.markdown("---")
st.markdown("💡 Adjust the inputs and click **Predict Performance** to see the results.")