import streamlit as st
import numpy as np
import pickle

# Load saved model and scaler
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("💓 Heart Disease Prediction App")
st.write("Enter the patient's health information below:")

# Input fields
age = st.number_input('Age', min_value=1, max_value=120)
sex = st.selectbox('Sex (0 = female, 1 = male)', [0, 1])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200)
chol = st.number_input('Cholesterol', min_value=100, max_value=600)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)', [0, 1])
restecg = st.selectbox('Resting ECG (0, 1, 2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=250)
exang = st.selectbox('Exercise Induced Angina (1 = yes, 0 = no)', [0, 1])
oldpeak = st.number_input('Oldpeak (ST depression)', format="%.1f")
slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)', [0, 1, 2])

# Combine input into array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 0:
        st.success("💚 Good News: The patient does NOT have heart disease.")
    else:
        st.error("⚠️ Alert: The patient MAY have heart disease. Please consult a doctor.")
