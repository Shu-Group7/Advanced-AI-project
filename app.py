import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Depression Prediction System")

st.write("Enter student details to predict depression risk")

# --- USER INPUTS ---
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 15, 40)
city = st.number_input("City Code", 1, 50)

profession = st.selectbox("Are you a student?", ["Yes", "No"])

academic_pressure = st.slider("Academic Pressure", 1, 5)
cgpa = st.number_input("CGPA", 0.0, 10.0)

study_satisfaction = st.slider("Study Satisfaction", 1, 5)
sleep_duration = st.slider("Sleep Duration (1=Low, 5=High)", 1, 5)

diet = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])

degree = st.number_input("Degree Code", 1, 10)

suicidal = st.selectbox("Suicidal Thoughts?", ["Yes", "No"])

work_hours = st.slider("Work/Study Hours", 0, 12)
financial_stress = st.slider("Financial Stress", 1, 5)

family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])


# --- ENCODING INPUT ---
def encode_inputs():
    return pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Age': [age],
        'City': [city],
        'Working Professional or Student': [1 if profession == "Yes" else 0],
        'Academic Pressure': [academic_pressure],
        'CGPA': [cgpa],
        'Study Satisfaction': [study_satisfaction],
        'Sleep Duration': [sleep_duration],
        'Dietary Habits': [1 if diet == "Healthy" else 0],
        'Degree': [degree],
        'Have you ever had suicidal thoughts ?': [1 if suicidal == "Yes" else 0],
        'Work/Study Hours': [work_hours],
        'Financial Stress': [financial_stress],
        'Family History of Mental Illness': [1 if family_history == "Yes" else 0]
    })


# --- PREDICTION BUTTON ---
if st.button("Predict"):
    input_data = encode_inputs()
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The student is likely Depressed")
    else:
        st.success("✅ The student is Not Depressed")