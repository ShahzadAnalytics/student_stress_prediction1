import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model

model = joblib.load(open("student.pkl", "rb"))

st.set_page_config(page_title="Student Stress Prediction", page_icon="🧠")
st.title("🧠 Student Stress Level Prediction")
st.write("Predict stress level of students based on multiple factors")

st.divider()

# ---------------- INPUT FEATURES ----------------
# Customize input widgets based on type: numeric / categorical

anxiety_level = st.slider("Anxiety Level (0-10)", 0, 10, 5)
self_esteem = st.slider("Self Esteem (0-10)", 0, 10, 5)
mental_health_history = st.selectbox("Mental Health History", ["No", "Yes"])
depression = st.slider("Depression Level (0-10)", 0, 10, 5)
headache = st.selectbox("Headache Frequency", ["No", "Sometimes", "Often"])
blood_pressure = st.slider("Blood Pressure (mmHg)", 80, 180, 120)
sleep_quality = st.slider("Sleep Quality (0-10)", 0, 10, 5)
breathing_problem = st.selectbox("Breathing Problem", ["No", "Yes"])
noise_level = st.slider("Noise Level (0-10)", 0, 10, 5)
living_conditions = st.selectbox("Living Conditions", ["Poor", "Average", "Good"])
safety = st.selectbox("Safety", ["Low", "Medium", "High"])
basic_needs = st.selectbox("Basic Needs Met", ["Low", "Medium", "High"])
academic_performance = st.slider("Academic Performance (0-10)", 0, 10, 5)
study_load = st.slider("Study Load (hours/day)", 0, 15, 5)
teacher_student_relationship = st.slider("Teacher-Student Relationship (0-10)", 0, 10, 5)
future_career_concerns = st.slider("Future Career Concerns (0-10)", 0, 10, 5)
social_support = st.slider("Social Support (0-10)", 0, 10, 5)
peer_pressure = st.slider("Peer Pressure (0-10)", 0, 10, 5)
extracurricular_activities = st.selectbox("Extracurricular Activities", ["None", "Some", "Many"])
bullying = st.selectbox("Experience of Bullying", ["No", "Yes"])

# ---------------- ENCODING CATEGORICAL ----------------
# Map categorical values to numbers (must match model training)
binary_map = {"No": 0, "Yes": 1}
level_map = {"Low": 0, "Medium": 1, "High": 2}
freq_map = {"No": 0, "Sometimes": 1, "Often": 2}
cond_map = {"Poor": 0, "Average": 1, "Good": 2}
activities_map = {"None": 0, "Some": 1, "Many": 2}

mental_health_history = binary_map[mental_health_history]
headache = freq_map[headache]
breathing_problem = binary_map[breathing_problem]
living_conditions = cond_map[living_conditions]
safety = level_map[safety]
basic_needs = level_map[basic_needs]
extracurricular_activities = activities_map[extracurricular_activities]
bullying = binary_map[bullying]

# ---------------- FEATURE ARRAY ----------------
input_data = np.array([[
    anxiety_level, self_esteem, mental_health_history, depression, headache,
    blood_pressure, sleep_quality, breathing_problem, noise_level,
    living_conditions, safety, basic_needs, academic_performance,
    study_load, teacher_student_relationship, future_career_concerns,
    social_support, peer_pressure, extracurricular_activities, bullying
]])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Stress Level"):
    prediction = model.predict(input_data)[0]

    stress_map = {0: "🟢 Low Stress", 1: "🟡 Medium Stress", 2: "🔴 High Stress"}
    st.subheader("Prediction Result")
    st.success(f"Stress Level: **{stress_map[prediction]}**")
