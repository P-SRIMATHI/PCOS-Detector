import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import shap
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from fpdf import FPDF

# Load API Key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def generate_report(prediction_prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, "High probability of PCOS detected." if prediction_prob > 0.5 else "No significant risk of PCOS detected.")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Lifestyle Changes:\n- Maintain a balanced diet\n- Exercise regularly\n- Manage stress\n- Get enough sleep")
    report_path = "PCOS_Report.pdf"
    pdf.output(report_path)
    return report_path

st.title("PCOS Prediction App")
st.markdown("<h2 style='text-align: center;'>Welcome to the PCOS Detection App</h2>", unsafe_allow_html=True)

# Create a grid layout for 6 clickable boxes
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

# 1st row - Prediction, Data Visualizations, Health Gamification
with col1:
    if st.button("🔮 **PCOS Prediction**", help="Click here to predict the risk of PCOS based on your data.", key="prediction", use_container_width=True):
        # Your Prediction code here (already present)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
        bmi = calculate_bmi(weight, height)
        st.write(f"Calculated BMI: {bmi:.2f}")

        # Prediction logic here...
        # After prediction, show report
        report_path = generate_report(prediction_prob)
        with open(report_path, "rb") as file:
            st.download_button("Download Report", file, file_name="PCOS_Report.pdf")

with col2:
    if st.button("📊 **Data Visualizations**", help="View data distributions, feature importance, and model impact.", key="visualizations", use_container_width=True):
        # Data Visualization content (graphs, charts)
        st.header("Data Visualizations")
        # Your visualization code here...

with col3:
    if st.button("🎮 **Health Gamification**", help="Track your health goals, earn points, and improve your lifestyle.", key="gamification", use_container_width=True):
        # Health gamification (Water intake, Steps, etc.)
        st.title("Health Gamification")
        # Your health gamification code here...

# 2nd row - Trivia Quiz, Support, Chatbot
with col4:
    if st.button("🧠 **Trivia Quiz**", help="Test your knowledge about PCOS in a fun trivia quiz!", key="quiz", use_container_width=True):
        # Trivia quiz content
        st.header("Trivia Quiz")
        questions = {
            "What is a common symptom of PCOS?": ["Irregular periods", "Acne", "Hair loss"],
            "Which hormone is often imbalanced in PCOS?": ["Insulin", "Estrogen", "Progesterone"],
            "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "High sugar diet"]
        }
        quiz_score = 0  # Initialize quiz score
        for question, options in questions.items():
            answer = st.radio(question, options)
            if answer == options[0]:
                quiz_score += 1
        st.write(f"Your final quiz score: {quiz_score}/{len(questions)}")

with col5:
    if st.button("🤝 **Community Support**", help="Share experiences, ask questions, and interact with the community.", key="support", use_container_width=True):
        # Community Support content (Post questions/experiences)
        st.header("Community Support")
        new_post = st.text_area("Post your experience or ask a question:")
        if st.button("Submit Post"):
            if new_post:
                st.session_state.posts.append(new_post)
                st.success("Post submitted successfully!")
            else:
                st.warning("Please write something to post.")
        
        if st.session_state.posts:
            st.write("### Community Posts:")
            for idx, post in enumerate(st.session_state.posts, 1):
                st.write(f"{idx}. {post}")

with col6:
    if st.button("💬 **Chatbot**", help="Ask anything about PCOS, and get instant answers from our chatbot.", key="chatbot", use_container_width=True):
        # Chatbot content (Ask about PCOS)
        st.header("Chatbot")
        user_question = st.text_input("Ask me anything about PCOS:")
        if user_question:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_question}])
            st.write(response["choices"][0]["message"]["content"])
