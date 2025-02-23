import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import shap
import openai
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from fpdf import FPDF

# Load API Key securely
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def load_data():
    file_path = "PCOS_data.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: File '{file_path}' not found. Please upload it.")
        return None
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

df = load_data()

def preprocess_data(df):
    df.columns = df.columns.str.replace(" ", "_")
    selected_columns = ["AMH", "betaHCG", "FSH_mIUmL"]
    X = df[selected_columns]
    y_column = "PCOS_YN" if "PCOS_YN" in df.columns else "PCOS_(Y/N)"
    y = df[y_column].astype(int)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled, scaler, selected_columns

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def generate_report(prediction_prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, "High probability of PCOS detected." if prediction_prob > 0.5 else "No significant risk of PCOS detected.")
    report_path = "PCOS_Report.pdf"
    pdf.output(report_path)
    return report_path

if df is not None:
    X, y, scaler, feature_columns = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    st.title("PCOS Prediction App")
    st.header("1. PCOS Prediction")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
    bmi = calculate_bmi(weight, height)
    st.write(f"Calculated BMI: {bmi:.2f}")
    user_input = {col: st.number_input(f"{col}", value=float(X[col].mean())) for col in feature_columns}
    if st.button("Submit Prediction"):
        input_df = pd.DataFrame([user_input])
        input_df[feature_columns] = scaler.transform(input_df[feature_columns])
        prediction_prob = model.predict_proba(input_df)[0][1]
        prediction = "PCOS Detected" if prediction_prob > 0.5 else "No PCOS Detected"
        st.success(prediction)
        report_path = generate_report(prediction_prob)
        with open(report_path, "rb") as file:
            st.download_button("Download Report", file, file_name="PCOS_Report.pdf")
    
    st.header("2. Data Visualizations")
    st.subheader("Feature Distributions")
    for column in feature_columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)
    
    st.subheader("Case Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    ax.set_xticklabels(["No PCOS", "PCOS"])
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.header("3. Chatbot")
    user_question = st.text_input("Ask me anything about PCOS:")
    if user_question:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_question}])
        st.write(response["choices"][0]["message"]["content"])
    
    st.header("4. Trivia Quiz")
    questions = {"What is a common symptom of PCOS?": ["Irregular periods", "Acne", "Hair loss"]}
    score = 0
    for question, options in questions.items():
        answer = st.radio(question, options)
        if answer == options[0]:
            score += 1
    st.write(f"Your final score: {score}/{len(questions)}")
