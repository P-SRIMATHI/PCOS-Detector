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
st.header("1. PCOS Prediction")

weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
bmi = calculate_bmi(weight, height)
st.write(f"Calculated BMI: {bmi:.2f}")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_data.csv")
    df.columns = df.columns.str.replace(" ", "_").str.replace("(__|\(|\)|-|:|,)", "", regex=True)
    return df

df = load_data()

if df is not None:
    possible_features = ["AMH", "betaHCG", "FSH"]
    selected_features = [col for col in df.columns if any(feature in col for feature in possible_features)]
    
    if not selected_features:
        st.error("None of the selected features are found in the dataset! Please check column names.")
        st.write("Columns in dataset:", df.columns.tolist())
        st.stop()

    df = df.dropna()
    X = df[selected_features]
    y = df[df.columns[df.columns.str.contains("PCOS")][0]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    user_input = {col: st.number_input(f"{col}", value=float(pd.to_numeric(X.iloc[:, i], errors="coerce").mean(skipna=True) or 0)) for i, col in enumerate(selected_features)}
    
    if st.button("Submit Prediction"):
        input_df = pd.DataFrame([user_input])
        prediction_prob = model.predict_proba(input_df)[0][1] if len(model.predict_proba(input_df).shape) > 1 else model.predict_proba(input_df)[0]
        prediction = "PCOS Detected" if prediction_prob > 0.5 else "No PCOS Detected"
        st.success(prediction)
        report_path = generate_report(prediction_prob)
        with open(report_path, "rb") as file:
            st.download_button("Download Report", file, file_name="PCOS_Report.pdf")

    # Graphs
    st.header("2. Data Visualizations")
    st.subheader("Case Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    ax.set_xticklabels(["No PCOS", "PCOS"])
    st.pyplot(fig)
    
    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=selected_features, y=feature_importances, ax=ax)
    st.pyplot(fig)
    
    st.subheader("SHAP Model Impact")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, feature_names=selected_features, show=False)
    st.pyplot(fig)
    
    # Chatbot
    st.header("3. Chatbot")
    user_question = st.text_input("Ask me anything about PCOS:")
    if user_question:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_question}])
        st.write(response["choices"][0]["message"]["content"])
    
    # Trivia Quiz
    st.header("4. Trivia Quiz")
    questions = {
        "What is a common symptom of PCOS?": ["Irregular periods", "Acne", "Hair loss"],
        "Which hormone is often imbalanced in PCOS?": ["Insulin", "Estrogen", "Progesterone"],
        "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "High sugar diet"]
    }
    score = 0
    for question, options in questions.items():
        answer = st.radio(question, options)
        if answer == options[0]:
            score += 1
    st.write(f"Your final score: {score}/{len(questions)}")
