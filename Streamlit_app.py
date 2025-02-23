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

        # Column name corrections
        column_mapping = {
            "I___betaHCGmIUmL": "I_beta_HCG_mIU_mL",
            "II____betaHCGmIUmL": "II_beta_HCG_mIU_mL",
            "AMHngmL": "AMH_ng_mL"
        }
        df.rename(columns=column_mapping, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

df = load_data()

def preprocess_data(df):
    required_columns = ["I_beta_HCG_mIU_mL", "II_beta_HCG_mIU_mL", "AMH_ng_mL"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"⚠️ Required features missing! Missing columns: {missing_columns}")
        return None, None, None, None

    X = df[required_columns]
    y = df["PCOS_YN"].astype(int)

    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled, scaler, required_columns

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def generate_report(prediction_prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
    pdf.ln(10)

    if prediction_prob > 0.5:
        pdf.multi_cell(0, 10, "⚠️ High Probability of PCOS Detected! Here are some important measures:")
    else:
        pdf.multi_cell(0, 10, "✅ No significant risk of PCOS detected. General health recommendations:")

    pdf.ln(5)
    pdf.multi_cell(0, 10, "📌 Diet Recommendations:")
    pdf.multi_cell(0, 10, "- Eat more fiber and protein.")
    pdf.multi_cell(0, 10, "- Avoid processed and sugary foods.")
    pdf.multi_cell(0, 10, "- Include omega-3 fatty acids (flaxseeds, walnuts, fish).")

    pdf.ln(5)
    pdf.multi_cell(0, 10, "🏋️‍♀️ Exercise Tips:")
    pdf.multi_cell(0, 10, "- Engage in at least 30 minutes of daily exercise.")
    pdf.multi_cell(0, 10, "- Strength training and yoga are beneficial.")

    pdf.ln(5)
    pdf.multi_cell(0, 10, "🌿 Lifestyle Changes:")
    pdf.multi_cell(0, 10, "- Manage stress with meditation and sleep.")
    pdf.multi_cell(0, 10, "- Maintain a healthy sleep cycle.")
    pdf.multi_cell(0, 10, "- Stay hydrated and avoid excess caffeine.")

    report_path = "PCOS_Report.pdf"
    pdf.output(report_path)
    st.success("✅ Report generated successfully!")
    return report_path

if df is not None:
    X, y, scaler, feature_columns = preprocess_data(df)

    if X is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        st.title("PCOS Prediction App")

        st.sidebar.header("User Input")
        weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
        height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
        bmi = calculate_bmi(weight, height)
        st.sidebar.write(f"Calculated BMI: {bmi:.2f}")

        user_input = {col: st.sidebar.number_input(f"{col}", value=float(X[col].mean())) for col in feature_columns}

        if st.sidebar.button("Submit"):
            input_df = pd.DataFrame([user_input])
            input_df[feature_columns] = scaler.transform(input_df[feature_columns])
            prediction_prob = model.predict_proba(input_df)[0][1]
            prediction = 1 if prediction_prob > 0.5 else 0

            st.write("### Prediction:")
            if prediction == 1:
                st.error(f"⚠️ PCOS Detected (Confidence: {prediction_prob:.2%})")
            else:
                st.success(f"✅ No PCOS Detected (Confidence: {1 - prediction_prob:.2%})")

            report_path = generate_report(prediction_prob)
            with open(report_path, "rb") as file:
                st.download_button("📄 Download Personalized Report", file, file_name="PCOS_Report.pdf")

        st.subheader("📊 PCOS Case Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=y, palette=["red", "green"], ax=ax)
        ax.set_xticklabels(["Negative", "Positive"])
        st.pyplot(fig)

        # SHAP Explanation Graph
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        st.subheader("📉 SHAP Values: Feature Impact on Prediction")
        shap.summary_plot(shap_values[1], X_test, show=False)
        st.pyplot()

        # PCOS Trivia Quiz
        st.header("🧠 PCOS Trivia Quiz")
        questions = {
            "What is PCOS?": ["A hormonal disorder", "A type of cancer", "A viral infection", "A digestive disorder"],
            "Which symptom is common in PCOS?": ["Irregular periods", "Fever", "Low blood pressure", "Hearing loss"],
            "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "Avoiding all fats", "Sleeping less"],
            "Which hormone is often elevated in PCOS?": ["Testosterone", "Insulin", "Cortisol", "Thyroxine"],
            "Can PCOS affect fertility?": ["Yes", "No"]
        }
        score = 0
        for question, options in questions.items():
            answer = st.radio(question, options, key=question)
            if answer == options[0]:
                score += 1
        st.write(f"Your Score: {score}/{len(questions)}")

else:
    st.write("Please upload the required CSV file.")
