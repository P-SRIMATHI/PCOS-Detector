import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import shap
import openai
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
        df.columns = df.columns.str.replace(r"[^\w\s]", "", regex=True)  # Remove special characters
        df.columns = df.columns.str.replace("-", "_").str.replace(" ", "_")  # Standardize names
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

df = load_data()

def preprocess_data(df):
    if df is None:
        return None, None, None, None  

    required_columns = ["I_beta_HCG_mIU_mL", "II_beta_HCG_mIU_mL", "AMH_ng_mL"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"⚠️ Required features missing from dataset! Missing columns: {missing_columns}")
        st.write("Columns available in dataset:", df.columns.tolist())
        return None, None, None, None
    
    X = df[required_columns].copy()
    y = df["PCOS_YN"].astype(int) if "PCOS_YN" in df.columns else None

    if y is None:
        st.error("⚠️ 'PCOS (Y/N)' column is missing!")
        return None, None, None, None

    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled, scaler, required_columns

if df is not None:
    X, y, scaler, feature_columns = preprocess_data(df)
    
    if X is None or y is None:
        st.error("❌ Dataset is missing required values. Please check your file!")
    else:
        st.success("✅ Data Loaded Successfully!")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # Sidebar Inputs
        st.sidebar.header("User Input")
        weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
        height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)

        bmi = weight / ((height / 100) ** 2)
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

            # Generate Report
            def generate_report(prediction_prob):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
                pdf.ln(10)

                if prediction_prob > 0.5:
                    pdf.multi_cell(0, 10, "High probability of PCOS detected. Recommended measures:")
                else:
                    pdf.multi_cell(0, 10, "No PCOS detected. General health recommendations:")

                pdf.ln(5)
                pdf.multi_cell(0, 10, "✅ Maintain a healthy diet rich in fiber and protein.")
                pdf.multi_cell(0, 10, "✅ Regular physical activity like yoga and strength training.")
                pdf.multi_cell(0, 10, "✅ Manage stress through meditation and adequate sleep.")
                pdf.multi_cell(0, 10, "✅ Stay hydrated and limit processed foods.")

                report_path = "PCOS_Report.pdf"
                pdf.output(report_path)
                return report_path

            report_path = generate_report(prediction_prob)
            with open(report_path, "rb") as file:
                st.download_button("Download Personalized Report", file, file_name="PCOS_Report.pdf")

        # 📊 **Graphs**
        st.subheader("PCOS Case Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=y, palette=["red", "green"], ax=ax)
        ax.set_xticklabels(["Negative", "Positive"])
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feat_importances = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
        fig, ax = plt.subplots()
        feat_importances.plot(kind='bar', ax=ax, color="blue")
        st.pyplot(fig)

        # SHAP Graph
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        st.subheader("SHAP Value Analysis")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_columns, show=False)
        st.pyplot(fig)

        # 🧠 **Trivia Quiz**
        st.header("PCOS Trivia Quiz")
        questions = {
            "What is PCOS?": ["A hormonal disorder", "A type of cancer", "A viral infection", "A digestive disorder"],
            "Which symptom is common in PCOS?": ["Irregular periods", "Fever", "Low blood pressure", "Hearing loss"],
            "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "Avoiding all fats", "Sleeping less"],
            "Which hormone is commonly affected in PCOS?": ["Insulin", "Thyroxine", "Estrogen", "Adrenaline"],
            "What diet is recommended for PCOS management?": ["High fiber, low sugar", "Only dairy", "High sugar, low protein", "Only meat"],
        }

        score = 0
        for question, options in questions.items():
            answer = st.radio(question, options, key=question)
            if answer == options[0]:
                score += 1
        
        st.write(f"Your Score: {score}/{len(questions)}")
