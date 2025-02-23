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
from dotenv import load_dotenv  # Load environment variables
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
    potential_columns = [col for col in df.columns if "betaHCG" in col or "AMH" in col]
    
    if len(potential_columns) < 3:
        alternative_columns = ["FSH_mIUmL", "LH_mIUmL", "TSH_mIUL", "PRL_ngmL", "Vit_D3_ngmL"]
        selected_columns = potential_columns + [col for col in alternative_columns if col in df.columns]
    else:
        selected_columns = potential_columns[:3]
    
    if len(selected_columns) < 3:
        st.warning(f"Using alternative features: {selected_columns}")
    
    X = df[selected_columns]
    
    y_column = "PCOS_YN"
    if y_column not in df.columns:
        for alt_col in ["PCOS", "PCOS_(Y/N)"]:
            if alt_col in df.columns:
                y_column = alt_col
                break
        else:
            raise KeyError("No suitable PCOS target variable found!")
    
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
    
    if prediction_prob > 0.5:
        pdf.multi_cell(0, 10, "Based on your input, there is a high probability of PCOS. Below are some personalized recommendations:")
    else:
        pdf.multi_cell(0, 10, "No significant risk of PCOS detected. However, here are some general health tips:")
    
    pdf.ln(5)
    pdf.multi_cell(0, 10, "Diet Plan:")
    pdf.multi_cell(0, 10, "- Include more fiber and protein in your diet.")
    pdf.multi_cell(0, 10, "- Avoid processed and sugary foods.")
    pdf.multi_cell(0, 10, "- Increase omega-3 fatty acids intake (flaxseeds, walnuts, fish).")
    
    pdf.ln(5)
    pdf.multi_cell(0, 10, "Exercise Recommendations:")
    pdf.multi_cell(0, 10, "- Engage in at least 30 minutes of moderate exercise daily.")
    pdf.multi_cell(0, 10, "- Strength training and yoga are beneficial.")
    
    pdf.ln(5)
    pdf.multi_cell(0, 10, "Lifestyle Changes:")
    pdf.multi_cell(0, 10, "- Manage stress through meditation and adequate sleep.")
    pdf.multi_cell(0, 10, "- Maintain a healthy sleep cycle.")
    pdf.multi_cell(0, 10, "- Stay hydrated and avoid excessive caffeine.")
    
    report_path = "PCOS_Report.pdf"
    pdf.output(report_path)
    st.success("Report generated successfully!")
    return report_path

if df is not None:
    X, y, scaler, feature_columns = preprocess_data(df)
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
            st.error(f"PCOS Detected (Confidence: {prediction_prob:.2%})")
        else:
            st.success(f"No PCOS Detected (Confidence: {1 - prediction_prob:.2%})")
        
        report_path = generate_report(prediction_prob)
        with open(report_path, "rb") as file:
            st.download_button("Download Personalized Report", file, file_name="PCOS_Report.pdf")
    
    # Display chatbot
    st.header("PCOS Chatbot")
    st.write("Ask me anything about PCOS!")
    user_question = st.text_input("Your Question:")
    if user_question:
        response = openai.Completion.create(engine="text-davinci-003", prompt=user_question, max_tokens=100)
        st.write(response["choices"][0]["text"].strip())
    
    # Display SHAP values safely
    st.header("SHAP Value Plot")
    explainer = shap.Explainer(model, X_train, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test[:50])
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test[:50], show=False)
    st.pyplot(fig)
