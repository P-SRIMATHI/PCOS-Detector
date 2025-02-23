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

# Load API Key
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
        
        # Rename columns to match expected format
        df.rename(columns={
            "I___betaHCGmIUmL": "I_beta_HCG_mIU_mL",
            "II____betaHCGmIUmL": "II_beta_HCG_mIU_mL",
            "AMHngmL": "AMH_ng_mL"
        }, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

df = load_data()

def preprocess_data(df):
    required_columns = ["I_beta_HCG_mIU_mL", "II_beta_HCG_mIU_mL", "AMH_ng_mL"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"Required features missing! Missing columns: {missing_cols}")
        return None, None, None, None
    
    X = df[required_columns]
    y = df["PCOS_YN"].astype(int)
    
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled, scaler, required_columns

if df is not None:
    X, y, scaler, feature_columns = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    st.title("PCOS Prediction App")
    
    # Sidebar Input
    st.sidebar.header("User Input")
    user_input = {col: st.sidebar.number_input(f"{col}", value=float(X[col].mean())) for col in feature_columns}
    
    if st.sidebar.button("Submit"):
        input_df = pd.DataFrame([user_input])
        input_df[feature_columns] = scaler.transform(input_df[feature_columns])
        prediction_prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prediction_prob > 0.5 else 0  
        
        st.write("### Prediction:")
        if prediction == 1:
            st.error(f"PCOS Detected (Confidence: {prediction_prob:.2%}")
        else:
            st.success(f"No PCOS Detected (Confidence: {1 - prediction_prob:.2%}")
        
        # Generate Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, "Personalized Recommendations:")
        
        report_path = "PCOS_Report.pdf"
        pdf.output(report_path)
        
        with open(report_path, "rb") as file:
            st.download_button("Download Report", file, file_name="PCOS_Report.pdf")
    
    # PCOS Case Distribution
    st.subheader("PCOS Case Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette=["red", "green"], ax=ax)
    ax.set_xticklabels(["Negative", "Positive"])
    st.pyplot(fig)
    
    # SHAP Graph
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    st.subheader("Feature Importance (SHAP)")
    shap.summary_plot(shap_values[1], X_test, feature_names=feature_columns)
    
    # Trivia Quiz
    st.header("PCOS Trivia Quiz")
    questions = {
        "What is PCOS?": ["A hormonal disorder", "A type of cancer", "A viral infection", "A digestive disorder"],
        "Which symptom is common in PCOS?": ["Irregular periods", "Fever", "Low blood pressure", "Hearing loss"],
        "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "Avoiding all fats", "Sleeping less"],
        "Which hormone is usually elevated in PCOS?": ["Testosterone", "Estrogen", "Insulin", "Cortisol"],
        "What dietary change is recommended for PCOS?": ["Low glycemic diet", "High sugar intake", "More dairy", "More red meat"]
    }
    
    score = 0
    for question, options in questions.items():
        answer = st.radio(question, options, key=question)
        if answer == options[0]:
            score += 1
    
    st.write(f"Your Score: {score}/{len(questions)}")
    
    # Chatbot
    st.header("PCOS Health Assistant")
    user_question = st.text_input("Ask me anything about PCOS")
    if st.button("Ask"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a health assistant providing PCOS information."},
                      {"role": "user", "content": user_question}]
        )
        st.write(response['choices'][0]['message']['content'])
else:
    st.write("Please upload the required CSV file.")
