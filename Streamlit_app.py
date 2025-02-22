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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set up OpenAI API Key (Replace with actual key)
openai.api_key = "YOUR_OPENAI_API_KEY"

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
    required_columns = [col for col in df.columns if "beta-HCG" in col or "AMH" in col]
    if len(required_columns) < 3:
        raise KeyError(f"Missing required columns: Expected at least 3, found {len(required_columns)}")
    
    X = df[required_columns]
    y = df["PCOS (Y/N)"].astype(int)
    
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler, required_columns

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

if df is not None:
    X, y, scaler, feature_columns = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Streamlit UI
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
        prediction_prob = model.predict_proba(input_df)[0][1]  # Probability of PCOS
        prediction = 1 if prediction_prob > 0.6 else 0  # Adjusted threshold to 0.6

        st.write("### Prediction:")
        if prediction == 1:
            st.error(f"PCOS Detected (Confidence: {prediction_prob:.2%})")
            st.write("### Analysis and Suggestions:")
            st.write("- PCOS is a hormonal disorder common among women of reproductive age.")
            st.write("- Symptoms include irregular periods, weight gain, and acne.")
            st.write("- It can lead to complications like infertility and metabolic disorders.")
            st.write("### Recommendations:")
            st.write("- Maintain a balanced diet and exercise regularly.")
            st.write("- Consult a gynecologist for further evaluation.")
            st.write("- Monitor blood sugar and hormonal levels frequently.")
        else:
            st.success(f"No PCOS Detected (Confidence: {1 - prediction_prob:.2%})")
            st.write("### General Analysis Report:")
            st.write("- Your hormone levels are within the expected range.")
            st.write("- Your weight and height are within the normal range.")
    
    # Display Graphs After Prediction
    st.subheader("Feature Importance (SHAP Values)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Select the class index for PCOS predictions
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)
    
    st.subheader("PCOS Case Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette=["red", "green"], ax=ax)
    ax.set_xticklabels(["Negative", "Positive"])
    st.pyplot(fig)
    
    st.subheader("Feature Importance from Model")
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x=feat_imp_df["Importance"], y=feat_imp_df["Feature"], ax=ax)
    st.pyplot(fig)
    
    # Chatbot Integration
    st.subheader("💬 PCOS Chatbot")
    user_query = st.text_input("Ask me anything about PCOS:")
    if st.button("Get Answer"):
        if user_query:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant specialized in PCOS-related topics."},
                          {"role": "user", "content": user_query}]
            )
            answer = response["choices"][0]["message"]["content"]
            st.write("*Chatbot:*", answer)
        else:
            st.warning("Please enter a question before clicking Get Answer.")
else:
    st.write("Please upload the required CSV file.")
