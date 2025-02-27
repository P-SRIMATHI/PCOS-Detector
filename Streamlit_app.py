import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lime.lime_tabular import LimeTabularExplainer

# Page Configuration
st.set_page_config(page_title="PCOS Prediction Tool", layout="wide")
st.title("🩺 PCOS Prediction Tool")

# Sidebar Navigation
menu = st.sidebar.radio("Navigate to:", ["Home", "Upload Data", "Model Training", "Prediction", "Insights", "PCOS Quiz", "About"])

# File Upload Handler
def load_data(file):
    df = pd.read_csv(file)
    return df

# Store uploaded file in session_state
if "dataset" not in st.session_state:
    st.session_state.dataset = None

if menu == "Home":
    st.header("Welcome to the PCOS Prediction Tool")
    st.write("This tool predicts PCOS based on non-invasive features using Machine Learning.")

elif menu == "Upload Data":
    st.header("Upload PCOS Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        st.session_state.dataset = load_data(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(st.session_state.dataset.head())

elif menu == "Model Training":
    st.header("Train Machine Learning Model")

    if st.session_state.dataset is not None:
        df = st.session_state.dataset.copy()

        # Assuming "PCOS" is the target variable
        X = df.drop(columns=["PCOS"])
        y = df["PCOS"]

        # Handling missing values
        X.fillna(X.mean(), inplace=True)

        # Encoding categorical variables
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col])

        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Oversampling using SMOTE
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Training with Ensemble Learning
        rf = RandomForestClassifier()
        xgb = XGBClassifier()
        lgbm = LGBMClassifier()

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)

        # Evaluate models
        accuracy_rf = accuracy_score(y_test, rf.predict(X_test))
        accuracy_xgb = accuracy_score(y_test, xgb.predict(X_test))
        accuracy_lgbm = accuracy_score(y_test, lgbm.predict(X_test))

        st.write(f"Random Forest Accuracy: {accuracy_rf:.2f}")
        st.write(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
        st.write(f"LightGBM Accuracy: {accuracy_lgbm:.2f}")

        # Save the best model
        best_model = max([(rf, accuracy_rf), (xgb, accuracy_xgb), (lgbm, accuracy_lgbm)], key=lambda x: x[1])[0]
        with open("pcos_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        st.success("Model trained and saved successfully!")
    else:
        st.warning("Please upload data first in the 'Upload Data' section.")

elif menu == "Prediction":
    st.header("PCOS Prediction")

    if os.path.exists("pcos_model.pkl"):
        with open("pcos_model.pkl", "rb") as f:
            model = pickle.load(f)

        st.write("Enter patient details to predict PCOS")

        age = st.slider("Age", 15, 50, 25)
        bmi = st.slider("BMI", 15, 40, 25)
        waist_hip_ratio = st.slider("Waist-Hip Ratio", 0.6, 1.2, 0.85)
        irregular_periods = st.radio("Do you have irregular menstrual cycles?", ["Yes", "No"])
        acne = st.radio("Do you have acne or oily skin?", ["Yes", "No"])
        weight_gain = st.radio("Do you experience sudden weight gain?", ["Yes", "No"])

        # Convert categorical inputs to numerical
        features = np.array([
            age, bmi, waist_hip_ratio, 
            1 if irregular_periods == "Yes" else 0,
            1 if acne == "Yes" else 0,
            1 if weight_gain == "Yes" else 0
        ]).reshape(1, -1)

        prediction = model.predict(features)

        st.write("Prediction Result:")
        if prediction[0] == 1:
            st.error("PCOS Detected! Consult a doctor for further analysis.")
        else:
            st.success("No PCOS detected. Keep maintaining a healthy lifestyle.")

    else:
        st.error("Model file not found. Please train the model first.")

elif menu == "PCOS Quiz":
    st.header("Take the PCOS Risk Quiz")

    questions = [
        "
