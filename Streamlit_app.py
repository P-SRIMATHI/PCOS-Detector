import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_infertility.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

def preprocess_data(df):
    required_columns = [col for col in df.columns if "beta-HCG" in col or "AMH" in col]
    X = df[required_columns]
    y = df["PCOS (Y/N)"].astype(int)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

X, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

# Streamlit Sidebar Navigation
st.sidebar.title("PCOS Prediction App")
section = st.sidebar.radio("Navigation", ["Home", "Non-Invasive Indicators", "Clinical Analysis", "Quiz", "ML Model Training", "Prediction", "Accuracy", "Suggestions & Precautions", "Conclusion"])

if section == "Home":
    st.title("Welcome to the PCOS Prediction App")
    st.write("This app helps predict PCOS risk using machine learning.")
    st.image("pcos_image.jpg")

elif section == "Non-Invasive Indicators":
    st.title("Non-Invasive Indicators of PCOS")
    st.write("- Irregular menstrual cycles\n- Weight gain\n- Acne\n- Hair thinning\n- Mood swings")

elif section == "Clinical Analysis":
    st.title("Clinical Analysis for PCOS")
    st.write("- Hormone level tests\n- Ultrasound scan\n- Blood sugar levels")

elif section == "Quiz":
    st.title("PCOS Risk Assessment Quiz")
    answers = []
    q1 = st.radio("Do you experience irregular periods?", ["Yes", "No"])
    q2 = st.radio("Do you have excessive hair growth (face/body)?", ["Yes", "No"])
    q3 = st.radio("Do you experience weight gain?", ["Yes", "No"])
    if st.button("Submit Quiz"):
        risk_score = sum([q1 == "Yes", q2 == "Yes", q3 == "Yes"])
        st.write(f"Your PCOS risk score is: {risk_score}/3")

elif section == "ML Model Training":
    st.title("ML Model Training")
    st.write("RandomForest Model trained with 80% training data.")
    st.write("Features: Beta-HCG, AMH levels, etc.")

elif section == "Prediction":
    st.title("PCOS Prediction")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
    bmi = calculate_bmi(weight, height)
    st.write(f"Calculated BMI: {bmi:.2f}")
    user_input = {col: st.number_input(col, value=float(X[col].mean())) for col in X.columns}
    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        input_df[X.columns] = scaler.transform(input_df[X.columns])
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("PCOS Detected")
        else:
            st.success("No PCOS Detected")

elif section == "Accuracy":
    st.title("Model Accuracy")
    accuracy = model.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy:.2%}")

elif section == "Suggestions & Precautions":
    st.title("Health Recommendations")
    st.write("- Maintain a balanced diet\n- Exercise regularly\n- Monitor hormone levels")

elif section == "Conclusion":
    st.title("Conclusion")
    st.write("This tool helps in early PCOS detection. Consult a doctor for confirmation.")
