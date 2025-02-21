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
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    return df

df = load_data()

def preprocess_data(df):
    required_columns = [col for col in df.columns if "beta-HCG" in col or "AMH" in col]
    if len(required_columns) < 3:
        raise KeyError(f"Missing required columns: Expected at least 3, found {len(required_columns)}")
    
    X = df[required_columns]
    y = df["PCOS (Y/N)"].astype(int)  # Ensure target column is numeric
    
    # Convert to numeric and handle missing values
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.median(), inplace=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

X, y, scaler = preprocess_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

# Streamlit UI
st.title("PCOS Prediction App")

st.sidebar.header("User Input")
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)

bmi = calculate_bmi(weight, height)
st.sidebar.write(f"Calculated BMI: {bmi:.2f}")

user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.number_input(f"{col}", value=float(X[col].mean()))

if st.sidebar.button("Submit"):
    input_df = pd.DataFrame([user_input])
    input_df[X.columns] = scaler.transform(input_df[X.columns])
    prediction = model.predict(input_df)[0]

    st.write("### Prediction:")
    if prediction == 1:
        st.error("PCOS Detected")
        st.write("### Analysis and Suggestions:")
        st.write("- PCOS is a hormonal disorder common among women of reproductive age.")
        st.write("- Symptoms include irregular periods, weight gain, and acne.")
        st.write("- It can lead to complications like infertility and metabolic disorders.")
        st.write("### Recommendations:")
        st.write("- Maintain a balanced diet and exercise regularly.")
        st.write("- Consult a gynecologist for further evaluation.")
        st.write("- Monitor blood sugar and hormonal levels frequently.")
    else:
        st.success("No PCOS Detected")
        st.write("### General Analysis Report:")
        st.write("- Your hormone levels are within the expected range.")
        st.write("- Your weight and height are within the normal range.")
