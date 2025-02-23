import os
import pandas as pd
import numpy as np
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

# Calculate BMI function
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

# Generate a simple report
def generate_report(prediction_prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, "PCOS Detected" if prediction_prob > 0.5 else "No PCOS Detected.")
    report_path = "PCOS_Report.pdf"
    pdf.output(report_path)
    return report_path

# Load data and handle column issues
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_data.csv")
    df.columns = df.columns.str.strip()  # Remove extra spaces
    df.columns = df.columns.str.replace(" ", "_")  # Replace spaces with underscores
    return df

# Load dataset
df = load_data()

# Define features (adjust based on your dataset)
selected_features = ["AMH", "betaHCG", "FSH"]  # Adjust this based on your actual columns
X = df[selected_features]
y = df["PCOS"]  # Assuming 'PCOS' is the target column

# Preprocessing and Model Training
df = df.dropna()
X_scaled = StandardScaler().fit_transform(X)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Streamlit App Layout
st.title("PCOS Prediction App")

# Sidebar for tab navigation
tab = st.sidebar.radio("Select a Section", ["PCOS Prediction 🩺", "Recipes 🍲", "Mood Tracker 😊"])

# PCOS Prediction Section
if tab == "PCOS Prediction 🩺":
    st.header("1. PCOS Prediction 🩺")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
    bmi = calculate_bmi(weight, height)
    st.write(f"Calculated BMI: {bmi:.2f}")

    user_input = {col: st.number_input(f"{col}", value=float(df[col].mean())) for col in selected_features}

    if st.button("Submit Prediction"):
        input_df = pd.DataFrame([user_input])
        prediction_proba = model.predict_proba(input_df)
        prediction_prob = prediction_proba[0][1]

        prediction = "PCOS Detected" if prediction_prob > 0.5 else "No PCOS Detected"
        st.success(prediction)

        # Generate and download report
        report_path = generate_report(prediction_prob)
        with open(report_path, "rb") as file:
            st.download_button("Download Report", file, file_name="PCOS_Report.pdf")

# Recipes Section
elif tab == "Recipes 🍲":
    st.header("2. PCOS Recipes 🍲")
    st.write("Here are some healthy recipes to help manage PCOS:")
    recipes = [
        {"name": "Salmon and Avocado Salad", "ingredients": ["Salmon", "Avocado", "Olive oil", "Lemon"], "instructions": "Mix all ingredients in a bowl."},
        {"name": "Chia Pudding", "ingredients": ["Chia seeds", "Almond milk", "Honey"], "instructions": "Mix chia seeds with almond milk and let it sit overnight."},
    ]
    for recipe in recipes:
        st.subheader(recipe["name"])
        st.write("Ingredients: " + ", ".join(recipe["ingredients"]))
        st.write("Instructions: " + recipe["instructions"])

# Mood Tracker Section
elif tab == "Mood Tracker 😊":
    st.header("3. Mood Tracker 😊")
    mood = st.selectbox("How are you feeling today?", ["Happy", "Neutral", "Sad", "Anxious"])
    st.write(f"Your mood: {mood}")
