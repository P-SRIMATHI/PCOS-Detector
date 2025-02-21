import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
    
    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

if df is not None:
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    st.title("PCOS Prediction App")
    st.sidebar.header("User Input")
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
    height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
    bmi = calculate_bmi(weight, height)
    st.sidebar.write(f"Calculated BMI: {bmi:.2f}")
    
    user_input = {col: st.sidebar.number_input(f"{col}", value=float(X[col].mean())) for col in X.columns}
    
    if st.sidebar.button("Submit"):
        input_df = pd.DataFrame([user_input])
        input_df[X.columns] = scaler.transform(input_df[X.columns])
        prediction = model.predict(input_df)[0]

        st.write("### Prediction:")
        if prediction == 1:
            st.error("PCOS Detected")
        else:
            st.success("No PCOS Detected")
        
        # SHAP Summary Plot
        st.subheader("Feature Importance (SHAP Values)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[1], X_test, show=False)
        st.pyplot(fig)
        
        # Visualization of PCOS cases
        st.subheader("PCOS Case Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=y, palette=["red", "green"], ax=ax)
        ax.set_xticklabels(["Negative", "Positive"])
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importance from Model")
        importances = model.feature_importances_
        feature_names = X.columns
        feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x=feat_imp_df["Importance"], y=feat_imp_df["Feature"], ax=ax)
        st.pyplot(fig)

else:
    st.write("Please upload the required CSV file.")
