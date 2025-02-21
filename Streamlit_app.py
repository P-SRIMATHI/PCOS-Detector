 import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    file_path = "PCOS_infertility.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: File '{file_path}' not found. Please upload it.")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

df = load_data()
if df is not None:
    st.write("Data Loaded Successfully!")
else:
    st.write("Please upload the required CSV file.")
