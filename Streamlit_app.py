import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# Initialize session state variables for gamification and community
if "score" not in st.session_state:
    st.session_state.score = 0
if "posts" not in st.session_state:
    st.session_state.posts = []
if "health_points" not in st.session_state:
    st.session_state.health_points = 0
if "water_intake" not in st.session_state:
    st.session_state.water_intake = 0
if "steps_walked" not in st.session_state:
    st.session_state.steps_walked = 0

def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def generate_report(prediction_prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "PCOS Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, "High probability of PCOS detected." if prediction_prob > 0.5 else "No significant risk of PCOS detected.")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Lifestyle Changes:\n- Maintain a balanced diet\n- Exercise regularly\n- Manage stress\n- Get enough sleep")
    report_path = "PCOS_Report.pdf"
    pdf.output(report_path)
    return report_path

# Streamlit UI
st.title("Welcome to the PCOS Prediction App")  # Add the welcome title

# Display 6 clickable boxes for sections
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    data_viz_button = st.button("2. Data Visualization 📊")
with col2:
    health_game_button = st.button("3. Health Gamification 🎮")
with col3:
    prediction_button = st.button("1. PCOS Prediction 🤖")
with col4:
    trivia_button = st.button("4. Trivia Quiz ❓")
with col5:
    support_button = st.button("5. Community Support 💬")
with col6:
    chatbot_button = st.button("6. Chatbot 🤖")

# Handle Prediction Section
if prediction_button:
    st.header("PCOS Prediction")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
    bmi = calculate_bmi(weight, height)
    st.write(f"Calculated BMI: {bmi:.2f}")

    # Load Data
    @st.cache_data
    def load_data():
        df = pd.read_csv("PCOS_data.csv")
        df.columns = df.columns.str.replace(" ", "_").str.replace("(__|\(|\)|-|:|,)", "", regex=True)
        return df

    df = load_data()

    if df is not None:
        possible_features = ["AMH", "betaHCG", "FSH"]
        selected_features = [col for col in df.columns if any(feature in col for feature in possible_features)]

        if not selected_features:
            st.error("None of the selected features are found in the dataset! Please check column names.")
            st.write("Columns in dataset:", df.columns.tolist())
            st.stop()

        df = df.dropna()
        X = df[selected_features]
        y = df[df.columns[df.columns.str.contains("PCOS")][0]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        user_input = {col: st.number_input(f"{col}", value=float(pd.to_numeric(X.iloc[:, i], errors="coerce").mean(skipna=True) or 0)) for i, col in enumerate(selected_features)}

        if st.button("Submit Prediction"):
            input_df = pd.DataFrame([user_input])
            prediction_proba = model.predict_proba(input_df)

            if isinstance(prediction_proba, np.ndarray) and len(prediction_proba.shape) == 2 and prediction_proba.shape[1] > 1:
                prediction_prob = prediction_proba[0][1]
            else:
                prediction_prob = prediction_proba[0]

            prediction = "PCOS Detected" if prediction_prob > 0.5 else "No PCOS Detected"
            st.success(prediction)

            report_path = generate_report(prediction_prob)
            with open(report_path, "rb") as file:
                st.download_button("Download Report", file, file_name="PCOS_Report.pdf")

            # AI-powered Alerts
            st.header("AI-powered Alerts")
            if prediction_prob > 0.8:
                st.warning("High risk of PCOS detected. Consider consulting a healthcare professional.")
            elif prediction_prob > 0.5:
                st.info("Moderate risk of PCOS detected. Lifestyle changes are recommended.")

# Handle Data Visualization Section
if data_viz_button:
    st.header("2. Data Visualizations")
    
    # Case Distribution - Count Plot
    st.subheader("Case Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size for better visibility
    sns.countplot(x=y, ax=ax)
    ax.set_xticklabels(["No PCOS", "PCOS"])
    st.pyplot(fig)  # Display plot

    # Feature Importance - Bar Plot
    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size for better visibility
    sns.barplot(x=selected_features, y=feature_importances, ax=ax)
    ax.set_ylabel("Importance")
    st.pyplot(fig)  # Display plot

    # SHAP Model Impact - SHAP Summary Plot
    st.subheader("SHAP Model Impact")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size for better visibility
    shap.summary_plot(shap_values, X_test, feature_names=selected_features, show=False)
    st.pyplot(fig)  # Display plot

# Handle Health Gamification Section
if health_game_button:
    st.header("Health Gamification")
    st.subheader("Track Your Progress and Earn Points!")

    # Track Water Intake
    st.subheader("Track Your Water Intake")
    water_glasses = st.slider("How many glasses of water did you drink today?", min_value=0, max_value=15)
    st.session_state.water_intake = water_glasses

    # Reward for Drinking Water
    if st.session_state.water_intake >= 8:
        st.session_state.health_points += 10
        st.success("Great job! You've completed your water intake goal! +10 points")
    else:
        st.warning(f"Drink more water! You've had {st.session_state.water_intake} glasses.")

    # Track Steps
    st.subheader("Track Your Steps")
    steps = st.slider("How many steps did you walk today?", min_value=0, max_value=20000)
    st.session_state.steps_walked = steps

    # Reward for Walking Steps
    if st.session_state.steps_walked >= 10000:
        st.session_state.health_points += 20
        st.success("Amazing! You've reached 10,000 steps! +20 points")
    else:
        st.warning(f"You're doing well! You've walked {st.session_state.steps_walked} steps today.")

    # Display Total Health Points
    st.write(f"Total Health Points: {st.session_state.health_points}")

    # Challenge Participation
    if st.session_state.health_points >= 50:
        st.balloons()

# Handle Trivia Quiz Section
if trivia_button:
    st.header("Trivia Quiz")
    questions = {
        "What is a common symptom of PCOS?": ["Irregular periods", "Acne", "Hair loss"],
        "Which hormone is often imbalanced in PCOS?": ["Insulin", "Estrogen", "Progesterone"],
        "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "High sugar diet"],
        "Which condition often occurs alongside PCOS?": ["Obesity", "Hypertension", "Cystic fibrosis"]
    }

    quiz_score = 0
    for question, options in questions.items():
        answer = st.radio(question, options)
        if answer == options[0]:
            quiz_score += 1
    st.write(f"Your final quiz score: {quiz_score}/{len(questions)}")
    st.session_state.score += quiz_score  # Add quiz score to session score

# Handle Community Support Section
if support_button:
    st.header("Community Support")
    new_post = st.text_area("Post your experience or ask a question:")
    if st.button("Submit Post"):
        if new_post:
            st.session_state.posts.append(new_post)
            st.success("Post submitted successfully!")
        else:
            st.warning("Please write something to post.")

    if st.session_state.posts:
        st.write("### Community Posts:")
        for idx, post in enumerate(st.session_state.posts, 1):
            st.write(f"{idx}. {post}")

# Handle Chatbot Section
if chatbot_button:
    st.header("Chatbot")
    user_question = st.text_input("Ask me anything about PCOS:")
    if user_question:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_question}])
        st.write(response["choices"][0]["message"]["content"])
