import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from fpdf import FPDF
import openai

# Set up OpenAI API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state variables for gamification
if "score" not in st.session_state:
    st.session_state.score = 0
if "health_points" not in st.session_state:
    st.session_state.health_points = 0
if "water_intake" not in st.session_state:
    st.session_state.water_intake = 0
if "steps_walked" not in st.session_state:
    st.session_state.steps_walked = 0
if "posts" not in st.session_state:
    st.session_state.posts = []

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_data.csv")
    df.columns = df.columns.str.replace(" ", "_").str.replace("(__|\(|\)|-|:|,)", "", regex=True)
    return df

# Calculate BMI for prediction
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

# Generate report for the prediction results
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

# Main Streamlit app code
st.title("PCOS Prediction App")

# PCOS Prediction Section
st.header("1. PCOS Prediction 🩺")
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
bmi = calculate_bmi(weight, height)
st.write(f"Calculated BMI: {bmi:.2f}")

df = load_data()

# Preprocessing and model setup
selected_features = ["AMH", "betaHCG", "FSH"]
df = df.dropna()
X = df[selected_features]
y = df[df.columns[df.columns.str.contains("PCOS")][0]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

user_input = {col: st.number_input(f"{col}", value=float(pd.to_numeric(X.iloc[:, i], errors="coerce").mean(skipna=True) or 0)) for i, col in enumerate(selected_features)}

if st.button("Submit Prediction"):
    input_df = pd.DataFrame([user_input])
    prediction_proba = model.predict_proba(input_df)
    prediction_prob = prediction_proba[0][1]
    prediction = "PCOS Detected" if prediction_prob > 0.5 else "No PCOS Detected"
    st.success(prediction)

    report_path = generate_report(prediction_prob)
    with open(report_path, "rb") as file:
        st.download_button("Download Report", file, file_name="PCOS_Report.pdf")

    if prediction_prob > 0.8:
        st.warning("High risk of PCOS detected. Consider consulting a healthcare professional.")
    elif prediction_prob > 0.5:
        st.info("Moderate risk of PCOS detected. Lifestyle changes are recommended.")

# Data Visualization Section
st.header("2. Data Visualization 📊")
df = load_data()

st.subheader("Case Distribution")
fig, ax = plt.subplots()
sns.countplot(x=y, ax=ax)
ax.set_xticklabels(["No PCOS", "PCOS"])
st.pyplot(fig)

st.subheader("Feature Importance")
feature_importances = model.feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=selected_features, y=feature_importances, ax=ax)
st.pyplot(fig)

st.subheader("SHAP Model Impact")
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, feature_names=selected_features, show=False)
st.pyplot(fig)

# Health Gamification Section
st.header("3. Health Gamification 🏃‍♀️")
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

st.write(f"Total Health Points: {st.session_state.health_points}")

# Trivia Quiz Section
st.header("4. Trivia Quiz 🧠")
questions = {
    "What is a common symptom of PCOS?": ["Irregular periods", "Acne", "Hair loss"],
    "Which hormone is often imbalanced in PCOS?": ["Insulin", "Estrogen", "Progesterone"],
    "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "High sugar diet"]
}

quiz_score = 0
for question, options in questions.items():
    answer = st.radio(question, options)
    if answer == options[0]:
        quiz_score += 1
st.write(f"Your final quiz score: {quiz_score}/{len(questions)}")

# Community Support Section
st.header("5. Community Support 🤝")
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

# Chatbot Section
st.header("6. Chatbot 🤖")
user_question = st.text_input("Ask me anything about PCOS:")
if user_question:
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_question}])
    st.write(response["choices"][0]["message"]["content"])
