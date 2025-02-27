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

# Initialize session state variables for gamification and community
if "score" not in st.session_state:
    st.session_state.score = 0
if "posts" not in st.session_state:
    st.session_state.posts = []

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

# Load Data and handle column issues
@st.cache_data
def load_data():
    df = pd.read_csv("PCOS_data.csv")
    df.columns = df.columns.str.strip()  
    df.columns = df.columns.str.replace(" ", "_")  
    return df

# Load dataset and prepare for prediction
df = load_data()

# Define the features you're interested in (ensure they exist in your dataset)
possible_features = ["AMH", "betaHCG", "FSH"]
selected_features = [col for col in df.columns if any(feature in col for feature in possible_features)]

if not selected_features:
    st.error("None of the selected features are found in the dataset! Please check column names.")
    st.stop()

# Preprocessing and Model Training
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

# Streamlit App Layout and Features
st.title("PCOS Prediction App")
st.header("1. PCOS Prediction 🩺")

weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
bmi = calculate_bmi(weight, height)
st.write(f"Calculated BMI: {bmi:.2f}")

# User input and Prediction
user_input = {col: st.number_input(f"{col}", value=float(pd.to_numeric(X.iloc[:, i], errors="coerce").mean(skipna=True) or 0)) for i, col in enumerate(selected_features)}

if st.button("Submit Prediction"):
    input_df = pd.DataFrame([user_input])
    prediction_proba = model.predict_proba(input_df)

    # Check if prediction_proba is a NumPy array and has a shape attribute
    if isinstance(prediction_proba, np.ndarray) and len(prediction_proba.shape) == 2 and prediction_proba.shape[1] > 1:
        prediction_prob = prediction_proba[0][1]  # Probability of PCOS
    else:
        prediction_prob = prediction_proba[0]  # If only one probability value is returned

    prediction = "PCOS Detected" if prediction_prob > 0.5 else "No PCOS Detected"
    st.success(prediction)

    # Generate and provide report
    report_path = generate_report(prediction_prob)
    with open(report_path, "rb") as file:
        st.download_button("Download Report", file, file_name="PCOS_Report.pdf")

    # AI-powered Alerts (based on model prediction)
    st.header("AI-powered Alerts")
    if prediction_prob > 0.8:
        st.warning("High risk of PCOS detected. Consider consulting a healthcare professional.")
    elif prediction_prob > 0.5:
        st.info("Moderate risk of PCOS detected. Lifestyle changes are recommended.")

# Graphs and Data Visualization
st.header("2. Data Visualizations 📊")
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
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, feature_names=selected_features, show=False)
st.pyplot(fig)

# Health Gamification Section
st.header("3. Health Gamification 🎮")
if "health_points" not in st.session_state:
    st.session_state.health_points = 0
if "water_intake" not in st.session_state:
    st.session_state.water_intake = 0
if "steps_walked" not in st.session_state:
    st.session_state.steps_walked = 0

# Track Water Intake
water_glasses = st.slider("How many glasses of water did you drink today?", min_value=0, max_value=15)
st.session_state.water_intake = water_glasses

# Reward for Drinking Water
if st.session_state.water_intake >= 8:
    st.session_state.health_points += 10  # Give points for completing water goal
    st.success("Great job! You've completed your water intake goal! +10 points")
else:
    st.warning(f"Drink more water! You've had {st.session_state.water_intake} glasses.")

# Track Steps (example challenge)
steps = st.slider("How many steps did you walk today?", min_value=0, max_value=20000)
st.session_state.steps_walked = steps

# Reward for Walking Steps
if st.session_state.steps_walked >= 10000:
    st.session_state.health_points += 20  # Give points for walking 10,000 steps
    st.success("Amazing! You've reached 10,000 steps! +20 points")
else:
    st.warning(f"You're doing well! You've walked {st.session_state.steps_walked} steps today.")

# Display Total Health Points
st.write(f"Total Health Points: {st.session_state.health_points}")

# Celebration if points exceed 40
if st.session_state.health_points > 40:
    st.balloons() 

# Display Total Health Points
st.write(f"Total Health Points: {st.session_state.health_points}")

# Community Support: User can post questions and share experiences
st.header("4. Community Support 💬")
new_post = st.text_area("Post your experience or ask a question:")
if st.button("Submit Post"):
    if new_post:
        st.session_state.posts.append(new_post)
        st.success("Post submitted successfully!")
    else:
        st.warning("Please write something to post.")

# Display Community Posts
if st.session_state.posts:
    st.write("### Community Posts:")
    for idx, post in enumerate(st.session_state.posts, 1):
        st.write(f"{idx}. {post}")

 # Trivia Quiz Section
st.header("6. Trivia Quiz 🧠")
questions = {
    "What is a common symptom of PCOS?": ["Irregular periods", "Acne", "Hair loss"],
    "Which hormone is often imbalanced in PCOS?": ["Insulin", "Estrogen", "Progesterone"],
    "What lifestyle change can help manage PCOS?": ["Regular exercise", "Skipping meals", "High sugar diet"]
}

quiz_score = 0  # Initialize quiz score
for question, options in questions.items():
    answer = st.radio(question, options)
    if answer == options[0]:
        quiz_score += 1
st.write(f"Your final quiz score: {quiz_score}/{len(questions)}")

st.session_state.score += quiz_score  # Add quiz score to session score

# Mood Tracker (last section)
st.header("7. Mood Tracker 😊")
mood = st.selectbox("How do you feel today?", ["Happy", "Excited", "Neutral", "Sad", "Anxious"])

# Display mood and give advice
if mood:
    mood_advice = {
        "Happy": "Keep up the great energy! 🌟",
        "Excited": "Enjoy the excitement! 🌈",
        "Neutral": "It's okay to feel neutral, take it easy. ☁️",
        "Sad": "Take care of yourself, things will get better. 💙",
        "Anxious": "Try some deep breaths, you're doing well. 🌱"
    }
    st.write(f"You are feeling: {mood}")
    st.write(mood_advice.get(mood, "Stay strong!"))

# Recipes Section (last section)
st.header("8. PCOS-Friendly Recipes 🍲")
recipes = [
    {"name": "Spinach & Chickpea Curry", "ingredients": ["Spinach", "Chickpeas", "Coconut milk", "Garlic", "Ginger"]},
    {"name": "Oats Pancakes", "ingredients": ["Oats", "Eggs", "Banana", "Almond milk"]},
    {"name": "Greek Yogurt Salad", "ingredients": ["Greek Yogurt", "Cucumber", "Olives", "Olive oil", "Lemon"]},
]

# Display recipes
for recipe in recipes:
    st.subheader(recipe["name"])
    st.write("Ingredients:", ", ".join(recipe["ingredients"]))
    import streamlit as st
import streamlit.components.v1 as components

# Function to display a clickable 3D PCOS model
def interactive_3d_display():
    st.header("🩺 Explore PCOS in 3D")
    
    # Embed the 3D model using an iframe
    model_url = "https://sketchfab.com/models/62bfb490ad344caaaea675da9df7ba34/embed"
    
    st.write("Rotate, zoom, and explore the PCOS-related anatomy interactively.")
    components.iframe(model_url, height=500)

# Call the function in your Streamlit app
interactive_3d_display()
