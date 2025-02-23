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
if "moods" not in st.session_state:
    st.session_state.moods = []
if "recipes" not in st.session_state:
    st.session_state.recipes = []

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
    df.columns = df.columns.str.strip()  # Remove extra spaces
    df.columns = df.columns.str.replace(" ", "_")  # Replace spaces with underscores
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

# Chatbot Section
st.header("5. Chatbot 🤖")
user_question = st.text_input("Ask me anything about PCOS:")
if user_question:
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_question}])
    st.write(response["choices"][0]["message"]["content"])
# Mood Tracker Section
st.header("6. Mood Tracker 😊")
mood = st.selectbox("How are you feeling today?", ["Happy", "Sad", "Anxious", "Neutral", "Excited"])
mood_notes = st.text_area("Add a note about your mood today (optional):")
if st.button("Submit Mood"):
    mood_data = {"Mood": mood, "Note": mood_notes, "Date": pd.to_datetime("today").strftime("%Y-%m-%d")}
    st.session_state.moods.append(mood_data)

# Display Mood History with Trend
if st.session_state.moods:
    mood_df = pd.DataFrame(st.session_state.moods)
    st.write(mood_df)

    # Check if 'Mood' column exists before proceeding with mapping
    if 'Mood' in mood_df.columns:
        mood_df['Mood_Code'] = mood_df['Mood'].map({"Happy": 5, "Excited": 4, "Neutral": 3, "Sad": 2, "Anxious": 1})

        # Plot Mood Trend (simplified version)
        mood_trend_fig = plt.figure(figsize=(10, 5))
        plt.plot(mood_df['Date'], mood_df['Mood_Code'], marker='o', linestyle='-', color='b')
        plt.title("Mood Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Mood (Scale: 1-5)")
        plt.xticks(rotation=45)
        st.pyplot(mood_trend_fig)
    else:
        st.warning("No mood data available yet.")
# Make sure to initialize recipes in session state if it doesn't exist
if "recipes" not in st.session_state:
    st.session_state.recipes = []

# PCOS Recipes Section
st.header("7. PCOS Recipes 🍲")

# Sample Recipes (this can be expanded later with a real dataset or API)
recipes = [
    {
        "name": "Low GI Salad",
        "ingredients": ["Lettuce", "Tomatoes", "Cucumbers", "Olive Oil", "Lemon"],
        "instructions": "Chop all ingredients and mix them together. Drizzle with olive oil and lemon juice for added flavor.",
        "benefits": "Low glycemic index, helps in managing insulin levels and promotes overall health."
    },
    {
        "name": "Avocado Toast",
        "ingredients": ["Whole wheat bread", "Avocado", "Lemon juice", "Chili flakes", "Olive oil"],
        "instructions": "Toast the bread, spread mashed avocado on top, drizzle with lemon juice and olive oil. Sprinkle chili flakes for extra flavor.",
        "benefits": "Healthy fats, promotes balanced blood sugar levels and reduces inflammation."
    },
    {
        "name": "Chia Pudding",
        "ingredients": ["Chia seeds", "Almond milk", "Honey", "Vanilla extract", "Berries"],
        "instructions": "Mix chia seeds with almond milk and refrigerate overnight. Top with honey, vanilla extract, and fresh berries.",
        "benefits": "High in omega-3s, fiber, and antioxidants which support hormone balance and digestion."
    }
]

# Display recipes
for recipe in recipes:
    st.subheader(recipe["name"])
    st.write(f"**Ingredients:** {', '.join(recipe['ingredients'])}")
    st.write(f"**Instructions:** {recipe['instructions']}")
    st.write(f"**Health Benefits:** {recipe['benefits']}")
    st.markdown("---")

# Option for user to submit their own recipes
st.subheader("Submit Your Own Recipe ✍️")
recipe_name = st.text_input("Recipe Name")
ingredients = st.text_area("Ingredients (comma separated)")
instructions = st.text_area("Instructions")
benefits = st.text_area("Health Benefits")

if st.button("Submit Recipe"):
    if recipe_name and ingredients and instructions and benefits:
        new_recipe = {
            "name": recipe_name,
            "ingredients": ingredients.split(","),
            "instructions": instructions,
            "benefits": benefits
        }
        st.session_state.recipes.append(new_recipe)
        st.success("Recipe submitted successfully! 🌟")
    else:
        st.warning("Please fill in all the fields before submitting.")
        
# Display user-submitted recipes (ensure there are recipes in session state)
if st.session_state.recipes:
    st.write("### User-Submitted Recipes")
    for recipe in st.session_state.recipes:
        st.subheader(recipe["name"])
        st.write(f"**Ingredients:** {', '.join(recipe['ingredients'])}")
        st.write(f"**Instructions:** {recipe['instructions']}")
        st.write(f"**Health Benefits:** {recipe['benefits']}")
        st.markdown("---")


 
