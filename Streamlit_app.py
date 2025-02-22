import pandas as pd
import numpy as np
import speech_recognition as sr
import pyttsx3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load Dataset (Replace with actual file path)
df = pd.read_csv("PCOS_data.csv")

# Preprocessing: Handling missing values
df = df.apply(pd.to_numeric, errors='coerce')  # Convert all to numeric
df.fillna(df.median(numeric_only=True), inplace=True)

# Checking actual column names
print(df.columns)

# Replace target column name with correct one
target_column = "PCOS (Yes/No)"  # Adjust this based on actual dataset
X = df.drop(columns=[target_column])
y = df[target_column]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Splitting data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model Training: Random Forest & XGBoost
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(rf, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

# Best Model Selection
best_rf = grid_search.best_estimator_

# Model Evaluation
y_pred = best_rf.predict(X_test)
y_probs = best_rf.predict_proba(X_test)[:, 1]
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_probs))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig("precision_recall_curve.png")
plt.show()

# Explainability using SHAP
explainer = shap.Explainer(best_rf, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png")

# Function to Generate Suggestions
def generate_suggestions(prediction_prob):
    if prediction_prob > 0.7:
        return "High risk: Consult a gynecologist. Focus on a healthy diet, exercise, and regular check-ups."
    elif prediction_prob > 0.4:
        return "Moderate risk: Maintain a balanced diet and active lifestyle. Monitor symptoms and consider medical consultation."
    else:
        return "Low risk: Keep up with healthy habits. Regular monitoring is recommended."

# Function to Generate Lifestyle & Diet Plan
def generate_lifestyle_plan(prediction_prob):
    if prediction_prob > 0.7:
        return "Low-GI foods, high-fiber diet, regular exercise (strength & cardio), stress management."
    elif prediction_prob > 0.4:
        return "Balanced diet with lean protein, healthy fats, moderate carbs, and light exercise."
    else:
        return "Maintain a healthy diet with portion control and stay active."

# Generate PDF Report
def generate_pdf_report(y_test, y_pred, y_probs, filename="PCOS_Report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="PCOS Prediction Report", ln=True, align='C')
    pdf.ln(10)
    
    # Adding Prediction Results
    pdf.cell(200, 10, txt=f"AUC-ROC Score: {roc_auc_score(y_test, y_probs):.2f}", ln=True)
    pdf.cell(200, 10, txt="Classification Report:", ln=True)
    pdf.multi_cell(0, 10, txt=classification_report(y_test, y_pred))
    pdf.ln(10)
    
    # Adding Suggestion & Lifestyle Plan
    for idx, prob in enumerate(y_probs[:5]):  # Limit to 5 examples
        pdf.cell(200, 10, txt=f"Sample {idx+1}: Risk Score = {prob:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Suggestion: {generate_suggestions(prob)}", ln=True)
        pdf.cell(200, 10, txt=f"Lifestyle & Diet Plan: {generate_lifestyle_plan(prob)}", ln=True)
        pdf.ln(5)
    
    pdf.output(filename)
    print("PDF Report Generated!")

# Generate Report
generate_pdf_report(y_test, y_pred, y_probs)

# Voice Assistant for PCOS-related Queries
def voice_assistant():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    
    engine.say("Hello! You can ask me about PCOS symptoms, diet plans, or suggestions.")
    engine.runAndWait()
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
        try:
            query = recognizer.recognize_google(audio).lower()
            print("User said:", query)
            
            if "symptoms" in query:
                response = "Common symptoms of PCOS include irregular periods, excessive hair growth, acne, and weight gain."
            elif "diet" in query:
                response = "A balanced diet with fiber, lean proteins, and low carbs can help manage PCOS."
            elif "lifestyle" in query:
                response = "A healthy lifestyle with regular exercise, stress management, and balanced meals is crucial for PCOS management."
            else:
                response = "I can provide information on PCOS symptoms, lifestyle tips, and more. Just ask!"
            
            print("Assistant:", response)
            engine.say(response)
            engine.runAndWait()
        
        except sr.UnknownValueError:
            print("Sorry, I didn't understand. Please try again.")
            engine.say("Sorry, I didn't understand. Please try again.")
            engine.runAndWait()

# Run the Voice Assistant
voice_assistant()
