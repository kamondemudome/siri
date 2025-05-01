import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime, timedelta
import os
import io
import json
import tempfile

# Set page config for a wide layout and custom title
st.set_page_config(page_title="Diabetes Risk Dashboard", layout="wide")

# Path to the shared settings file and model
SETTINGS_FILE = "settings.json"
MODEL_PATH = "new_diabetes_rf_model.pkl"

# Define features and ranges early, since they are needed for SHAP explainer initialization
FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'PhysActivity', 'Fruits', 
    'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
    'Sex', 'Age', 'Education', 'Income'
]

FEATURE_RANGES = {
    'HighBP': (0, 1), 'HighChol': (0, 1), 'CholCheck': (0, 1), 'Smoker': (0, 1),
    'PhysActivity': (0, 1), 'Fruits': (0, 1), 'Veggies': (0, 1), 'HvyAlcoholConsump': (0, 1),
    'NoDocbcCost': (0, 1), 'Sex': (0, 1), 'GenHlth': (1, 5), 'MentHlth': (0, 30),
    'PhysHlth': (0, 30), 'Age': (1, 13), 'Education': (1, 6), 'Income': (1, 8),
    'BMI': (10, 100)
}

# Ensure the directory exists
def ensure_directory_exists():
    directory = os.path.dirname(SETTINGS_FILE)
    if not directory:
        return
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            st.write(f"Created directory: {directory}")
    except Exception as e:
        st.error(f"Error creating directory {directory}: {str(e)}")
        raise

# Initialize the settings file with default values if it doesn't exist
def initialize_settings():
    default_settings = {
        "theme": "Light",
        "font_size": "Medium",
        "accent_color": "Blue"
    }
    try:
        ensure_directory_exists()
        if not os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "w") as f:
                json.dump(default_settings, f)
            st.write(f"Created settings file at: {SETTINGS_FILE}")
        return default_settings
    except Exception as e:
        st.warning(f"Error initializing settings file at {SETTINGS_FILE}: {str(e)}. Using default settings.")
        return default_settings

# Load settings from the JSON file
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Settings file not found at {SETTINGS_FILE}. Initializing with default settings.")
        return initialize_settings()
    except Exception as e:
        st.warning(f"Error loading settings from {SETTINGS_FILE}: {str(e)}. Using default settings.")
        return {
            "theme": "Light",
            "font_size": "Medium",
            "accent_color": "Blue"
        }

# Save settings to the JSON file
def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        st.warning(f"Error saving settings to {SETTINGS_FILE}: {str(e)}")

# Function to apply theme, font size, and accent color
def apply_settings(settings):
    theme = settings["theme"]
    font_size = settings["font_size"]
    accent_color = settings["accent_color"]

    if theme == "Dark":
        theme_css = """
        <style>
        /* General styling */
        body {
            font-family: 'Arial', sans-serif;
        }

        /* Main content styling */
        .stApp {
            background-color: #1e1e1e;
            color: #ecf0f1;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ecf0f1;
        }
        .stMarkdown p {
            color: #ecf0f1;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #2c3e50 !important;
        }
        section[data-testid="stSidebar"] .css-1v3fvcr {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .css-1v3fvcr:hover {
            background-color: #34495e !important;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stCheckbox > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stSlider > label {
            color: #ecf0f1 !important;
        }

        /* Card styling for sections */
        .card {
            background-color: #2c3e50;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }

        /* Form and input styling */
        .stForm {
            background-color: #2c3e50;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #34495e;
            color: #ecf0f1;
            border: 1px solid #34495e;
            border-radius: 5px;
        }

        /* Button styling */
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
        }
        .stButton button:hover {
            background-color: #2980b9;
        }

        /* Link styling */
        a {
            color: #66b3ff;
            text-decoration: none;
        }
        a:hover {
            color: #99ccff;
            text-decoration: underline;
        }

        /* Expander styling */
        .stExpander {
            background-color: #34495e;
            color: #ecf0f1;
            border-radius: 5px;
        }
        .stExpander summary {
            color: #ecf0f1 !important;
        }

        /* Footer styling */
        .footer {
            color: #bdc3c7;
            text-align: center;
            padding: 20px 0;
            margin-top: 20px;
            border-top: 1px solid #34495e;
        }
        .footer .message {
            font-weight: bold;
            font-size: 16px;
        }
        .footer .copyright, .footer .developer {
            font-size: 12px;
        }

        /* Health Card Styling */
        .health-card {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
            text-align: center;
        }
        .health-card h3 {
            color: #ecf0f1;
            margin-bottom: 10px;
        }
        .health-card .risk-level {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .health-card .goal {
            font-style: italic;
            color: #bdc3c7;
        }

        /* Slider Styling */
        .stSlider .css-1qrvynf {
            background-color: #34495e !important;
        }

        /* Plot Styling */
        .stPlotlyChart, .stPyplot {
            background-color: #2c3e50;
            border-radius: 5px;
            padding: 10px;
        }

        /* Disclaimer styling */
        .disclaimer {
            color: #bdc3c7;
        }

        /* Health Avatar Styling */
        .health-avatar {
            text-align: center;
            font-size: 48px;
            margin-bottom: 20px;
        }

        /* Metric Card Styling */
        .metric-card {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        .metric-card h4 {
            color: #ecf0f1;
            margin-bottom: 5px;
        }
        .metric-card .value {
            font-size: 20px;
            font-weight: bold;
            color: #3498db;
        }
        </style>
        """
    else:  # Light Mode
        theme_css = """
        <style>
        /* General styling */
        body {
            font-family: 'Arial', sans-serif;
        }

        /* Main content styling */
        .stApp {
            background-color: #f4f7fa;
            color: #34495e;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
        }
        .stMarkdown p {
            color: #34495e;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #2c3e50 !important;
        }
        section[data-testid="stSidebar"] .css-1v3fvcr {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .css-1v3fvcr:hover {
            background-color: #34495e !important;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stCheckbox > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stSlider > label {
            color: #ecf0f1 !important;
        }

        /* Card styling for sections */
        .card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Form and input styling */
        .stForm {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #ecf0f1;
            color: #34495e;
            border: 1px solid #d1d8e0;
            border-radius: 5px;
        }

        /* Button styling */
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
        }
        .stButton button:hover {
            background-color: #2980b9;
        }

        /* Link styling */
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            color: #2980b9;
            text-decoration: underline;
        }

        /* Expander styling */
        .stExpander {
            background-color: #ecf0f1;
            color: #34495e;
            border-radius: 5px;
        }
        .stExpander summary {
            color: #34495e !important;
        }

        /* Footer styling */
        .footer {
            color: #7f8c8d;
            text-align: center;
            padding: 20px 0;
            margin-top: 20px;
            border-top: 1px solid #d1d8e0;
        }
        .footer .message {
            font-weight: bold;
            font-size: 16px;
        }
        .footer .copyright, .footer .developer {
            font-size: 12px;
        }

        /* Health Card Styling */
        .health-card {
            background: linear-gradient(135deg, #ffffff, #ecf0f1);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .health-card h3 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .health-card .risk-level {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .health-card .goal {
            font-style: italic;
            color: #7f8c8d;
        }

        /* Slider Styling */
        .stSlider .css-1qrvynf {
            background-color: #ecf0f1 !important;
        }

        /* Plot Styling */
        .stPlotlyChart, .stPyplot {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 10px;
        }

        /* Disclaimer styling */
        .disclaimer {
            color: #7f8c8d;
        }

        /* Health Avatar Styling */
        .health-avatar {
            text-align: center;
            font-size: 48px;
            margin-bottom: 20px;
        }

        /* Metric Card Styling */
        .metric-card {
            background: linear-gradient(135deg, #ffffff, #ecf0f1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .metric-card h4 {
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .metric-card .value {
            font-size: 20px;
            font-weight: bold;
            color: #3498db;
        }
        </style>
        """

    font_sizes = {
        "Small": "12px",
        "Medium": "16px",
        "Large": "20px"
    }
    font_size_css = f"""
    <style>
    .stApp, .stMarkdown, .stText, .stSelectbox, .stSlider, .stNumberInput, .stButton button {{
        font-size: {font_sizes.get(font_size, "16px")} !important;
    }}
    </style>
    """

    accent_colors = {
        "Blue": "#3498db",
        "Green": "#2ecc71",
        "Red": "#e74c3c"
    }
    accent_color_css = f"""
    <style>
    .stButton button {{
        background-color: {accent_colors.get(accent_color, "#3498db")} !important;
        border-color: {accent_colors.get(accent_color, "#3498db")} !important;
    }}
    .stButton button:hover {{
        background-color: {accent_colors.get(accent_color, "#3498db")} !important;
        opacity: 0.8;
    }}
    a {{
        color: {accent_colors.get(accent_color, "#3498db")} !important;
    }}
    a:hover {{
        color: {accent_colors.get(accent_color, "#3498db")} !important;
        opacity: 0.8;
        text-decoration: underline;
    }}
    .health-card .risk-level {{
        color: {accent_colors.get(accent_color, "#3498db")} !important;
    }}
    .metric-card .value {{
        color: {accent_colors.get(accent_color, "#3498db")} !important;
    }}
    </style>
    """

    return theme_css + font_size_css + accent_color_css

# Simple prediction function (for Home page quick detection)
def quick_predict_diabetes(bmi, phys_activity, fruits, age):
    try:
        base_risk = 0.3
        if bmi > 30:
            base_risk += 0.2
        if phys_activity == 0:
            base_risk += 0.15
        if fruits == 0:
            base_risk += 0.1
        if age > 9:  # Age category > 60
            base_risk += 0.15
        risk_prob = min(max(base_risk, 0.0), 1.0)
        
        if risk_prob < 0.3:
            return "Low risk (<30%)", risk_prob
        elif risk_prob < 0.5:
            return "Moderate risk (30-50%)", risk_prob
        else:
            return "High risk (>50%)", risk_prob
    except Exception as e:
        st.error(f"Error in quick prediction: {str(e)}")
        return "Error calculating risk", 0.0

# Load the trained model (for Diabetes Detection Tool)
model = None
explainer = None
try:
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at {MODEL_PATH}. The 'Diabetes Detection Tool' section will be disabled. "
            "Please ensure the file 'new_diabetes_rf_model.pkl' is placed in the same directory as this script."
        )
    else:
        model = joblib.load(MODEL_PATH)
        # Verify the model is a binary classifier
        if not hasattr(model, 'n_classes_'):
            st.error("Loaded model does not appear to be a classifier. Please ensure the model is a scikit-learn binary classifier.")
            model = None
        elif model.n_classes_ != 2:
            st.error(f"Model is not a binary classifier. Expected 2 classes, but found {model.n_classes_}.")
            model = None
        else:
            try:
                # Use KernelExplainer instead of TreeExplainer to support any model type
                # KernelExplainer requires a function that returns predictions and some background data
                # We use a small sample of background data (mean values of features) for approximation
                background_data = pd.DataFrame(
                    {feat: [0.5 if FEATURE_RANGES[feat] == (0, 1) else sum(FEATURE_RANGES[feat]) / 2]
                     for feat in FEATURES},
                    columns=FEATURES
                )
                def model_predict(data):
                    return model.predict_proba(data)[:, 1]
                explainer = shap.KernelExplainer(model_predict, background_data)
            except Exception as e:
                st.warning(f"Error initializing SHAP explainer: {str(e)}. SHAP explanations will be skipped.")
                explainer = None
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATH}: {str(e)}. The 'Diabetes Detection Tool' section will be disabled.")

# Define feature descriptions (for Diabetes Detection Tool)
FEATURE_FULL_NAMES = {
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'CholCheck': 'Cholesterol Check in Last 5 Years',
    'BMI': 'Body Mass Index (BMI)',
    'Smoker': 'Smoking History (100+ Cigarettes)',
    'PhysActivity': 'Physical Activity in Past 30 Days',
    'Fruits': 'Daily Fruit Consumption',
    'Veggies': 'Daily Vegetable Consumption',
    'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
    'NoDocbcCost': 'Unable to See Doctor Due to Cost',
    'GenHlth': 'General Health Rating',
    'MentHlth': 'Days of Poor Mental Health',
    'PhysHlth': 'Days of Poor Physical Health',
    'Sex': 'Sex',
    'Age': 'Age Category',
    'Education': 'Education Level',
    'Income': 'Income Category'
}

FEATURE_DESCRIPTIONS = {
    'HighBP': 'Do you have high blood pressure?',
    'HighChol': 'Do you have high cholesterol?',
    'CholCheck': 'Have you had a cholesterol check in the last 5 years?',
    'BMI': 'What is your Body Mass Index (BMI)?',
    'Smoker': 'Have you smoked at least 100 cigarettes in your life?',
    'PhysActivity': 'Have you done physical activity in the past 30 days?',
    'Fruits': 'Do you consume fruit 1 or more times per day?',
    'Veggies': 'Do you consume vegetables 1 or more times per day?',
    'HvyAlcoholConsump': 'Do you engage in heavy alcohol consumption?',
    'NoDocbcCost': 'Were you unable to see a doctor due to cost in the past 12 months?',
    'GenHlth': 'How would you rate your general health? (1 = Excellent, 5 = Poor)',
    'MentHlth': 'How many days in the past 30 days was your mental health not good?',
    'PhysHlth': 'How many days in the past 30 days was your physical health not good?',
    'Sex': 'What is your sex?',
    'Age': 'What is your age category?',
    'Education': 'What is your education level?',
    'Income': 'What is your income category?'
}

FEATURE_TOOLTIPS = {
    'HighBP': 'Select "Yes" if you have been diagnosed with high blood pressure.',
    'HighChol': 'Select "Yes" if you have been diagnosed with high cholesterol.',
    'CholCheck': 'Select "Yes" if you’ve had your cholesterol checked in the last 5 years.',
    'BMI': 'Enter your BMI (e.g., 25.0). BMI = weight (kg) / height (m)^2.',
    'Smoker': 'Select "Yes" if you have smoked at least 100 cigarettes in your lifetime.',
    'PhysActivity': 'Select "Yes" if you’ve done any physical activity (e.g., walking, exercise) in the past 30 days.',
    'Fruits': 'Select "Yes" if you eat fruit at least once per day.',
    'Veggies': 'Select "Yes" if you eat vegetables at least once per day.',
    'HvyAlcoholConsump': 'Select "Yes" if you drink more than 14 drinks per week (men) or 7 drinks per week (women).',
    'NoDocbcCost': 'Select "Yes" if you couldn’t see a doctor due to cost in the past 12 months.',
    'GenHlth': 'Rate your overall health on a scale from 1 (excellent) to 5 (poor).',
    'MentHlth': 'Enter the number of days (0-30) you experienced poor mental health (e.g., stress, depression) in the past 30 days.',
    'PhysHlth': 'Enter the number of days (0-30) you experienced poor physical health (e.g., illness, injury) in the past 30 days.',
    'Sex': 'Select your biological sex.',
    'Age': 'Select your age category (e.g., 1 = 18-24, 9 = 55-59, 13 = 80+).',
    'Education': 'Select your highest level of education (e.g., 1 = Never attended, 6 = College graduate).',
    'Income': 'Select your annual income category (e.g., 1 = < Ksh 10000, 8 = Ksh 1000000+).'
}

AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"
}

EDUCATION_LABELS = {
    1: "Never attended school", 2: "Grades 1-8", 3: "Grades 9-11",
    4: "High school graduate", 5: "Some college", 6: "College graduate"
}

INCOME_LABELS = {
    1: "< Ksh 10000", 2: "Ksh 10000-15000", 3: "Ksh 15000-25000", 4: "Ksh 25000-35000",
    5: "Ksh 35000-50000", 6: "Ksh 50000-75000", 7: "Ksh 75000-100000", 8: "Ksh 1000000+"
}

def predict_diabetes(user_data, threshold, show_debug):
    if model is None:
        st.error("Prediction model is not available. Please ensure the model file is loaded correctly.")
        return None, None, None

    try:
        # Validate user data
        for feature, value in user_data.items():
            min_val, max_val = FEATURE_RANGES[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(f"Value for {feature} ({value}) is out of expected range [{min_val}, {max_val}].")

        input_df = pd.DataFrame([user_data], columns=FEATURES)
        input_df = input_df.astype({
            'HighBP': int, 'HighChol': int, 'CholCheck': int, 'Smoker': int,
            'PhysActivity': int, 'Fruits': int, 'Veggies': int, 'HvyAlcoholConsump': int,
            'NoDocbcCost': int, 'Sex': int, 'GenHlth': int, 'MentHlth': int,
            'PhysHlth': int, 'Age': int, 'Education': int, 'Income': int,
            'BMI': float
        })
        prob = model.predict_proba(input_df)[:, 1][0]
        prediction = 1 if prob >= threshold else 0
        
        shap_values = None
        if explainer is not None:
            try:
                # Compute SHAP values using KernelExplainer
                shap_values_raw = explainer.shap_values(input_df)
                if show_debug:
                    st.write("Debug: Type of shap_values_raw:", str(type(shap_values_raw)))
                    st.write("Debug: Structure of shap_values_raw:", shap_values_raw.shape if isinstance(shap_values_raw, np.ndarray) else [arr.shape for arr in shap_values_raw])
                
                # Handle SHAP output for binary classification
                if isinstance(shap_values_raw, np.ndarray):
                    if len(shap_values_raw.shape) == 2 and shap_values_raw.shape[0] == 1:
                        # Shape (1, n_features)
                        shap_values = shap_values_raw[0, :]
                    elif len(shap_values_raw.shape) == 1:
                        # Shape (n_features,)
                        shap_values = shap_values_raw
                    else:
                        st.warning("Unexpected SHAP values structure. SHAP explanations will be skipped.")
                        shap_values = None
                else:
                    st.warning("Unexpected SHAP values structure. SHAP explanations will be skipped.")
                    shap_values = None
                
                if shap_values is not None and len(shap_values.shape) != 1:
                    st.warning("SHAP values are not in the expected 1D format. SHAP explanations will be skipped.")
                    shap_values = None
                
                if show_debug and shap_values is not None:
                    st.write("Debug: Final shape of shap_values after processing:", shap_values.shape)
            except Exception as e:
                st.warning(f"Error processing SHAP values: {str(e)}. SHAP explanations will be skipped.")
                shap_values = None
        
        if shap_values is None:
            # Fallback: create a zero array for SHAP values
            shap_values = np.zeros(len(FEATURES))
        
        return prob, prediction, shap_values
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def categorize_risk(prob):
    try:
        if prob < 0.3:
            return "Low risk (<30%)"
        elif prob < 0.5:
            return "Moderate risk (30-50%)"
        else:
            return "High risk (>50%)"
    except Exception as e:
        st.error(f"Error categorizing risk: {str(e)}")
        return "Error"

def get_health_avatar(prob):
    try:
        if prob < 0.3:
            return "😊 **You're doing great!** (Low Risk)"
        elif prob < 0.5:
            return "😐 **Keep an eye on your health.** (Moderate Risk)"
        else:
            return "😟 **Take action to reduce your risk!** (High Risk)"
    except Exception as e:
        st.error(f"Error generating health avatar: {str(e)}")
        return "⚠️ Error"

def get_health_tips(user_data, shap_values):
    try:
        tips = []
        top_features = sorted(zip(FEATURES, shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]
        for feature, shap_value in top_features:
            if shap_value > 0:
                if feature == 'GenHlth' and user_data['GenHlth'] >= 4:
                    tips.append("Your general health rating is poor. Consider scheduling a check-up with your doctor to address any underlying health issues.")
                elif feature == 'BMI' and user_data['BMI'] >= 30:
                    tips.append("Your BMI is high. Consulting a dietitian or starting a weight management program may help reduce your risk.")
                elif feature == 'HighBP' and user_data['HighBP'] == 1:
                    tips.append("High blood pressure increases your risk. Monitor your blood pressure regularly and discuss management options with your doctor.")
                elif feature == 'PhysActivity' and user_data['PhysActivity'] == 0:
                    tips.append("Lack of physical activity increases your risk. Aim for at least 150 minutes of moderate exercise per week, such as brisk walking.")
                elif feature == 'HvyAlcoholConsump' and user_data['HvyAlcoholConsump'] == 1:
                    tips.append("Heavy alcohol consumption increases your risk. Consider reducing your alcohol intake to within recommended limits.")
        return tips
    except Exception as e:
        st.warning(f"Error generating health tips: {str(e)}")
        return []

def save_to_csv(user_data, prob, prediction):
    try:
        record = user_data.copy()
        record['Probability'] = prob
        record['Prediction'] = categorize_risk(prob)
        record['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        df = pd.DataFrame([record])
        history_file = "prediction_history.csv"
        if not os.path.exists(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)
    except Exception as e:
        st.warning(f"Error saving prediction history: {str(e)}")

def load_prediction_history():
    history_file = "prediction_history.csv"
    try:
        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error loading prediction history: {str(e)}")
        return pd.DataFrame()

def generate_pdf_report(user_data, prob, prediction, shap_values):
    try:
        # Use a temporary file to avoid write permission issues
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            filename = tmp.name
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        
        def check_new_page(y_position, space_needed=50):
            if y_position < space_needed:
                c.showPage()
                return height - 50
            return y_position

        styles = getSampleStyleSheet()
        style_normal = styles['Normal']
        style_normal.fontSize = 10
        style_normal.leading = 12

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, height - 50, "Diabetes Risk Prediction Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        y_position = height - 100

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Section 1: User Inputs")
        y_position -= 20

        table_data = [["Feature", "Value"]]
        for feature, value in user_data.items():
            feature_name = FEATURE_FULL_NAMES[feature]
            if feature in ['Age', 'Education', 'Income']:
                if feature == 'Age':
                    display_value = AGE_LABELS[value]
                elif feature == 'Education':
                    display_value = EDUCATION_LABELS[value]
                elif feature == 'Income':
                    display_value = INCOME_LABELS[value]
            elif feature == 'Sex':
                display_value = 'Male' if value == 1 else 'Female'
            elif feature in ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost']:
                display_value = 'Yes' if value == 1 else 'No'
            else:
                display_value = str(value)
            feature_paragraph = Paragraph(feature_name, style_normal)
            table_data.append([feature_paragraph, display_value])

        table = Table(table_data, colWidths=[3.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        table_width, table_height = table.wrap(width - 100, height)
        y_position = check_new_page(y_position, table_height + 20)
        table.drawOn(c, 50, y_position - table_height)
        y_position -= (table_height + 20)

        y_position = check_new_page(y_position, 100)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Section 2: Prediction Results")
        y_position -= 20
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, f"Probability of diabetes: {prob:.2%}")
        y_position -= 15
        c.drawString(50, y_position, f"Risk Level: {categorize_risk(prob)}")
        y_position -= 30

        y_position = check_new_page(y_position, 150)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Section 3: Key Factors Influencing the Prediction")
        y_position -= 20

        table_data = [["Feature", "Value", "Impact", "SHAP Value"]]
        top_features = sorted(zip(FEATURES, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feature, shap_value in top_features:
            feature_name = FEATURE_FULL_NAMES[feature]
            impact = "increases" if shap_value > 0 else "decreases"
            value = user_data[feature]
            if feature in ['Age', 'Education', 'Income']:
                if feature == 'Age':
                    display_value = AGE_LABELS[value]
                elif feature == 'Education':
                    display_value = EDUCATION_LABELS[value]
                elif feature == 'Income':
                    display_value = INCOME_LABELS[value]
            elif feature == 'Sex':
                display_value = 'Male' if value == 1 else 'Female'
            elif feature in ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost']:
                display_value = 'Yes' if value == 1 else 'No'
            else:
                display_value = str(value)
            feature_paragraph = Paragraph(feature_name, style_normal)
            table_data.append([feature_paragraph, display_value, impact, f"{shap_value:.3f}"])

        table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        table_width, table_height = table.wrap(width - 100, height)
        y_position = check_new_page(y_position, table_height + 20)
        table.drawOn(c, 50, y_position - table_height)
        y_position -= (table_height + 20)

        y_position = check_new_page(y_position, 300)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Section 4: Feature Impact Visualization")
        y_position -= 20

        feature_names = [f"{feat}: {user_data[feat]}" for feat in FEATURES]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=shap_values, y=feature_names, palette='coolwarm')
        plt.xlabel("SHAP Value (Impact on Prediction)")
        plt.title("Feature Contributions to Diabetes Risk Prediction")
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        img = Image(img_buffer, width=6*inch, height=3*inch)
        y_position = check_new_page(y_position, 3.5*inch)
        img.drawOn(c, 50, y_position - 3*inch)
        y_position -= 3.5*inch

        y_position = check_new_page(y_position, 150)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Section 5: Health Tips")
        y_position -= 20
        c.setFont("Helvetica", 12)
        tips = get_health_tips(user_data, shap_values)
        if tips:
            for tip in tips:
                lines = []
                current_line = ""
                for word in tip.split():
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.strip())
                
                for line in lines:
                    y_position = check_new_page(y_position, 15)
                    c.drawString(50, y_position, f"- {line}")
                    y_position -= 15
        else:
            y_position = check_new_page(y_position, 15)
            c.drawString(50, y_position, "No specific health tips based on your inputs. Maintain a healthy lifestyle to reduce your risk.")
            y_position -= 15

        y_position = check_new_page(y_position, 50)
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.grey)
        c.drawString(50, y_position, "Disclaimer: This prediction is for informational purposes only.")
        y_position -= 15
        c.drawString(50, y_position, "Please consult a healthcare professional for a medical diagnosis.")
        
        c.showPage()
        c.save()
        return filename
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None

# Sidebar navigation
st.sidebar.title("Diabetes Risk Dashboard")

# Page selection using a selectbox
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Diabetes Detection Tool", "Awareness", "Preventive Measures", "Reports & Progress", "Community Support"]
)

# Personalization settings in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Personalization")
with st.sidebar.expander("Customize Settings", expanded=True):
    settings = load_settings()
    theme = st.selectbox("Choose Theme", ["Light", "Dark"], index=0 if settings.get("theme", "Light") == "Light" else 1)
    font_size = st.selectbox("Font Size", ["Small", "Medium", "Large"], index=["Small", "Medium", "Large"].index(settings.get("font_size", "Medium")))
    accent_color = st.selectbox("Accent Color", ["Blue", "Green", "Red"], index=["Blue", "Green", "Red"].index(settings.get("accent_color", "Blue")))

# Daily Health Challenge (moved to sidebar for all pages)
st.sidebar.markdown("---")
st.sidebar.header("Daily Health Challenge")
if 'challenge_completed' not in st.session_state:
    st.session_state['challenge_completed'] = False
if 'challenge_accepted' not in st.session_state:
    st.session_state['challenge_accepted'] = False

daily_challenges = [
    "Drink 8 glasses of water today.",
    "Walk 5,000 steps today.",
    "Eat 2 servings of vegetables today.",
    "Get 7-8 hours of sleep tonight.",
    "Avoid sugary drinks today."
]
if 'current_challenge' not in st.session_state:
    st.session_state['current_challenge'] = np.random.choice(daily_challenges)

if not st.session_state['challenge_accepted']:
    st.sidebar.write(f"**Today’s Challenge:** {st.session_state['current_challenge']}")
    if st.sidebar.button("Accept Challenge"):
        st.session_state['challenge_accepted'] = True
        st.sidebar.success("Challenge accepted! Complete it before the day ends.")
else:
    st.sidebar.write(f"**Your Challenge:** {st.session_state['current_challenge']}")
    if not st.session_state['challenge_completed']:
        if st.sidebar.button("Mark as Completed"):
            st.session_state['challenge_completed'] = True
            st.sidebar.success("🎉 Congratulations! You've earned the 'Daily Health Star' badge!")
    else:
        st.sidebar.write("🎉 **Daily Health Star** badge earned!")

# Initialize session state with settings
if "settings" not in st.session_state:
    st.session_state["settings"] = settings

# Update settings if they have changed
new_settings = {
    "theme": theme,
    "font_size": font_size,
    "accent_color": accent_color
}
if new_settings != st.session_state["settings"]:
    st.session_state["settings"] = new_settings
    save_settings(new_settings)

# Apply the settings
st.markdown(apply_settings(st.session_state["settings"]), unsafe_allow_html=True)

# Define the footer HTML
footer_html = """
<div class="footer">
    <div class="message">Empower Your Health Journey – Stay Ahead of Diabetes!</div>
    <div class="copyright">© 2025 Diabetes Risk Dashboard</div>
    <div class="developer">Developed by Kamonde K. Mudome</div>
</div>
"""

# Main content based on the selected page
# --- Start of "Home" Page ---
if page == "Home":
    st.title("🏠 Home")
    st.markdown("### Your Personalized Health Dashboard")
    
    # Widget 1: Health Status Overview
    st.markdown('<div class="health-card">', unsafe_allow_html=True)
    st.markdown("#### Health Status Overview")
    if "latest_risk" not in st.session_state:
        st.session_state["latest_risk"] = "Moderate risk (30-50%)"
        st.session_state["latest_risk_prob"] = 0.45
    st.markdown(f'<div class="risk-level">{st.session_state["latest_risk"]}</div>', unsafe_allow_html=True)
    daily_goals = [
        "Drink 8 glasses of water.",
        "Walk 10,000 steps.",
        "Eat 2 servings of vegetables.",
        "Get 7-8 hours of sleep.",
        "Avoid sugary drinks today."
    ]
    if "selected_goal" not in st.session_state:
        st.session_state["selected_goal"] = np.random.choice(daily_goals)
    st.markdown(f'<div class="goal">Today’s Goal: {st.session_state["selected_goal"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Widget 2: Interactive Risk Assessment Tool
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Interactive Risk Assessment Tool")
    st.write("Adjust the sliders to see how lifestyle changes can impact your diabetes risk.")
    
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.slider("Your BMI", 15.0, 40.0, 25.0, step=0.1, help="Body Mass Index (weight in kg / height in m²)")
        phys_activity = st.selectbox("Physical Activity (past 30 days)", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Have you done physical activity in the past 30 days?")
    with col2:
        fruits = st.selectbox("Daily Fruit Consumption", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", help="Do you eat fruit at least once per day?")
        age = st.slider("Age Category", 1, 13, 5, help="1 = 18-24, 13 = 80+")
    
    risk_level, risk_prob = quick_predict_diabetes(bmi, phys_activity, fruits, age)
    st.write(f"**Simulated Risk Level:** {risk_level}")
    st.write(f"**Risk Probability:** {risk_prob:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Widget 3: Quick Detection Tool
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Quick Detection Tool")
    with st.form("quick_detection_form"):
        st.write("Enter basic information for a quick risk assessment:")
        quick_bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        quick_phys_activity = st.selectbox("Physical Activity", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="quick_phys")
        quick_fruits = st.selectbox("Daily Fruits", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="quick_fruits")
        quick_age = st.slider("Age Category", 1, 13, 5, key="quick_age")
        submitted = st.form_submit_button("Assess Risk")
    
    if submitted:
        quick_risk_level, quick_risk_prob = quick_predict_diabetes(quick_bmi, quick_phys_activity, quick_fruits, quick_age)
        st.session_state["latest_risk"] = quick_risk_level
        st.session_state["latest_risk_prob"] = quick_risk_prob
        st.write(f"**Quick Assessment Result:** {quick_risk_level}")
        st.write(f"**Probability:** {quick_risk_prob:.2%}")
        st.write("For a detailed analysis, visit the Diabetes Detection Tool section from the sidebar.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Widget 4: Insights & Tips Feed
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Insights & Tips Feed")
    with st.expander("Latest Health Tips"):
        st.write("- **Tip 1**: Replace sugary drinks with water or herbal tea to reduce your risk of diabetes.")
        st.write("- **Tip 2**: Aim for 30 minutes of moderate exercise, like brisk walking, most days of the week.")
    with st.expander("Featured Article: Understanding Diabetes"):
        st.write("""
        Diabetes is a chronic condition that affects how your body turns food into energy. There are two main types:
        - **Type 1 Diabetes**: An autoimmune condition where the body does not produce insulin.
        - **Type 2 Diabetes**: The body either resists insulin or doesn’t produce enough, often linked to lifestyle factors.
        Learn more about symptoms, causes, and management strategies in the Awareness section.
        """)
    with st.expander("Video: Healthy Eating Tips"):
        st.write("*(Video placeholder)* Watch a 5-minute guide on maintaining a balanced diet to prevent diabetes.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Widget 5: Community Widget
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Community Highlights")
    st.write("**Trending Discussions:**")
    st.write("- *'What are the best exercises for managing blood sugar?'* - 15 replies")
    st.write("- *'How can I reduce my BMI effectively?'* - 10 replies")
    st.write("Join the conversation in the Community Support section!")
    st.markdown('</div>', unsafe_allow_html=True)

    # Add Footer
    st.markdown(footer_html, unsafe_allow_html=True)
# --- End of "Home" Page ---

# --- Start of "Diabetes Detection Tool" Page ---
elif page == "Diabetes Detection Tool":
    st.title("🩺 Diabetes Risk Prediction Tool")
    if model is None:
        st.error(
            "The prediction model is not available. Please ensure 'new_diabetes_rf_model.pkl' is in the correct directory "
            "and restart the app. You can still explore other sections like Home, Awareness, and Preventive Measures."
        )
    else:
        st.markdown("""
        This tool predicts your risk of diabetes based on health, lifestyle, and demographic factors.
        Please fill out the form below to get your prediction.
        """)

        # Sidebar settings for Diabetes Detection Tool
        st.sidebar.markdown("---")
        st.sidebar.header("Prediction Settings")
        threshold = st.sidebar.slider("Prediction Threshold", 0.3, 0.5, 0.4, 0.05, 
                                    help="Adjust the threshold for classifying high vs. low risk. Lower values increase recall (more high-risk predictions).")
        save_history = st.sidebar.checkbox("Save prediction to history", value=False, 
                                        help="Save your inputs and prediction to a CSV file for record-keeping.")
        show_debug = st.sidebar.checkbox("Show debug output", value=False, help="Show debug information for developers.")

        # Form for user input
        with st.form("input_form"):
            st.header("Enter Your Information")
            
            # Health Section
            st.subheader("Health Information")
            col1, col2 = st.columns(2)
            with col1:
                user_data = {}
                user_data['HighBP'] = st.selectbox(FEATURE_DESCRIPTIONS['HighBP'], [0, 1], 
                                                format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                help=FEATURE_TOOLTIPS['HighBP'])
                user_data['HighChol'] = st.selectbox(FEATURE_DESCRIPTIONS['HighChol'], [0, 1], 
                                                    format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                    help=FEATURE_TOOLTIPS['HighChol'])
                user_data['CholCheck'] = st.selectbox(FEATURE_DESCRIPTIONS['CholCheck'], [0, 1], 
                                                    format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                    help=FEATURE_TOOLTIPS['CholCheck'])
                user_data['BMI'] = st.number_input(FEATURE_DESCRIPTIONS['BMI'], min_value=10.0, max_value=100.0, 
                                                value=25.0, step=0.1, help=FEATURE_TOOLTIPS['BMI'])
            with col2:
                user_data['GenHlth'] = st.slider(FEATURE_DESCRIPTIONS['GenHlth'], 1, 5, 3, 
                                                help=FEATURE_TOOLTIPS['GenHlth'])
                user_data['MentHlth'] = st.slider(FEATURE_DESCRIPTIONS['MentHlth'], 0, 30, 0, 
                                                help=FEATURE_TOOLTIPS['MentHlth'])
                user_data['PhysHlth'] = st.slider(FEATURE_DESCRIPTIONS['PhysHlth'], 0, 30, 0, 
                                                help=FEATURE_TOOLTIPS['PhysHlth'])
                user_data['NoDocbcCost'] = st.selectbox(FEATURE_DESCRIPTIONS['NoDocbcCost'], [0, 1], 
                                                        format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                        help=FEATURE_TOOLTIPS['NoDocbcCost'])

            # Lifestyle Section
            st.subheader("Lifestyle Information")
            col3, col4 = st.columns(2)
            with col3:
                user_data['Smoker'] = st.selectbox(FEATURE_DESCRIPTIONS['Smoker'], [0, 1], 
                                                format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                help=FEATURE_TOOLTIPS['Smoker'])
                user_data['PhysActivity'] = st.selectbox(FEATURE_DESCRIPTIONS['PhysActivity'], [0, 1], 
                                                        format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                        help=FEATURE_TOOLTIPS['PhysActivity'])
                user_data['Fruits'] = st.selectbox(FEATURE_DESCRIPTIONS['Fruits'], [0, 1], 
                                                format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                help=FEATURE_TOOLTIPS['Fruits'])
            with col4:
                user_data['Veggies'] = st.selectbox(FEATURE_DESCRIPTIONS['Veggies'], [0, 1], 
                                                    format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                    help=FEATURE_TOOLTIPS['Veggies'])
                user_data['HvyAlcoholConsump'] = st.selectbox(FEATURE_DESCRIPTIONS['HvyAlcoholConsump'], [0, 1], 
                                                            format_func=lambda x: 'No' if x == 0 else 'Yes', 
                                                            help=FEATURE_TOOLTIPS['HvyAlcoholConsump'])

            # Demographics Section
            st.subheader("Demographic Information")
            col5, col6 = st.columns(2)
            with col5:
                user_data['Sex'] = st.selectbox(FEATURE_DESCRIPTIONS['Sex'], [0, 1], 
                                                format_func=lambda x: 'Female' if x == 0 else 'Male', 
                                                help=FEATURE_TOOLTIPS['Sex'])
                user_data['Age'] = st.selectbox(FEATURE_DESCRIPTIONS['Age'], list(range(1, 14)), 
                                                format_func=lambda x: AGE_LABELS[x], 
                                                help=FEATURE_TOOLTIPS['Age'])
            with col6:
                user_data['Education'] = st.selectbox(FEATURE_DESCRIPTIONS['Education'], list(range(1, 7)), 
                                                    format_func=lambda x: EDUCATION_LABELS[x], 
                                                    help=FEATURE_TOOLTIPS['Education'])
                user_data['Income'] = st.selectbox(FEATURE_DESCRIPTIONS['Income'], list(range(1, 9)), 
                                                format_func=lambda x: INCOME_LABELS[x], 
                                                help=FEATURE_TOOLTIPS['Income'])

            # Form buttons
            col7, col8 = st.columns(2)
            with col7:
                submitted = st.form_submit_button("Predict")
            with col8:
                reset = st.form_submit_button("Reset")

        if reset:
            st.rerun()

        if submitted:
            prob, prediction, shap_values = predict_diabetes(user_data, threshold, show_debug)
            if prob is not None:
                # Always save to history regardless of save_history checkbox
                save_to_csv(user_data, prob, prediction)

                st.write("### Prediction Results")
                st.markdown(f'<div class="health-avatar">{get_health_avatar(prob)}</div>', unsafe_allow_html=True)
                st.write(f"**Probability of diabetes:** {prob:.2%}")
                st.write(f"**Risk Level:** {categorize_risk(prob)}")

                st.write("### Key Factors Influencing the Prediction")
                top_features = sorted(zip(FEATURES, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
                for feature, shap_value in top_features:
                    impact = "increases" if shap_value > 0 else "decreases"
                    st.write(f"- **{feature}**: {user_data[feature]} ({impact} risk, SHAP value: {shap_value:.3f})")

                st.write("#### Feature Impact Visualization")
                feature_names = [f"{feat}: {user_data[feat]}" for feat in FEATURES]
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=shap_values, y=feature_names, palette='coolwarm')
                plt.xlabel("SHAP Value (Impact on Prediction)")
                plt.title("Feature Contributions to Diabetes Risk Prediction")
                if st.session_state["settings"]["theme"] == "Dark":
                    ax.set_facecolor('#2c3e50')
                    fig.set_facecolor('#2c3e50')
                    ax.tick_params(colors='#ecf0f1')
                    ax.xaxis.label.set_color('#ecf0f1')
                    ax.yaxis.label.set_color('#ecf0f1')
                    ax.title.set_color('#ecf0f1')
                else:
                    ax.set_facecolor('#ffffff')
                    fig.set_facecolor('#ffffff')
                    ax.tick_params(colors='#34495e')
                    ax.xaxis.label.set_color('#34495e')
                    ax.yaxis.label.set_color('#34495e')
                    ax.title.set_color('#2c3e50')
                st.pyplot(fig)
                plt.close(fig)  # Close the figure to free memory

                st.write("### Health Tips")
                tips = get_health_tips(user_data, shap_values)
                if tips:
                    for tip in tips:
                        st.write(f"- {tip}")
                else:
                    st.write("No specific health tips based on your inputs. Maintain a healthy lifestyle to reduce your risk.")

                pdf_file = generate_pdf_report(user_data, prob, prediction, shap_values)
                if pdf_file:
                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            "Download PDF Report", 
                            f, 
                            file_name=f"diabetes_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", 
                            mime="application/pdf"
                        )
                    # Clean up the temporary file
                    try:
                        os.remove(pdf_file)
                    except Exception as e:
                        st.warning(f"Error cleaning up temporary PDF file: {str(e)}")

                st.markdown('<div class="disclaimer">**Note:** This prediction is for informational purposes only. Please consult a healthcare professional for a medical diagnosis.</div>', unsafe_allow_html=True)

    # Add Footer
    st.markdown(footer_html, unsafe_allow_html=True)
# --- End of "Diabetes Detection Tool" Page ---

# --- Start of "Awareness" Page ---
elif page == "Awareness":
    st.title("📚 Awareness")
    st.markdown("### Educational Content on Diabetes")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Articles")
    with st.expander("What is Diabetes?"):
        st.write("""
        Diabetes is a chronic condition that affects how your body turns food into energy. There are two main types:
        - **Type 1 Diabetes**: An autoimmune condition where the body does not produce insulin.
        - **Type 2 Diabetes**: The body either resists insulin or doesn’t produce enough, often linked to lifestyle factors.
        Learn more about symptoms, causes, and management strategies.
        """)
    with st.expander("Risk Factors for Diabetes"):
        st.write("""
        Key risk factors include:
        - High BMI (>30)
        - Lack of physical activity
        - Poor diet (low fruit/vegetable intake)
        - Family history of diabetes
        - High blood pressure or cholesterol
        Understanding these factors can help you take preventive steps.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Videos")
    st.write("**Understanding Diabetes (Video)** - A 5-minute overview of diabetes causes and management. *(Link to be added)*")
    st.write("**Healthy Eating Tips (Video)** - Tips to maintain a balanced diet to prevent diabetes. *(Link to be added)*")
    st.markdown('</div>', unsafe_allow_html=True)

    # Add Footer
    st.markdown(footer_html, unsafe_allow_html=True)
# --- End of "Awareness" Page ---

# --- Start of "Preventive Measures" Page ---
elif page == "Preventive Measures":
    st.title("🛡️ Preventive Measures")
    st.markdown("### Personalized Tips and Health Plans")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Tips to Reduce Diabetes Risk")
    st.write("- **Maintain a Healthy Weight**: Aim for a BMI below 25. Regular exercise and a balanced diet can help.")
    st.write("- **Stay Active**: Engage in at least 150 minutes of moderate exercise per week, such as brisk walking.")
    st.write("- **Eat a Balanced Diet**: Include more fruits, vegetables, and whole grains while reducing processed foods and sugars.")
    st.write("- **Monitor Your Health**: Regular check-ups for blood pressure, cholesterol, and blood sugar levels can help detect issues early.")
    st.write("- **Limit Alcohol**: Reduce alcohol consumption to within recommended limits (e.g., up to 1 drink per day for women, 2 for men).")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Sample Health Plan")
    st.write("**Daily Routine:**")
    st.write("- Morning: 30-minute walk")
    st.write("- Meals: Include a serving of vegetables in every meal")
    st.write("- Evening: 15-minute stretching or yoga")
    st.write("**Weekly Goals:**")
    st.write("- Exercise: 5 days of moderate activity")
    st.write("- Diet: Reduce sugary drinks to 1 per week")
    st.markdown('</div>', unsafe_allow_html=True)

    # Add Footer
    st.markdown(footer_html, unsafe_allow_html=True)
# --- End of "Preventive Measures" Page ---

# --- Start of "Reports & Progress" Page ---
elif page == "Reports & Progress":
    st.title("📊 Reports & Progress")
    st.markdown("### Track Your Diabetes Risk Over Time")
    
    # Load prediction history
    history_df = load_prediction_history()
    
    # Overview Section
    st.markdown('<div class="health-card">', unsafe_allow_html=True)
    st.markdown("#### Progress Overview")
    if not history_df.empty:
        avg_risk = history_df['Probability'].mean()
        min_risk = history_df['Probability'].min()
        max_risk = history_df['Probability'].max()
        trend = "Decreasing 📉" if history_df['Probability'].iloc[-1] < history_df['Probability'].iloc[0] else "Increasing 📈"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Average Risk")
            st.markdown(f'<div class="value">{avg_risk:.2%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Lowest Risk")
            st.markdown(f'<div class="value">{min_risk:.2%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Highest Risk")
            st.markdown(f'<div class="value">{max_risk:.2%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Trend")
            st.markdown(f'<div class="value">{trend}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No prediction history available. Use the Diabetes Detection Tool to start tracking your risk.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Past Detection Results Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Past Detection Results")
    if not history_df.empty:
        st.write("**Your Prediction History (Sortable & Filterable):**")
        
        # Add date range filter
        min_date = history_df['Timestamp'].min().date()
        max_date = history_df['Timestamp'].max().date()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter by date range
        filtered_df = history_df[
            (history_df['Timestamp'].dt.date >= start_date) & 
            (history_df['Timestamp'].dt.date <= end_date)
        ]
        
        # Filter by risk level
        risk_levels = ["All", "Low risk (<30%)", "Moderate risk (30-50%)", "High risk (>50%)"]
        selected_risk = st.selectbox("Filter by Risk Level", risk_levels)
        if selected_risk != "All":
            filtered_df = filtered_df[filtered_df['Prediction'] == selected_risk]
        
        if not filtered_df.empty:
            # Prepare display DataFrame
            display_df = filtered_df[['Timestamp', 'Prediction', 'Probability']].copy()
            display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.2%}")
            display_df['Details'] = ""
            
            # Display the table
            st.dataframe(
                display_df,
                column_config={
                    "Timestamp": "Date & Time",
                    "Prediction": "Risk Level",
                    "Probability": "Risk Probability",
                    "Details": st.column_config.TextColumn("Details")
                },
                use_container_width=True
            )
            
            # Add expanders for each row
            for idx, row in filtered_df.iterrows():
                with st.expander(f"Details for {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                    for feature in FEATURES:
                        value = row[feature]
                        if feature in ['Age', 'Education', 'Income']:
                            if feature == 'Age':
                                display_value = AGE_LABELS[value]
                            elif feature == 'Education':
                                display_value = EDUCATION_LABELS[value]
                            elif feature == 'Income':
                                display_value = INCOME_LABELS[value]
                        elif feature == 'Sex':
                            display_value = 'Male' if value == 1 else 'Female'
                        elif feature in ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost']:
                            display_value = 'Yes' if value == 1 else 'No'
                        else:
                            display_value = str(value)
                        st.write(f"- **{FEATURE_FULL_NAMES[feature]}**: {display_value}")
            
            # Download button for filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Prediction History as CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.write("No predictions match the selected filters.")
    else:
        st.write("No prediction history available. Use the Diabetes Detection Tool to start tracking your risk.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Health Trends Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Health Trends Over Time")
    if not history_df.empty:
        # Time range filter
        time_range = st.selectbox("Select Time Range", ["Last 30 Days", "Last 90 Days", "All Time"])
        filtered_trend_df = history_df.copy()
        if time_range == "Last 30 Days":
            filtered_trend_df = filtered_trend_df[filtered_trend_df['Timestamp'] >= datetime.now() - timedelta(days=30)]
        elif time_range == "Last 90 Days":
            filtered_trend_df = filtered_trend_df[filtered_trend_df['Timestamp'] >= datetime.now() - timedelta(days=90)]
        
        if not filtered_trend_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the risk probability with a filled area
            ax.fill_between(
                filtered_trend_df['Timestamp'],
                filtered_trend_df['Probability'],
                color='blue',
                alpha=0.1
            )
            ax.plot(
                filtered_trend_df['Timestamp'],
                filtered_trend_df['Probability'],
                marker='o',
                color='blue',
                label='Diabetes Risk Probability'
            )
            
            # Calculate and plot moving average
            if len(filtered_trend_df) >= 3:
                moving_avg = filtered_trend_df['Probability'].rolling(window=3, min_periods=1).mean()
                ax.plot(
                    filtered_trend_df['Timestamp'],
                    moving_avg,
                    color='orange',
                    linestyle='--',
                    label='3-Point Moving Average'
                )
            
            # Highlight significant changes
            if len(filtered_trend_df) >= 2:
                changes = filtered_trend_df['Probability'].diff().abs()
                significant_change = changes > 0.1  # Threshold for significant change
                for idx, (timestamp, change, prob) in enumerate(zip(
                    filtered_trend_df['Timestamp'][significant_change],
                    changes[significant_change],
                    filtered_trend_df['Probability'][significant_change]
                )):
                    ax.annotate(
                        f"Change: {change:.2%}",
                        (timestamp, prob),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        color='red'
                    )
            
            ax.set_xlabel("Date")
            ax.set_ylabel("Risk Probability")
            ax.set_title("Your Diabetes Risk Over Time")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if st.session_state["settings"]["theme"] == "Dark":
                ax.set_facecolor('#2c3e50')
                fig.set_facecolor('#2c3e50')
                ax.tick_params(colors='#ecf0f1')
                ax.xaxis.label.set_color('#ecf0f1')
                ax.yaxis.label.set_color('#ecf0f1')
                ax.title.set_color('#ecf0f1')
            else:
                ax.set_facecolor('#ffffff')
                fig.set_facecolor('#ffffff')
                ax.tick_params(colors='#34495e')
                ax.xaxis.label