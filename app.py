import streamlit as st
import pandas as pd
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
from datetime import datetime
import os
import numpy as np
import io
import json

# Path to the shared settings file (relative to the parent directory)
SETTINGS_FILE = "../settings.json"

# Path to the model file (relative to the parent directory)
MODEL_PATH = "../new_diabetes_rf_model.pkl"

# Ensure the directory exists
def ensure_directory_exists():
    directory = os.path.dirname(SETTINGS_FILE)
    # If directory is empty, the file is in the current working directory; no need to create a directory
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
        st.error(f"Error initializing settings file at {SETTINGS_FILE}: {str(e)}")
        return default_settings

# Load settings from the JSON file
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Settings file not found at {SETTINGS_FILE}. Using default settings.")
        return initialize_settings()
    except Exception as e:
        st.error(f"Error loading settings from {SETTINGS_FILE}: {str(e)}")
        return {
            "theme": "Light",
            "font_size": "Medium",
            "accent_color": "Blue"
        }

# Function to apply theme, font size, and accent color (copied from index.py for consistency)
def apply_settings(settings):
    theme = settings["theme"]
    font_size = settings["font_size"]
    accent_color = settings["accent_color"]

    # Theme CSS
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
        section[data-testid="stSidebar"] .stCheckbox > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stSlider > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox > label {
            color: #ecf0f1 !important;
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
        .stSlider .css-1qrvynf {
            background-color: #34495e !important;
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

        /* Plot styling */
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

        /* Home Icon Styling */
        .home-icon {
            position: absolute;
            top: 10px;
            left: 10px;
            font-size: 24px;
            cursor: pointer;
        }
        .home-icon a {
            color: #ecf0f1;
            text-decoration: none;
        }
        .home-icon a:hover {
            color: #99ccff;
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
        section[data-testid="stSidebar"] .stCheckbox > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stSlider > label {
            color: #ecf0f1 !important;
        }
        section[data-testid="stSidebar"] .stSelectbox > label {
            color: #ecf0f1 !important;
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
        .stSlider .css-1qrvynf {
            background-color: #ecf0f1 !important;
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

        /* Plot styling */
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

        /* Home Icon Styling */
        .home-icon {
            position: absolute;
            top: 10px;
            left: 10px;
            font-size: 24px;
            cursor: pointer;
        }
        .home-icon a {
            color: #34495e;
            text-decoration: none;
        }
        .home-icon a:hover {
            color: #2980b9;
        }
        </style>
        """

    # Font Size CSS
    font_sizes = {
        "Small": "12px",
        "Medium": "16px",
        "Large": "20px"
    }
    font_size_css = f"""
    <style>
    .stApp, .stMarkdown, .stText, .stSelectbox, .stSlider, .stNumberInput, .stButton button {{
        font-size: {font_sizes[font_size]} !important;
    }}
    </style>
    """

    # Accent Color CSS
    accent_colors = {
        "Blue": "#3498db",
        "Green": "#2ecc71",
        "Red": "#e74c3c"
    }
    accent_color_css = f"""
    <style>
    .stButton button {{
        background-color: {accent_colors[accent_color]} !important;
        border-color: {accent_colors[accent_color]} !important;
    }}
    .stButton button:hover {{
        background-color: {accent_colors[accent_color]} !important;
        opacity: 0.8;
    }}
    a {{
        color: {accent_colors[accent_color]} !important;
    }}
    a:hover {{
        color: {accent_colors[accent_color]} !important;
        opacity: 0.8;
        text-decoration: underline;
    }}
    </style>
    """

    # Apply all CSS
    return theme_css + font_size_css + accent_color_css

# Load the trained model
try:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the file 'new_diabetes_rf_model.pkl' is placed in the correct directory and the path is updated in the script.")
        st.stop()
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATH}: {str(e)}")
    st.stop()

# Verify the model is a binary classifier
if model.n_classes_ != 2:
    st.error(f"Model is not a binary classifier. Expected 2 classes, but found {model.n_classes_}.")
    st.stop()

# Initialize SHAP explainer
try:
    explainer = shap.TreeExplainer(model)
except Exception as e:
    st.error(f"Error initializing SHAP explainer: {str(e)}")
    st.stop()

# Define features and descriptions
FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'PhysActivity', 'Fruits', 
    'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
    'Sex', 'Age', 'Education', 'Income'
]

# Full feature names for the PDF report (not in sentence form)
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

# Feature descriptions for data collection (in sentence form)
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
    'Income': 'Select your annual income category (e.g., 1 = < $10,000, 8 = $75,000+).'
}

FEATURE_RANGES = {
    'HighBP': (0, 1), 'HighChol': (0, 1), 'CholCheck': (0, 1), 'Smoker': (0, 1),
    'PhysActivity': (0, 1), 'Fruits': (0, 1), 'Veggies': (0, 1), 'HvyAlcoholConsump': (0, 1),
    'NoDocbcCost': (0, 1), 'Sex': (0, 1), 'GenHlth': (1, 5), 'MentHlth': (0, 30),
    'PhysHlth': (0, 30), 'Age': (1, 13), 'Education': (1, 6), 'Income': (1, 8),
    'BMI': (10, 100)
}

# Descriptive labels for Age, Education, and Income
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"
}

EDUCATION_LABELS = {
    1: "Never attended school", 2: "Grades 1-8", 3: "Grades 9-11",
    4: "High school graduate", 5: "Some college", 6: "College graduate"
}

INCOME_LABELS = {
    1: "< $10,000", 2: "$10,000-$15,000", 3: "$15,000-$25,000", 4: "$25,000-$35,000",
    5: "$35,000-$50,000", 6: "$50,000-$75,000", 7: "$75,000-$100,000", 8: "$100,000+"
}

def predict_diabetes(user_data, threshold, show_debug):
    try:
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
        
        # Get SHAP values
        shap_values = explainer.shap_values(input_df)
        if show_debug:
            st.write("Debug: Type of shap_values:", str(type(shap_values)))
            st.write("Debug: Shape of shap_values:", shap_values.shape if isinstance(shap_values, np.ndarray) else [arr.shape for arr in shap_values])
        
        # Handle SHAP values based on their structure
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            if shap_values.shape[0] == 1 and shap_values.shape[2] == 2:
                shap_values = np.transpose(shap_values, (2, 0, 1))
            shap_values = shap_values[1]
            shap_values = shap_values[0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
            shap_values = shap_values[0]
        else:
            raise ValueError(f"Unexpected SHAP values structure: {type(shap_values)}.")
        
        if show_debug:
            st.write("Debug: Final shape of shap_values after processing:", shap_values.shape)
        
        return prob, prediction, shap_values
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def categorize_risk(prob):
    if prob < 0.3:
        return "Low risk (<30%)"
    elif prob < 0.5:
        return "Moderate risk (30-50%)"
    else:
        return "High risk (>50%)"

def get_health_avatar(prob):
    if prob < 0.3:
        return "😊 **You're doing great!** (Low Risk)"
    elif prob < 0.5:
        return "😐 **Keep an eye on your health.** (Moderate Risk)"
    else:
        return "😟 **Take action to reduce your risk!** (High Risk)"

def get_health_tips(user_data, shap_values):
    tips = []
    top_features = sorted(zip(FEATURES, shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]
    for feature, shap_value in top_features:
        if shap_value > 0:
            if feature == 'GenHlth' and user_data['GenHlth'] >= 4:
                tips.append("Your general health rating is poor. Consider scheduling a check-up with your doctor to address any underlying health issues.")
            elif feature == 'BMI' and user_data['BMI'] >= 30:
                tips.append("Your BMI is high. Consulting a dietitian or starting a weight management program may help reduce your diabetes risk.")
            elif feature == 'HighBP' and user_data['HighBP'] == 1:
                tips.append("High blood pressure increases your risk. Monitor your blood pressure regularly and discuss management options with your doctor.")
            elif feature == 'PhysActivity' and user_data['PhysActivity'] == 0:
                tips.append("Lack of physical activity increases your risk. Aim for at least 150 minutes of moderate exercise per week, such as brisk walking.")
            elif feature == 'HvyAlcoholConsump' and user_data['HvyAlcoholConsump'] == 1:
                tips.append("Heavy alcohol consumption increases your risk. Consider reducing your alcohol intake to within recommended limits.")
    return tips

def save_to_csv(user_data, prob, prediction):
    record = user_data.copy()
    record['Probability'] = prob
    record['Prediction'] = categorize_risk(prob)
    record['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    df = pd.DataFrame([record])
    if not os.path.exists('../prediction_history.csv'):
        df.to_csv('../prediction_history.csv', index=False)
    else:
        df.to_csv('../prediction_history.csv', mode='a', header=False, index=False)

def load_prediction_history():
    if os.path.exists('../prediction_history.csv'):
        try:
            df = pd.read_csv('../prediction_history.csv')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except Exception as e:
            st.error(f"Error loading prediction history: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

def generate_pdf_report(user_data, prob, prediction, shap_values):
    filename = f"diabetes_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
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

def main():
    # Load and apply settings
    settings = load_settings()
    st.markdown(apply_settings(settings), unsafe_allow_html=True)

    # Home Icon (links to index.py, the home page)
    st.markdown(
        '<div class="home-icon">'
        '<a href="../index.py" title="Back to Home">🏠</a>'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("Diabetes Risk Prediction Tool")
    st.markdown("""
    This tool predicts your risk of diabetes based on health, lifestyle, and demographic factors.
    Please fill out the form below to get your prediction.
    """)

    # Sidebar for settings
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Prediction Threshold", 0.3, 0.5, 0.4, 0.05, 
                                 help="Adjust the threshold for classifying high vs. low risk. Lower values increase recall (more high-risk predictions).")
    save_history = st.sidebar.checkbox("Save prediction to history", value=False, 
                                       help="Save your inputs and prediction to a CSV file for record-keeping.")
    show_debug = st.sidebar.checkbox("Show debug output", value=False, help="Show debug information for developers.")

    # Navigation Links (using st.page_link for multi-page navigation)
    st.sidebar.header("Navigation")
    st.page_link("index.py", label="Go to Home")
    st.page_link("pages/control_panel.py", label="Go to Control Panel")

    # Health Challenge Feature
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
        st.experimental_rerun()

    if submitted:
        prob, prediction, shap_values = predict_diabetes(user_data, threshold, show_debug)
        if prob is not None:
            if save_history:
                save_to_csv(user_data, prob, prediction)

            st.write("### Prediction Results")
            # Health Avatar
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
            if settings["theme"] == "Dark":
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

            st.write("### Health Tips")
            tips = get_health_tips(user_data, shap_values)
            if tips:
                for tip in tips:
                    st.write(f"- {tip}")
            else:
                st.write("No specific health tips based on your inputs. Maintain a healthy lifestyle to reduce your risk.")

            pdf_file = generate_pdf_report(user_data, prob, prediction, shap_values)
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF Report", f, file_name=pdf_file, mime="application/pdf")

            st.markdown('<div class="disclaimer">**Note:** This prediction is for informational purposes only. Please consult a healthcare professional for a medical diagnosis.</div>', unsafe_allow_html=True)

    # Risk Timeline Section
    st.write("### Your Risk Timeline")
    history_df = load_prediction_history()
    if not history_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history_df['Timestamp'], history_df['Probability'], marker='o', color='blue', label='Diabetes Risk Probability')
        ax.set_xlabel("Date")
        ax.set_ylabel("Risk Probability")
        ax.set_title("Your Diabetes Risk Over Time")
        ax.grid(True)
        if settings["theme"] == "Dark":
            ax.set_facecolor('#2c3e50')
            fig.set_facecolor('#2c3e50')
            ax.tick_params(colors='#ecf0f1')
            ax.xaxis.label.set_color('#ecf0f1')
            ax.yaxis.label.set_color('#ecf0f1')
            ax.title.set_color('#ecf0f1')
            ax.spines['bottom'].set_color('#ecf0f1')
            ax.spines['top'].set_color('#ecf0f1')
            ax.spines['left'].set_color('#ecf0f1')
            ax.spines['right'].set_color('#ecf0f1')
            ax.legend(facecolor='#2c3e50', edgecolor='#ecf0f1', labelcolor='#ecf0f1')
        else:
            ax.set_facecolor('#ffffff')
            fig.set_facecolor('#ffffff')
            ax.tick_params(colors='#34495e')
            ax.xaxis.label.set_color('#34495e')
            ax.yaxis.label.set_color('#34495e')
            ax.title.set_color('#2c3e50')
            ax.spines['bottom'].set_color('#34495e')
            ax.spines['top'].set_color('#34495e')
            ax.spines['left'].set_color('#34495e')
            ax.spines['right'].set_color('#34495e')
            ax.legend(facecolor='#ffffff', edgecolor='#34495e', labelcolor='#34495e')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("No prediction history available. Check the 'Save prediction to history' option to start tracking your risk over time.")

    # Footer
    st.markdown("---")
    if settings["theme"] == "Dark":
        st.markdown('<div class="disclaimer">**Diabetes Risk Prediction Tool** | Developed by [Your Name/Team] | © 2025</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="disclaimer">**Diabetes Risk Prediction Tool** | Developed by [Your Name/Team] | © 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()