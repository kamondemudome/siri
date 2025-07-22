import os
import json
import pandas as pd
import numpy as np
import joblib
import shap
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Path to the shared settings file and model
SETTINGS_FILE = "D:/Myproject/settings.json"
MODEL_PATH = "D:/Myproject/new_diabetes_rf_model.pkl"

# Define features and ranges
FEATURES = [
    'HighBP', 'HighChol', 'BMI', 'GenHlth', 'Smoker',
    'PhysActivity', 'Fruits', 'Veggies', 'Age', 'Income'
]

FEATURE_RANGES = {
    'HighBP': (0, 1), 'HighChol': (0, 1), 'BMI': (10, 100),
    'GenHlth': (1, 5), 'Smoker': (0, 1), 'PhysActivity': (0, 1),
    'Fruits': (0, 1), 'Veggies': (0, 1), 'Age': (1, 13), 'Income': (1, 8)
}

FEATURE_FULL_NAMES = {
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'BMI': 'Body Mass Index (BMI)',
    'GenHlth': 'General Health Rating',
    'Smoker': 'Smoking History (100+ Cigarettes)',
    'PhysActivity': 'Physical Activity in Past 30 Days',
    'Fruits': 'Daily Fruit Consumption',
    'Veggies': 'Daily Vegetable Consumption',
    'Age': 'Age Category',
    'Income': 'Income Category'
}

FEATURE_DESCRIPTIONS = {
    'HighBP': 'Do you have high blood pressure?',
    'HighChol': 'Do you have high cholesterol?',
    'BMI': 'What is your Body Mass Index (BMI)?',
    'GenHlth': 'How would you rate your general health? (1 = Excellent, 5 = Poor)',
    'Smoker': 'Have you smoked at least 100 cigarettes in your life?',
    'PhysActivity': 'Have you done physical activity in the past 30 days?',
    'Fruits': 'Do you consume fruit 1 or more times per day?',
    'Veggies': 'Do you consume vegetables 1 or more times per day?',
    'Age': 'What is your age category?',
    'Income': 'What is your income category?'
}

FEATURE_TOOLTIPS = {
    'HighBP': 'Select "Yes" if you have been diagnosed with high blood pressure.',
    'HighChol': 'Select "Yes" if you have been diagnosed with high cholesterol.',
    'BMI': 'Enter your BMI (e.g., 25.0). BMI = weight (kg) / height (m)^2.',
    'GenHlth': 'Rate your overall health on a scale from 1 (excellent) to 5 (poor).',
    'Smoker': 'Select "Yes" if you have smoked at least 100 cigarettes in your lifetime.',
    'PhysActivity': 'Select "Yes" if you’ve done any physical activity (e.g., walking, exercise) in the past 30 days.',
    'Fruits': 'Select "Yes" if you eat fruit at least once per day.',
    'Veggies': 'Select "Yes" if you eat vegetables at least once per day.',
    'Age': 'Select your age category (e.g., 1 = 18-24, 13 = 80+).',
    'Income': 'Select your annual income category (e.g., 1 = < Ksh 10000, 8 = Ksh 1000000+).'
}

AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"
}

INCOME_LABELS = {
    1: "< Ksh 10000", 2: "Ksh 10000-15000", 3: "Ksh 15000-25000", 4: "Ksh 25000-35000",
    5: "Ksh 35000-50000", 6: "Ksh 50000-75000", 7: "Ksh 75000-100000", 8: "Ksh 1000000+"
}

EDUCATION_LABELS = {
    1: "Never attended school", 2: "Grades 1-8", 3: "Grades 9-11", 4: "High school graduate",
    5: "Some college", 6: "College graduate"
}

def ensure_directory_exists():
    directory = os.path.dirname(SETTINGS_FILE)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def initialize_settings():
    default_settings = {
        "theme": "Light",
        "font_size": "Medium",
        "accent_color": "Blue"
    }
    ensure_directory_exists()
    with open(SETTINGS_FILE, "w") as f:
        json.dump(default_settings, f)
    return default_settings

def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return initialize_settings()
    except Exception as e:
        print(f"Error loading settings from {SETTINGS_FILE}: {str(e)}")
        return initialize_settings()

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        print(f"Error saving settings to {SETTINGS_FILE}: {str(e)}")

def apply_settings(settings):
    theme = settings["theme"]
    font_size = settings["font_size"]
    accent_color = settings["accent_color"]

    theme_css = """
    <style>
    body { font-family: 'Arial', sans-serif; }
    .stApp { background-color: #f4f7fa; color: #34495e; padding-bottom: 60px; }
    h1, h2, h3, h4, h5, h6 { color: #2c3e50; }
    .stMarkdown p { color: #34495e; }
    section[data-testid="stSidebar"] { background-color: #ecf0f1 !important; }
    .card { background: linear-gradient(135deg, #ffffff, #ecf0f1); border-radius: 15px; padding: 20px; margin: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align: center; }
    .health-card { background: linear-gradient(135deg, #ffffff, #ecf0f1); border-radius: 15px; padding: 25px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align: center; }
    .metric-card { background: linear-gradient(135deg, #ffffff, #ecf0f1); border-radius: 10px; padding: 15px; margin: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align: center; }
    .stForm { background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] { background-color: #ecf0f1; color: #34495e; border: 1px solid #d1d8e0; border-radius: 5px; }
    .stButton button { background-color: #3498db; color: white; border-radius: 10px; padding: 10px 20px; border: none; font-weight: bold; }
    .stButton button:hover { background-color: #2980b9; }
    a { color: #3498db; text-decoration: none; }
    .stExpander { background-color: #ecf0f1; color: #34495e; border-radius: 5px; }
    .footer { color: #7f8c8d; text-align: center; padding: 15px 0; border-top: 1px solid #d1d8e0; width: 100%; position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); background-color: #ffffff; z-index: 1000; height: 60px; }
    .footer .message { font-weight: bold; font-size: 16px; }
    .footer .copyright, .footer .developer { font-size: 12px; }
    .disclaimer { color: #7f8c8d; }
    .health-avatar { text-align: center; font-size: 48px; margin-bottom: 20px; }
    </style>
    """ if theme == "Light" else """
    <style>
    body { font-family: 'Arial', sans-serif; }
    .stApp { background-color: #1e1e1e; color: #ecf0f1; padding-bottom: 60px; }
    h1, h2, h3, h4, h5, h6 { color: #ecf0f1; }
    .stMarkdown p { color: #ecf0f1; }
    section[data-testid="stSidebar"] { background-color: #2c3e50 !important; }
    .card { background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 15px; padding: 20px; margin: 10px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5); text-align: center; }
    .health-card { background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 15px; padding: 25px; margin-bottom: 20px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5); text-align: center; }
    .metric-card { background: linear-gradient(135deg, #2c3e50, #34495e); border-radius: 10px; padding: 15px; margin: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); text-align: center; }
    .stForm { background-color: #2c3e50; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] { background-color: #34495e; color: #ecf0f1; border: 1px solid #34495e; border-radius: 5px; }
    .stButton button { background-color: #3498db; color: white; border-radius: 10px; padding: 10px 20px; border: none; font-weight: bold; }
    .stButton button:hover { background-color: #2980b9; }
    a { color: #66b3ff; text-decoration: none; }
    .stExpander { background-color: #34495e; color: #ecf0f1; border-radius: 5px; }
    .footer { color: #bdc3c7; text-align: center; padding: 15px 0; border-top: 1px solid #34495e; width: 100%; position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); background-color: #2c3e50; z-index: 1000; height: 60px; }
    .footer .message { font-weight: bold; font-size: 16px; }
    .footer .copyright, .footer .developer { font-size: 12px; }
    .disclaimer { color: #bdc3c7; }
    .health-avatar { text-align: center; font-size: 48px; margin-bottom: 20px; }
    </style>
    """

    font_sizes = {"Small": "12px", "Medium": "16px", "Large": "20px"}
    font_size_css = f"""
    <style>
    .stApp, .stMarkdown, .stText, .stSelectbox, .stSlider, .stNumberInput, .stButton button {{
        font-size: {font_sizes.get(font_size, "16px")} !important;
    }}
    </style>
    """

    accent_colors = {"Blue": "#3498db", "Green": "#2ecc71", "Red": "#e74c3c"}
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
    .health-card .risk-level {{
        color: {accent_colors.get(accent_color, "#3498db")} !important;
    }}
    .metric-card .value {{
        color: {accent_colors.get(accent_color, "#3498db")} !important;
    }}
    </style>
    """

    return theme_css + font_size_css + accent_color_css

def quick_predict_diabetes(bmi, phys_activity, fruits, age):
    try:
        base_risk = 0.3
        if bmi > 30:
            base_risk += 0.2
        if phys_activity == 0:
            base_risk += 0.15
        if fruits == 0:
            base_risk += 0.1
        if age > 9:
            base_risk += 0.15
        risk_prob = min(max(base_risk, 0.0), 1.0)
        if risk_prob < 0.3:
            return "Low risk (<30%)", risk_prob
        elif risk_prob < 0.5:
            return "Moderate risk (30-50%)", risk_prob
        else:
            return "High risk (>50%)", risk_prob
    except Exception as e:
        print(f"Error in quick prediction: {str(e)}")
        return "Error", 0.0

def predict_diabetes(user_data, threshold, show_debug, model, explainer=None):
    if model is None:
        raise ValueError("Prediction model is not available.")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
        if not hasattr(model, 'predict_proba'):
            raise AttributeError("Model does not support predict_proba.")

        form_features = FEATURES
        input_data = {}
        for feature in form_features:
            if feature not in user_data:
                raise ValueError(f"Missing required feature: {feature}")
            value = user_data[feature]
            min_val, max_val = FEATURE_RANGES[feature]
            if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                print(f"Invalid value for {feature}: {value}. Clamping to range [{min_val}, {max_val}]")
                value = max(min(float(value), max_val), min_val)
            input_data[feature] = float(value)

        df_input = pd.DataFrame([input_data], columns=form_features)
        if show_debug:
            print(f"Input DataFrame:\n{df_input}")

        model_features = getattr(model, 'feature_names_in_', form_features)
        df_input = df_input.reindex(columns=model_features, fill_value=0.0)
        if show_debug:
            print(f"Model expected features: {model_features}")
            print(f"Aligned Input DataFrame:\n{df_input}")

        scaler = StandardScaler()
        X_input = scaler.fit_transform(df_input)
        if show_debug:
            print(f"Scaled Input:\n{X_input}")

        prob_output = model.predict_proba(X_input)
        if show_debug:
            print(f"predict_proba output type: {type(prob_output)}")
            print(f"predict_proba output shape: {prob_output.shape if hasattr(prob_output, 'shape') else 'No shape'}")
            print(f"predict_proba output: {prob_output}")

        if isinstance(prob_output, np.ndarray) and prob_output.ndim == 2 and prob_output.shape[1] >= 2:
            prob = prob_output[0, 1]
        else:
            raise ValueError(f"Unexpected predict_proba output: {prob_output}")

        prediction = 1 if prob >= threshold else 0

        shap_values = None
        if explainer is not None:
            try:
                shap_values_raw = explainer.shap_values(X_input)
                if isinstance(shap_values_raw, np.ndarray):
                    if len(shap_values_raw.shape) == 2 and shap_values_raw.shape[0] == 1:
                        shap_values = shap_values_raw[0, :len(form_features)]
                    elif len(shap_values_raw.shape) == 1 and len(shap_values_raw) == len(form_features):
                        shap_values = shap_values_raw
                if shap_values is None or len(shap_values) != len(form_features):
                    shap_values = np.zeros(len(form_features))
            except Exception as e:
                if show_debug:
                    print(f"SHAP computation failed: {str(e)}")
                shap_values = np.zeros(len(form_features))
        else:
            shap_values = np.zeros(len(form_features))

        if show_debug:
            print(f"Probability: {prob:.4f}, Prediction: {prediction}, SHAP Values: {shap_values}")

        return prob, prediction, shap_values

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def categorize_risk(prob):
    try:
        if prob < 0.3:
            return "Low risk (<30%)"
        elif prob < 0.5:
            return "Moderate risk (30-50%)"
        else:
            return "High risk (>50%)"
    except Exception as e:
        print(f"Error categorizing risk: {str(e)}")
        return "Unknown risk"

def get_health_avatar(prob):
    try:
        if prob < 0.3:
            return "😊 **You're doing great!** (Low Risk)"
        elif prob < 0.5:
            return "😐 **Keep an eye on your health.** (Moderate Risk)"
        else:
            return "😟 **Take action to reduce your risk!** (High Risk)"
    except Exception as e:
        print(f"Error generating health avatar: {str(e)}")
        return "😕 **Error in risk assessment**"

def get_health_tips(user_data, shap_values):
    try:
        tips = []
        top_features = sorted(zip(FEATURES, shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]
        for feature, shap_value in top_features:
            if shap_value > 0:
                if feature == 'GenHlth' and user_data['GenHlth'] >= 4:
                    tips.append("Your general health rating is poor. Consider scheduling a check-up.")
                elif feature == 'BMI' and user_data['BMI'] >= 30:
                    tips.append("Your BMI is high. Consulting a dietitian may help reduce your risk.")
                elif feature == 'HighBP' and user_data['HighBP'] == 1:
                    tips.append("High blood pressure increases your risk. Monitor it regularly.")
                elif feature == 'PhysActivity' and user_data['PhysActivity'] == 0:
                    tips.append("Aim for 150 minutes of moderate exercise per week.")
                elif feature == 'Smoker' and user_data['Smoker'] == 1:
                    tips.append("Consider quitting smoking with support from a healthcare provider.")
        return tips if tips else ["Maintain a healthy lifestyle to reduce your risk."]
    except Exception as e:
        print(f"Error generating health tips: {str(e)}")
        return ["Unable to generate specific health tips."]

def save_to_csv(user_data, prob, prediction):
    try:
        record = user_data.copy()
        record['Probability'] = prob
        record['Prediction'] = categorize_risk(prob)
        record['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.DataFrame([record])
        history_file = "D:/Myproject/prediction_history.csv"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        if not os.path.exists(history_file):
            df.to_csv(history_file, index=False)
        else:
            df.to_csv(history_file, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error saving prediction history: {str(e)}")

def load_prediction_history():
    history_file = "D:/Myproject/prediction_history.csv"
    try:
        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading prediction history: {str(e)}")
        return pd.DataFrame()

def generate_pdf_report(user_data, prob, prediction, shap_values):
    try:
        import tempfile
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
            if feature in ['Age', 'Income']:
                display_value = AGE_LABELS[value] if feature == 'Age' else INCOME_LABELS[value]
            elif feature in ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies']:
                display_value = 'Yes' if value == 1 else 'No'
            else:
                display_value = str(value)
            table_data.append([Paragraph(feature_name, style_normal), display_value])

        table = Table(table_data, colWidths=[3.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        table_width, table_height = table.wrap(width - 100, height)
        y_position = check_new_page(y_position, table_height + 20)
        table.drawOn(c, 50, y_position - table_height)
        y_position -= (table_height + 20)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Section 2: Prediction Results")
        y_position -= 20
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, f"Probability of diabetes: {prob:.2%}")
        y_position -= 15
        c.drawString(50, y_position, f"Risk Level: {categorize_risk(prob)}")
        y_position -= 30

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Section 3: Key Factors")
        y_position -= 20
        table_data = [["Feature", "Value", "Impact", "SHAP Value"]]
        top_features = sorted(zip(FEATURES, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feature, shap_value in top_features:
            feature_name = FEATURE_FULL_NAMES[feature]
            impact = "increases" if shap_value > 0 else "decreases"
            value = user_data[feature]
            if feature in ['Age', 'Income']:
                display_value = AGE_LABELS[value] if feature == 'Age' else INCOME_LABELS[value]
            elif feature in ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies']:
                display_value = 'Yes' if value == 1 else 'No'
            else:
                display_value = str(value)
            table_data.append([Paragraph(feature_name, style_normal), display_value, impact, f"{shap_value:.3f}"])

        table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        table_width, table_height = table.wrap(width - 100, height)
        y_position = check_new_page(y_position, table_height + 20)
        table.drawOn(c, 50, y_position - table_height)
        y_position -= (table_height + 20)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Section 4: Health Tips")
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
            c.drawString(50, y_position, "No specific health tips. Maintain a healthy lifestyle.")
            y_position -= 15

        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.grey)
        c.drawString(50, y_position, "Disclaimer: This prediction is for informational purposes only.")
        y_position -= 15
        c.drawString(50, y_position, "Please consult a healthcare professional for a medical diagnosis.")

        c.showPage()
        c.save()
        return filename
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return None

def initialize_explainer(model):
    if model is not None:
        try:
            background_data = pd.DataFrame({
                'HighBP': [0, 1], 'HighChol': [0, 1], 'BMI': [20.0, 30.0], 'GenHlth': [1, 5],
                'Smoker': [0, 1], 'PhysActivity': [0, 1], 'Fruits': [0, 1], 'Veggies': [0, 1],
                'Age': [1, 13], 'Income': [1, 8]
            })
            def model_predict(data):
                return model.predict_proba(data)[:, 1]
            return shap.KernelExplainer(model_predict, background_data)
        except Exception as e:
            print(f"Error initializing SHAP explainer: {str(e)}")
            return None
    return None