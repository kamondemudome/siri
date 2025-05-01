import streamlit as st
import pandas as pd
import os
import json

# Set page config
st.set_page_config(page_title="Control Panel - Diabetes Risk Prediction", layout="wide")

# Path to the shared settings file
SETTINGS_FILE = "d:\\Myproject\\settings.json"

# Load settings from the JSON file
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Default settings if file doesn't exist
        return {
            "theme": "Light",
            "font_size": "Medium",
            "accent_color": "Blue"
        }

# Function to apply settings
def apply_settings(settings):
    theme = settings["theme"]
    font_size = settings["font_size"]
    accent_color = settings["accent_color"]

    # Theme CSS
    if theme == "Dark":
        css = """
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
        .css-1d391kg {
            background-color: #2c3e50 !important;
        }
        .css-1d391kg .css-1v3fvcr {
            color: #ecf0f1 !important;
        }
        .css-1d391kg .css-1v3fvcr:hover {
            background-color: #34495e !important;
        }
        .css-1d391kg .stTextInput > label {
            color: #ecf0f1 !important;
        }
        .css-1d391kg .stSelectbox > label {
            color: #ecf0f1 !important;
        }
        .css-1d391kg .stCheckbox > label {
            color: #ecf0f1 !important;
        }

        /* Input styling */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
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

        /* Table styling */
        .stDataFrame {
            background-color: #2c3e50;
            border-radius: 5px;
            padding: 10px;
        }
        .stDataFrame table {
            background-color: #2c3e50;
            color: #ecf0f1;
        }
        .stDataFrame th {
            background-color: #34495e !important;
            color: #ecf0f1 !important;
        }
        .stDataFrame td {
            color: #ecf0f1 !important;
        }

        /* Error message styling */
        .error-message {
            color: #ff6b6b;
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
        </style>
        """
    else:  # Light Mode
        css = """
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
        .css-1d391kg {
            background-color: #2c3e50 !important;
        }
        .css-1d391kg .css-1v3fvcr {
            color: #ecf0f1 !important;
        }
        .css-1d391kg .css-1v3fvcr:hover {
            background-color: #34495e !important;
        }
        .css-1d391kg .stTextInput > label {
            color: #ecf0f1 !important;
        }
        .css-1d391kg .stSelectbox > label {
            color: #ecf0f1 !important;
        }
        .css-1d391kg .stCheckbox > label {
            color: #ecf0f1 !important;
        }

        /* Input styling */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
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

        /* Table styling */
        .stDataFrame {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 10px;
        }
        .stDataFrame table {
            background-color: #ffffff;
            color: #34495e;
        }
        .stDataFrame th {
            background-color: #ecf0f1 !important;
            color: #34495e !important;
        }
        .stDataFrame td {
            color: #34495e !important;
        }

        /* Error message styling */
        .error-message {
            color: #e74c3c;
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
    return css + font_size_css + accent_color_css

# Load and apply settings
settings = load_settings()
st.markdown(apply_settings(settings), unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.header("Navigation")
st.sidebar.markdown("[Go to Dashboard](http://localhost:8501)", unsafe_allow_html=True)
st.sidebar.markdown("[Go to Detection Tool](http://localhost:8502)", unsafe_allow_html=True)

# Admin Access
st.title("Control Panel - Diabetes Risk Prediction")
password = st.text_input("Enter Admin Password", type="password")

# Password for admin access
ADMIN_PASSWORD = "admin123"

if password != ADMIN_PASSWORD:
    st.markdown('<div class="error-message">Incorrect password. Please try again.</div>', unsafe_allow_html=True)
else:
    st.success("Access granted!")
    
    # Load data
    if os.path.exists('prediction_history.csv'):
        df = pd.read_csv('prediction_history.csv')
    else:
        st.error("No prediction history found. Please make some predictions first.")
        st.stop()

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(df)

    # Data Cleaning Options
    st.subheader("Data Cleaning Options")
    remove_duplicates = st.checkbox("Remove Duplicates")
    handle_missing = st.selectbox("Handle Missing Values", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"])

    # Filter Data
    st.subheader("Filter Data")
    columns = df.columns.tolist()
    filter_column = st.selectbox("Select Column to Filter", columns)
    unique_values = df[filter_column].unique()
    filter_value = st.selectbox(f"Select {filter_column} Value", unique_values)

    # Apply cleaning and filtering
    df_cleaned = df.copy()
    
    if remove_duplicates:
        df_cleaned = df_cleaned.drop_duplicates()
    
    if handle_missing != "None":
        if handle_missing == "Drop Rows":
            df_cleaned = df_cleaned.dropna()
        elif handle_missing == "Fill with Mean":
            for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
        elif handle_missing == "Fill with Median":
            for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    df_filtered = df_cleaned[df_cleaned[filter_column] == filter_value]

    # Display cleaned and filtered data
    st.subheader("Cleaned and Filtered Data")
    st.dataframe(df_filtered)

    # Download cleaned data
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=csv,
        file_name='cleaned_prediction_history.csv',
        mime='text/csv'
    )