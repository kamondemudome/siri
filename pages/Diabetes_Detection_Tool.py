import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from datetime import datetime
import joblib
from utils import predict_diabetes, categorize_risk, get_health_avatar, get_health_tips, generate_pdf_report, save_to_csv, apply_settings, load_settings, initialize_explainer
from constants import FEATURE_FULL_NAMES, FEATURE_DESCRIPTIONS, FEATURE_TOOLTIPS, AGE_LABELS, INCOME_LABELS, FEATURES, MODEL_PATH
from database import init_db

# Initialize database
init_db()

# Login page for unauthenticated users
def login_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Login Required")
    st.write("Please log in through the home page to access the Diabetes Detection Tool.")
    st.markdown('<a href="index.py">Go to Login</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Load model and explainer at startup
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            st.session_state["prediction_model"] = model
            # Initialize SHAP explainer
            explainer = initialize_explainer(model)
            st.session_state["explainer"] = explainer
            print(f"Model loaded from {MODEL_PATH}")
            if hasattr(model, 'feature_names_in_'):
                print(f"Model features: {model.feature_names_in_}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.session_state["prediction_model"] = None
            st.session_state["explainer"] = None
    else:
        st.error(f"Model file 'new_diabetes_rf_model.pkl' not found in {MODEL_PATH}. Please ensure it exists.")
        st.session_state["prediction_model"] = None
        st.session_state["explainer"] = None

# Main Diabetes Detection Tool content
def main_app():
    # Display username and logout button at top-right
    if st.session_state.get("logged_in", False):
        st.markdown(
            f"""
            <div class="user-logout-container">
                <div class="username-container">
                    <span class="username">{st.session_state.get("username", "User")}</span>
                </div>
                <div class="logout-button-container">
                    <div id="logout-button-placeholder"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.container():
            # Place the Streamlit button in the placeholder div using CSS positioning
            st.markdown(
                """
                <style>
                .user-logout-container {
                    position: fixed;
                    top: 10px;
                    right: 200px;
                    z-index: 1000;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .username-container {
                    display: flex;
                    align-items: center;
                }
                .logout-button-container {
                    display: inline-block;
                    width: auto;
                    height: auto;
                    line-height: normal;
                }
                #logout-button-placeholder .stButton>button {
                    background-color: #F7A072;
                    color: white;
                    border-radius: 10px;
                    padding: 8px 16px;
                    border: none;
                    font-weight: 500;
                    font-size: 1em;
                    transition: background-color 0.2s, transform 0.2s;
                    margin: 0;
                    width: auto;
                    height: auto;
                }
                #logout-button-placeholder .stButton>button:hover {
                    background-color: #F5A46B;
                    transform: scale(1.05);
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            if st.button("Logout", key="logout_button"):
                st.session_state.clear()
                st.session_state["page"] = "login"
                st.query_params.clear()
                st.rerun()

    st.title("🩺 Diabetes Risk Prediction Tool")
    st.markdown('<div class="animated-text">Predict Your Diabetes Risk with Ease</div>', unsafe_allow_html=True)

    if st.session_state.get("prediction_model") is None:
        st.error(
            f"The prediction model is not available. Please ensure 'new_diabetes_rf_model.pkl' is in {MODEL_PATH} "
            "and restart the app. You can still explore other sections like Home, Awareness, and Community Support."
        )
        st.write("Debug: Model in session state:", st.session_state.get("prediction_model") is not None)
    else:
        st.markdown(f"""
        Welcome, {st.session_state.get("username", "User")}! This tool predicts your risk of diabetes based on 10 common health and lifestyle indicators.
        Please fill out the form below to get your prediction.
        """)

        # Sidebar settings
        st.sidebar.markdown("---")
        st.sidebar.header("Prediction Settings")
        threshold = st.sidebar.slider(
            "Prediction Threshold", 0.3, 0.5, 0.4, 0.05,
            help="Adjust the threshold for classifying high vs. low risk. Lower values increase recall (more high-risk predictions)."
        )
        save_history = st.sidebar.checkbox(
            "Save prediction to history", value=False,
            help="Save your inputs and prediction to your personal history."
        )
        show_debug = st.sidebar.checkbox(
            "Show debug output", value=False,
            help="Show debug information for developers."
        )

        # Form for user input
        with st.form("input_form"):
            st.header("Enter Your Information")

            # Health and Lifestyle Section
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Health and Lifestyle Information")
            col1, col2 = st.columns(2)
            with col1:
                user_data = {}
                user_data['HighBP'] = st.selectbox(
                    FEATURE_DESCRIPTIONS['HighBP'], [0, 1],
                    format_func=lambda x: 'No' if x == 0 else 'Yes',
                    help=FEATURE_TOOLTIPS['HighBP']
                )
                user_data['HighChol'] = st.selectbox(
                    FEATURE_DESCRIPTIONS['HighChol'], [0, 1],
                    format_func=lambda x: 'No' if x == 0 else 'Yes',
                    help=FEATURE_TOOLTIPS['HighChol']
                )
                st.markdown("**What is your BMI (Body Mass Index)?**")
                height_col, weight_col = st.columns(2)
                with height_col:
                    height = st.number_input(
                        "Height (meters)", min_value=0.5, max_value=2.5, value=1.7, step=0.01,
                        help="Enter your height in meters (e.g., 1.70 for 170 cm)."
                    )
                with weight_col:
                    weight = st.number_input(
                        "Weight (kilograms)", min_value=10.0, max_value=300.0, value=70.0, step=0.1,
                        help="Enter your weight in kilograms (e.g., 70.0 for 70 kg)."
                    )
                if height > 0:
                    bmi = round(weight / (height ** 2), 1)
                    if np.isfinite(bmi):
                        user_data['BMI'] = bmi
                    else:
                        user_data['BMI'] = 25.0
                    st.write(f"**Calculated BMI:** {user_data['BMI']}")
                else:
                    user_data['BMI'] = 25.0
                    st.write("**Calculated BMI:** 25.0 (Please enter a valid height)")
                user_data['GenHlth'] = st.radio(
                    FEATURE_DESCRIPTIONS['GenHlth'], options=[5, 4, 3, 2, 1],
                    format_func=lambda x: {5: "Poor", 4: "Fair", 3: "Good", 2: "Very Good", 1: "Excellent"}[x],
                    help=FEATURE_TOOLTIPS['GenHlth'], index=2, horizontal=True
                )
            with col2:
                user_data['Smoker'] = st.selectbox(
                    FEATURE_DESCRIPTIONS['Smoker'], [0, 1],
                    format_func=lambda x: 'No' if x == 0 else 'Yes',
                    help=FEATURE_TOOLTIPS['Smoker']
                )
                user_data['PhysActivity'] = st.selectbox(
                    FEATURE_DESCRIPTIONS['PhysActivity'], [0, 1],
                    format_func=lambda x: 'No' if x == 0 else 'Yes',
                    help=FEATURE_TOOLTIPS['PhysActivity']
                )
                user_data['Fruits'] = st.selectbox(
                    FEATURE_DESCRIPTIONS['Fruits'], [0, 1],
                    format_func=lambda x: 'No' if x == 0 else 'Yes',
                    help=FEATURE_TOOLTIPS['Fruits']
                )
                user_data['Veggies'] = st.selectbox(
                    FEATURE_DESCRIPTIONS['Veggies'], [0, 1],
                    format_func=lambda x: 'No' if x == 0 else 'Yes',
                    help=FEATURE_TOOLTIPS['Veggies']
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # Demographics Section
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Demographic Information")
            col3, col4 = st.columns(2)
            with col3:
                user_data['Age'] = st.selectbox(
                    FEATURE_DESCRIPTIONS['Age'], list(range(1, 14)),
                    format_func=lambda x: AGE_LABELS[x],
                    help=FEATURE_TOOLTIPS['Age']
                )
            with col4:
                user_data['Income'] = st.selectbox(
                    FEATURE_DESCRIPTIONS['Income'], list(range(1, 9)),
                    format_func=lambda x: INCOME_LABELS[x],
                    help=FEATURE_TOOLTIPS['Income']
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # Form buttons
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            with col5:
                st.markdown('<div class="action-button-wrapper">', unsafe_allow_html=True)
                submitted = st.form_submit_button("Predict", help="Click to get your diabetes risk prediction")
                st.markdown('</div>', unsafe_allow_html=True)
            with col6:
                st.markdown('<div class="action-button-wrapper">', unsafe_allow_html=True)
                reset = st.form_submit_button("Reset", help="Click to clear all inputs and start over")
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if reset:
            st.rerun()

        if submitted:
            try:
                model = st.session_state["prediction_model"]
                explainer = st.session_state.get("explainer")
                # Validate user_data
                expected_features = FEATURES
                for feature in expected_features:
                    if feature not in user_data or not np.isfinite(user_data[feature]):
                        st.error(f"Invalid or missing value for {feature}. Please check your inputs.")
                        return
                prob, prediction, shap_values = predict_diabetes(user_data, threshold, show_debug, model, explainer)
                if prob is not None:
                    # Save to history if requested
                    if save_history:
                        save_to_csv(user_data, prob, prediction)
                        st.success("Prediction saved to your personal history.")

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("### Prediction Results")
                    st.markdown(f'<div class="health-avatar">{get_health_avatar(prob)}</div>', unsafe_allow_html=True)
                    st.write(f"**Probability of diabetes:** {prob:.2%}")
                    st.write(f"**Risk Level:** {categorize_risk(prob)}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("### Key Factors Influencing the Prediction")
                    top_features = sorted(zip(FEATURES, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
                    for feature, shap_value in top_features:
                        feature_name = FEATURE_FULL_NAMES[feature]
                        value = user_data[feature]
                        if feature in ['Age', 'Income']:
                            display_value = AGE_LABELS[value] if feature == 'Age' else INCOME_LABELS[value]
                        elif feature in ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies']:
                            display_value = 'Yes' if value == 1 else 'No'
                        else:
                            display_value = str(value)
                        impact = "increases" if shap_value > 0 else "decreases"
                        st.write(f"- **{feature_name}**: {display_value} ({impact} risk, SHAP value: {shap_value:.3f})")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("#### Feature Impact Visualization")
                    feature_names = [f"{FEATURE_FULL_NAMES[feat]}: {user_data[feat] if feat not in ['Age', 'Income'] else (AGE_LABELS[user_data[feat]] if feat == 'Age' else INCOME_LABELS[user_data[feat]])}" for feat in FEATURES]
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x=shap_values, y=feature_names, palette='coolwarm')
                    plt.xlabel("SHAP Value (Impact on Prediction)")
                    plt.title("Feature Contributions to Diabetes Risk Prediction")
                    if st.session_state["settings"].get("theme", "Light") == "Dark":
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
                    plt.close(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("### Health Tips")
                    tips = get_health_tips(user_data, shap_values)
                    if tips:
                        for tip in tips:
                            st.write(f"- {tip}")
                    else:
                        st.write("No specific health tips based on your inputs. Maintain a healthy lifestyle to reduce your risk.")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    pdf_file = generate_pdf_report(user_data, prob, prediction, shap_values)
                    if pdf_file:
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                "Download PDF Report",
                                f,
                                file_name=f"diabetes_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                help="Click to download your personalized report"
                            )
                        try:
                            os.remove(pdf_file)
                        except Exception as e:
                            st.warning(f"Error cleaning up temporary PDF file: {str(e)}")
                    st.markdown('<div class="disclaimer">**Note:** This prediction is for informational purposes only. Please consult a healthcare professional for a medical diagnosis.</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

    # Footer
    st.markdown("""
    <div class="footer">
        <div class="message">Empower Your Health Journey – Stay Ahead of Diabetes!</div>
        <div class="copyright">© 2025 Diabetes Risk Dashboard</div>
        <div class="developer">Developed by Kamonde K. Mudome</div>
    </div>
    """, unsafe_allow_html=True)

# Main function with authentication check
def main():
    st.set_page_config(page_title="Diabetes Detection Tool", layout="wide")
    # Apply settings for consistent styling
    st.markdown(apply_settings(load_settings()), unsafe_allow_html=True)
    # CSS styling (aligned with index.py and Community_Support.py)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
    body { font-family: 'Poppins', sans-serif; }
    .stApp {
        background-color: #F9F5F0;
        color: #1A252F;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        padding-top: 60px; /* Add padding to avoid overlap with fixed header */
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #1A252F;
            color: #F9F5F0;
        }
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #1A252F;
    }
    @media (prefers-color-scheme: dark) {
        .card {
            background-color: #2D3748;
            color: #F9F5F0;
        }
    }
    .card:hover {
        transform: scale(1.02);
        transition: transform 0.2s ease-in-out;
    }
    .action-button {
        background-color: #F7A072;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
        transition: transform 0.2s;
        font-size: 1.1em;
        width: 100%;
        cursor: pointer;
    }
    .action-button:hover {
        background-color: #F5A46B;
        transform: scale(1.05);
    }
    .action-button-wrapper {
        margin: 0;
        padding: 0;
    }
    .animated-text {
        margin-left: 2em;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .st-expander {
        color: #1A252F;
    }
    @media (prefers-color-scheme: dark) {
        .st-expander {
            color: #F9F5F0;
        }
    }
    h1 {
        color: #1A252F !important;
    }
    @media (prefers-color-scheme: dark) {
        h1 {
            color: #F9F5F0 !important;
        }
    }
    .footer {
        text-align: center;
        padding: 10px 0;
        background-color: #F9F5F0;
        color: #1A252F;
        margin-top: auto;
    }
    @media (prefers-color-scheme: dark) {
        .footer {
            background-color: #1A252F;
            color: #F9F5F0;
        }
    }
    .footer div {
        margin: 5px 0;
    }
    .disclaimer {
        color: #7f8c8d;
        font-size: 0.9em;
    }
    @media (prefers-color-scheme: dark) {
        .disclaimer {
            color: #bdc3c7;
        }
    }
    .health-avatar {
        font-size: 48px;
        text-align: center;
        margin-bottom: 20px;
    }
    .username {
        font-weight: 500;
        font-size: 1.1em;
        color: #1A252F;
    }
    @media (prefers-color-scheme: dark) {
        .username {
            color: #F9F5F0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "settings" not in st.session_state:
        st.session_state["settings"] = {"theme": "Light"}

    # Load model if not already loaded
    if "prediction_model" not in st.session_state:
        load_model()

    if st.session_state["logged_in"]:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()