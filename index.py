import streamlit as st
from datetime import datetime
from utils import get_health_avatar, quick_predict_diabetes, apply_settings, load_settings
from constants import MODEL_PATH
from xai_sdk import Client
from database import init_db, register_user, authenticate_user
import os
import requests
from dotenv import load_dotenv
import joblib

# Load environment variables
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    st.error("XAI_API_KEY not found. Please set it in a .env file.")
    st.stop()

# Initialize xAI Client
try:
    client = Client(api_key=XAI_API_KEY, api_host="https://api.x.ai/v1")
except Exception as e:
    st.session_state.setdefault("ai_errors", []).append(f"xAI SDK Error: {str(e)}")
    client = None

# Initialize database
init_db()

# Function to query xAI
def call_ai_api(user_input):
    try:
        if client:
            response = client.chat.create(
                model="grok-3",
                messages=[{"role": "user", "content": user_input}],
                max_tokens=200
            )
            return response.choices[0].message.content if hasattr(response, 'choices') and response.choices else "No valid response from SDK."
    except Exception as e:
        st.session_state.setdefault("ai_errors", []).append(f"SDK Error: {str(e)}")

    try:
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": user_input}],
            "max_tokens": 200
        }
        response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] if response.json().get("choices") else "No valid response from API."
    except requests.exceptions.HTTPError as e:
        st.session_state.setdefault("ai_errors", []).append(f"API Error: {str(e)} - Check endpoint or API key.")
        return f"Error: {str(e)}"
    except Exception as e:
        st.session_state.setdefault("ai_errors", []).append(f"Error: {str(e)}")
        return f"Error: {str(e)}"

# Load the prediction model with enhanced debugging
def load_prediction_model():
    if "prediction_model" not in st.session_state:
        if not os.path.exists(MODEL_PATH):
            st.session_state["prediction_model"] = None
        else:
            try:
                with open(MODEL_PATH, "rb") as model_file:
                    model = joblib.load(model_file)
                    st.session_state["prediction_model"] = model
            except PermissionError:
                st.session_state["prediction_model"] = None
            except Exception:
                st.session_state["prediction_model"] = None
    return st.session_state["prediction_model"]

# Login page
def login_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Diabetes Risk Dashboard")
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        user_id = authenticate_user(username, password)
        if user_id:
            st.session_state["user_id"] = user_id
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
            st.session_state["page"] = "home"
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    if st.button("Need an account? Register here"):
        st.session_state["page"] = "register"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Registration page
def register_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Diabetes Risk Dashboard")
    st.subheader("Register")
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
    if submit:
        if password != confirm_password:
            st.error("Passwords do not match.")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            if register_user(username, password, email):
                st.success("Registration successful! Please log in.")
                st.session_state["page"] = "login"
                st.rerun()
            else:
                st.error("Username or email already exists.")
    if st.button("Already have an account? Login here"):
        st.session_state["page"] = "login"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Main app content
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

    # Sidebar navigation
    st.sidebar.title(f"Welcome, {st.session_state.get('username', 'User')}")
    # Show Admin Dashboard option only for admin user
    nav_options = ["Home", "Diabetes Detection Tool", "Reports & Progress", "Community Support", "Diabetes Education"]
    if st.session_state.get('username') == "admin":
        nav_options.append("Admin Dashboard")
    page = st.sidebar.radio("Navigate", nav_options)
    st.session_state["current_page"] = page

    # Display content based on selected page
    if page == "Home":
        st.title("🏠 Welcome to Your Health Hub")
        st.markdown(f'<div class="card">Welcome, {st.session_state.get("username", "User")}</div>', unsafe_allow_html=True)

        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Current Risk Level")
                risk_prob = st.session_state.get("latest_risk_prob", 0.45)
                st.markdown(f'<div class="health-avatar">{get_health_avatar(risk_prob)}</div>', unsafe_allow_html=True)
                catchy_message = "🌟 Your health is your power—take one step today, and own your tomorrow!"
                st.write(f"**{catchy_message}**")
                st.markdown('<a href="pages/Diabetes_Detection_Tool.py">🧪 Take Full Assessment</a>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("What is Diabetes?")
                st.write("Diabetes is a chronic condition where the body either doesn’t produce enough insulin or can’t use it properly. This results in too much blood sugar staying in your bloodstream, which over time can cause serious health problems like heart disease, vision loss, and kidney issues. Awareness, early detection, and healthy living are key to preventing or managing diabetes effectively.")
                st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Quick Health Check")
                model = load_prediction_model()
                if model is None:
                    st.error(f"The prediction model is not available. Please ensure 'new_diabetes_rf_model.pkl' is in {MODEL_PATH} and restart the app. You can still explore other sections like Home, Awareness, and Preventive Measures.")
                else:
                    with st.form("quick_check_form"):
                        c1, c2 = st.columns(2)
                        bmi = c1.slider("Your BMI", 15.0, 40.0, 25.0, step=0.1)
                        activity = c1.selectbox("Physical Activity", ["No", "Yes"])
                        fruit = c2.selectbox("Daily Fruits", ["No", "Yes"])
                        age = c2.slider("Age", 18, 80, 30)
                        submit = st.form_submit_button("Predict Risk")
                    if submit:
                        level, prob = quick_predict_diabetes(bmi, activity == "Yes", fruit == "Yes", (age - 18)//7 + 1)
                        st.session_state.update({
                            "latest_bmi": bmi,
                            "latest_activity": "Active" if activity == "Yes" else "Inactive",
                            "latest_fruits": "High" if fruit == "Yes" else "Low",
                            "latest_risk": level,
                            "latest_risk_prob": prob
                        })
                        st.success(f"Risk Level: {level} ({prob:.2%})")
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Unlock the Secrets to a Vibrant Life")
                st.markdown('<div class="intriguing-image">', unsafe_allow_html=True)
                st.image("D:/Myproject/pages/healthy_lifestyle.jpg", width=300)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Health Tips")
        st.info("💧 **Stay hydrated** – Aim for 8 glasses of water daily.")
        st.info("🏃 **Move more** – Physical activity cuts diabetes risk by 30%.")
        st.info("📚 **Learn** – Explore our education section for in-depth content.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ai-card">', unsafe_allow_html=True)
        st.subheader("Ask AI About Your Health")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        if "ai_messages" not in st.session_state:
            st.session_state["ai_messages"] = []
        if "ai_errors" not in st.session_state:
            st.session_state["ai_errors"] = []
        for error in st.session_state["ai_errors"]:
            st.write(f"*Error*: {error}")
        for message in st.session_state["ai_messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        st.markdown('<div class="clear-button">', unsafe_allow_html=True)
        if st.button("Clear Chat"):
            st.session_state["ai_messages"] = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        if prompt := st.chat_input("Ask me anything about your health!"):
            st.session_state["ai_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                response = call_ai_api(prompt)
                st.write(response)
            st.session_state["ai_messages"].append({"role": "assistant", "content": response})
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Redirect to other pages using markdown links (Streamlit multi-page app handles these)
    elif page == "Diabetes Detection Tool":
        st.markdown('<meta http-equiv="refresh" content="0;url=pages/Diabetes_Detection_Tool.py">', unsafe_allow_html=True)
    elif page == "Reports & Progress":
        st.markdown('<meta http-equiv="refresh" content="0;url=pages/Reports_Progress.py">', unsafe_allow_html=True)
    elif page == "Community Support":
        st.markdown('<meta http-equiv="refresh" content="0;url=pages/Community_Support.py">', unsafe_allow_html=True)
    elif page == "Diabetes Education":
        st.markdown('<meta http-equiv="refresh" content="0;url=pages/Diabetes_Education.py">', unsafe_allow_html=True)
    elif page == "Admin Dashboard":
        st.markdown('<meta http-equiv="refresh" content="0;url=pages/Admin.py">', unsafe_allow_html=True)

# Page Layout
def main():
    st.set_page_config(page_title="Diabetes Risk Dashboard", layout="wide")
    st.markdown(apply_settings(load_settings()), unsafe_allow_html=True)

    # CSS styling
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
    .card, .health-card, .ai-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
        color: #1A252F;
        margin-bottom: 20px;
    }
    @media (prefers-color-scheme: dark) {
        .card, .health-card, .ai-card {
            background-color: #2D3748;
            color: #F9F5F0;
        }
    }
    .card:hover, .health-card:hover, .ai-card:hover {
        transform: scale(1.02);
    }
    .health-avatar {
        font-size: 50px;
        text-align: left;
        margin-bottom: 20px;
        animation: bounce 2s infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .intriguing-image {
        position: relative;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(26, 115, 232, 0.7);
            opacity: 0.9;
        }
        50% {
            transform: scale(1.05);
            box-shadow: 0 0 15px 5px rgba(255, 87, 34, 0.7);
            opacity: 1;
        }
        100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(26, 115, 232, 0.7);
            opacity: 0.9;
        }
    }
    .metric {
        font-weight: 600;
        font-size: 16px;
        color: #627D98;
    }
    @media (prefers-color-scheme: dark) {
        .metric {
            color: #A0AEC0;
        }
    }
    .value {
        font-size: 22px;
        font-weight: bold;
        color: #1A73E8;
    }
    @media (prefers-color-scheme: dark) {
        .value {
            color: #4C9AFF;
        }
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        background-color: #E8ECEF;
        border-radius: 5px;
        padding: 10px;
        box-shadow: inset 0 1px 6px rgba(0, 0, 0, 0.05);
        color: #1A252F;
    }
    @media (prefers-color-scheme: dark) {
        .chat-container {
            background-color: #4A5568;
            color: #F9F5F0;
        }
    }
    .clear-button {
        position: absolute;
        bottom: 10px;
        right: 10px;
    }
    button {
        background-color: #F7A072;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
        transition: transform 0.2s;
        font-size: 1.1em;
    }
    button:hover {
        background-color: #F5A46B;
        transform: scale(1.05);
    }
    .card, .health-card, .ai-card, .stForm {
        width: 100%;
        margin: 10px 0;
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
    .stSidebar {
        margin-top: 60px; /* Adjust sidebar to avoid overlap with fixed header */
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state["page"] = "login"
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "settings" not in st.session_state:
        st.session_state["settings"] = {"theme": "Light"}

    # Show only login or register form if not logged in
    if not st.session_state["logged_in"]:
        if st.session_state["page"] == "register":
            register_page()
        else:
            login_page()
    else:
        main_app()

    # Footer (always shown)
    st.markdown("""
    <div class="footer">
        <div class="message">Empower Your Health Journey – Stay Ahead of Diabetes!</div>
        <div class="copyright">© 2025 Diabetes Risk Dashboard</div>
        <div class="developer">Developed by Kamonde K. Mudome</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()