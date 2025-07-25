import streamlit as st
import os
from database import init_db
from utils import apply_settings, load_settings

# Initialize database
init_db()

# Login page for unauthenticated users
def login_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Login Required")
    st.write("Please log in through the home page to access the Diabetes Education page.")
    st.markdown('<a href="index.py">Go to Login</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Diabetes Education content
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

    st.title("📚🛡️ Diabetes Awareness & Prevention")
    st.markdown('<div class="animated-text">Comprehensive Guide to Understanding and Preventing Diabetes</div>', unsafe_allow_html=True)
    st.markdown(f"Welcome, {st.session_state.get('username', 'User')}!", unsafe_allow_html=True)

    # Education & Tips and Reduce Diabetes Risk Side by Side
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Education & Tips")
        with st.expander("What is Diabetes?"):
            st.write("""
            Diabetes is a chronic condition that affects how your body turns food into energy. There are two main types:
            - **Type 1 Diabetes**: An autoimmune condition where the body does not produce insulin.
            - **Type 2 Diabetes**: The body either resists insulin or doesn’t produce enough, often linked to lifestyle factors.
            Learn more about symptoms, causes, and management strategies.
            """)
        with st.expander("Risk Factors & Prevention Tips"):
            st.write("""
            **Risk Factors:**
            - High BMI (>30)
            - Lack of physical activity
            - Poor diet (low fruit/vegetable intake)
            - Family history of diabetes
            - High blood pressure or cholesterol
            **Prevention Tips:**
            - Maintain a Healthy Weight: Aim for a BMI below 25 with exercise and a balanced diet.
            - Stay Active: Engage in at least 150 minutes of moderate exercise per week, like brisk walking.
            - Eat a Balanced Diet: Include more fruits, vegetables, and whole grains; reduce processed foods and sugars.
            - Monitor Your Health: Regular check-ups for blood pressure, cholesterol, and blood sugar.
            - Limit Alcohol: Keep consumption within recommended limits (e.g., 1 drink/day for women, 2 for men).
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Reduce Diabetes Risk")
        st.write("""
        - Maintain a Healthy Weight: Aim for a BMI below 25. Regular exercise and a balanced diet can help.
        - Stay Active: Engage in at least 150 minutes of moderate exercise per week, such as brisk walking.
        - Eat a Balanced Diet: Include more fruits, vegetables, and whole grains while reducing processed foods and sugars.
        - Monitor Your Health: Regular check-ups for blood pressure, cholesterol, and blood sugar levels can help detect issues early.
        - Limit Alcohol: Reduce alcohol consumption to within recommended limits (e.g., up to 1 drink per day for women, 2 for men).
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Resources & Plans Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Resources & Plans")
    st.markdown('<div class="animated-text">Educational Videos:</div>', unsafe_allow_html=True)
    col_video1, col_video2 = st.columns([1, 1])
    with col_video1:
        try:
            st.video("https://www.youtube.com/watch?v=wZAjVQWbMlE&t=4s", start_time=4)
        except Exception as e:
            st.error(f"Error loading video: {str(e)}. Please check the URL or your internet connection.")
    with col_video2:
        try:
            st.video("https://www.youtube.com/watch?v=TQo9NNYl1DY")
        except Exception as e:
            st.error(f"Error loading video: {str(e)}. Please check the URL or your internet connection.")
    col_plan, col_ebook = st.columns([1, 1])
    with col_plan:
        with st.expander("Sample Health Plan"):
            st.write("**Daily Routine:**")
            st.write("- Morning: 30-minute walk")
            st.write("- Meals: Include a serving of vegetables in every meal")
            st.write("- Evening: 15-minute stretching or yoga")
            st.write("**Weekly Goals:**")
            st.write("- Exercise: 5 days of moderate activity")
            st.write("- Diet: Reduce sugary drinks to 1 per week")
    with col_ebook:
        st.markdown("#### Recommended eBook")
        st.write("Download our free eBook: *Mastering Diabetes: In-Depth Insights for Understanding and Managing Diabetes* for comprehensive guidance.")
        file_path = os.path.normpath("D:/Myproject/Mastering_Diabetes.pdf")
        if not os.path.exists(file_path):
            st.error(f"The eBook file 'Mastering_Diabetes.pdf' was not found at {file_path}. "
                     f"Current working directory: {os.getcwd()}. "
                     f"Please ensure the file exists in D:/Myproject/ and is readable.")
        else:
            try:
                with open(file_path, "rb") as file:
                    st.download_button(
                        label="Download eBook",
                        data=file.read(),
                        file_name="Mastering_Diabetes.pdf",
                        mime="application/pdf",
                        key="download_ebook_button",
                        help="Click to download the Mastering Diabetes eBook"
                    )
            except PermissionError:
                st.error(f"Permission denied when accessing 'Mastering_Diabetes.pdf' at {file_path}. "
                         f"Please check file permissions and ensure it is not locked by another process.")
            except Exception as e:
                st.error(f"Error accessing 'Mastering_Diabetes.pdf' at {file_path}: {str(e)}. "
                         f"Please ensure the file exists and is readable.")
    st.markdown('</div>', unsafe_allow_html=True)

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
    st.set_page_config(page_title="Diabetes Education", layout="wide")
    st.markdown(apply_settings(load_settings()), unsafe_allow_html=True)
    # CSS styling (aligned with other files)
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
    .download-button {
        background-color: #F7A072;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
        transition: transform 0.2s;
        font-size: 1.1em;
    }
    .download-button:hover {
        background-color: #F5A46B;
        transform: scale(1.05);
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
    """, unsafe_allow_html=True)

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "settings" not in st.session_state:
        st.session_state["settings"] = {"theme": "Light"}

    if st.session_state["logged_in"]:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()