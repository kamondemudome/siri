import streamlit as st
import os
import time
from database import init_db, save_volunteer_application
from utils import apply_settings, load_settings

# Initialize database
init_db()

# Login page (minimal, to redirect unauthenticated users)
def login_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Login Required")
    st.write("Please log in through the home page to access the Community Support Hub.")
    st.markdown('<a href="index.py">Go to Login</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Community Support content
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

    st.title("🤝 Community Support Hub")
    st.markdown('<div class="animated-text">Connect, Learn, and Support Your Health Journey</div>', unsafe_allow_html=True)

    # Social Support, Resources, and Doctor Support Side by Side
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Join a Support Community")
            st.write("Connect with others for peer support. Share your journey or find encouragement.")
            if st.button("Join Peer Chat"):
                st.write("Redirecting to the Diabetes.co.uk Forum, a global diabetes support community...")
                st.markdown('<meta http-equiv="refresh" content="1;url=https://www.diabetes.co.uk/forum/">', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Local Health Resources")
            locations = ["Select a location", "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]
            selected_location = st.selectbox("Select your location", locations)
            if selected_location != "Select a location":
                st.write(f"Finding resources near {selected_location}...")
                if selected_location == "Nairobi":
                    st.write("- Local Clinic: Nairobi Medical Centre, 456 Health Ave.")
                    st.write("- Transportation: Call 1-800-NAIROBI for assistance.")
                elif selected_location == "Mombasa":
                    st.write("- Local Clinic: Mombasa Health Hub, 789 Ocean Rd.")
                    st.write("- Transportation: Call 1-800-MOMBASA for assistance.")
                elif selected_location == "Kisumu":
                    st.write("- Local Clinic: Kisumu Wellness Clinic, 321 Lake St.")
                    st.write("- Transportation: Call 1-800-KISUMU for assistance.")
                elif selected_location == "Nakuru":
                    st.write("- Local Clinic: Nakuru Health Center, 654 Hill Rd.")
                    st.write("- Transportation: Call 1-800-NAKURU for assistance.")
                elif selected_location == "Eldoret":
                    st.write("- Local Clinic: Eldoret Care Facility, 987 Valley St.")
                    st.write("- Transportation: Call 1-800-ELDORET for assistance.")
            else:
                st.write("Please select a location to see available resources.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Talk to a Doctor")
            st.markdown('<div class="doctor-card">', unsafe_allow_html=True)
            st.markdown("""
            <a href="https://api.whatsapp.com/send/?phone=+254756551551&text=Hi&type=phone_number&app_absent=0" target="_blank" style="text-decoration: none;">
                <div class="whatsapp-button">
                    <span role="img" aria-label="whatsapp">💬</span> Join WhatsApp Group
                </div>
            </a>
            <a href="https://zuri.health/doctors" target="_blank" style="text-decoration: none;">
                <div class="doctors-site">
                    Our Doctors Site
                </div>
            </a>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Health Education, Get Involved, and Telehealth Options Side by Side
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Health Education")
            st.write("Explore guides on diabetes management and wellness.")
            file_path = os.path.normpath("D:/Myproject/Mastering_Diabetes.pdf")
            if not os.path.exists(file_path):
                st.error(f"The guide file 'Mastering_Diabetes.pdf' was not found at {file_path}. "
                         f"Current working directory: {os.getcwd()}. "
                         f"Please ensure the file exists in D:/Myproject/ and is readable.")
            else:
                try:
                    with open(file_path, "rb") as file:
                        st.download_button(
                            label="Download Diabetes Guide",
                            data=file.read(),
                            file_name="Mastering_Diabetes.pdf",
                            mime="application/pdf",
                            key="download_guide_button",
                            help="Click to download the Diabetes Guide"
                        )
                except PermissionError:
                    st.error(f"Permission denied when accessing 'Mastering_Diabetes.pdf' at {file_path}. "
                             f"Please check file permissions and ensure it is not locked by another process.")
                except Exception as e:
                    st.error(f"Error accessing 'Mastering_Diabetes.pdf' at {file_path}: {str(e)}. "
                             f"Please ensure the file exists and is readable.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Get Involved")
            st.write("Volunteer or learn about our partners.")
            if 'show_form' not in st.session_state:
                st.session_state.show_form = False
            if 'show_success' not in st.session_state:
                st.session_state.show_success = False
            if 'success_message' not in st.session_state:
                st.session_state.success_message = ""
            if st.button("Get Involved"):
                st.session_state.show_form = True
                st.session_state.show_success = False
            if st.session_state.show_form:
                with st.form(key="volunteer_form"):
                    name = st.text_input("Your Name")
                    email = st.text_input("Your Email")
                    message = st.text_area("Why do you want to get involved? (e.g., skills, availability)")
                    submit_button = st.form_submit_button("Submit Volunteer Application")
                    if submit_button:
                        if name and email:
                            user_id = st.session_state.get("user_id")
                            if user_id:
                                try:
                                    save_volunteer_application(user_id, name, email, message)
                                    st.session_state.success_message = f"Thank you, {name}! Your application has been received. We will contact you at {email} soon."
                                    st.session_state.show_form = False
                                    st.session_state.show_success = True
                                    time.sleep(5)
                                    st.session_state.show_success = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error saving application: {str(e)}")
                            else:
                                st.error("You must be logged in to submit an application.")
                        else:
                            st.error("Please fill in both your name and email.")
            if st.session_state.show_success:
                st.success(st.session_state.success_message)
            st.write("Partners: Local Health Org, Community Clinic Network")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Telehealth Options")
            if 'telehealth_expanded' not in st.session_state:
                st.session_state.telehealth_expanded = False
            if st.button("Learn About Telehealth"):
                st.session_state.telehealth_expanded = not st.session_state.telehealth_expanded
            if st.session_state.telehealth_expanded:
                st.write("""
                **What is Telehealth?**  
                Telehealth is the use of digital information and communication technologies, such as computers and mobile devices, to access healthcare services remotely. It includes virtual consultations with healthcare providers, remote monitoring of health conditions, and access to health education.

                **Benefits of Telehealth:**  
                - Convenient access to care from home.  
                - Reduced travel time and costs.  
                - Improved management of chronic conditions like diabetes.  
                - Enhanced privacy and flexibility for patients.

                For a detailed guide, download the document below.
                """)
                file_path = os.path.normpath("D:/Myproject/Telehealth_book.pdf")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as file:
                        st.download_button(
                            label="Download Telehealth Book",
                            data=file.read(),
                            file_name="Telehealth_book.pdf",
                            mime="application/pdf",
                            key="download_telehealth_button",
                            help="Click to download the Telehealth Guide"
                        )
                else:
                    st.error(f"The Telehealth_book.pdf file was not found at {file_path}. "
                             f"Current working directory: {os.getcwd()}. "
                             f"Please ensure the file exists in D:/Myproject/ and is readable.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <div class="message">Empower Your Health Journey – Stay Ahead of Diabetes!</div>
        <div class="copyright">© 2025 Diabetes Risk Dashboard</div>
        <div class="developer">Developed by Kamonde K. Mudome</div>
    </div>
    """, unsafe_allow_html=True)

# Page Layout
def main():
    st.set_page_config(page_title="Community Support", layout="wide")
    # Apply settings for consistent styling
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
    .card, .doctor-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #1A252F;
    }
    @media (prefers-color-scheme: dark) {
        .card, .doctor-card {
            background-color: #2D3748;
            color: #F9F5F0;
        }
    }
    .doctor-card {
        text-align: center;
        padding: 10px;
    }
    .card:hover, .doctor-card:hover {
        transform: scale(1.02);
        transition: transform 0.2s ease-in-out;
    }
    .whatsapp-button, .doctors-site {
        background-color: #25D366;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        display: inline-block;
        font-weight: 500;
        margin: 10px 0;
        text-decoration: none;
        transition: transform 0.2s, background-color 0.2s;
    }
    .doctors-site {
        background-color: #2c3e50;
    }
    .whatsapp-button:hover {
        background-color: #1DA851;
        transform: scale(1.05);
    }
    .doctors-site:hover {
        background-color: #23374d;
        transform: scale(1.05);
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
    .card, .stForm {
        width: 100%;
        margin: 10px 0;
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