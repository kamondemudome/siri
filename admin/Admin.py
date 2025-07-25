import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime

# Adjust Python path to include Myproject directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from database import init_db, authenticate_user, DB_PATH
    from constants import FEATURES, FEATURE_FULL_NAMES, AGE_LABELS, INCOME_LABELS
except ModuleNotFoundError as e:
    st.error(f"Module import error: {str(e)}. Ensure database.py and constants.py are in D:/Myproject/")
    raise

# Initialize database
init_db()

# Function to get the count of registered users
def get_user_count():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        return count
    except sqlite3.Error as e:
        st.error(f"Error retrieving user count: {str(e)}")
        return 0
    finally:
        conn.close()

# Function to get all user predictions
def get_all_predictions():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT u.username, p.* 
            FROM prediction_history p 
            JOIN users u ON p.user_id = u.id
            """,
            conn,
            parse_dates=['Timestamp']
        )
        return df
    except sqlite3.Error as e:
        st.error(f"Error retrieving predictions: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

# Function to get all usernames
def get_all_usernames():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username != 'admin'")
        usernames = [row[0] for row in cursor.fetchall()]
        return usernames
    except sqlite3.Error as e:
        st.error(f"Error retrieving usernames: {str(e)}")
        return []
    finally:
        conn.close()

# Function to delete a user and their predictions
def delete_user(username):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()
        if user_id:
            user_id = user_id[0]
            cursor.execute("DELETE FROM prediction_history WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM volunteer_applications WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            st.success(f"User '{username}' and their data have been deleted.")
        else:
            st.error(f"User '{username}' not found.")
    except sqlite3.Error as e:
        st.error(f"Error deleting user: {str(e)}")
    finally:
        conn.close()

# Function to get prediction statistics
def get_prediction_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prediction_history")
        total_predictions = cursor.fetchone()[0]
        df = pd.read_sql_query("SELECT Prediction FROM prediction_history", conn)
        risk_counts = df['Prediction'].value_counts().to_dict() if not df.empty else {}
        return total_predictions, risk_counts
    except sqlite3.Error as e:
        st.error(f"Error retrieving prediction stats: {str(e)}")
        return 0, {}
    finally:
        conn.close()

# Function to get recent activity (last 5 predictions)
def get_recent_activity():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT u.username, p.Timestamp, p.Prediction 
            FROM prediction_history p 
            JOIN users u ON p.user_id = u.id 
            ORDER BY p.Timestamp DESC 
            LIMIT 5
            """,
            conn,
            parse_dates=['Timestamp']
        )
        return df
    except sqlite3.Error as e:
        st.error(f"Error retrieving recent activity: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

# Function to get database status
def get_database_status():
    try:
        if os.path.exists(DB_PATH):
            file_size = os.path.getsize(DB_PATH) / (1024 * 1024)  # Size in MB
            last_modified = datetime.fromtimestamp(os.path.getmtime(DB_PATH))
            return file_size, last_modified
        return 0, None
    except Exception as e:
        st.error(f"Error retrieving database status: {str(e)}")
        return 0, None

# Login page for admin access
def login_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Admin Login")
    with st.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        user_id = authenticate_user(username, password)
        if user_id and username == "admin":
            st.session_state["admin_logged_in"] = True
            st.session_state["admin_username"] = username
            st.session_state["user_id"] = user_id
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["current_page"] = "Admin Dashboard"
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password. Only the admin user can access this page.")
    st.markdown('</div>', unsafe_allow_html=True)

# Main Admin Dashboard content
def main_app():
    # Sidebar navigation
    st.sidebar.title(f"Welcome, {st.session_state.get('admin_username', 'Admin')}")
    nav_options = ["Admin Dashboard", "Home", "Diabetes Detection Tool", "Reports & Progress", "Community Support", "Diabetes Education"]
    page = st.sidebar.radio("Navigate", nav_options, key="admin_nav")
    st.session_state["current_page"] = page

    # Handle navigation
    if page != "Admin Dashboard":
        page_map = {
            "Home": "index.py",
            "Diabetes Detection Tool": "pages/Diabetes_Detection_Tool.py",
            "Reports & Progress": "pages/Reports_Progress.py",
            "Community Support": "pages/Community_Support.py",
            "Diabetes Education": "pages/Diabetes_Education.py"
        }
        if page in page_map:
            st.session_state["logged_in"] = True
            st.session_state["username"] = st.session_state["admin_username"]
            if "user_id" not in st.session_state:
                st.session_state["user_id"] = authenticate_user(st.session_state["admin_username"])
            if st.session_state["user_id"] is None:
                st.error("Authentication failed. Please log in again.")
                st.session_state.clear()
                st.session_state["page"] = "login"
                st.rerun()
            # Streamlit's multi-page app handles navigation
            st.info(f"Navigating to {page}. If this page does not load, ensure {page_map[page]} exists and handles st.session_state['logged_in'] correctly.")
            return

    # Display username and logout button
    st.markdown(
        f"""
        <div class="user-logout-container">
            <div class="username-container">
                <span class="username">{st.session_state.get("admin_username", "Admin")}</span>
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
            st.rerun()

    st.title("🛠️ Admin Dashboard")
    st.markdown('<div class="animated-text">Manage Users and Predictions</div>', unsafe_allow_html=True)

    # User Statistics and Manage Users side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("User Statistics")
        user_count = get_user_count()
        total_predictions, risk_counts = get_prediction_stats()
        # Side-by-side metrics for Registered Users and Total Predictions
        col_metrics1, col_metrics2 = st.columns(2)
        with col_metrics1:
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Registered Users")
            st.markdown(f'<div class="value">{user_count}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_metrics2:
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Total Predictions")
            st.markdown(f'<div class="value">{total_predictions}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        # Risk Level Distribution below
        if risk_counts:
            st.markdown("##### Risk Level Distribution")
            for risk, count in risk_counts.items():
                st.markdown(f"- {risk}: {count} ({count/total_predictions:.2%})")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Manage Users")
        usernames = get_all_usernames()
        if usernames:
            selected_user = st.selectbox("Select User to Delete", [""] + usernames, key="delete_user_select")
            if selected_user:
                if st.button(f"Delete User '{selected_user}'", key="delete_user_button"):
                    delete_user(selected_user)
                    st.rerun()
        else:
            st.write("No users available to delete.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent Activity and Database Status side by side
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recent Activity")
        recent_df = get_recent_activity()
        if not recent_df.empty:
            recent_df['Timestamp'] = recent_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(
                recent_df[['username', 'Timestamp', 'Prediction']],
                use_container_width=True
            )
        else:
            st.write("No recent activity available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Database Status")
        file_size, last_modified = get_database_status()
        st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("##### Database File Size")
        st.markdown(f'<div class="value">{file_size:.2f} MB</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if last_modified:
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Last Modified")
            st.markdown(f'<div class="value">{last_modified.strftime('%Y-%m-%d %H:%M:%S')}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.write("Database file not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction History
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("All User Predictions")
    predictions_df = get_all_predictions()
    if not predictions_df.empty:
        # Format the DataFrame for display
        display_df = predictions_df.copy()
        display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.2%}")
        for feature in ['Age', 'Income']:
            if feature in display_df.columns:
                display_df[feature] = display_df[feature].map(
                    AGE_LABELS if feature == 'Age' else INCOME_LABELS
                )
        for feature in ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies']:
            if feature in display_df.columns:
                display_df[feature] = display_df[feature].apply(lambda x: 'Yes' if x == 1 else 'No')
        if 'GenHlth' in display_df.columns:
            display_df['GenHlth'] = display_df['GenHlth'].apply(lambda x: {1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'}.get(x, 'Unknown'))
        display_df = display_df.rename(columns={**FEATURE_FULL_NAMES, 'Timestamp': 'Date & Time', 'Prediction': 'Risk Level'})
        
        st.dataframe(
            display_df[['username', 'Date & Time', 'Risk Level', 'Probability'] + list(FEATURE_FULL_NAMES.values())],
            use_container_width=True
        )

        # Select users for prediction download
        st.subheader("Download Predictions for Model Retraining")
        selected_users = st.multiselect(
            "Select Users for Prediction Download (leave empty for all users)",
            usernames,
            key="download_users_select"
        )
        if st.button("Download Selected Predictions as CSV", key="download_predictions_button"):
            if selected_users:
                download_df = predictions_df[predictions_df['username'].isin(selected_users)]
            else:
                download_df = predictions_df
            # Include only the 11 features needed for retraining
            model_columns = FEATURES + ['Prediction']
            if all(col in download_df.columns for col in model_columns):
                csv = download_df[model_columns].to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name=f"predictions_for_retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download selected predictions for model retraining",
                    key="download_button"
                )
            else:
                st.error("Error: Not all required features are present in the prediction data.")
    else:
        st.write("No predictions available in the database.")
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
    st.set_page_config(page_title="Admin Dashboard", layout="wide")
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
        padding-top: 60px;
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #1A252F;
            color: #F9F5F0;
        }
    }
    .card, .metric-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #1A252F;
    }
    @media (prefers-color-scheme: dark) {
        .card, .metric-card {
            background-color: #2D3748;
            color: #F9F5F0;
        }
    }
    .card:hover, .metric-card:hover {
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
    .metric-card {
        text-align: center;
        padding: 15px;
        background: #f8f9fa;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #34495e;
        }
    }
    .value {
        font-size: 1.5em;
        font-weight: bold;
        color: #2c3e50;
    }
    @media (prefers-color-scheme: dark) {
        .value {
            color: #ecf0f1;
        }
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
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    # Show login page or main app
    if not st.session_state["admin_logged_in"]:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()