import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="Control Panel - Diabetes Risk Prediction", layout="wide")

# Simple admin authentication
def check_password():
    """Returns `True` if the user enters the correct password."""
    def password_entered():
        """Checks whether the entered password is correct."""
        if st.session_state["password"] == "admin123":  # Simple password for demo purposes
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show password input
        st.text_input("Enter Admin Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input("Enter Admin Password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password. Please try again.")
        return False
    else:
        # Password correct
        return True

# Main app logic
if check_password():
    st.title("Control Panel - Diabetes Risk Prediction")
    st.markdown("""
    Welcome to the Control Panel. Here, admins can oversee the app's functionality, view collected user data, clean and manipulate it, and prepare it for retraining the model.
    """)

    # Load the prediction history data
    st.header("Prediction History Data")
    if os.path.exists('prediction_history.csv'):
        df = pd.read_csv('prediction_history.csv')
        if df.empty:
            st.warning("No data available in prediction_history.csv.")
        else:
            st.write("### Raw Data")
            st.dataframe(df)

            # Data Cleaning and Manipulation Section
            st.header("Data Cleaning and Manipulation")
            
            # Remove duplicates
            if st.checkbox("Remove Duplicates"):
                df_cleaned = df.drop_duplicates()
                st.write(f"Removed {len(df) - len(df_cleaned)} duplicate rows.")
                df = df_cleaned
                st.dataframe(df)

            # Handle missing values
            st.subheader("Handle Missing Values")
            missing_action = st.selectbox("Action for Missing Values", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"])
            if missing_action != "None":
                if missing_action == "Drop Rows":
                    df_cleaned = df.dropna()
                    st.write(f"Dropped {len(df) - len(df_cleaned)} rows with missing values.")
                elif missing_action == "Fill with Mean":
                    df_cleaned = df.fillna(df.mean(numeric_only=True))
                    st.write("Filled missing values with column means (for numeric columns).")
                elif missing_action == "Fill with Median":
                    df_cleaned = df.fillna(df.median(numeric_only=True))
                    st.write("Filled missing values with column medians (for numeric columns).")
                df = df_cleaned
                st.dataframe(df)

            # Filter data
            st.subheader("Filter Data")
            filter_column = st.selectbox("Select Column to Filter", df.columns)
            if filter_column:
                unique_values = df[filter_column].unique()
                filter_value = st.selectbox(f"Select Value for {filter_column}", unique_values)
                df_filtered = df[df[filter_column] == filter_value]
                st.write(f"Filtered data where {filter_column} = {filter_value}")
                df = df_filtered
                st.dataframe(df)

            # Download cleaned data
            st.header("Download Cleaned Data")
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Cleaned Data as CSV",
                    data=csv,
                    file_name="cleaned_prediction_history.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No data available to download after cleaning/filtering.")

    else:
        st.error("prediction_history.csv not found. No user data has been saved yet.")

if __name__ == "__main__":
    pass