import streamlit as st

# Define the footer HTML
footer_html = """
<div class="footer">
    <div class="message">Empower Your Health Journey – Stay Ahead of Diabetes!</div>
    <div class="copyright">© 2025 Diabetes Risk Dashboard</div>
    <div class="developer">Developed by Kamonde K. Mudome</div>
</div>
"""

# Diabetes Education Page Content
def main():
    st.set_page_config(page_title="Diabetes Education", layout="wide")
    st.title("📚🛡️ Diabetes Awareness & Prevention")
    st.markdown("### Comprehensive Guide to Understanding and Preventing Diabetes")

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
    }
    @media (prefers-color-scheme: dark) { 
        .card { 
            background-color: #2D3748; 
            color: #F9F5F0; 
        } 
    }
    .card:hover { transform: scale(1.02); transition: transform 0.2s ease-in-out; }
    .footer { 
        text-align: center; 
        padding: 10px 0; 
        background-color: #F9F5F0; 
        color: #1A252F; 
        margin-top: auto; /* Pushes footer to the end */
    }
    @media (prefers-color-scheme: dark) { 
        .footer { 
            background-color: #1A252F; 
            color: #F9F5F0; 
        } 
    }
    .footer div { margin: 5px 0; }
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
    </style>
    """, unsafe_allow_html=True)

    # Education & Tips and Reduce Diabetes Risk Side by Side
    col1, col2 = st.columns([1, 1])  # Equal width columns
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

    # Resources & Plans Section with Videos, Sample Health Plan, and eBook Download
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Resources & Plans")
    st.markdown('<div class="animated-text">Educational Videos:</div>', unsafe_allow_html=True)
    col_video1, col_video2 = st.columns([1, 1])  # Side-by-side columns for videos
    with col_video1:
        st.video("https://www.youtube.com/watch?v=wZAjVQWbMlE&t=4s", start_time=4)
    with col_video2:
        st.video("https://www.youtube.com/watch?v=TQo9NNYl1DY")
    col_plan, col_ebook = st.columns([1, 1])  # Side-by-side columns for Plan and eBook
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
        with open("D:/Myproject/Mastering Diabetes.pdf", "rb") as file:
            pdf_data = file.read()
        st.download_button(
            label="Download eBook",
            data=pdf_data,
            file_name="Mastering Diabetes.pdf",
            mime="application/pdf",
            key="download_ebook_button",
            help="Click to download the Mastering Diabetes eBook"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Add Footer
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()