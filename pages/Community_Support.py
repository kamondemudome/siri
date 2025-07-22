import streamlit as st
import os
import time

st.set_page_config(page_title="Community Support", layout="wide")

st.title("🤝 Community Support Hub")

st.markdown("""
<style>
.card { background-color: #FFFFFF; border-radius: 15px; padding: 20px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); margin-bottom: 40px; }
@media (prefers-color-scheme: dark) { .card { background-color: #2D3748; color: #F9F5F0; } }
.doctor-card { 
    background-color: #FFFFFF; 
    border-radius: 15px; 
    padding: 10px; 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
    margin-top: 20px; 
    text-align: center; 
}
@media (prefers-color-scheme: dark) { 
    .doctor-card { 
        background-color: #2D3748; 
        color: #F9F5F0; 
    } 
}
.doctor-card:hover { transform: scale(1.02); transition: transform 0.2s ease-in-out; }
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
.doctors-site { background-color: #2c3e50; }
.whatsapp-button:hover, .doctors-site:hover {
    transform: scale(1.05); 
    background-color: #1DA851; 
}
.doctors-site:hover { background-color: #23374d; }
</style>
""", unsafe_allow_html=True)

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
        # Location selection dropdown
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
        if st.button("Download Diabetes Guide"):
            st.write("Download link placeholder...")
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
                        st.session_state.success_message = f"Thank you, {name}! Your application has been received. We will contact you at {email} soon."
                        st.session_state.show_form = False
                        st.session_state.show_success = True
                        time.sleep(5)  # Display success message for 5 seconds
                        st.session_state.show_success = False
                        st.rerun()  # Refresh to hide the message
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
            file_path = "D:/Myproject/Telehealth_book.pdf"
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    st.download_button(
                        label="Download Telehealth Book",
                        data=file.read(),
                        file_name="Telehealth_book.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("The Telehealth_book.pdf file is not found in D:/Myproject/. Please ensure the file exists.")
        st.markdown('</div>', unsafe_allow_html=True)