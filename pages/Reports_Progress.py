import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from utils import load_prediction_history, AGE_LABELS, EDUCATION_LABELS, INCOME_LABELS, FEATURE_FULL_NAMES

# Define the footer HTML
footer_html = """
<div class="footer">
    <div class="message">Empower Your Health Journey – Stay Ahead of Diabetes!</div>
    <div class="copyright">© 2025 Diabetes Risk Dashboard</div>
    <div class="developer">Developed by Kamonde K. Mudome</div>
</div>
"""

# Reports & Progress Page Content
def main():
    st.set_page_config(page_title="Reports & Progress", layout="wide")
    st.markdown('<style>body { background-color: #f0f2f5; }</style>', unsafe_allow_html=True)

    # Centered Title with Slight Offset
    st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        margin-left: 10%;
        font-size: 2em;
        color: #2c3e50;
    }
    @media (prefers-color-scheme: dark) {
        .centered-title { color: #ecf0f1; }
    }
    </style>
    <div class="centered-title">📊 Diabetes Risk Dashboard</div>
    """, unsafe_allow_html=True)

    # Load prediction history
    history_df = load_prediction_history()

    # Talk to a Doctor Section
    st.markdown('<div class="doctor-card">', unsafe_allow_html=True)
    st.markdown("#### Talk to a Doctor")
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("""
        <a href="https://api.whatsapp.com/send/?phone=+254756551551&text=Hi&type=phone_number&app_absent=0" target="_blank" style="text-decoration: none;">
            <div class="whatsapp-button">
                <span role="img" aria-label="whatsapp">💬</span> Join WhatsApp Group
            </div>
        </a>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown("""
        <a href="https://zuri.health/doctors" target="_blank" style="text-decoration: none;">
            <div class="doctors-site">
                Our Doctors Site
            </div>
        </a>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Progress Overview (Hero Section)
    st.markdown('<div class="health-card">', unsafe_allow_html=True)
    st.markdown("#### Your Progress Snapshot")
    if not history_df.empty:
        avg_risk = history_df['Probability'].mean()
        min_risk = history_df['Probability'].min()
        max_risk = history_df['Probability'].max()
        trend = "Decreasing 📉" if history_df['Probability'].iloc[-1] < history_df['Probability'].iloc[0] else "Increasing 📈"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Avg. Risk")
            st.markdown(f'<div class="value">{avg_risk:.2%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Min. Risk")
            st.markdown(f'<div class="value">{min_risk:.2%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Max. Risk")
            st.markdown(f'<div class="value">{max_risk:.2%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("##### Trend")
            st.markdown(f'<div class="value">{trend}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No prediction history available. Use the Diabetes Detection Tool to start tracking your risk.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Two-Column Layout for Past Detection and Health Trends
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Past Detection Results")
        # Filters under the title
        min_date = history_df['Timestamp'].min().date() if not history_df.empty else datetime.now().date()
        max_date = history_df['Timestamp'].max().date() if not history_df.empty else datetime.now().date()
        col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
        with col_f1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col_f2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        with col_f3:
            risk_levels = ["Ascending", "Descending", "Low Level", "Middle Level", "High Level"]
            selected_risk = st.selectbox("Sort/Filter by Risk", risk_levels)

        if not history_df.empty:
            # Filter by date range
            filtered_df = history_df[
                (history_df['Timestamp'].dt.date >= start_date) &
                (history_df['Timestamp'].dt.date <= end_date)
            ]

            # Filter or sort by selected risk level
            if selected_risk == "Ascending":
                filtered_df = filtered_df.sort_values(by="Probability", ascending=True)
            elif selected_risk == "Descending":
                filtered_df = filtered_df.sort_values(by="Probability", ascending=False)
            elif selected_risk == "Low Level":
                filtered_df = filtered_df[filtered_df['Probability'] < 0.3]
            elif selected_risk == "Middle Level":
                filtered_df = filtered_df[(filtered_df['Probability'] >= 0.3) & (filtered_df['Probability'] <= 0.5)]
            elif selected_risk == "High Level":
                filtered_df = filtered_df[filtered_df['Probability'] > 0.5]

            if not filtered_df.empty:
                # Prepare display DataFrame
                display_df = filtered_df[['Timestamp', 'Prediction', 'Probability']].copy()
                display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.2%}")
                display_df['Details'] = ""

                # Display the table
                st.dataframe(
                    display_df,
                    column_config={
                        "Timestamp": "Date & Time",
                        "Prediction": "Risk Level",
                        "Probability": "Risk Probability",
                        "Details": st.column_config.TextColumn("Details")
                    },
                    use_container_width=True
                )

                # Add expanders for each row
                for idx, row in filtered_df.iterrows():
                    with st.expander(f"Details for {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                        table_data = []
                        for feature in ['HighBP', 'HighChol', 'BMI', 'GenHlth', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'Age', 'Income']:
                            value = row[feature]
                            if feature in ['Age', 'Income']:
                                display_value = AGE_LABELS[value] if feature == 'Age' else INCOME_LABELS[value]
                            elif feature in ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Fruits', 'Veggies']:
                                display_value = 'Yes' if value == 1 else 'No'
                            else:
                                display_value = str(value)
                            table_data.append({
                                "Feature": FEATURE_FULL_NAMES[feature],
                                "Value": display_value
                            })

                        table_df = pd.DataFrame(table_data)
                        st.table(table_df)

                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction History as CSV",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.write("No predictions match the selected filters.")
        else:
            st.write("No prediction history available. Use the Diabetes Detection Tool to start tracking your risk.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Health Trends Over Time")
        # Time Range Filter under the title
        time_range = st.selectbox("Select Time Range", ["Last 30 Days", "Last 90 Days", "All Time"])
        if not history_df.empty:
            # Filter by time range
            filtered_trend_df = history_df.copy()
            if time_range == "Last 30 Days":
                filtered_trend_df = filtered_trend_df[filtered_trend_df['Timestamp'] >= datetime.now() - timedelta(days=30)]
            elif time_range == "Last 90 Days":
                filtered_trend_df = filtered_trend_df[filtered_trend_df['Timestamp'] >= datetime.now() - timedelta(days=90)]

            if not filtered_trend_df.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.fill_between(
                    filtered_trend_df['Timestamp'],
                    filtered_trend_df['Probability'],
                    color='blue',
                    alpha=0.1
                )
                ax.plot(
                    filtered_trend_df['Timestamp'],
                    filtered_trend_df['Probability'],
                    marker='o',
                    color='blue',
                    label='Diabetes Risk Probability'
                )
                window_size = min(7, len(filtered_trend_df))
                moving_avg = filtered_trend_df['Probability'].rolling(window=window_size, min_periods=1).mean()
                ax.plot(filtered_trend_df['Timestamp'], moving_avg, color='red', label=f'{window_size}-Day Moving Avg')

                ax.set_title("Diabetes Risk Trend")
                ax.set_ylabel("Risk Probability")
                ax.legend()
                if st.session_state.get("settings", {}).get("theme", "Light") == "Dark":
                    ax.set_facecolor('#2c3e50')
                    fig.set_facecolor('#2c3e50')
                    ax.tick_params(colors='#ecf0f1')
                    ax.xaxis.label.set_color('#ecf0f1')
                    ax.yaxis.label.set_color('#ecf0f1')
                    ax.title.set_color('#ecf0f1')
                    ax.legend().set_facecolor('#34495e')
                    ax.legend().set_edgecolor('#34495e')
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

    # Add Footer with Modern Styling
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
    .card, .health-card { 
        background-color: #FFFFFF; 
        border-radius: 15px; 
        padding: 20px; 
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px; 
    }
    @media (prefers-color-scheme: dark) { 
        .card, .health-card { 
            background-color: #2D3748; 
            color: #F9F5F0; 
        } 
    }
    .card:hover, .health-card:hover { transform: scale(1.02); transition: transform 0.2s ease-in-out; }
    .doctor-card { 
        background-color: #FFFFFF; 
        border-radius: 15px; 
        padding: 20px; 
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px; 
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
        transition: transform 0.2s, background-color 0.2s; 
    }
    .doctors-site { background-color: #2c3e50; }
    .whatsapp-button:hover, .doctors-site:hover {
        transform: scale(1.05); 
        background-color: #1DA851; 
    }
    .doctors-site:hover { background-color: #23374d; }
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
    .metric-card {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background: #f8f9fa;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card { background: #34495e; }
    }
    .value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
    @media (prefers-color-scheme: dark) { .value { color: #ecf0f1; } }
    .centered-title {
        text-align: center;
        margin-left: 10%;
        font-size: 2em;
        color: #2c3e50;
    }
    @media (prefers-color-scheme: dark) {
        .centered-title { color: #ecf0f1; }
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()