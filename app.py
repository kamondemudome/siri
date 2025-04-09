import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('new_diabetes_rf_model.pkl')

# Define features and descriptions
FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'PhysActivity', 'Fruits', 
    'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
    'Sex', 'Age', 'Education', 'Income'
]

FEATURE_DESCRIPTIONS = {
    'HighBP': 'Do you have high blood pressure?',
    'HighChol': 'Do you have high cholesterol?',
    'CholCheck': 'Have you had a cholesterol check in the last 5 years?',
    'BMI': 'What is your Body Mass Index (BMI)?',
    'Smoker': 'Have you smoked at least 100 cigarettes in your life?',
    'PhysActivity': 'Have you done physical activity in the past 30 days?',
    'Fruits': 'Do you consume fruit 1 or more times per day?',
    'Veggies': 'Do you consume vegetables 1 or more times per day?',
    'HvyAlcoholConsump': 'Do you engage in heavy alcohol consumption?',
    'NoDocbcCost': 'Were you unable to see a doctor due to cost in the past 12 months?',
    'GenHlth': 'How would you rate your general health? (1 = Excellent, 5 = Poor)',
    'MentHlth': 'How many days in the past 30 days was your mental health not good?',
    'PhysHlth': 'How many days in the past 30 days was your physical health not good?',
    'Sex': 'What is your sex?',
    'Age': 'What is your age category?',
    'Education': 'What is your education level?',
    'Income': 'What is your income category?'
}

def predict_diabetes(user_data):
    input_df = pd.DataFrame([user_data], columns=FEATURES)
    input_df = input_df.astype({
        'HighBP': int, 'HighChol': int, 'CholCheck': int, 'Smoker': int,
        'PhysActivity': int, 'Fruits': int, 'Veggies': int, 'HvyAlcoholConsump': int,
        'NoDocbcCost': int, 'Sex': int, 'GenHlth': int, 'MentHlth': int,
        'PhysHlth': int, 'Age': int, 'Education': int, 'Income': int,
        'BMI': float
    })
    prob = model.predict_proba(input_df)[:, 1][0]
    threshold = 0.4
    prediction = 1 if prob >= threshold else 0
    return prob, prediction

def main():
    st.title("Diabetes Risk Prediction Tool")
    st.write("Enter your information below to predict your risk of diabetes.")

    user_data = {}
    with st.form("input_form"):
        for feature in FEATURES:
            if feature in ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'PhysActivity', 'Fruits', 
                           'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost']:
                user_data[feature] = st.selectbox(FEATURE_DESCRIPTIONS[feature], [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
            elif feature == 'Sex':
                user_data[feature] = st.selectbox(FEATURE_DESCRIPTIONS[feature], [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
            elif feature == 'GenHlth':
                user_data[feature] = st.slider(FEATURE_DESCRIPTIONS[feature], 1, 5, 3)
            elif feature in ['MentHlth', 'PhysHlth']:
                user_data[feature] = st.slider(FEATURE_DESCRIPTIONS[feature], 0, 30, 0)
            elif feature == 'Age':
                user_data[feature] = st.selectbox(FEATURE_DESCRIPTIONS[feature], list(range(1, 14)), format_func=lambda x: f"Category {x}")
            elif feature == 'Education':
                user_data[feature] = st.selectbox(FEATURE_DESCRIPTIONS[feature], list(range(1, 7)), format_func=lambda x: f"Level {x}")
            elif feature == 'Income':
                user_data[feature] = st.selectbox(FEATURE_DESCRIPTIONS[feature], list(range(1, 9)), format_func=lambda x: f"Level {x}")
            elif feature == 'BMI':
                user_data[feature] = st.number_input(FEATURE_DESCRIPTIONS[feature], min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        prob, prediction = predict_diabetes(user_data)
        st.write("### Prediction Results")
        st.write(f"**Probability of diabetes:** {prob:.2%}")
        st.write(f"**Prediction:** {'High risk of diabetes' if prediction == 1 else 'Low risk of diabetes'}")
        
        st.write("### Key Factors Influencing the Prediction")
        if user_data['GenHlth'] >= 4:
            st.write(f"- Poor general health (GenHlth = {user_data['GenHlth']}) may increase your risk.")
        if user_data['BMI'] >= 30:
            st.write(f"- High BMI ({user_data['BMI']}) may increase your risk.")
        if user_data['HighBP'] == 1:
            st.write(f"- High blood pressure may increase your risk.")
        if user_data['Age'] >= 9:
            st.write(f"- Older age (category {user_data['Age']}) may increase your risk.")
        
        st.write("\n**Note:** This prediction is for informational purposes only. Please consult a healthcare professional for a medical diagnosis.")

if __name__ == "__main__":
    main()