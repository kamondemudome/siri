import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('new_diabetes_rf_model.pkl')

# Define the features and their expected ranges
FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'PhysActivity', 'Fruits', 
    'Veggies', 'HvyAlcoholConsump', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
    'Sex', 'Age', 'Education', 'Income'
]

FEATURE_DESCRIPTIONS = {
    'HighBP': 'Do you have high blood pressure? (0 = No, 1 = Yes)',
    'HighChol': 'Do you have high cholesterol? (0 = No, 1 = Yes)',
    'CholCheck': 'Have you had a cholesterol check in the last 5 years? (0 = No, 1 = Yes)',
    'BMI': 'What is your Body Mass Index (BMI)? (e.g., 26.0)',
    'Smoker': 'Have you smoked at least 100 cigarettes in your life? (0 = No, 1 = Yes)',
    'PhysActivity': 'Have you done physical activity in the past 30 days? (0 = No, 1 = Yes)',
    'Fruits': 'Do you consume fruit 1 or more times per day? (0 = No, 1 = Yes)',
    'Veggies': 'Do you consume vegetables 1 or more times per day? (0 = No, 1 = Yes)',
    'HvyAlcoholConsump': 'Do you engage in heavy alcohol consumption? (0 = No, 1 = Yes)',
    'NoDocbcCost': 'Were you unable to see a doctor due to cost in the past 12 months? (0 = No, 1 = Yes)',
    'GenHlth': 'How would you rate your general health? (1 = Excellent, 5 = Poor)',
    'MentHlth': 'How many days in the past 30 days was your mental health not good? (0-30)',
    'PhysHlth': 'How many days in the past 30 days was your physical health not good? (0-30)',
    'Sex': 'What is your sex? (0 = Female, 1 = Male)',
    'Age': 'What is your age category? (1 = 18-24, 2 = 25-29, ..., 13 = 80+)',
    'Education': 'What is your education level? (1 = Never attended, 2 = Grades 1-8, ..., 6 = College graduate)',
    'Income': 'What is your income category? (1 = < $10,000, 2 = $10,000-$15,000, ..., 8 = $75,000+)'
}

FEATURE_RANGES = {
    'HighBP': (0, 1), 'HighChol': (0, 1), 'CholCheck': (0, 1), 'Smoker': (0, 1),
    'PhysActivity': (0, 1), 'Fruits': (0, 1), 'Veggies': (0, 1), 'HvyAlcoholConsump': (0, 1),
    'NoDocbcCost': (0, 1), 'Sex': (0, 1), 'GenHlth': (1, 5), 'MentHlth': (0, 30),
    'PhysHlth': (0, 30), 'Age': (1, 13), 'Education': (1, 6), 'Income': (1, 8),
    'BMI': (10, 100)  # Reasonable range for BMI
}

def get_user_input():
    user_data = {}
    print("Please enter the following information to predict your diabetes risk:\n")
    
    for feature in FEATURES:
        while True:
            try:
                value = float(input(f"{FEATURE_DESCRIPTIONS[feature]}: "))
                min_val, max_val = FEATURE_RANGES[feature]
                if min_val <= value <= max_val:
                    user_data[feature] = value
                    break
                else:
                    print(f"Value must be between {min_val} and {max_val}. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    return user_data

def predict_diabetes(user_data):
    # Convert user data to DataFrame
    input_df = pd.DataFrame([user_data], columns=FEATURES)
    
    # Ensure correct data types
    input_df = input_df.astype({
        'HighBP': int, 'HighChol': int, 'CholCheck': int, 'Smoker': int,
        'PhysActivity': int, 'Fruits': int, 'Veggies': int, 'HvyAlcoholConsump': int,
        'NoDocbcCost': int, 'Sex': int, 'GenHlth': int, 'MentHlth': int,
        'PhysHlth': int, 'Age': int, 'Education': int, 'Income': int,
        'BMI': float
    })
    
    # Predict probability
    prob = model.predict_proba(input_df)[:, 1][0]
    
    # Apply threshold
    threshold = 0.4  # Based on the best threshold from validation
    prediction = 1 if prob >= threshold else 0
    
    return prob, prediction

def explain_prediction(prob, prediction, user_data):
    print("\nPrediction Results:")
    print(f"Probability of diabetes: {prob:.2%}")
    print(f"Prediction: {'High risk of diabetes' if prediction == 1 else 'Low risk of diabetes'}")
    
    # Simple explanation based on top features
    top_features = ['GenHlth', 'BMI', 'HighBP', 'Age']  # From feature importances
    print("\nKey factors influencing the prediction:")
    for feature in top_features:
        value = user_data[feature]
        if feature == 'GenHlth' and value >= 4:
            print(f"- Poor general health (GenHlth = {value}) may increase your risk.")
        elif feature == 'BMI' and value >= 30:
            print(f"- High BMI ({value}) may increase your risk.")
        elif feature == 'HighBP' and value == 1:
            print(f"- High blood pressure may increase your risk.")
        elif feature == 'Age' and value >= 9:  # Age 55+
            print(f"- Older age (category {value}) may increase your risk.")

def main():
    print("Welcome to the Diabetes Risk Prediction Tool\n")
    
    # Get user input
    user_data = get_user_input()
    
    # Make prediction
    prob, prediction = predict_diabetes(user_data)
    
    # Explain the prediction
    explain_prediction(prob, prediction, user_data)
    
    print("\nNote: This prediction is for informational purposes only. Please consult a healthcare professional for a medical diagnosis.")

if __name__ == "__main__":
    main()