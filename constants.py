# constants.py

# Path to the shared settings file and model
SETTINGS_FILE = "D:/Myproject/settings.json"
MODEL_PATH = "D:/Myproject/new_diabetes_rf_model.pkl"

# Features used for diabetes prediction
FEATURES = [
    'HighBP', 'HighChol', 'BMI', 'GenHlth', 'Smoker',
    'PhysActivity', 'Fruits', 'Veggies', 'Age', 'Income'
]

# Feature ranges for validation
FEATURE_RANGES = {
    'HighBP': (0, 1), 'HighChol': (0, 1), 'BMI': (10, 100),
    'GenHlth': (1, 5), 'Smoker': (0, 1), 'PhysActivity': (0, 1),
    'Fruits': (0, 1), 'Veggies': (0, 1), 'Age': (1, 13), 'Income': (1, 8)
}

# Full names for features
FEATURE_FULL_NAMES = {
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'BMI': 'Body Mass Index (BMI)',
    'GenHlth': 'General Health Rating',
    'Smoker': 'Smoking History (100+ Cigarettes)',
    'PhysActivity': 'Physical Activity in Past 30 Days',
    'Fruits': 'Daily Fruit Consumption',
    'Veggies': 'Daily Vegetable Consumption',
    'Age': 'Age Category',
    'Income': 'Income Category'
}

# Descriptions for input fields
FEATURE_DESCRIPTIONS = {
    'HighBP': 'Do you have high blood pressure?',
    'HighChol': 'Do you have high cholesterol?',
    'BMI': 'What is your Body Mass Index (BMI)?',
    'GenHlth': 'How would you rate your general health? (1 = Excellent, 5 = Poor)',
    'Smoker': 'Have you smoked at least 100 cigarettes in your life?',
    'PhysActivity': 'Have you done physical activity in the past 30 days?',
    'Fruits': 'Do you consume fruit 1 or more times per day?',
    'Veggies': 'Do you consume vegetables 1 or more times per day?',
    'Age': 'What is your age category?',
    'Income': 'What is your income category?'
}

# Tooltips for input fields
FEATURE_TOOLTIPS = {
    'HighBP': 'Select "Yes" if you have been diagnosed with high blood pressure.',
    'HighChol': 'Select "Yes" if you have been diagnosed with high cholesterol.',
    'BMI': 'Enter your BMI (e.g., 25.0). BMI = weight (kg) / height (m)^2.',
    'GenHlth': 'Rate your overall health on a scale from 1 (excellent) to 5 (poor).',
    'Smoker': 'Select "Yes" if you have smoked at least 100 cigarettes in your lifetime.',
    'PhysActivity': 'Select "Yes" if you’ve done any physical activity (e.g., walking, exercise) in the past 30 days.',
    'Fruits': 'Select "Yes" if you eat fruit at least once per day.',
    'Veggies': 'Select "Yes" if you eat vegetables at least once per day.',
    'Age': 'Select your age category (e.g., 1 = 18-24, 13 = 80+).',
    'Income': 'Select your annual income category (e.g., 1 = < Ksh 10000, 8 = Ksh 1000000+).'
}

# Age group labels
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"
}

# Income level labels
INCOME_LABELS = {
    1: "< Ksh 10000", 2: "Ksh 10000-15000", 3: "Ksh 15000-25000", 4: "Ksh 25000-35000",
    5: "Ksh 35000-50000", 6: "Ksh 50000-75000", 7: "Ksh 75000-100000", 8: "Ksh 1000000+"
}

# Education level labels (included for completeness, though unused in current app)
EDUCATION_LABELS = {
    1: "Never attended school", 2: "Grades 1-8", 3: "Grades 9-11", 4: "High school graduate",
    5: "Some college", 6: "College graduate"
}