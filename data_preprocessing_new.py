import pandas as pd
import numpy as np

def load_and_clean_new_data(file_path):
    # Load the new dataset
    df = pd.read_csv(file_path)
    
    # Ensure column names
    expected_columns = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
                        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 
                        'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    df.columns = expected_columns
    
    # Check for missing values
    print("Missing values:\n", df.isnull().sum())
    
    # Replace invalid values (e.g., BMI = 0) with NaN and impute with median
    df['BMI'] = df['BMI'].replace(0, np.nan)
    df['BMI'] = df['BMI'].fillna(df['BMI'].median())
    
    # Ensure correct data types
    df = df.astype({
        'Diabetes_binary': int, 'HighBP': int, 'HighChol': int, 'CholCheck': int, 'BMI': float,
        'Smoker': int, 'Stroke': int, 'HeartDiseaseorAttack': int, 'PhysActivity': int, 
        'Fruits': int, 'Veggies': int, 'HvyAlcoholConsump': int, 'AnyHealthcare': int, 
        'NoDocbcCost': int, 'GenHlth': int, 'MentHlth': int, 'PhysHlth': int, 'DiffWalk': int, 
        'Sex': int, 'Age': int, 'Education': int, 'Income': int
    })
    
    # Check class balance
    print(f"Class balance (Diabetes_binary 0/1): {df['Diabetes_binary'].value_counts(normalize=True)}")
    
    # Print row count
    print(f"Total rows in dataset: {len(df)}")
    
    # Save cleaned data
    df.to_csv('new_dataset_cleaned.csv', index=False)
    return df

if __name__ == "__main__":
    df = load_and_clean_new_data('new_dataset.csv')
    print(f"New data cleaned and saved as 'new_dataset_cleaned.csv' with {len(df)} rows")