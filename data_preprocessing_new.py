import pandas as pd
import numpy as np
import os

def load_and_clean_new_data(file_path):
    # Check if input file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' not found. Please ensure it exists.")
    
    try:
        # Load the new dataset
        print(f"Loading dataset from '{file_path}'...")
        df = pd.read_csv(file_path)
        
        # Ensure column names for required 10 features + target
        expected_columns = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'GenHlth', 'Smoker', 
                           'PhysActivity', 'Fruits', 'Veggies', 'Age', 'Income']
        if not all(col in df.columns for col in expected_columns):
            missing_cols = [col for col in expected_columns if col not in df.columns]
            raise ValueError(f"Missing expected columns: {missing_cols}")
        df = df[expected_columns]  # Keep only the 10 features + target
        
        # Check for missing values
        print("Missing values before cleaning:\n", df.isnull().sum())
        
        # Replace invalid values (e.g., BMI = 0 or negative) with NaN and impute with median
        df['BMI'] = df['BMI'].apply(lambda x: np.nan if x <= 0 else x)
        if df['BMI'].isnull().any():
            median_bmi = df['BMI'].median()
            df['BMI'] = df['BMI'].fillna(median_bmi)
            print(f"Imputed {df['BMI'].isnull().sum()} missing BMI values with median: {median_bmi}")
        
        # Ensure valid ranges for features
        feature_ranges = {
            'HighBP': (0, 1), 'HighChol': (0, 1), 'BMI': (10, 100),
            'GenHlth': (1, 5), 'Smoker': (0, 1), 'PhysActivity': (0, 1),
            'Fruits': (0, 1), 'Veggies': (0, 1), 'Age': (1, 13), 'Income': (1, 8)
        }
        for feature, (min_val, max_val) in feature_ranges.items():
            invalid_mask = (df[feature] < min_val) | (df[feature] > max_val)
            if invalid_mask.any():
                print(f"Found {invalid_mask.sum()} invalid values for {feature}. Replacing with median/mode.")
                if feature == 'BMI':
                    df.loc[invalid_mask, feature] = df[feature].median()
                else:
                    df.loc[invalid_mask, feature] = df[feature].mode()[0]
        
        # Ensure correct data types
        df = df.astype({
            'Diabetes_binary': int, 'HighBP': int, 'HighChol': int, 'BMI': float,
            'GenHlth': int, 'Smoker': int, 'PhysActivity': int, 'Fruits': int,
            'Veggies': int, 'Age': int, 'Income': int
        })
        
        # Check class balance
        print(f"Class balance (Diabetes_binary 0/1): {df['Diabetes_binary'].value_counts(normalize=True)}")
        
        # Print row count
        print(f"Total rows in dataset: {len(df)}")
        
        # Save cleaned data
        output_path = 'D:/Myproject/new_dataset_cleaned.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"New data cleaned and saved as '{output_path}' with {len(df)} rows")
        return df
    
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    df = load_and_clean_new_data('D:/Myproject/combined_dataset.csv')