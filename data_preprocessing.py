import pandas as pd
import numpy as np

def load_and_clean_data(file_path1, file_path2, file_path3, file_path4):
    df_pima1 = pd.read_csv(file_path1)
    df_pima4 = pd.read_csv(file_path2)
    df_diabetes = pd.read_csv(file_path3)
    df_p1 = pd.read_csv(file_path4)
    
    expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df_pima1.columns = expected_columns
    df_pima4.columns = expected_columns
    df_diabetes.columns = expected_columns
    df_p1.columns = expected_columns
    
    # Combine datasets
    df = pd.concat([df_pima1, df_pima4, df_diabetes, df_p1], ignore_index=True)
    print(f"Total rows before deduplication: {len(df)}")
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"Total rows after deduplication: {len(df)}")
    
    # Replace 0s with NaN
    for col in ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']:
        df[col] = df[col].replace(0, np.nan)
    
    # Impute missing values with median
    for col in ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']:
        df[col] = df[col].fillna(df[col].median())
    
    # Ensure correct data types
    df = df.astype({
        'Pregnancies': int, 'Glucose': float, 'BloodPressure': float, 
        'SkinThickness': float, 'Insulin': float, 'BMI': float, 
        'DiabetesPedigreeFunction': float, 'Age': int, 'Outcome': int
    })
    
    df.to_csv('pima_cleaned.csv', index=False)
    return df

if __name__ == "__main__":
    df = load_and_clean_data('pima1.csv', 'pima4.csv', 'diabetes.csv', 'p1.csv')
    print(f"Combined data cleaned and saved as 'pima_cleaned.csv' with {len(df)} rows")