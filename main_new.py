import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set to your CPU's core count

from data_preprocessing_new import load_and_clean_new_data
from model_training_new import train_model
from model_evaluation_new import evaluate_model

def main():
    try:
        # Define paths
        input_data_path = "D:/Myproject/combined_dataset.csv"
        cleaned_data_path = "D:/Myproject/new_dataset_cleaned.csv"
        
        # Ensure output directory exists
        os.makedirs("D:/Myproject", exist_ok=True)
        
        # Process combined dataset
        print("Starting data preprocessing...")
        df = load_and_clean_new_data(input_data_path)
        print(f"Total rows in cleaned dataset: {len(df)}")
        
        # Train model
        print("Starting model training...")
        model, X_train, X_val, X_test, y_train, y_val, y_test = train_model(cleaned_data_path)
        
        # Evaluate model
        print("\nModel Evaluation:")
        evaluate_model(model, X_val, y_val, X_test, y_test)
        print("Model creation and evaluation completed successfully.")
        
    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()