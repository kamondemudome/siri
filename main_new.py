import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set to your CPU's core count

from data_preprocessing_new import load_and_clean_new_data
from model_training_new import train_model
from model_evaluation_new import evaluate_model

def main():
    # Process combined dataset
    df = load_and_clean_new_data('combined_dataset.csv')
    print(f"Total rows in cleaned dataset: {len(df)}")
    model, X_train, X_val, X_test, y_train, y_val, y_test = train_model('new_dataset_cleaned.csv')
    print("\nModel Evaluation:")
    evaluate_model(model, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()