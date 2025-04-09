from data_preprocessing import load_and_clean_data
from model_training import train_model
from model_evaluation import evaluate_model

def main():
    # Process combined dataset with four files
    df = load_and_clean_data('pima1.csv', 'pima4.csv', 'diabetes.csv', 'p1.csv')
    model, X_train, X_val, X_test, y_train, y_val, y_test = train_model('pima_cleaned.csv')
    print("\nModel Evaluation:")
    evaluate_model(model, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()