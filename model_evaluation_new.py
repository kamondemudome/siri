from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import numpy as np

def evaluate_model(model, X_val, y_val, X_test, y_test):
    # Get predicted probabilities
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    # Test multiple thresholds
    thresholds = [0.3, 0.4, 0.5]
    for threshold in thresholds:
        y_val_pred = (y_val_prob >= threshold).astype(int)
        print(f"\nValidation Metrics (threshold={threshold}):")
        print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
        print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
        print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
        print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")
    
    # Choose the best threshold (e.g., based on F1-score) and evaluate on test set
    best_threshold = 0.4  # Adjust based on validation results
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)
    print(f"\nTest Accuracy (threshold={best_threshold}): {accuracy_score(y_test, y_test_pred):.4f}")

if __name__ == "__main__":
    from model_training_new import train_model
    model, _, X_val, X_test, _, y_val, y_test = train_model('new_dataset_cleaned.csv')
    print("\nModel Evaluation:")
    evaluate_model(model, X_val, y_val, X_test, y_test)