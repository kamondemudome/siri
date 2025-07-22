from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(model, X_val, y_val, X_test, y_test):
    try:
        # Verify model output structure
        sample_input = X_val.iloc[:1]
        prob_output = model.predict_proba(sample_input)
        print(f"Sample predict_proba output shape: {prob_output.shape}")
        print(f"Sample predict_proba output: {prob_output}")
        if prob_output.shape != (1, 2):
            raise ValueError("Model predict_proba output shape is not (n_samples, 2). Expected binary classification output.")
        
        # Get predicted probabilities
        y_val_prob = model.predict_proba(X_val)[:, 1]
        
        # Test multiple thresholds and find the best F1-score
        thresholds = [0.3, 0.4, 0.5]
        best_f1 = 0
        best_threshold = 0.4
        for threshold in thresholds:
            y_val_pred = (y_val_prob >= threshold).astype(int)
            current_f1 = f1_score(y_val, y_val_pred)
            print(f"\nValidation Metrics (threshold={threshold}):")
            print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
            print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
            print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
            print(f"F1-Score: {current_f1:.4f}")
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        
        print(f"\nBest threshold based on F1-score: {best_threshold} (F1: {best_f1:.4f})")
        
        # Evaluate on test set with the best threshold
        y_test_prob = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= best_threshold).astype(int)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"\nTest Metrics (threshold={best_threshold}):")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
        
        if test_accuracy < 0.7:
            print("Warning: Test accuracy is below 0.7. The model may not perform reliably in the app.")
        
        return test_accuracy
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    from model_training_new import train_model
    try:
        model, _, X_val, X_test, _, y_val, y_test = train_model('D:/Myproject/new_dataset_cleaned.csv')
        print("\nModel Evaluation:")
        evaluate_model(model, X_val, y_val, X_test, y_test)
        print("Evaluation completed successfully.")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")