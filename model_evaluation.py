from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def evaluate_model(model, X_val, y_val, X_test, y_test):
    y_val_pred = model.predict(X_val)
    print("Validation Metrics:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
    print(f"F1-Score: {f1_score(y_val, y_val_pred):.4f}")
    y_test_pred = model.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

if __name__ == "__main__":
    from model_training import train_model
    model, _, X_val, X_test, _, y_val, y_test = train_model('pima_cleaned.csv')
    print("\nModel Evaluation:")
    evaluate_model(model, X_val, y_val, X_test, y_test)