import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data_path):
    # Load cleaned data
    df = pd.read_csv(data_path)
    
    # Features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize and train the model with class weighting
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(rf_model, 'diabetes_rf_model.pkl')
    
    return rf_model, X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    model, *_ = train_model('pima_cleaned.csv')
    print("Model trained and saved as 'diabetes_rf_model.pkl'")