import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import joblib
import os
from sklearn.metrics import accuracy_score

# Set LOKY_MAX_CPU_COUNT to avoid Windows warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust based on your CPU cores

def train_model(data_path):
    try:
        # Check if input file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input file '{data_path}' not found. Please ensure it exists.")
        
        print("Loading cleaned data...")
        df = pd.read_csv(data_path)
        
        # Features and target (limited to the 10 required features)
        features = ['HighBP', 'HighChol', 'BMI', 'GenHlth', 'Smoker', 
                    'PhysActivity', 'Fruits', 'Veggies', 'Age', 'Income']
        if not all(col in df.columns for col in features + ['Diabetes_binary']):
            missing_cols = [col for col in features + ['Diabetes_binary'] if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
        X = df[features]
        y = df['Diabetes_binary']
        
        # Verify feature data types and ranges
        feature_ranges = {
            'HighBP': (0, 1), 'HighChol': (0, 1), 'BMI': (10, 100),
            'GenHlth': (1, 5), 'Smoker': (0, 1), 'PhysActivity': (0, 1),
            'Fruits': (0, 1), 'Veggies': (0, 1), 'Age': (1, 13), 'Income': (1, 8)
        }
        for feature, (min_val, max_val) in feature_ranges.items():
            invalid_mask = (X[feature] < min_val) | (X[feature] > max_val)
            if invalid_mask.any():
                print(f"Warning: {invalid_mask.sum()} invalid values for {feature}. Replacing with median/mode.")
                if feature == 'BMI':
                    X.loc[invalid_mask, feature] = X[feature].median()
                else:
                    X.loc[invalid_mask, feature] = X[feature].mode()[0]
        
        # Split data
        print("Splitting data into train, validation, and test sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Apply SMOTE with a custom sampling strategy
        print("Applying SMOTE for class balance...")
        smote = SMOTE(sampling_strategy=0.333, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Training set class balance after SMOTE: {pd.Series(y_train_resampled).value_counts(normalize=True)}")
        print(f"Training set size after SMOTE: {len(X_train_resampled)} rows")
        
        # Subsample the SMOTE-balanced data for tuning (5% to speed up)
        print("Creating subsample for hyperparameter tuning...")
        np.random.seed(42)
        subsample_idx = np.random.choice(len(X_train_resampled), size=int(0.05 * len(X_train_resampled)), replace=False)
        X_train_subsample = X_train_resampled.iloc[subsample_idx]
        y_train_subsample = y_train_resampled.iloc[subsample_idx]
        print(f"Subsampled training set size for tuning: {len(X_train_subsample)} rows")
        
        # Define parameter distribution for RandomizedSearchCV
        param_dist = {
            'n_estimators': randint(50, 150),
            'max_depth': randint(8, 15),
            'min_samples_split': randint(2, 6),
            'min_samples_leaf': randint(1, 3),
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Initialize model with custom class weights
        class_weight = {0: 1, 1: 2}
        rf = RandomForestClassifier(class_weight=class_weight, random_state=42, n_jobs=-1)
        
        # Randomized search
        print("Starting hyperparameter tuning with RandomizedSearchCV...")
        random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, scoring='f1', random_state=42, n_jobs=-1)
        random_search.fit(X_train_subsample, y_train_subsample)
        
        print("Hyperparameter tuning completed. Best parameters:", random_search.best_params_)
        rf_model = RandomForestClassifier(**random_search.best_params_, class_weight=class_weight, random_state=42, n_jobs=-1)
        print("Training model on full resampled data...")
        rf_model.fit(X_train_resampled, y_train_resampled)
        
        # Validate model on validation set
        y_val_pred = rf_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        if val_accuracy < 0.7:
            print("Warning: Model validation accuracy is below 0.7. Consider adjusting hyperparameters or data.")
        
        # Print feature importances
        feature_importances = pd.DataFrame({
            'Feature': features,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print("\nFeature Importances:\n", feature_importances)
        
        # Save the model to the correct path
        output_path = 'D:/Myproject/new_diabetes_rf_model.pkl'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving model to '{output_path}'...")
        joblib.dump(rf_model, output_path)
        print(f"Model trained and saved as '{output_path}'")
        
        # Verify model output structure
        sample_input = X_val.iloc[:1]
        prob_output = rf_model.predict_proba(sample_input)
        print(f"Sample predict_proba output shape: {prob_output.shape}")
        print(f"Sample predict_proba output: {prob_output}")
        if prob_output.shape != (1, 2):
            print("Warning: Model predict_proba output shape is not (n_samples, 2). This may cause issues in the app.")
        
        return rf_model, X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    model, *_ = train_model('D:/Myproject/new_dataset_cleaned.csv')
    print("Model training and saving completed.")