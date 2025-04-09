import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import joblib

def train_model(data_path):
    try:
        # Load cleaned data
        df = pd.read_csv(data_path)
        
        # Features and target
        X = df.drop('Diabetes_binary', axis=1)
        y = df['Diabetes_binary']
        
        # Drop low-importance features
        low_importance_features = ['HeartDiseaseorAttack', 'AnyHealthcare', 'Stroke', 'DiffWalk']
        X = X.drop(columns=low_importance_features)
        print(f"Dropped low-importance features: {low_importance_features}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Apply SMOTE with a custom sampling strategy (e.g., 3:1 negative to positive)
        smote = SMOTE(sampling_strategy=0.333, random_state=42)  # 0.333 means 1 positive for every 3 negatives
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Training set class balance after SMOTE: {pd.Series(y_train_resampled).value_counts(normalize=True)}")
        print(f"Training set size after SMOTE: {len(X_train_resampled)} rows")
        
        # Subsample the SMOTE-balanced data for tuning (10% to speed up)
        np.random.seed(42)  # Set seed for reproducibility
        subsample_idx = np.random.choice(len(X_train_resampled), size=int(0.1 * len(X_train_resampled)), replace=False)
        X_train_subsample = X_train_resampled.iloc[subsample_idx]
        y_train_subsample = y_train_resampled.iloc[subsample_idx]
        print(f"Subsampled training set size for tuning: {len(X_train_subsample)} rows")
        
        # Define parameter distribution for RandomizedSearchCV
        param_dist = {
            'n_estimators': randint(50, 150),
            'max_depth': randint(8, 15),
            'min_samples_split': randint(2, 6),
            'min_samples_leaf': randint(1, 3),
            'max_features': ['sqrt', 'log2', None]  # Added max_features
        }
        
        # Initialize model with custom class weights
        class_weight = {0: 1, 1: 2}  # Give more weight to the positive class
        rf = RandomForestClassifier(class_weight=class_weight, random_state=42, n_jobs=1)
        
        # Randomized search
        random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, scoring='f1', random_state=42, n_jobs=1)
        random_search.fit(X_train_subsample, y_train_subsample)
        
        print("Best parameters from RandomizedSearchCV:", random_search.best_params_)
        rf_model = RandomForestClassifier(**random_search.best_params_, class_weight=class_weight, random_state=42, n_jobs=1)
        rf_model.fit(X_train_resampled, y_train_resampled)  # Train on full SMOTE data with best parameters
        
        # Print feature importances
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print("\nFeature Importances:\n", feature_importances)
        
        # Save the model
        joblib.dump(rf_model, 'new_diabetes_rf_model.pkl')
        
        return rf_model, X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    model, *_ = train_model('new_dataset_cleaned.csv')
    print("Model trained and saved as 'new_diabetes_rf_model.pkl'")