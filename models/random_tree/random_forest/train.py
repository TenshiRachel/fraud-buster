from src.process_data import get_train_data
from models.random_tree.random_forest.model import get_rf_classifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score

# for progression bar visualisation
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# PARALLELISE TRAINING
from joblib import parallel_backend

def train_rf(feature_eng):
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=feature_eng)
    
    n_estimators = 500
    rf_classifier = get_rf_classifier()

    print("Starting training...")
    # Train incrementally with a progress bar
    with tqdm(total=n_estimators, desc="Training Progress") as pbar:
        for i in range(1, n_estimators + 1):
            rf_classifier.n_estimators = i  # Increment tree count
            with parallel_backend('loky'):  # Parallel processing
                rf_classifier.fit(X_train, y_train)
            pbar.update(1)  # Update progress bar
  
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    # Print results
    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)
    print(f"\nBalanced Accuracy: {balanced_accuracy:.4f}")
    print(f"AUC-ROC Score: {auc_roc:.4f}\n")  # Important for class imbalance

    return accuracy, balanced_accuracy
