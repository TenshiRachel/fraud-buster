from src.process_data import get_train_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score

# improve fields balancing
import pandas as pd
from imblearn.combine import SMOTETomek  # Improved balancing
import os

# for progression bar visualisation
from tqdm import tqdm
import time  # Just for simulation purposes

import warnings
warnings.filterwarnings('ignore')

# Use SMOTE + Tomek Links for better resampling
smote_tomek = SMOTETomek(random_state=42)

def resample_save(X_train, y_train):
    print("Applying SMOTE + Tomek...")
    with tqdm(total=2, desc="Processing") as pbar:
        time.sleep(1)  # Simulating SMOTE step
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
        pbar.update(1)

        time.sleep(1)  # Simulating Tomek step
        pbar.update(1)

    print(f"this is X_train_resampled: {X_train_resampled} and y_train_resampled: {y_train_resampled}")

    # Convert to DataFrame
    X_train_resampled_df = pd.DataFrame(X_train_resampled)
    y_train_resampled_df = pd.DataFrame(y_train_resampled)
    
    print("Saving Resampled data into CSV...")
    # Save to CSV
    X_train_resampled_df.to_csv("./src/X_train_resampled.csv", index=False)
    y_train_resampled_df.to_csv("./src/y_train_resampled.csv", index=False)

    print("✅ Resampled data saved to CSV.")

    return X_train_resampled, y_train_resampled


def get_resample():
    # Load back the data
    X_train_resampled = pd.read_csv("./src/X_train_resampled.csv")
    y_train_resampled = pd.read_csv("./src/y_train_resampled.csv")

    print("✅ Resampled data loaded successfully.")

    return X_train_resampled, y_train_resampled


def train_rf(n_estimators=100):
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=True)
    print("got train data")
    print(f"this is y_train: {y_train} and y_test: {y_test}")

    # balance the attributes
    if os.path.exists("./src/X_train_resampled.csv") and os.path.exists("./src/y_train_resampled.csv"):
        print("Loading existing resampled data...")
        X_train_resampled, y_train_resampled = get_resample()
        print("dataset retrieved!")
        print(f"this is x_resampled: {X_train_resampled} and y_resampled: {y_train_resampled}")
        
    else:
        X_train_resampled, y_train_resampled = resample_save(X_train, y_train)

    
    # specify fields
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,  # more trees for better stability
        class_weight="balanced",  # adjust class weight impact (either balanced or balanced_subsample)
        max_depth=25,  # prevent overfitting
        min_samples_leaf=5,  # reduce variance
        max_features="sqrt",  # use sqrt features per split
        random_state=42
    )

    print("Starting training...")
    for i in tqdm(range(1, n_estimators + 1), desc="Training Progress"):
        rf_classifier.n_estimators = i  # incrementally increase trees
        rf_classifier.fit(X_train_resampled, y_train_resampled)
  
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
    print(f"\nAUC-ROC Score: {auc_roc:.4f}")  # Important for class imbalance
