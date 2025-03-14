from src.process_data import get_train_data
from models.random_tree.random_forest.model import get_rf_classifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score

# for progression bar visualisation
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# PARALLELISE TRAINING
from joblib import parallel_backend

def train_rf(n_estimators=300, max_depth=20, min_samples_leaf=15, max_features="sqrt", class_weight="balanced_subsample", feature_engineering=False):
    """
    Trains a Random Forest classifier incrementally while displaying a progress bar.
    
    Parameters:
    n_estimators (int): Number of trees in the forest.
    max_depth (int): Maximum depth of each tree.
    min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
    max_features (str or int): Number of features to consider for the best split.
    class_weight (str or dict): Weights associated with classes to handle imbalance.
    feature_engineering (bool): Whether to use feature engineering when loading data.
    
    Returns:
    None
    """
    
    # Load training and testing data
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=feature_engineering)
    print("Dataset retrieved!")

    # Initialize the Random Forest classifier with given hyperparameters
    rf_classifier = get_rf_classifier(
        n_estimators=n_estimators,
        warm_start=True,  # Allows incremental learning
        n_jobs=-1,  # Utilize all available CPU cores
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=42
    )

    print("Starting training...")
    
    # Train incrementally with a progress bar
    with tqdm(total=n_estimators, desc="Training Progress") as pbar:
        for i in range(1, n_estimators + 1):
            rf_classifier.n_estimators = i  # Increment the number of trees
            with parallel_backend('loky'):  # Parallel processing for efficiency
                rf_classifier.fit(X_train, y_train)
            pbar.update(1)  # Update the progress bar
    
    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)
    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Print results
    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)
    print(f"\nBalanced Accuracy: {balanced_accuracy:.4f}")
    print(f"AUC-ROC Score: {auc_roc:.4f}\n")  # Important for class imbalance
    print(f"PR AUC Score: {pr_auc:.4f}\n")  # Important for class imbalance

    return accuracy, balanced_accuracy