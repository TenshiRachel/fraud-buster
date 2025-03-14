from src.process_data import get_train_data
from src.eval import print_metrics
from models.gradient_boost.lightGBM.model import lightgbm_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, average_precision_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import numpy as np


# Hyperparameters
max_depth = 7
num_leaves = 100
num_estimators = 500
learning_rate = 0.1


def train_lgbm(feature_engineering):
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=feature_engineering) 

    model = lightgbm_model(max_depth=max_depth, num_leaves = num_leaves, n_estimators=num_estimators, learning_rate=learning_rate) 
    
    # Train model
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability for positive class

    # Accuracy Metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Evaluation Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    roc_pr = average_precision_score(y_test, y_prob)
    class_report = classification_report(y_test, y_pred)

    # Print metrics
    print_metrics(accuracy, balanced_accuracy, roc_auc, roc_pr, class_report)


# Hyperparameter tuning
def find_best_hyperparameters():
    # Load dataset with feature engineering enabled
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=True)
    
    # Convert y_train to 1D array if it's a DataFrame or column vector
    if hasattr(y_train, 'values'):
        y_train = y_train.values.ravel()  # For pandas Series/DataFrame
    elif isinstance(y_train, np.ndarray) and y_train.ndim > 1:
        y_train = y_train.ravel()  # For numpy arrays with more than 1 dimension
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Class distribution: {np.bincount(y_train.astype(int))}")
    
    # Define LightGBM classifier
    lgbm = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='dart',
        random_state=42,
        n_jobs=-1,
    )

    # Define the hyperparameter grid
    param_grid = {
        'num_leaves': [50, 100, 150],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.1],  
        'n_estimators': [100, 200, 500],
        'scale_pos_weight': [5, 10, 20]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='balanced_accuracy'
    )

    # Fit GridSearchCV
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)

    # Print best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_