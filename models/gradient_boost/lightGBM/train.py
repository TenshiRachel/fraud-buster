from src.process_data import get_train_data
from models.gradient_boost.lightGBM.model import lightgbm_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Hyperparameters
max_depth = 1
num_estimators = 50
learning_rate = 0.1


def train_lgbm():
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42)

    model = lightgbm_model(max_depth=max_depth, num_estimators=num_estimators, learning_rate=learning_rate)
    
    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability for positive class

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    return y_test, y_pred, y_prob

# from src.process_data import get_train_data
# from models.lightGBM.model import lightgbm_model
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from skopt import BayesSearchCV
# import lightgbm as lgb

# # Define hyperparameter search space
# param_space = {
#     'num_leaves': (16, 64),  # Reduce complexity
#     'max_depth': (3, 8),  # Avoid excessive depth
#     'learning_rate': (0.01, 0.2, 'log-uniform'),
#     'n_estimators': (100, 500),  # Limit boosting rounds
#     'lambda_l1': (0, 1),  
#     'lambda_l2': (0, 1),  
#     'min_data_in_leaf': (5, 50),  # Lower min leaf samples
#     'scale_pos_weight': (10, 50)  # Handle class imbalance
# }

# def train_lgbm():
#     # Load training data
#     X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42)

#     # Initialize LightGBM model
#     base_model = lgb.LGBMClassifier(objective='binary', class_weight='balanced', force_row_wise=True, random_state=42)

#     # Perform Bayesian Optimization
#     bayes_search = BayesSearchCV(
#     base_model,
#     param_space,
#     n_iter=20,  # Reduce if it's running indefinitely
#     cv=3,
#     scoring='roc_auc',
#     n_jobs=-1,
#     verbose=1,
#     error_score=0  # Skip broken iterations
#     )

#     # Train with hyperparameter tuning
#     bayes_search.fit(X_train, y_train)

#     # Get best model with tuned parameters
#     best_model = bayes_search.best_estimator_

#     # Predictions
#     y_pred = best_model.predict(X_test)
#     y_prob = best_model.predict_proba(X_test)[:, 1]  # Probability for positive class

#     # Evaluation Metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_prob)

#     print(f"\n🔍 Best Hyperparameters: {bayes_search.best_params_}")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"AUC-ROC: {roc_auc:.4f}")

#     return y_test, y_pred, y_prob, best_model